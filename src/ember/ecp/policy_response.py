"""Frozen full-layer PI0.5 responses in ECP owner coordinates."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.ecp.observer import ActionLayerStateCapture, TargetOwnerProjector
from ember.functional_adaptation.functional_response import FunctionalResponseError
from ember.lora import LoRAContract, functional_lora_call
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
)


@dataclass(frozen=True)
class CapturedPolicyResponse:
    flow: torch.Tensor
    owner_basis: torch.Tensor


@dataclass(frozen=True)
class OwnerResolvedResponseLoss:
    loss: torch.Tensor
    normalized_disagreement: torch.Tensor
    active_owner_fraction: torch.Tensor


@dataclass(frozen=True)
class FrozenOwnerResponseTargets:
    source: torch.Tensor
    experts: torch.Tensor

    def to(self, device: torch.device) -> "FrozenOwnerResponseTargets":
        return FrozenOwnerResponseTargets(
            source=self.source.to(device, non_blocking=True),
            experts=self.experts.to(device, non_blocking=True),
        )


def owner_response_targets_from_payload(
    value: Mapping[str, Any], *, expert_count: int
) -> FrozenOwnerResponseTargets | None:
    fields = tuple(
        value.get(name)
        for name in (
            "source_owner_response",
            "expert_owner_responses",
        )
    )
    if all(field is None for field in fields):
        return None
    if any(field is None for field in fields):
        raise FunctionalResponseError("owner-resolved response payload is partial")
    source, experts = (field.float() for field in fields)
    if (
        source.ndim != 4
        or experts.ndim != 5
        or experts.shape != (expert_count, *source.shape)
    ):
        raise FunctionalResponseError("owner-resolved response payload changed shape")
    return FrozenOwnerResponseTargets(
        source=source,
        experts=experts,
    )


def _dct_basis(*, horizon: int, count: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(horizon, device=device, dtype=torch.float32) + 0.5
    rows = [torch.ones(horizon, device=device) / math.sqrt(horizon)]
    for frequency in range(1, count):
        rows.append(
            math.sqrt(2.0 / horizon)
            * torch.cos(math.pi * frequency * positions / horizon)
        )
    return torch.stack(rows)


def _capture_policy_response(
    *,
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    projector: TargetOwnerProjector,
    policy_seed: int,
    horizon_basis: int,
    detach: bool,
) -> CapturedPolicyResponse:
    """Capture every Action layer before reducing the 50-token horizon."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise FunctionalResponseError("policy-support capture requires a frozen PI0.5")
    if any(parameter.requires_grad for parameter in projector.parameters()):
        raise FunctionalResponseError("policy-support projector must be frozen")
    core = policy.model
    expert = core.paligemma_with_expert.gemma_expert.model
    action_inputs: list[torch.Tensor] = []
    flows: list[torch.Tensor] = []

    def capture_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        action_inputs.append(inputs[0].detach() if detach else inputs[0])

    def capture_flow(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        flows.append(output.detach() if detach else output)

    input_handle = core.action_in_proj.register_forward_pre_hook(capture_input)
    output_handle = core.action_out_proj.register_forward_hook(capture_flow)
    try:
        grad_context = torch.no_grad() if detach else nullcontext()
        with grad_context, scoped_policy_randomness(
            policy_seed, next(policy.parameters()).device
        ):
            with scoped_policy_flow_noise_sampling(
                policy, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME
            ):
                with scoped_policy_flow_time_sampling(
                    policy, INDEPENDENT_BETA_TIME_SAMPLING_SCHEME
                ):
                    with ActionLayerStateCapture(expert, detach=detach) as layers:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            functional_lora_call(policy, state, contract, batch)
        layer_states = layers.stacked()
    finally:
        input_handle.remove()
        output_handle.remove()
    if (
        len(action_inputs) != 1
        or len(flows) != 1
        or action_inputs[0].shape[1:] != (50, 32)
        or flows[0].shape != action_inputs[0].shape
    ):
        raise FunctionalResponseError("policy-support Action response topology changed")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        owner = projector(layer_states, flows[0], action_inputs[0])
    basis = _dct_basis(
        horizon=int(owner.shape[2]), count=horizon_basis, device=owner.device
    )
    compressed = torch.einsum("bohd,ph->bopd", owner.float(), basis)
    if compressed.shape[1:] != (38, horizon_basis, 128):
        raise FunctionalResponseError("policy-support owner response changed shape")
    return CapturedPolicyResponse(flow=flows[0].float(), owner_basis=compressed)


@torch.no_grad()
def capture_policy_response(
    *,
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    projector: TargetOwnerProjector,
    policy_seed: int,
    horizon_basis: int,
) -> CapturedPolicyResponse:
    return _capture_policy_response(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        projector=projector,
        policy_seed=policy_seed,
        horizon_basis=horizon_basis,
        detach=True,
    )


def differentiable_policy_response(
    *,
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    projector: TargetOwnerProjector,
    policy_seed: int,
    horizon_basis: int,
) -> CapturedPolicyResponse:
    """Expose flow and owner-resolved responses with gradients to one LoRA state."""

    return _capture_policy_response(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        projector=projector,
        policy_seed=policy_seed,
        horizon_basis=horizon_basis,
        detach=False,
    )


def owner_resolved_response_distillation_loss(
    *,
    candidate: torch.Tensor,
    source: torch.Tensor,
    experts: torch.Tensor,
    expert_weights: torch.Tensor,
    outcome_weight: float,
) -> OwnerResolvedResponseLoss:
    """Match successful-policy effects in the frozen owner/layer response space."""

    if (
        candidate.ndim != 4
        or source.shape != candidate.shape
        or experts.ndim != 5
        or experts.shape[1:] != candidate.shape
        or expert_weights.shape != (experts.shape[0],)
    ):
        raise FunctionalResponseError("owner-resolved response topology changed")
    weights = expert_weights.to(candidate).float().clamp_min(1e-4)
    weights = weights / weights.sum()
    source = source.to(candidate).float()
    expert_delta = experts.to(candidate).float() - source[None]
    target_delta = torch.einsum("m,mbopd->bopd", weights, expert_delta)
    candidate_delta = candidate.float() - source
    error = (candidate_delta - target_delta).square().mean(dim=(0, 2, 3))
    signal = target_delta.square().mean(dim=(0, 2, 3))
    disagreement = torch.einsum(
        "m,mbopd->bopd",
        weights,
        (expert_delta - target_delta[None]).square(),
    ).mean(dim=(0, 2, 3))
    global_signal = signal.mean().clamp_min(1e-8)
    confidence = signal / (signal + disagreement + 0.05 * global_signal)
    normalized_error = error / (signal + 0.05 * global_signal)
    loss = (
        (confidence * normalized_error).sum()
        / confidence.sum().clamp_min(1e-6)
        * float(outcome_weight)
    )
    return OwnerResolvedResponseLoss(
        loss=loss,
        normalized_disagreement=disagreement.mean() / global_signal,
        active_owner_fraction=(confidence > 0.1).float().mean(),
    )
