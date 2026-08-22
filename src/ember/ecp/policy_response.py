"""Frozen full-layer PI0.5 responses in ECP owner coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.ecp.observer import ActionLayerStateCapture, TargetOwnerProjector
from ember.functional_adaptation.functional_response import FunctionalResponseError
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    functional_lora_call,
    validate_lora_state,
)
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
    target_inputs: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class TargetActivationEffectLoss:
    loss: torch.Tensor
    normalized_disagreement: torch.Tensor
    active_owner_fraction: torch.Tensor


@dataclass(frozen=True)
class FrozenTargetActivationEffects:
    reference_inputs: Mapping[str, torch.Tensor]
    expert_effects: Mapping[str, torch.Tensor]

    def to(self, device: torch.device) -> "FrozenTargetActivationEffects":
        return FrozenTargetActivationEffects(
            reference_inputs={
                name: value.to(device, non_blocking=True)
                for name, value in self.reference_inputs.items()
            },
            expert_effects={
                name: value.to(device, non_blocking=True)
                for name, value in self.expert_effects.items()
            },
        )


def target_activation_effects_from_payload(
    value: Mapping[str, Any], *, contract: LoRAContract, expert_count: int
) -> FrozenTargetActivationEffects | None:
    fields = tuple(
        value.get(name)
        for name in (
            "reference_target_inputs",
            "expert_target_effects",
        )
    )
    if all(field is None for field in fields):
        return None
    if any(field is None for field in fields):
        raise FunctionalResponseError("target-local activation payload is partial")
    reference, experts = (dict(field) for field in fields)
    expected = {target.name for target in contract.targets}
    if set(reference) != expected or set(experts) != expected:
        raise FunctionalResponseError("target-local activation owners changed")
    for target in contract.targets:
        target_input = reference[target.name]
        target_effect = experts[target.name]
        if (
            target_input.ndim != 3
            or target_input.shape[-1] != target.in_features
            or target_effect.shape
            != (expert_count, *target_input.shape[:-1], target.out_features)
            or not torch.isfinite(target_input).all()
            or not torch.isfinite(target_effect).all()
        ):
            raise FunctionalResponseError(
                f"target-local activation payload changed at {target.name}"
            )
    return FrozenTargetActivationEffects(
        reference_inputs={name: value.float() for name, value in reference.items()},
        expert_effects={name: value.float() for name, value in experts.items()},
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
    capture_target_inputs: bool,
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
    target_inputs: dict[str, list[torch.Tensor]] = {
        target.name: [] for target in contract.targets
    }

    def capture_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        action_inputs.append(inputs[0].detach())

    def capture_flow(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        flows.append(output.detach())

    def capture_target(name: str):
        def hook(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            target_inputs[name].append(inputs[0].detach())

        return hook

    input_handle = core.action_in_proj.register_forward_pre_hook(capture_input)
    output_handle = core.action_out_proj.register_forward_hook(capture_flow)
    target_handles = []
    if capture_target_inputs:
        modules = dict(policy.named_modules())
        for target in contract.targets:
            module = modules.get(target.name)
            if module is None:
                raise FunctionalResponseError(
                    f"policy target module is unavailable: {target.name}"
                )
            target_handles.append(
                module.register_forward_pre_hook(capture_target(target.name))
            )
    try:
        with torch.no_grad(), scoped_policy_randomness(
            policy_seed, next(policy.parameters()).device
        ):
            with scoped_policy_flow_noise_sampling(
                policy, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME
            ):
                with scoped_policy_flow_time_sampling(
                    policy, INDEPENDENT_BETA_TIME_SAMPLING_SCHEME
                ):
                    with ActionLayerStateCapture(expert, detach=True) as layers:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            functional_lora_call(policy, state, contract, batch)
        layer_states = layers.stacked()
    finally:
        input_handle.remove()
        output_handle.remove()
        for handle in target_handles:
            handle.remove()
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
    compressed_inputs = None
    if capture_target_inputs:
        compressed_inputs = {}
        for target in contract.targets:
            values = target_inputs[target.name]
            if (
                len(values) != 1
                or values[0].ndim != 3
                or values[0].shape[1] != owner.shape[2]
                or values[0].shape[2] != target.in_features
            ):
                raise FunctionalResponseError(
                    f"policy target input topology changed at {target.name}"
                )
            compressed_inputs[target.name] = torch.einsum(
                "bhd,ph->bpd", values[0].float(), basis
            )
    return CapturedPolicyResponse(
        flow=flows[0].float(),
        owner_basis=compressed,
        target_inputs=compressed_inputs,
    )


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
    capture_target_inputs: bool = False,
) -> CapturedPolicyResponse:
    return _capture_policy_response(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        projector=projector,
        policy_seed=policy_seed,
        horizon_basis=horizon_basis,
        capture_target_inputs=capture_target_inputs,
    )


def lora_activation_effects(
    *,
    state: Mapping[str, torch.Tensor],
    reference_inputs: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    """Apply every LoRA target to the same detached policy-native inputs."""

    validate_lora_state(state, contract)
    if set(reference_inputs) != {target.name for target in contract.targets}:
        raise FunctionalResponseError("target-local reference inputs changed")
    scale = float(contract.alpha) / float(contract.rank)
    result = {}
    for target in contract.targets:
        value = reference_inputs[target.name]
        if value.ndim != 3 or value.shape[-1] != target.in_features:
            raise FunctionalResponseError(
                f"target-local reference input changed at {target.name}"
            )
        a = state[target.name + LORA_A_SUFFIX].float()
        b = state[target.name + LORA_B_SUFFIX].float()
        result[target.name] = (
            torch.einsum("bpi,ri,or->bpo", value.float(), a, b) * scale
        )
    return result


def target_activation_effect_distillation_loss(
    *,
    candidate_state: Mapping[str, torch.Tensor],
    targets: FrozenTargetActivationEffects,
    contract: LoRAContract,
    expert_weights: torch.Tensor,
    outcome_weight: float,
) -> TargetActivationEffectLoss:
    """Match gauge-invariant owner-local LoRA effects on frozen inputs."""

    expert_count = next(iter(targets.expert_effects.values())).shape[0]
    if expert_weights.shape != (expert_count,):
        raise FunctionalResponseError("target-local expert weights changed")
    first = next(iter(targets.reference_inputs.values()))
    weights = expert_weights.to(first).float().clamp_min(1e-4)
    weights = weights / weights.sum()
    candidate = lora_activation_effects(
        state=candidate_state,
        reference_inputs=targets.reference_inputs,
        contract=contract,
    )
    errors = []
    signals = []
    disagreements = []
    for target in contract.targets:
        experts = targets.expert_effects[target.name].to(
            candidate[target.name]
        ).float()
        if experts.shape[0] != expert_count:
            raise FunctionalResponseError("target-local expert count changed")
        consensus = torch.einsum("m,mbpo->bpo", weights, experts)
        errors.append((candidate[target.name] - consensus).square().mean())
        signals.append(consensus.square().mean())
        disagreements.append(
            torch.einsum(
                "m,mbpo->bpo",
                weights,
                (experts - consensus[None]).square(),
            ).mean()
        )
    error = torch.stack(errors)
    signal = torch.stack(signals)
    disagreement = torch.stack(disagreements)
    global_signal = signal.mean().clamp_min(1e-8)
    confidence = signal / (signal + disagreement + 0.05 * global_signal)
    normalized_error = error / (signal + 0.05 * global_signal)
    loss = (
        (confidence * normalized_error).sum()
        / confidence.sum().clamp_min(1e-6)
        * float(outcome_weight)
    )
    return TargetActivationEffectLoss(
        loss=loss,
        normalized_disagreement=disagreement.mean() / global_signal,
        active_owner_fraction=(confidence > 0.1).float().mean(),
    )
