"""Frozen full-layer PI0.5 responses in ECP owner coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

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


def _dct_basis(*, horizon: int, count: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(horizon, device=device, dtype=torch.float32) + 0.5
    rows = [torch.ones(horizon, device=device) / math.sqrt(horizon)]
    for frequency in range(1, count):
        rows.append(
            math.sqrt(2.0 / horizon)
            * torch.cos(math.pi * frequency * positions / horizon)
        )
    return torch.stack(rows)


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
    """Capture every Action layer before reducing the 50-token horizon."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise FunctionalResponseError("policy-support capture requires a frozen PI0.5")
    core = policy.model
    expert = core.paligemma_with_expert.gemma_expert.model
    action_inputs: list[torch.Tensor] = []
    flows: list[torch.Tensor] = []

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

    input_handle = core.action_in_proj.register_forward_pre_hook(capture_input)
    output_handle = core.action_out_proj.register_forward_hook(capture_flow)
    try:
        with scoped_policy_randomness(policy_seed, next(policy.parameters()).device):
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
