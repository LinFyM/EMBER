"""Policy-functional flow responses for expert-to-decoder distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ember.lora import LoRAContract, functional_lora_call
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
)


class FunctionalResponseError(RuntimeError):
    """Raised when PI0.5 no longer exposes one full flow response."""


@dataclass(frozen=True)
class FunctionalResponseTarget:
    """Frozen identity and expert responses for one exact probe query."""

    identity: torch.Tensor
    expert: torch.Tensor

    @property
    def adapter_energy(self) -> torch.Tensor:
        return (self.expert.float() - self.identity.float()).square().mean()


def pi05_flow_response(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_seed: int,
) -> torch.Tensor:
    """Return the complete Action-Expert velocity response before MSE reduction."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise FunctionalResponseError("functional response policy must be frozen")
    try:
        model = policy.model
        output_projection = model.action_out_proj
        device = next(policy.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise FunctionalResponseError("functional response requires a PI0.5 policy") from error
    captured: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output)

    handle = output_projection.register_forward_hook(capture)
    try:
        with scoped_policy_randomness(policy_seed, device):
            with scoped_policy_flow_noise_sampling(
                policy,
                INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            ):
                with scoped_policy_flow_time_sampling(
                    policy,
                    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
                ):
                    functional_lora_call(policy, state, contract, batch)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].ndim != 3:
        raise FunctionalResponseError(
            "PI0.5 functional probe did not expose one token-level flow response"
        )
    return captured[0]


@torch.no_grad()
def build_functional_response_target(
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    expert_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_seed: int,
) -> FunctionalResponseTarget:
    """Cache the source and expert sides of one paired functional probe."""

    identity = pi05_flow_response(
        policy,
        identity_state,
        contract,
        batch,
        policy_seed=policy_seed,
    )
    expert = pi05_flow_response(
        policy,
        expert_state,
        contract,
        batch,
        policy_seed=policy_seed,
    )
    return FunctionalResponseTarget(identity=identity, expert=expert)


def functional_response_distillation_loss(
    policy: torch.nn.Module,
    candidate_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    target: FunctionalResponseTarget,
    *,
    policy_seed: int,
) -> torch.Tensor:
    """Match expert-induced flow changes without letting source behavior dominate."""

    candidate = pi05_flow_response(
        policy,
        candidate_state,
        contract,
        batch,
        policy_seed=policy_seed,
    )
    expert = target.expert.to(candidate)
    adapter_energy = target.adapter_energy.to(candidate.device)
    return (candidate.float() - expert.float()).square().mean() / adapter_energy.clamp_min(
        1e-8
    )
