"""Exact PI0.5 execution-observation policy effects for ECP Stage 1."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils.checkpoint import checkpoint

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.observer import ECPNativeObserver


@dataclass(frozen=True)
class ExecutionPolicyPrefix:
    """Official image/language prefix embeddings for real rollout observations."""

    embeddings: torch.Tensor
    padding: torch.Tensor


@dataclass(frozen=True)
class PolicyEffectResponse:
    """Antithetic-averaged native owner, flow, and integrated action response."""

    owner: torch.Tensor
    flow: torch.Tensor
    action: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "PolicyEffectResponse":
        return PolicyEffectResponse(
            owner=self.owner.index_select(0, indices),
            flow=self.flow.index_select(0, indices),
            action=self.action.index_select(0, indices),
        )

    def to(self, *args: Any, **kwargs: Any) -> "PolicyEffectResponse":
        return PolicyEffectResponse(
            owner=self.owner.to(*args, **kwargs),
            flow=self.flow.to(*args, **kwargs),
            action=self.action.to(*args, **kwargs),
        )


@dataclass(frozen=True)
class PolicyEffectParticles:
    """Native owner, flow, and action responses with the probe axis intact."""

    owner: torch.Tensor
    flow: torch.Tensor
    action: torch.Tensor

    def mean_response(self) -> PolicyEffectResponse:
        return PolicyEffectResponse(
            owner=self.owner.mean(1),
            flow=self.flow.mean(1),
            action=self.action.mean(1),
        )

    def to(self, *args: Any, **kwargs: Any) -> "PolicyEffectParticles":
        return PolicyEffectParticles(
            owner=self.owner.to(*args, **kwargs),
            flow=self.flow.to(*args, **kwargs),
            action=self.action.to(*args, **kwargs),
        )


def _autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def prepare_execution_policy_prefix(
    policy: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> ExecutionPolicyPrefix:
    """Embed the exact prefix used by ``PI05Policy.predict_action_chunk``."""

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    images, image_masks = policy._preprocess_images(batch)
    with _autocast(tokens.device):
        embeddings, padding, _ = policy.model.embed_prefix(
            images, image_masks, tokens, masks
        )
    return ExecutionPolicyPrefix(
        embeddings=embeddings.detach(),
        padding=padding.detach(),
    )


def _prepare_prefix_cache(
    policy: torch.nn.Module,
    prefix: ExecutionPolicyPrefix,
) -> Any:
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    core = policy.model
    attention = torch.zeros_like(prefix.padding)
    mask = core._prepare_attention_masks_4d(
        make_att_2d_masks(prefix.padding, attention)
    )
    positions = torch.cumsum(prefix.padding, dim=1) - 1
    bridge = core.paligemma_with_expert
    bridge.paligemma.model.language_model.config._attn_implementation = "eager"
    with torch.no_grad(), _autocast(prefix.embeddings.device):
        _, cache = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix.embeddings, None],
            use_cache=True,
        )
    return cache


@torch.no_grad()
def prepare_policy_effect_prefix_cache(
    policy: torch.nn.Module,
    prefix: ExecutionPolicyPrefix,
) -> Any:
    """Prepare the antithetic prefix cache once for repeated effect queries."""

    repeated = ExecutionPolicyPrefix(
        embeddings=prefix.embeddings.repeat_interleave(2, dim=0),
        padding=prefix.padding.repeat_interleave(2, dim=0),
    )
    return _prepare_prefix_cache(policy, repeated)


def dct_basis(device: torch.device, count: int = 4) -> torch.Tensor:
    """Orthonormal low-frequency basis over PI0.5's 50 action positions."""

    horizon = 50
    positions = torch.arange(horizon, device=device, dtype=torch.float32) + 0.5
    rows = [torch.ones(horizon, device=device) / horizon**0.5]
    rows.extend(
        (2.0 / horizon) ** 0.5 * torch.cos(torch.pi * frequency * positions / horizon)
        for frequency in range(1, count)
    )
    return torch.stack(rows)


def capture_policy_effect_particles(
    *,
    policy: torch.nn.Module,
    observer: ECPNativeObserver,
    lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    prefix: ExecutionPolicyPrefix,
    suffix_noise: torch.Tensor,
    denoising_steps: int,
    prepared_prefix_cache: Any | None = None,
) -> PolicyEffectParticles:
    """Run the official denoising path without averaging antithetic probes."""

    states = int(prefix.embeddings.shape[0])
    if (
        states <= 0
        or prefix.padding.shape != prefix.embeddings.shape[:2]
        or suffix_noise.shape != (states, 50, 32)
        or denoising_steps <= 0
    ):
        raise ValueError("ECP policy-effect execution panel changed")
    repeated = ExecutionPolicyPrefix(
        embeddings=prefix.embeddings.repeat_interleave(2, dim=0),
        padding=prefix.padding.repeat_interleave(2, dim=0),
    )
    noise = torch.stack((suffix_noise, -suffix_noise), dim=1).reshape(
        2 * states, 50, 32
    )
    batch_size = 2 * states
    needs_grad = any(value.requires_grad for value in state.values())
    state_names = tuple(state)
    state_values = tuple(state[name] for name in state_names)
    prefix_cache = (
        prepare_policy_effect_prefix_cache(policy, prefix)
        if prepared_prefix_cache is None
        else prepared_prefix_cache
    )

    def active_state(values: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        return dict(zip(state_names, values, strict=True))

    def first_step(
        x_t: torch.Tensor, *values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = active_state(values)
        with _autocast(x_t.device):
            observed = observer.observe_action_step(
                policy.model,
                repeated.padding,
                prefix_cache,
                x_t,
                torch.ones(batch_size, device=x_t.device),
                track_action_adapter_grad=needs_grad,
                action_adapter_context=lora.activate([active] * batch_size),
            )
        return observed.owner_lattice, observed.flow_velocity.float()

    def later_step(
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *values: torch.Tensor,
    ) -> torch.Tensor:
        active = active_state(values)
        grad_context = torch.enable_grad() if needs_grad else torch.no_grad()
        with (
            grad_context,
            lora.activate([active] * batch_size),
            _autocast(x_t.device),
        ):
            return policy.model.denoise_step(
                prefix_pad_masks=repeated.padding,
                past_key_values=prefix_cache,
                x_t=x_t,
                timestep=timestep,
            ).float()

    x_t = noise
    velocities = []
    actions = []
    owner_lattice = None
    dt = -1.0 / float(denoising_steps)
    for step in range(denoising_steps):
        timestep = torch.full(
            (batch_size,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=noise.device,
        )
        if step == 0:
            if needs_grad:
                owner_lattice, velocity = checkpoint(
                    first_step,
                    x_t,
                    *state_values,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                owner_lattice, velocity = first_step(x_t, *state_values)
        elif needs_grad:
            velocity = checkpoint(
                later_step,
                x_t,
                timestep,
                *state_values,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            velocity = later_step(x_t, timestep, *state_values)
        x_t = x_t + dt * velocity
        velocities.append(velocity)
        actions.append(x_t[..., :7])
    if owner_lattice is None:
        raise RuntimeError("ECP policy-effect denoising trajectory did not run")

    owner = torch.einsum(
        "bohd,ph->bopd", owner_lattice.float(), dct_basis(noise.device)
    ).reshape(states, 2, 38, 4, 128)
    flow = torch.stack(velocities, dim=1).reshape(states, 2, denoising_steps, 50, 32)
    action = torch.stack(actions, dim=1).reshape(states, 2, denoising_steps, 50, 7)
    return PolicyEffectParticles(
        owner=owner,
        flow=flow,
        action=action,
    )


def capture_policy_effect_response(
    *,
    policy: torch.nn.Module,
    observer: ECPNativeObserver,
    lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    prefix: ExecutionPolicyPrefix,
    suffix_noise: torch.Tensor,
    denoising_steps: int,
    prepared_prefix_cache: Any | None = None,
) -> PolicyEffectResponse:
    """Compatibility view that averages the two matched probe particles."""

    return capture_policy_effect_particles(
        policy=policy,
        observer=observer,
        lora=lora,
        state=state,
        prefix=prefix,
        suffix_noise=suffix_noise,
        denoising_steps=denoising_steps,
        prepared_prefix_cache=prepared_prefix_cache,
    ).mean_response()
