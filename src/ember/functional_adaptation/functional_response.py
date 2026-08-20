"""Policy-functional flow responses for expert-to-decoder distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.lora import LoRAContract, copy_task_lora_state_, functional_lora_call
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
def pi05_denoised_action_response(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_seed: int,
    num_steps: int = 10,
) -> torch.Tensor:
    """Return the paired full denoised action chunk under one installed LoRA."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise FunctionalResponseError("functional response policy must be frozen")
    try:
        device = next(policy.parameters()).device
        batch_size = int(batch[OBS_LANGUAGE_TOKENS].shape[0])
        chunk_size = int(policy.config.chunk_size)
        max_action_dim = int(policy.config.max_action_dim)
    except (AttributeError, KeyError, StopIteration) as error:
        raise FunctionalResponseError("action response requires a PI0.5 policy batch") from error
    if num_steps != 10:
        raise FunctionalResponseError("action response must use the official ten flow steps")
    generator = torch.Generator(device="cpu").manual_seed(int(policy_seed))
    noise = torch.randn(
        (batch_size, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device)
    copy_task_lora_state_(policy, state, contract)
    response = policy.predict_action_chunk(
        dict(batch), noise=noise, num_steps=num_steps
    )
    if response.shape != (batch_size, chunk_size, 7):
        raise FunctionalResponseError("PI0.5 action response shape changed")
    return response


def pi05_flow_action_jvp_response(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_seed: int,
) -> torch.Tensor:
    """Return one exact velocity JVP with respect to the noisy action sequence.

    The flow point, time, and unit-RMS tangent direction are paired by
    ``policy_seed``. The installed LoRA remains frozen: this measures a local
    policy response, not a gradient with respect to adapter parameters.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise FunctionalResponseError("functional response policy must be frozen")
    try:
        model = policy.model
        device = next(policy.parameters()).device
        images, image_masks = policy._preprocess_images(dict(batch))
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = policy.prepare_action(batch)
    except (AttributeError, KeyError, StopIteration) as error:
        raise FunctionalResponseError(
            "action JVP requires a PI0.5 policy batch"
        ) from error

    copy_task_lora_state_(policy, state, contract)
    with scoped_policy_randomness(policy_seed, device):
        with scoped_policy_flow_noise_sampling(
            policy,
            INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        ):
            with scoped_policy_flow_time_sampling(
                policy,
                INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            ):
                noise = model.sample_noise(actions.shape, device)
                time = model.sample_time(actions.shape[0], device)
                direction = model.sample_noise(actions.shape, device)
    direction = direction / direction.float().square().mean(
        dim=(1, 2), keepdim=True
    ).sqrt()
    noisy_actions = (
        time[:, None, None] * noise
        + (1.0 - time[:, None, None]) * actions
    )

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images,
            image_masks,
            tokens,
            token_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks,
            prefix_att_masks,
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    def velocity(value: torch.Tensor) -> torch.Tensor:
        return model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=value,
            timestep=time,
        )

    _, tangent = torch.func.jvp(
        velocity,
        (noisy_actions,),
        (direction,),
    )
    expected = (
        int(actions.shape[0]),
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    if tangent.shape != expected or not torch.isfinite(tangent).all():
        raise FunctionalResponseError(
            "PI0.5 action JVP response changed shape or is non-finite"
        )
    return tangent.float()


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
