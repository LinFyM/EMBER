"""Differentiable deployed PI05 action endpoint for reward preference."""

from __future__ import annotations

from typing import Mapping

import torch
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract, validate_lora_state
from ember.reward.protocol import RewardProtocolError


class Pi05EndpointPrefix(torch.nn.Module):
    """Build the one shared visual-language cache for endpoint integration."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, object]:
        model = self.policy.model
        images, image_masks = self.policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix, prefix_pad, prefix_attention = model.embed_prefix(
            images, image_masks, tokens, token_masks
        )
        attention = make_att_2d_masks(prefix_pad, prefix_attention)
        positions = torch.cumsum(prefix_pad, dim=1) - 1
        attention = model._prepare_attention_masks_4d(attention)
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        return prefix_pad, cache


class Pi05EndpointDenoiseStep(torch.nn.Module):
    """One action-expert step under an explicit functional LoRA state."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        prefix_pad: torch.Tensor,
        cache: object,
        action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy.model.denoise_step(
            prefix_pad_masks=prefix_pad,
            past_key_values=cache,
            x_t=action,
            timestep=time,
        )


def functional_executed_prefix_endpoint_action(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    noise: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Differentiate the actual deployed endpoint through generated LoRA tensors."""

    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("functional reward policy must remain frozen")
    if num_steps != 10 or policy.model._rtc_enabled():
        raise RewardProtocolError("endpoint preference requires the sealed PI05 solver")
    names = tuple(state)
    values = tuple(state.values())

    def functional_state(
        parameters: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        return {
            f"policy.{name}": value
            for name, value in zip(names, parameters, strict=True)
        }

    prefix_pad, cache = torch.func.functional_call(
        Pi05EndpointPrefix(policy),
        functional_state(values),
        args=(batch,),
        strict=False,
    )
    denoiser = Pi05EndpointDenoiseStep(policy)
    step_size = -1.0 / num_steps
    action = noise
    for step in range(num_steps):
        time = torch.full(
            (noise.shape[0],),
            1.0 + step * step_size,
            dtype=torch.float32,
            device=noise.device,
        )

        def denoise(
            value: torch.Tensor,
            *parameters: torch.Tensor,
            timestep: torch.Tensor = time,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                denoiser,
                functional_state(parameters),
                args=(prefix_pad, cache, value, timestep),
                strict=False,
            )

        velocity = (
            checkpoint(denoise, action, *values, use_reentrant=False)
            if torch.is_grad_enabled()
            else denoise(action, *values)
        )
        action = action + step_size * velocity
    action_dim = int(policy.config.output_features[ACTION].shape[0])
    return action[:, :, :action_dim]
