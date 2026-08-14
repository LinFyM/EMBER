"""Executed-prefix PI05 flow loss for on-policy binary reward replay."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.lora import LoRAContract, validate_lora_state
from ember.reward.protocol import RewardProtocolError


class Pi05ExecutedPrefixFlowLoss(torch.nn.Module):
    """Return one loss per replan chunk while masking unexecuted actions."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        valid = batch.get("executed_action_steps")
        if not isinstance(valid, torch.Tensor) or valid.ndim != 1:
            raise RewardProtocolError("reward batch lacks executed-action lengths")
        images, image_masks = self.policy._preprocess_images(batch)
        actions = self.policy.prepare_action(batch)
        losses = self.policy.model.forward(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            actions,
            noise,
            time,
        )
        action_dim = int(self.policy.config.output_features[ACTION].shape[0])
        losses = losses[:, :, :action_dim]
        valid = valid.to(device=losses.device, dtype=torch.long)
        if (
            valid.shape != (losses.shape[0],)
            or bool((valid <= 0).any())
            or bool((valid > losses.shape[1]).any())
        ):
            raise RewardProtocolError("reward executed-action mask is invalid")
        mask = (
            torch.arange(losses.shape[1], device=losses.device)[None] < valid[:, None]
        )
        return (losses * mask[:, :, None]).sum(dim=(1, 2)) / (valid * action_dim).to(
            dtype=losses.dtype
        )


def functional_executed_prefix_flow_loss(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Evaluate replay loss with differentiable generated LoRA tensors only."""

    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("functional reward policy must remain frozen")
    return torch.func.functional_call(
        Pi05ExecutedPrefixFlowLoss(policy),
        {f"policy.{name}": value for name, value in state.items()},
        args=(batch,),
        kwargs={"noise": noise, "time": time},
        strict=False,
    )
