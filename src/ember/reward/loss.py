"""Executed-prefix PI05 flow loss for binary-success reward replay."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ember.lora import LoRAContract, validate_lora_state
from ember.reward.protocol import RewardProtocolError


class Pi05ExecutedPrefixFlowLoss(torch.nn.Module):
    """Return one loss per replan chunk, masking every unexecuted action step."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        validate_prefix_values: bool = True,
        collect_details: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        valid = batch.get("executed_action_steps")
        if not isinstance(valid, torch.Tensor) or valid.ndim != 1:
            raise RewardProtocolError("PI05 reward batch lacks executed-action lengths")
        images, image_masks = self.policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.policy.prepare_action(batch)
        if noise is None:
            noise = self.policy.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.policy.model.sample_time(actions.shape[0], actions.device)
        if noise.shape != actions.shape or time.shape != (actions.shape[0],):
            raise RewardProtocolError("PI05 reward update noise/time shape changed")
        losses = self.policy.model.forward(
            images, image_masks, tokens, token_masks, actions, noise, time
        )
        action_dim = int(self.policy.config.output_features[ACTION].shape[0])
        losses = losses[:, :, :action_dim]
        valid = valid.to(device=losses.device, dtype=torch.long)
        if valid.shape[0] != losses.shape[0] or (
            validate_prefix_values
            and (
                bool((valid <= 0).any())
                or bool((valid > losses.shape[1]).any())
            )
        ):
            raise RewardProtocolError("PI05 reward executed-action mask is invalid")
        mask = torch.arange(losses.shape[1], device=losses.device)[None] < valid[:, None]
        numerator = (losses * mask[:, :, None]).sum(dim=(1, 2))
        per_chunk = numerator / (valid * action_dim).to(dtype=losses.dtype)
        if not collect_details:
            return per_chunk, {}
        return per_chunk, {
            "loss": float(per_chunk.mean().detach()),
            "successful_chunks": int(per_chunk.numel()),
            "executed_action_steps": int(valid.sum()),
            "masked_unexecuted_action_steps": int(
                losses.shape[0] * losses.shape[1] - valid.sum()
            ),
        }


def functional_executed_prefix_flow_loss(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    validate_prefix_values: bool = True,
    collect_details: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate executed-prefix loss with differentiable generated LoRA tensors."""

    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("functional reward policy must be physically frozen")
    replay = Pi05ExecutedPrefixFlowLoss(policy)
    prefixed = {f"policy.{name}": value for name, value in state.items()}
    return torch.func.functional_call(
        replay,
        prefixed,
        args=(batch,),
        kwargs={
            "noise": noise,
            "time": time,
            "validate_prefix_values": validate_prefix_values,
            "collect_details": collect_details,
        },
        strict=False,
    )
