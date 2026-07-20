"""Functional action loss from a Writer-generated task-local LoRA."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.lora import (
    SmolVLALoRAContract,
    functional_lora_call,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError


def prepare_frozen_writer_policy(
    policy: torch.nn.Module,
    contract: SmolVLALoRAContract,
) -> dict[str, torch.Tensor]:
    """Inject the sealed identity LoRA and freeze all physical policy state."""

    inject_task_lora(policy, contract)
    template = task_lora_state_dict(policy, clone=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("Writer functional policy must be fully frozen")
    return template


def writer_functional_action_loss(
    writer: CompleteLoRAWriter,
    policy: torch.nn.Module,
    contract: SmolVLALoRAContract,
    *,
    language_features: torch.Tensor,
    video_features: torch.Tensor,
    episode_offsets: torch.Tensor,
    batch: Mapping[str, Any],
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Run the frozen policy with the Writer's differentiable complete LoRA."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("functional action loss received a trainable policy")
    generated = writer(language_features, video_features, episode_offsets)
    output = functional_lora_call(
        policy,
        generated,
        contract,
        dict(batch),
        noise=noise,
        time=time,
    )
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 0
        or not isinstance(output[1], Mapping)
    ):
        raise WriterModelError("functional policy did not return a scalar loss")
    return output
