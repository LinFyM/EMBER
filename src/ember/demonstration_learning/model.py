"""Fresh complete public and video LoRA, compiled as one rank-144 adapter."""

from __future__ import annotations

from typing import Mapping

import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.model import CompleteLoRAWriter, DirectLoRAParameters, build_lora_tensor_specs


COMMON_RANK = 128
VIDEO_RANK = 16
COMPLETE_RANK = COMMON_RANK + VIDEO_RANK


def split_identity_template(full: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    """Partition one rank-144 identity template without changing its BA function."""
    if len(full) != 76:
        raise ValueError("complete identity must contain 38 A/B target pairs")
    common, video = {}, {}
    for name, value in full.items():
        is_a = name.endswith(LORA_A_SUFFIX)
        if not is_a and not name.endswith(LORA_B_SUFFIX):
            raise ValueError("identity contains a non-LoRA tensor")
        axis = 0 if is_a else 1
        if value.ndim != 2 or value.shape[axis] != COMPLETE_RANK:
            raise ValueError("rank-144 identity factor changed shape")
        common[name] = value.narrow(axis, 0, COMMON_RANK).clone().contiguous()
        video[name] = value.narrow(axis, COMMON_RANK, VIDEO_RANK).clone().contiguous()
        if not is_a and torch.count_nonzero(value):
            raise ValueError("identity LoRA-B must be exactly zero")
    return common, video


def concatenate_factors(common: Mapping[str, torch.Tensor],
                        video: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exactly Bc Ac + Bv Av, without cross terms or a second adapter."""
    if len(common) != 76 or set(common) != set(video):
        raise ValueError("complete 38-target shared/video factor set changed")
    result = {}
    for name, shared in common.items():
        conditioned = video[name]
        if name.endswith(LORA_A_SUFFIX):
            axis = 0
        elif name.endswith(LORA_B_SUFFIX):
            axis = 1
        else:
            raise ValueError("complete factor is outside LoRA A/B")
        if (shared.ndim != 2 or conditioned.ndim != 2
                or shared.shape[axis] != COMMON_RANK or conditioned.shape[axis] != VIDEO_RANK
                or shared.shape[1 - axis] != conditioned.shape[1 - axis]):
            raise ValueError(f"shared/video LoRA shapes differ: {name}")
        result[name] = torch.cat((shared, conditioned), dim=axis)
    return result


class CorrespondenceLoRA(torch.nn.Module):
    """One model for both P and I; only the event's legal reference changes."""

    def __init__(self, policy, model_config: Mapping, full_template: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        common_template, video_template = split_identity_template(full_template)
        self.common = DirectLoRAParameters(common_template)
        bridge = policy.model.paligemma_with_expert
        args = {key: model_config[key] for key in (
            "image_width", "expert_width", "program_width", "text_meta_lora_rank",
            "vl_meta_lora_rank", "action_meta_lora_rank", "patch_grounding_heads",
            "max_frames_per_encoder_call", "action_horizon", "padded_action_dim",
            "semantic_core_heads", "semantic_core_blocks", "frame_attention_initial_lambda",
            "procedure_heads", "procedure_blocks", "fusion_heads", "factor_hidden_width",
            "initialization_seed", "activation_checkpointing", "camera_view", "horizon_read",
        )}
        self.writer = CompleteLoRAWriter(
            build_lora_tensor_specs(video_template), template_state=video_template,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model, **args,
        )

    def forward(self, policy, condition: tuple) -> dict[str, torch.Tensor]:
        video = self.writer(*condition, policy=policy)
        return concatenate_factors(self.common(), video)
