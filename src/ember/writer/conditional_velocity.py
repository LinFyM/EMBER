"""One compiled rank-135 LoRA from a fresh common actor and legal video coefficient."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.model import CompleteLoRAWriter, DirectLoRAParameters
from ember.writer.procedure import RecurrentProcedureEncoder
from ember.writer.temporal import ContentCrossAttention, LanguageSemanticCore, RMSNorm
from ember.writer.video_program import Pi05LanguageAxialEncoder


COMMON_RANK = 128
CONDITION_RANK = 7
DEPLOY_RANK = COMMON_RANK + CONDITION_RANK
ACTION_OUT = "model.action_out_proj"


def compile_velocity_state(
    common: Mapping[str, torch.Tensor], coefficient: torch.Tensor, projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Concatenate factors so BA is exactly common BA plus E7 R U at action_out."""
    if coefficient.shape != (7, 256) or projection.shape != (256, 1024):
        raise ValueError("conditional velocity coefficient or shared projection changed")
    if len(common) != 76 or any(not name.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)) for name in common):
        raise ValueError("common actor lost the complete 38-target LoRA")
    result = {}
    for name, value in common.items():
        is_a = name.endswith(LORA_A_SUFFIX)
        if value.ndim != 2 or value.shape[0 if is_a else 1] != COMMON_RANK:
            raise ValueError(f"common actor rank changed: {name}")
        if is_a:
            extra = (coefficient.float() @ projection.float() if name.startswith(ACTION_OUT)
                     else value.new_zeros(CONDITION_RANK, value.shape[1]))
            if extra.shape[1] != value.shape[1]:
                raise ValueError("conditional action_out input width changed")
            result[name] = torch.cat((value, extra.to(value.dtype)), dim=0)
        else:
            extra = value.new_zeros(value.shape[0], CONDITION_RANK)
            if name.startswith(ACTION_OUT):
                if value.shape[0] != 32:
                    raise ValueError("native action_out output width changed")
                extra[:7] = torch.eye(CONDITION_RANK, device=value.device, dtype=value.dtype)
            result[name] = torch.cat((value, extra), dim=1)
    return result


class VelocityTeachingEncoder(torch.nn.Module):
    """Reuse the canonical native video, Core and Procedure owners without a compiler."""

    _validated_offsets = staticmethod(CompleteLoRAWriter._validated_offsets)
    _pack_video_program = CompleteLoRAWriter._pack_video_program
    _read_native_condition = CompleteLoRAWriter._read_native_condition
    encode_task = CompleteLoRAWriter.encode_task

    def __init__(self, policy: torch.nn.Module, model: Mapping[str, object]) -> None:
        super().__init__()
        if (model["camera_view"] != "agentview" or model["program_width"] != 256
                or model["procedure_blocks"] != 2):
            raise ValueError("conditional velocity teaching topology changed")
        bridge = policy.model.paligemma_with_expert
        self.program_width = 256
        self.camera_view = "agentview"
        self.initial_content_only = False
        self.semantic_encoder = Pi05LanguageAxialEncoder(
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
            image_width=model["image_width"], expert_width=model["expert_width"],
            program_width=256, text_meta_lora_rank=model["text_meta_lora_rank"],
            vl_meta_lora_rank=model["vl_meta_lora_rank"],
            action_meta_lora_rank=model["action_meta_lora_rank"],
            patch_grounding_heads=model["patch_grounding_heads"],
            max_frames_per_encoder_call=model["max_frames_per_encoder_call"],
            action_horizon=50, padded_action_dim=32,
            initialization_seed=model["initialization_seed"],
            activation_checkpointing=model["activation_checkpointing"],
            camera_view="agentview", horizon_read=model["horizon_read"],
        )
        self.semantic_core = LanguageSemanticCore(
            width=256, heads=model["semantic_core_heads"], blocks=2,
            frame_attention_initial_lambda=model["frame_attention_initial_lambda"],
            language_content_path=False,
        )
        self.procedure = RecurrentProcedureEncoder(
            width=256, expert_width=1024, heads=model["procedure_heads"],
            blocks=2, action_horizon=50,
        )


class ActionRowReadout(torch.nn.Module):
    """Seven learned action-row identities; actual C/P tokens supply both Values."""

    def __init__(self) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(7, 256))
        torch.nn.init.normal_(self.queries, std=0.02)
        self.core_norm = RMSNorm(256)
        self.core_read = ContentCrossAttention(width=256, heads=8, rotary_keys=False)
        self.read_norm = RMSNorm(256)
        self.procedure_norm = RMSNorm(256)
        self.procedure_read = ContentCrossAttention(width=256, heads=8, rotary_keys=True)
        self.core_mix = torch.nn.Linear(256, 256, bias=False)
        self.procedure_mix = torch.nn.Linear(256, 256, bias=False)
        self.output = torch.nn.Linear(256, 256, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, core, core_mask, procedure, positions, frame_mask) -> torch.Tensor:
        if (core.ndim != 3 or core.shape[0] != 1 or core.shape[-1] != 256
                or procedure.ndim != 3 or procedure.shape[:1] != core.shape[:1]
                or procedure.shape[-1] != 256):
            raise ValueError("action-row readout requires one real Core/Procedure condition")
        query = self.queries[None]
        c = self.core_read(query, self.core_norm(core), core, core_mask)
        p = self.procedure_read(
            query + self.read_norm(c), self.procedure_norm(procedure), procedure,
            frame_mask, positions,
        )
        return self.output(F.gelu(self.core_mix(c) + self.procedure_mix(p)))[0]


class ConditionalVelocityOperator(torch.nn.Module):
    """Fresh beta, native teaching path, action-row readout and common U."""

    def __init__(self, policy, model, common_template) -> None:
        super().__init__()
        # This order is part of the checkpointed initialization contract.
        self.common = DirectLoRAParameters(common_template)
        self.teaching = VelocityTeachingEncoder(policy, model)
        self.readout = ActionRowReadout()
        self.U = torch.nn.Linear(1024, 256, bias=False)

    def forward(self, policy, condition: tuple, *, frame_parallel_group=None):
        core, core_mask, procedure, positions, frame_mask, _ = self.teaching.encode_task(
            policy, *condition, frame_parallel_group=frame_parallel_group,
        )
        coefficient = self.readout(core, core_mask, procedure, positions, frame_mask)
        return compile_velocity_state(self.common(), coefficient, self.U.weight), coefficient
