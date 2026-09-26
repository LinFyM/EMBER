"""Bounded native q/v diagnostic reader with C0 teaching features.

This executes with precomputed teaching memory. It is not a deployable Writer
or a second task adapter; the only physical policy parameters stay frozen.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from ember.writer.model import CompleteLoRAWriter
from ember.writer.procedure import RecurrentProcedureEncoder
from ember.writer.runtime import autocast
from ember.writer.temporal import ContentCrossAttention, LanguageSemanticCore, RMSNorm
from ember.writer.video_program import Pi05LanguageAxialEncoder


@dataclass(frozen=True)
class TeachingMemory:
    mode: str
    core: torch.Tensor
    core_mask: torch.Tensor
    procedure: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    procedure_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"R_V", "R_L"}:
            raise ValueError("unknown native reader mode")
        if (self.core.ndim != 3 or self.core.shape[-1] != 256
                or self.core_mask.shape != self.core.shape[:2]
                or self.core_mask.dtype != torch.bool
                or not bool(self.core_mask.any(dim=1).all())):
            raise ValueError("invalid Core teaching memory")
        video = (self.procedure, self.positions, self.procedure_mask)
        if self.mode == "R_L":
            if any(value is not None for value in video):
                raise ValueError("language reader cannot receive video memory")
            return
        if any(value is None for value in video):
            raise ValueError("video reader requires complete ordered Procedure")
        if (self.procedure.ndim != 3 or self.procedure.shape[-1] != 256
                or self.procedure.shape[0] != self.core.shape[0]
                or self.positions.shape != self.procedure.shape[:2]
                or self.positions.dtype != torch.long
                or self.procedure_mask.shape != self.procedure.shape[:2]
                or self.procedure_mask.dtype != torch.bool
                or not bool(self.procedure_mask[:, 0].all())):
            raise ValueError("invalid ordered Procedure teaching memory")

    def values(self) -> tuple[torch.Tensor, ...]:
        return (self.core,) if self.mode == "R_L" else (self.core, self.procedure)

    def leaves(self) -> TeachingMemory:
        values = [value.detach().requires_grad_(True) for value in self.values()]
        return TeachingMemory(self.mode, values[0], self.core_mask,
                              values[1] if self.mode == "R_V" else None,
                              self.positions, self.procedure_mask)

    def expand(self, batch: int) -> TeachingMemory:
        if self.core.shape[0] == batch:
            return self
        if self.core.shape[0] != 1:
            raise ValueError("reader memory and execution batch differ")
        return TeachingMemory(
            self.mode, self.core.expand(batch, -1, -1), self.core_mask.expand(batch, -1),
            self.procedure.expand(batch, -1, -1) if self.procedure is not None else None,
            self.positions.expand(batch, -1) if self.positions is not None else None,
            self.procedure_mask.expand(batch, -1) if self.procedure_mask is not None else None,
        )


class NativeTeachingEncoder(torch.nn.Module):
    """C0's native text/E/H, Core and Procedure, without Compiler or heads."""

    # Share the exact C0 input validation and packing path rather than copying
    # its video-time rules into this short-lived diagnostic.
    _validated_offsets = staticmethod(CompleteLoRAWriter._validated_offsets)
    _pack_video_program = CompleteLoRAWriter._pack_video_program
    _read_native_condition = CompleteLoRAWriter._read_native_condition
    encode_task = CompleteLoRAWriter.encode_task

    def __init__(self, policy: torch.nn.Module, model: dict, *, mode: str) -> None:
        super().__init__()
        if mode not in {"R_V", "R_L"} or model["camera_view"] != "agentview":
            raise ValueError("native teaching reader requires R_V/R_L and agentview")
        bridge = policy.model.paligemma_with_expert
        self.mode = mode
        self.program_width = 256
        self.camera_view = "agentview"
        self.initial_content_only = False
        self.semantic_encoder = Pi05LanguageAxialEncoder(
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
            image_width=model["image_width"], expert_width=model["expert_width"],
            program_width=model["program_width"],
            text_meta_lora_rank=model["text_meta_lora_rank"],
            vl_meta_lora_rank=model["vl_meta_lora_rank"],
            action_meta_lora_rank=model["action_meta_lora_rank"],
            patch_grounding_heads=model["patch_grounding_heads"],
            max_frames_per_encoder_call=model["max_frames_per_encoder_call"],
            action_horizon=model["action_horizon"],
            padded_action_dim=model["padded_action_dim"],
            initialization_seed=model["initialization_seed"],
            activation_checkpointing=model["activation_checkpointing"],
            camera_view=model["camera_view"], horizon_read=model["horizon_read"],
        )
        self.semantic_core = LanguageSemanticCore(
            width=256, heads=model["semantic_core_heads"],
            blocks=model["semantic_core_blocks"],
            frame_attention_initial_lambda=model["frame_attention_initial_lambda"],
        )
        self.procedure = (RecurrentProcedureEncoder(
            width=256, expert_width=1024, heads=model["procedure_heads"],
            blocks=model["procedure_blocks"], action_horizon=50,
        ) if mode == "R_V" else None)
        if mode == "R_L":
            for module in (self.semantic_encoder.vl_meta_lora,
                           self.semantic_encoder.action_meta_lora,
                           self.semantic_encoder.interaction_projection,
                           self.semantic_encoder.patch_grounding,
                           self.semantic_core.frame_attention):
                module.requires_grad_(False)

    def forward(self, policy: torch.nn.Module, condition: tuple) -> TeachingMemory:
        if self.mode == "R_L":
            tokens, mask, span = condition
            text, valid = self.semantic_encoder.encode_text_only(policy, tokens, mask, span)
            return TeachingMemory("R_L", self.semantic_core.language_only(text, valid), valid)
        core, core_mask, procedure, positions, frame_mask, _ = self.encode_task(
            policy, *condition,
        )
        return TeachingMemory("R_V", core, core_mask, procedure, positions, frame_mask)


class NativeConditionalReader(torch.nn.Module):
    """Shared 256-wide nonlinear read, added at real layer-9 q/v outputs."""

    LAYER = 9

    def __init__(self) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(1024)
        self.query = torch.nn.Linear(1024, 256, bias=False)
        self.core_norm = RMSNorm(256)
        self.core_read = ContentCrossAttention(width=256, heads=8, rotary_keys=False)
        self.read_norm = RMSNorm(256)
        self.procedure_norm = RMSNorm(256)
        self.procedure_read = ContentCrossAttention(width=256, heads=8, rotary_keys=True)
        self.core_mix = torch.nn.Linear(256, 256, bias=False)
        self.procedure_mix = torch.nn.Linear(256, 256, bias=False)
        self.hidden_gate = torch.nn.Linear(1024, 256, bias=False)
        self.out_q = torch.nn.Linear(256, 2048, bias=False)
        self.out_v = torch.nn.Linear(256, 256, bias=False)
        torch.nn.init.zeros_(self.out_q.weight)
        torch.nn.init.zeros_(self.out_v.weight)

    def forward(self, hidden: torch.Tensor, memory: TeachingMemory, target: str) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (50, 1024) or target not in {"q", "v"}:
            raise ValueError("native reader requires real H50 layer-9 projection input")
        memory = memory.expand(hidden.shape[0])
        normalized = self.hidden_norm(hidden)
        query = self.query(normalized)
        core = self.core_read(query, self.core_norm(memory.core),
                              memory.core, memory.core_mask)
        if memory.mode == "R_V":
            mask = memory.procedure_mask[..., None]
            mean = (memory.procedure * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            centered = (memory.procedure - mean).masked_fill(~mask, 0)
            procedure = self.procedure_read(
                query + self.read_norm(core), self.procedure_norm(memory.procedure),
                centered, memory.procedure_mask, memory.positions,
            )
        else:
            procedure = torch.zeros_like(core)
        mixed = self.core_mix(core) + self.procedure_mix(procedure)
        value = F.gelu(self.hidden_gate(normalized)) * mixed
        return (self.out_q if target == "q" else self.out_v)(value)

    @contextmanager
    def installed(self, policy: torch.nn.Module, memory: TeachingMemory) -> Iterator[dict[str, int]]:
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise ValueError("native reader requires frozen source policy")
        layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[self.LAYER]
        counts = {"q": 0, "v": 0}
        handles = []
        for target, width in (("q", 2048), ("v", 256)):
            projection = getattr(layer.self_attn, f"{target}_proj")
            if projection.in_features != 1024 or projection.out_features != width:
                raise ValueError("native layer-9 target dimensions changed")

            def hook(_module, inputs, output, *, selected=target, expected=width):
                if output.ndim != 3 or output.shape[1:] != (50, expected):
                    raise ValueError("native target was not called on actual H50 suffix")
                counts[selected] += 1
                with autocast(output.device):
                    delta = self(inputs[0], memory, selected)
                return output + delta.to(output.dtype)

            handles.append(projection.register_forward_hook(hook))
        try:
            yield counts
        finally:
            for handle in handles:
                handle.remove()
