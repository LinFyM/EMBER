"""Compile centered Procedure values with bounded Core modulation."""

from __future__ import annotations

import math

import torch

from ember.writer.temporal import (
    RMSNorm,
    VariableEpisodeInputError,
    _apply_rope,
    _merge_heads,
    _split_heads,
)


class ContentCrossAttention(torch.nn.Module):
    """Cross-attend while keeping routing and positions out of value content."""

    def __init__(self, *, width: int, heads: int, rotary_keys: bool) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (rotary_keys and (width // heads) % 2)
        ):
            raise VariableEpisodeInputError("invalid content cross-attention")
        self.heads = int(heads)
        self.rotary_keys = bool(rotary_keys)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_key: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        valid_memory: torch.Tensor,
        memory_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query_key.ndim != 3
            or memory_key.ndim != 3
            or memory_value.shape != memory_key.shape
            or query_key.shape[0] != memory_key.shape[0]
            or query_key.shape[-1] != memory_key.shape[-1]
            or valid_memory.shape != memory_key.shape[:2]
            or valid_memory.dtype != torch.bool
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid content cross-attention batch")
        query = _split_heads(self.query(query_key), self.heads)
        key = _split_heads(self.key(memory_key), self.heads)
        if self.rotary_keys:
            if (
                memory_positions is None
                or memory_positions.shape != memory_key.shape[:2]
                or memory_positions.dtype != torch.long
            ):
                raise VariableEpisodeInputError("Procedure positions changed")
            query = _apply_rope(
                query,
                torch.zeros(
                    query_key.shape[:2],
                    dtype=torch.long,
                    device=query_key.device,
                ),
            )
            key = _apply_rope(key, memory_positions)
        elif memory_positions is not None:
            raise VariableEpisodeInputError("Core reader received positions")

        value = _split_heads(self.value(memory_value), self.heads)
        logits = torch.matmul(
            query.to(torch.float32),
            key.to(torch.float32).transpose(-1, -2),
        )
        logits = logits / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(
            ~valid_memory[:, None, None, :],
            float("-inf"),
        )
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights.to(value.dtype), value)
        return self.output(_merge_heads(attended)), weights


class CoreSlotReader(torch.nn.Module):
    """Read task Core content into 320 routed LoRA slots."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.memory_norm = RMSNorm(width)
        self.attention = ContentCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=False,
        )

    def forward(
        self,
        routing: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attention(
            routing,
            self.memory_norm(core),
            core,
            valid_core,
        )


class ProcedureSlotReader(torch.nn.Module):
    """Use Core-keyed attention to read raw time-centered Procedure values."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.core_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.attention = ContentCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=True,
        )

    def forward(
        self,
        routing: torch.Tensor,
        core_slots: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            procedure.ndim != 3
            or routing.ndim != 3
            or core_slots.shape != routing.shape
            or procedure.shape[0] != routing.shape[0]
            or procedure.shape[-1] != routing.shape[-1]
            or positions.shape != procedure.shape[:2]
            or positions.dtype != torch.long
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Procedure slot memory")
        normalized_core = self.core_norm(core_slots)
        mask = valid_procedure[..., None]
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        procedure_float = procedure.to(torch.float32)
        first_valid = valid_procedure.to(torch.long).argmax(dim=1)
        reference = procedure_float.gather(
            1,
            first_valid[:, None, None].expand(-1, 1, procedure.shape[-1]),
        )
        delta = (procedure_float - reference).masked_fill(~mask, 0.0)
        delta_mean = delta.sum(dim=1, keepdim=True) / count.to(torch.float32)
        centered = (delta - delta_mean).masked_fill(~mask, 0.0)
        centered = centered.to(procedure.dtype)
        slots, weights = self.attention(
            routing + normalized_core,
            self.memory_norm(procedure),
            centered,
            valid_procedure,
            positions,
        )
        return slots, normalized_core, centered, weights


class AmplitudePreservingSlotMixer(torch.nn.Module):
    """Mix slot directions, then restore every slot's pre-mixer RMS."""

    UNIT_FLOOR = 1e-6

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid slot mixer")
        self.heads = int(heads)
        self.attention_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    @classmethod
    def _unit_direction(
        cls,
        content: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physical_rms = (
            torch.linalg.vector_norm(
                content.to(torch.float32),
                dim=-1,
                keepdim=True,
            )
            / math.sqrt(content.shape[-1])
        )
        direction = content / physical_rms.clamp_min(cls.UNIT_FLOOR).to(
            content.dtype
        )
        return direction, physical_rms

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if content.shape != routing.shape or content.ndim != 3:
            raise VariableEpisodeInputError("invalid slot mixing batch")
        direction, input_rms = self._unit_direction(content)
        addressed = self.attention_norm(direction) + routing.to(direction.dtype)
        query = _split_heads(self.query(addressed), self.heads)
        key = _split_heads(self.key(addressed), self.heads)
        value = _split_heads(self.value(direction), self.heads)
        weights = torch.softmax(
            torch.matmul(
                query.to(torch.float32),
                key.to(torch.float32).transpose(-1, -2),
            )
            / math.sqrt(query.shape[-1]),
            dim=-1,
        )
        attended = torch.matmul(weights.to(value.dtype), value)
        mixed = direction + self.output(_merge_heads(attended))
        mixed = mixed + self.ffn(self.ffn_norm(mixed))
        mixed_direction, _ = self._unit_direction(mixed)
        output = mixed_direction * input_rms.to(mixed_direction.dtype)
        return output, weights, input_rms


class CoreKeyedProcedureCompiler(torch.nn.Module):
    """Compile Procedure as value; let Core only address and boundedly modulate it."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK
    CORE_MODULATION_FRACTION = 0.25

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise VariableEpisodeInputError("invalid Core-keyed Procedure compiler")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def parameter(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.query_table = parameter(self.QUERY_COUNT)
        self.module_identity = parameter(3)
        self.layer_identity = parameter(self.EXPERT_LAYERS)
        self.rank_identity = parameter(self.RANK)
        self.routing_norm = RMSNorm(width)
        self.core_reader = CoreSlotReader(width=width, heads=heads)
        self.procedure_reader = ProcedureSlotReader(width=width, heads=heads)
        self.core_gate = torch.nn.Linear(width, width, bias=False)
        torch.nn.init.zeros_(self.core_gate.weight)
        self.slot_mixer = AmplitudePreservingSlotMixer(
            width=width,
            heads=heads,
        )

    def _routing(self) -> torch.Tensor:
        expert = (
            self.query_table[: self.EXPERT_LAYERS * self.RANK].reshape(
                self.EXPERT_LAYERS,
                self.RANK,
                -1,
            )
            + self.module_identity[0]
            + self.layer_identity[:, None]
            + self.rank_identity[None]
        ).reshape(self.EXPERT_LAYERS * self.RANK, -1)
        action_in = (
            self.query_table[
                self.EXPERT_LAYERS * self.RANK :
                self.EXPERT_LAYERS * self.RANK + self.RANK
            ]
            + self.module_identity[1]
            + self.rank_identity
        )
        action_out = (
            self.query_table[-self.RANK :]
            + self.module_identity[2]
            + self.rank_identity
        )
        return torch.cat((expert, action_in, action_out), dim=0)

    @staticmethod
    def _validate_memories(
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> None:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or procedure.ndim != 3
            or positions.shape != procedure.shape[:2]
            or positions.dtype != torch.long
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or core.shape[0] != procedure.shape[0]
            or core.shape[-1] != procedure.shape[-1]
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Core/Procedure compiler memory")

    def fused_slots(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_memories(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
        )
        routing = self.routing_norm(self._routing())[None].expand(
            core.shape[0],
            -1,
            -1,
        )
        core_slots, core_attention = self.core_reader(
            routing,
            core,
            valid_core,
        )
        (
            procedure_slots,
            normalized_core,
            centered_procedure,
            procedure_attention,
        ) = self.procedure_reader(
            routing,
            core_slots,
            procedure,
            positions,
            valid_procedure,
        )
        core_modulation = self.CORE_MODULATION_FRACTION * torch.tanh(
            self.core_gate(normalized_core)
        )
        core_gate = 1.0 + core_modulation
        gated_procedure = procedure_slots * core_gate
        content, slot_attention, pre_mixer_rms = self.slot_mixer(
            gated_procedure,
            routing,
        )
        diagnostics = {
            "routing": routing,
            "core_slots": core_slots,
            "core_attention": core_attention,
            "procedure_centered": centered_procedure,
            "procedure_slots": procedure_slots,
            "procedure_attention": procedure_attention,
            "core_modulation": core_modulation,
            "core_gate": core_gate,
            "gated_procedure": gated_procedure,
            "pre_mixer_rms": pre_mixer_rms,
            "slot_attention": slot_attention,
            "fused_slots": content,
        }
        return content, diagnostics

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content, _ = self.fused_slots(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
        )
        expert_stop = self.EXPERT_LAYERS * self.RANK
        batch = core.shape[0]
        expert = content[:, :expert_stop].reshape(
            batch,
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
