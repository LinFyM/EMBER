"""Compile a semantic Core license with the full raw video Procedure program."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ember.writer.temporal import (
    RMSNorm,
    VariableEpisodeInputError,
    _apply_rope,
    _merge_heads,
    _split_heads,
)


class RawValueCrossAttention(torch.nn.Module):
    """Use learned Q/K/O addressing while preserving the memory as raw values."""

    def __init__(self, *, width: int, heads: int, rotary_keys: bool) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (rotary_keys and (width // heads) % 2)
        ):
            raise VariableEpisodeInputError("invalid raw-value cross-attention")
        self.heads = int(heads)
        self.rotary_keys = bool(rotary_keys)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
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
            raise VariableEpisodeInputError("invalid raw-value attention batch")
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

        value = _split_heads(memory_value, self.heads)
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
    """Read raw semantic Core values into the routed public-LoRA slots."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.memory_norm = RMSNorm(width)
        self.attention = RawValueCrossAttention(
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
    """Use the Core-derived address to read the complete raw Procedure program."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.core_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.attention = RawValueCrossAttention(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        slots, weights = self.attention(
            routing + normalized_core,
            self.memory_norm(procedure),
            procedure,
            valid_procedure,
            positions,
        )
        return slots, normalized_core, weights


class CoreProgramBilinearFusion(torch.nn.Module):
    """Require both the semantic Core license and Procedure program."""

    def __init__(self, *, width: int, hidden_width: int) -> None:
        super().__init__()
        if min(width, hidden_width) <= 0:
            raise VariableEpisodeInputError("invalid Core-Program fusion")
        self.core_projection = torch.nn.Linear(
            width,
            hidden_width,
            bias=False,
        )
        self.program_projection = torch.nn.Linear(
            width,
            hidden_width,
            bias=False,
        )
        self.output = torch.nn.Linear(hidden_width, width, bias=False)

    def forward(
        self,
        normalized_core_slots: torch.Tensor,
        procedure_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            normalized_core_slots.shape != procedure_slots.shape
            or normalized_core_slots.ndim != 3
        ):
            raise VariableEpisodeInputError("invalid Core-Program slot fusion")
        core_basis = F.silu(
            self.core_projection(normalized_core_slots)
        )
        procedure_program = self.program_projection(procedure_slots)
        fused = self.output(core_basis * procedure_program)
        return fused, core_basis, procedure_program


class ZeroPreservingSlotBlock(torch.nn.Module):
    """Coordinate LoRA slots without letting routing enter their value content."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid zero-preserving slot block")
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

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if content.shape != routing.shape or content.ndim != 3:
            raise VariableEpisodeInputError("invalid slot coordination batch")
        addressed = self.attention_norm(content) + routing.to(content.dtype)
        query = _split_heads(self.query(addressed), self.heads)
        key = _split_heads(self.key(addressed), self.heads)
        value = _split_heads(self.value(content), self.heads)
        weights = torch.softmax(
            torch.matmul(
                query.to(torch.float32),
                key.to(torch.float32).transpose(-1, -2),
            )
            / math.sqrt(query.shape[-1]),
            dim=-1,
        )
        attended = torch.matmul(weights.to(value.dtype), value)
        content = content + self.output(_merge_heads(attended))
        content = content + self.ffn(self.ffn_norm(content))
        return content, weights


class CoreProgramCompiler(torch.nn.Module):
    """Generate LoRA content only from a licensed raw Procedure program."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        bilinear_hidden_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(width, heads, bilinear_hidden_width) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid Core-Program compiler")
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
        self.bilinear_fusion = CoreProgramBilinearFusion(
            width=width,
            hidden_width=bilinear_hidden_width,
        )
        self.slot_block = ZeroPreservingSlotBlock(
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
            procedure_attention,
        ) = self.procedure_reader(
            routing,
            core_slots,
            procedure,
            positions,
            valid_procedure,
        )
        fused, core_basis, procedure_program = self.bilinear_fusion(
            normalized_core,
            procedure_slots,
        )
        content, slot_attention = self.slot_block(fused, routing)
        diagnostics = {
            "routing": routing,
            "core_slots": core_slots,
            "core_attention": core_attention,
            "normalized_core_slots": normalized_core,
            "procedure_slots": procedure_slots,
            "procedure_attention": procedure_attention,
            "core_basis": core_basis,
            "procedure_program": procedure_program,
            "bilinear_slots": fused,
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
