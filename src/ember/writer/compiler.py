"""Compile Semantic Core and ordered Procedure evidence into LoRA slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ember.writer.temporal import (
    RMSNorm,
    VariableEpisodeInputError,
    _apply_rope,
    _merge_heads,
    _split_heads,
)


class ContentCrossAttention(torch.nn.Module):
    """Cross-attend while routing/position affect only Q/K, never content V."""

    def __init__(self, *, width: int, heads: int, rotary_keys: bool) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid content cross-attention")
        if rotary_keys and (width // heads) % 2:
            raise VariableEpisodeInputError("rotary cross-attention head is odd")
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
    ) -> torch.Tensor:
        if (
            query_key.ndim != 3
            or memory_key.ndim != 3
            or memory_value.shape != memory_key.shape
            or query_key.shape[0] != memory_key.shape[0]
            or query_key.shape[-1] != memory_key.shape[-1]
            or valid_memory.shape != memory_key.shape[:2]
            or valid_memory.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid content cross-attention batch")
        query = _split_heads(self.query(query_key), self.heads)
        key = _split_heads(self.key(memory_key), self.heads)
        if self.rotary_keys:
            if memory_positions is None or memory_positions.shape != memory_key.shape[:2]:
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
            raise VariableEpisodeInputError("Core reader received frame positions")
        value = _split_heads(self.value(memory_value), self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(_merge_heads(attended))


class CoreSlotReader(torch.nn.Module):
    """Read language-axis Core content into 320 routed LoRA slots."""

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
    ) -> torch.Tensor:
        return self.attention(
            routing,
            self.memory_norm(core),
            core,
            valid_core,
        )


class ProcedureSlotReader(torch.nn.Module):
    """Read stream-centered Procedure content with a Core-conditioned query."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(width)
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
        stream_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            stream_types.shape != valid_procedure.shape
            or stream_types.dtype != torch.long
            or bool(((stream_types < 0) & valid_procedure).any())
            or bool(((stream_types > 1) & valid_procedure).any())
        ):
            raise VariableEpisodeInputError("invalid Procedure stream identities")
        centered = torch.zeros_like(procedure)
        for stream in (0, 1):
            active = valid_procedure & stream_types.eq(stream)
            counts = active.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (
                procedure.masked_fill(~active[..., None], 0.0).sum(
                    dim=1,
                    keepdim=True,
                )
                / counts[..., None].to(procedure.dtype)
            )
            centered = centered + (procedure - mean).masked_fill(
                ~active[..., None],
                0.0,
            )
        conditioned_query = routing + self.query_norm(core_slots)
        slots = self.attention(
            conditioned_query,
            self.memory_norm(procedure),
            centered,
            valid_procedure,
            positions,
        )
        return slots, conditioned_query, centered


class PostFusionSlotBlock(torch.nn.Module):
    """Coordinate already fused LoRA slots without reopening either memory path."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.self_norm = RMSNorm(width)
        self.self_attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )
        self.output_norm = RMSNorm(width)

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
    ) -> torch.Tensor:
        addressed = self.self_norm(content) + routing
        attended, _ = self.self_attention(
            addressed,
            addressed,
            content,
            need_weights=False,
        )
        content = content + attended
        content = content + self.ffn(self.ffn_norm(content))
        return self.output_norm(content)


class ProcedureContentCompiler(torch.nn.Module):
    """Let Procedure contribute content and gate full-rank Core context."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid slot-normalized compiler")
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
        self.procedure_modulation_norm = RMSNorm(width)
        self.procedure_modulation_hidden = torch.nn.Linear(
            width,
            2 * width,
            bias=False,
        )
        self.procedure_modulation_output = torch.nn.Linear(
            2 * width,
            2 * width,
            bias=False,
        )
        self.core_content_norm = RMSNorm(width)
        self.core_content = torch.nn.Linear(width, width, bias=False)
        self.post_fusion = PostFusionSlotBlock(width=width, heads=heads)

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
        stream_types: torch.Tensor,
    ) -> None:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or procedure.ndim != 3
            or positions.shape != procedure.shape[:2]
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or stream_types.shape != procedure.shape[:2]
            or stream_types.dtype != torch.long
            or core.shape[0] != procedure.shape[0]
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
        stream_types: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_memories(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
            stream_types,
        )
        routing = self.routing_norm(self._routing())[None].expand(
            core.shape[0],
            -1,
            -1,
        )
        core_slots = self.core_reader(routing, core, valid_core)
        procedure_slots, conditioned_query, centered = self.procedure_reader(
            routing,
            core_slots,
            procedure,
            positions,
            valid_procedure,
            stream_types,
        )
        hidden = F.gelu(
            self.procedure_modulation_hidden(
                self.procedure_modulation_norm(procedure_slots)
            )
        )
        gamma, beta = self.procedure_modulation_output(hidden).chunk(2, dim=-1)
        gated_core = torch.tanh(gamma) * self.core_content(
            self.core_content_norm(core_slots)
        )
        fused = procedure_slots + beta + gated_core
        output = self.post_fusion(fused, routing)
        return output, {
            "core_slots": core_slots,
            "conditioned_query": conditioned_query,
            "procedure_centered": centered,
            "procedure_slots": procedure_slots,
            "procedure_gamma": gamma,
            "procedure_beta": beta,
            "gated_core": gated_core,
            "pre_coordination_slots": fused,
            "fused_slots": output,
        }

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
        stream_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content, _ = self.fused_slots(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
            stream_types,
        )
        expert_stop = self.EXPERT_LAYERS * self.RANK
        expert = content[:, :expert_stop].reshape(
            core.shape[0],
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
