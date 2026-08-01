"""Asymmetric Core and Program readers for the AP-ADR Writer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ember.writer.semantic_program import (
    RMSNorm,
    SemanticProgramError,
    apply_two_axis_rope,
    merge_heads,
    split_heads,
)


class CoreCrossAttention(torch.nn.Module):
    """Let target queries select raw Core content with a private softmax."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise SemanticProgramError("invalid Core cross-attention")
        self.heads = int(heads)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_key: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
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
            raise SemanticProgramError("invalid target-only Core memory")
        attended = F.scaled_dot_product_attention(
            split_heads(self.query(query_key), self.heads),
            split_heads(self.key(memory_key), self.heads),
            split_heads(memory_value, self.heads),
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(merge_heads(attended))


class ProgramCrossAttention(torch.nn.Module):
    """Let target/rank queries select raw Program values using contextual keys."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 4:
            raise SemanticProgramError("invalid Program cross-attention")
        self.heads = int(heads)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_key: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        valid_memory: torch.Tensor,
        *,
        memory_qk_identity: torch.Tensor,
        endpoint_positions: torch.Tensor,
        semantic_positions: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query_key.ndim != 3
            or memory_key.ndim != 3
            or memory_value.shape != memory_key.shape
            or query_key.shape[0] != memory_key.shape[0]
            or query_key.shape[-1] != memory_key.shape[-1]
            or valid_memory.shape != memory_key.shape[:2]
            or valid_memory.dtype != torch.bool
            or memory_qk_identity.shape != memory_key.shape
            or endpoint_positions.shape != memory_key.shape[:2]
            or semantic_positions.shape != memory_key.shape[:2]
            or endpoint_positions.dtype != torch.long
            or semantic_positions.dtype != torch.long
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target/rank Program memory")
        query = split_heads(self.query(query_key), self.heads)
        key = split_heads(
            self.key(memory_key + memory_qk_identity), self.heads
        )
        query_positions = torch.zeros(
            query_key.shape[:2], dtype=torch.long, device=query_key.device
        )
        query = apply_two_axis_rope(query, query_positions, query_positions)
        key = apply_two_axis_rope(key, endpoint_positions, semantic_positions)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            split_heads(memory_value, self.heads),
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(merge_heads(attended))


class AsymmetricDualReader(torch.nn.Module):
    """Read Core once per target and Program once per target/rank coordinate."""

    TYPE_COUNT = 3

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        target_count: int,
        rank: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads, target_count, rank) <= 0 or width % heads:
            raise SemanticProgramError("invalid asymmetric reader dimensions")
        self.target_count = int(target_count)
        self.rank = int(rank)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def identity(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.target_identity = identity(target_count)
        self.rank_identity = identity(rank)
        self.program_type_identity = identity(self.TYPE_COUNT)
        self.target_identity_norm = RMSNorm(width)
        self.rank_identity_norm = RMSNorm(width)
        self.program_type_identity_norm = RMSNorm(width)
        self.core_norm = RMSNorm(width)
        self.program_norm = RMSNorm(width)
        self.core_reader = CoreCrossAttention(width=width, heads=heads)
        self.program_reader = ProgramCrossAttention(width=width, heads=heads)

    @staticmethod
    def _task_tokens(columns: int) -> int:
        if columns < 3 or (columns - 1) % 2:
            raise SemanticProgramError("reader Program column topology changed")
        return (columns - 1) // 2

    def _type_identity(self, columns: int) -> torch.Tensor:
        task_tokens = self._task_tokens(columns)
        identity = self.program_type_identity_norm(self.program_type_identity)
        return torch.cat(
            (
                identity[0:1],
                identity[1:2].expand(task_tokens, -1),
                identity[2:3].expand(task_tokens, -1),
            ),
            dim=0,
        )

    @classmethod
    def _semantic_ordinals(cls, columns: int, device: torch.device) -> torch.Tensor:
        task_tokens = cls._task_tokens(columns)
        token_ordinals = torch.arange(task_tokens, dtype=torch.long, device=device)
        return torch.cat(
            (torch.zeros(1, dtype=torch.long, device=device), token_ordinals, token_ordinals)
        )

    def compile_with_diagnostics(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        program_key: torch.Tensor,
        program_value: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or program_key.ndim != 4
            or program_value.shape != program_key.shape
            or core.shape[0] != program_key.shape[0]
            or core.shape[-1] != program_key.shape[-1]
            or endpoint_positions.shape != program_key.shape[:2]
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != program_key.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_semantics.shape
            != (program_key.shape[0], program_key.shape[2])
            or valid_semantics.dtype != torch.bool
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_intervals.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid asymmetric dual-reader memory")
        batch, intervals, columns, width = program_key.shape
        self._task_tokens(columns)
        target_query = self.target_identity_norm(self.target_identity)[None].expand(
            batch, -1, -1
        )
        core_read = self.core_reader(
            target_query,
            self.core_norm(core),
            core,
            valid_core,
        )

        rank_identity = self.rank_identity_norm(self.rank_identity)
        program_query = (
            target_query[:, :, None] + rank_identity[None, None]
        ).reshape(batch, self.target_count * self.rank, width)
        flat_key = program_key.reshape(batch, intervals * columns, width)
        flat_value = program_value.reshape(batch, intervals * columns, width)
        valid_program = (
            valid_intervals[:, :, None] & valid_semantics[:, None]
        ).reshape(batch, intervals * columns)
        first_positions = endpoint_positions[:, :, None].expand(
            batch, intervals, columns
        ).reshape(batch, intervals * columns)
        ordinals = self._semantic_ordinals(columns, program_key.device)
        second_positions = ordinals[None, None].expand(
            batch, intervals, columns
        ).reshape(batch, intervals * columns)
        type_identity = self._type_identity(columns).to(program_key.dtype)
        memory_identity = type_identity[None, None].expand(
            batch, intervals, columns, width
        ).reshape(batch, intervals * columns, width)
        program_read = self.program_reader(
            program_query,
            self.program_norm(flat_key),
            flat_value,
            valid_program,
            memory_qk_identity=memory_identity,
            endpoint_positions=first_positions,
            semantic_positions=second_positions,
        ).reshape(batch, self.target_count, self.rank, width)
        core_broadcast = core_read[:, :, None].expand(
            batch, self.target_count, self.rank, width
        )
        coordinates = torch.cat((core_broadcast, program_read), dim=-1)
        return coordinates, {
            "target_query": target_query,
            "program_query": program_query,
            "core_read": core_read,
            "program_read": program_read,
            "coordinates": coordinates,
        }

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        program_key: torch.Tensor,
        program_value: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> torch.Tensor:
        coordinates, _ = self.compile_with_diagnostics(
            core,
            valid_core,
            program_key,
            program_value,
            endpoint_positions,
            valid_intervals,
            valid_semantics,
        )
        return coordinates


class FactorHead(torch.nn.Module):
    """Decode one raw Core/Program coordinate into one public LoRA factor row."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        if min(input_width, hidden_width, output_width) <= 0:
            raise SemanticProgramError("invalid LoRA factor-head dimensions")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=False),
        )
        torch.nn.init.zeros_(self.network[-1].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise SemanticProgramError("factor head lost its rank dimension")
        return self.network(value)
