"""Single-stage target/rank reader for the Unified Causal Program."""

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


class RawValueCrossAttention(torch.nn.Module):
    """Cross-attend with learned Q/K/O and physical Program content as V."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 4:
            raise SemanticProgramError("invalid raw-value cross-attention")
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
        first_positions: torch.Tensor,
        second_positions: torch.Tensor,
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
            or first_positions.shape != memory_key.shape[:2]
            or second_positions.shape != memory_key.shape[:2]
            or first_positions.dtype != torch.long
            or second_positions.dtype != torch.long
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid raw-value cross-attention batch")
        query = split_heads(self.query(query_key), self.heads)
        key = split_heads(
            self.key(memory_key + memory_qk_identity), self.heads
        )
        query_positions = torch.zeros(
            query_key.shape[:2], dtype=torch.long, device=query_key.device
        )
        query = apply_two_axis_rope(query, query_positions, query_positions)
        key = apply_two_axis_rope(key, first_positions, second_positions)
        value = split_heads(memory_value, self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(merge_heads(attended))


class TargetRankProgramReader(torch.nn.Module):
    """Let each real policy target/rank coordinate read the full Program once."""

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
            raise SemanticProgramError("invalid target/rank reader dimensions")
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
        self.program_norm = RMSNorm(width)
        self.reader = RawValueCrossAttention(width=width, heads=heads)

    @staticmethod
    def _task_tokens(columns: int) -> int:
        if columns < 3 or (columns - 1) % 2:
            raise SemanticProgramError("reader Program column topology changed")
        return (columns - 1) // 2

    def _type_identity(self, columns: int) -> torch.Tensor:
        task_tokens = self._task_tokens(columns)
        identities = self.program_type_identity_norm(
            self.program_type_identity
        )
        return torch.cat(
            (
                identities[0:1].expand(task_tokens, -1),
                identities[1:2],
                identities[2:3].expand(task_tokens, -1),
            ),
            dim=0,
        )

    def compile_with_diagnostics(
        self,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            program.ndim != 4
            or endpoint_positions.shape != program.shape[:2]
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != program.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_semantics.shape != (program.shape[0], program.shape[2])
            or valid_semantics.dtype != torch.bool
            or program.shape[1] == 0
            or not bool(valid_intervals.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target/rank Program memory")
        batch, intervals, columns, width = program.shape
        self._task_tokens(columns)
        target_identity = self.target_identity_norm(self.target_identity)
        rank_identity = self.rank_identity_norm(self.rank_identity)
        query = (target_identity[:, None] + rank_identity[None]).reshape(
            1, self.target_count * self.rank, width
        ).expand(batch, -1, -1)
        flat_program = program.reshape(batch, intervals * columns, width)
        valid_program = (
            valid_intervals[:, :, None] & valid_semantics[:, None]
        ).reshape(batch, intervals * columns)
        first_positions = endpoint_positions[:, :, None].expand(
            batch, intervals, columns
        ).reshape(batch, intervals * columns)
        second_positions = torch.arange(
            columns, dtype=torch.long, device=program.device
        )[None, None].expand(batch, intervals, columns).reshape(
            batch, intervals * columns
        )
        type_identity = self._type_identity(columns).to(program.dtype)
        memory_identity = type_identity[None, None].expand(
            batch, intervals, columns, width
        ).reshape(batch, intervals * columns, width)
        coordinates = self.reader(
            query,
            self.program_norm(flat_program),
            flat_program,
            valid_program,
            memory_qk_identity=memory_identity,
            first_positions=first_positions,
            second_positions=second_positions,
        ).reshape(batch, self.target_count, self.rank, width)
        return coordinates, {
            "coordinate_query": query,
            "coordinates": coordinates,
        }

    def forward(
        self,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> torch.Tensor:
        coordinates, _ = self.compile_with_diagnostics(
            program,
            endpoint_positions,
            valid_intervals,
            valid_semantics,
        )
        return coordinates


class FactorHead(torch.nn.Module):
    """Decode one target/rank state into one public LoRA factor row."""

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
