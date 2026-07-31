"""Target-first, rank-last compiler for the Semantic Program Grid Writer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ember.writer.semantic_program import (
    RMSNorm,
    RawValueSelfAttention,
    SemanticProgramError,
    apply_two_axis_rope,
    merge_heads,
    split_heads,
)


class RawValueCrossAttention(torch.nn.Module):
    """Cross-attend with learned Q/K/O and physical memory content as V."""

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
        memory_qk_identity: torch.Tensor | None = None,
        first_positions: torch.Tensor | None = None,
        second_positions: torch.Tensor | None = None,
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
            raise SemanticProgramError("invalid raw-value cross-attention batch")
        if memory_qk_identity is not None:
            if memory_qk_identity.shape != memory_key.shape:
                raise SemanticProgramError("cross-attention identity shape changed")
            memory_key = memory_key + memory_qk_identity
        query = split_heads(self.query(query_key), self.heads)
        key = split_heads(self.key(memory_key), self.heads)
        using_positions = first_positions is not None or second_positions is not None
        if using_positions:
            if (
                first_positions is None
                or second_positions is None
                or first_positions.shape != memory_key.shape[:2]
                or second_positions.shape != memory_key.shape[:2]
                or first_positions.dtype != torch.long
                or second_positions.dtype != torch.long
            ):
                raise SemanticProgramError("invalid two-axis Program positions")
            zeros = torch.zeros(
                query_key.shape[:2], dtype=torch.long, device=query_key.device
            )
            query = apply_two_axis_rope(query, zeros, zeros)
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


class CoordinateMixer(torch.nn.Module):
    """Coordinate rank and target axes without imposing spectral geometry."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.rank_attention = RawValueSelfAttention(
            width=width, heads=heads, causal=False
        )
        self.target_attention = RawValueSelfAttention(
            width=width, heads=heads, causal=False
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def forward(
        self,
        content: torch.Tensor,
        target_identity: torch.Tensor,
        rank_identity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 4
            or target_identity.shape != (content.shape[1], content.shape[-1])
            or rank_identity.shape != (content.shape[2], content.shape[-1])
        ):
            raise SemanticProgramError("invalid target/rank coordinate batch")
        batch, targets, ranks, width = content.shape
        valid_rank = torch.ones(
            batch * targets, ranks, dtype=torch.bool, device=content.device
        )
        rank_positions = torch.arange(
            ranks, dtype=torch.long, device=content.device
        )[None].expand(batch * targets, -1)
        rank_qk = (target_identity[:, None] + rank_identity[None]).to(content.dtype)
        rank_qk = rank_qk[None].expand(batch, -1, -1, -1).reshape(
            batch * targets, ranks, width
        )
        ranked = content.reshape(batch * targets, ranks, width)
        ranked = ranked + self.rank_attention(
            ranked,
            valid_rank,
            positions=rank_positions,
            qk_identity=rank_qk,
        )
        content = ranked.reshape(batch, targets, ranks, width)

        targeted = content.permute(0, 2, 1, 3).reshape(
            batch * ranks, targets, width
        )
        valid_target = torch.ones(
            batch * ranks, targets, dtype=torch.bool, device=content.device
        )
        target_positions = torch.arange(
            targets, dtype=torch.long, device=content.device
        )[None].expand(batch * ranks, -1)
        target_qk = (rank_identity[:, None] + target_identity[None]).to(content.dtype)
        target_qk = target_qk[None].expand(batch, -1, -1, -1).reshape(
            batch * ranks, targets, width
        )
        targeted = targeted + self.target_attention(
            targeted,
            valid_target,
            positions=target_positions,
            qk_identity=target_qk,
        )
        content = targeted.reshape(batch, ranks, targets, width).permute(0, 2, 1, 3)
        return content + self.ffn(self.ffn_norm(content))


class TargetRankProgramCompiler(torch.nn.Module):
    """Read Core per target, then Program independently per target/rank."""

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
            raise SemanticProgramError("invalid target/rank compiler dimensions")
        self.target_count = int(target_count)
        self.rank = int(rank)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def identity(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.target_identity = identity(target_count)
        self.rank_identity = identity(rank)
        self.program_type_identity = identity(2)
        self.core_norm = RMSNorm(width)
        self.core_reader = RawValueCrossAttention(width=width, heads=heads)
        self.target_core_norm = RMSNorm(width)
        self.program_norm = RMSNorm(width)
        self.program_reader = RawValueCrossAttention(width=width, heads=heads)
        self.coordinate_mixer = CoordinateMixer(width=width, heads=heads)

    def compile_with_diagnostics(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or program.ndim != 4
            or program.shape[0] != core.shape[0]
            or program.shape[-1] != core.shape[-1]
            or endpoint_positions.shape != program.shape[:2]
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != program.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_semantics.shape != (program.shape[0], program.shape[2])
            or valid_semantics.dtype != torch.bool
            or core.shape[1] == 0
            or program.shape[1] == 0
            or program.shape[2] < 2
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_intervals.any(dim=1).all())
            or not bool(valid_semantics[:, 0].all())
        ):
            raise SemanticProgramError("invalid target/rank compiler memory")
        batch, intervals, columns, width = program.shape
        target_routing = self.target_identity[None].expand(batch, -1, -1)
        target_core = self.core_reader(
            target_routing,
            self.core_norm(core),
            core,
            valid_core,
        )

        normalized_target_core = self.target_core_norm(target_core)
        query = (
            self.target_identity[:, None]
            + self.rank_identity[None]
            + normalized_target_core[:, :, None]
        ).reshape(batch, self.target_count * self.rank, width)
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
        type_identity = torch.cat(
            (
                self.program_type_identity[:1],
                self.program_type_identity[1:2].expand(columns - 1, -1),
            ),
            dim=0,
        )
        memory_identity = type_identity[None, None].expand(
            batch, intervals, columns, width
        ).reshape(batch, intervals * columns, width)
        program_coordinates = self.program_reader(
            query,
            self.program_norm(flat_program),
            flat_program,
            valid_program,
            memory_qk_identity=memory_identity.to(flat_program.dtype),
            first_positions=first_positions,
            second_positions=second_positions,
        ).reshape(batch, self.target_count, self.rank, width)
        fused = target_core[:, :, None] + program_coordinates
        mixed = self.coordinate_mixer(
            fused,
            self.target_identity,
            self.rank_identity,
        )
        return mixed, {
            "target_core": target_core,
            "program_coordinates": program_coordinates,
            "fused_coordinates": fused,
            "mixed_coordinates": mixed,
        }

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> torch.Tensor:
        content, _ = self.compile_with_diagnostics(
            core,
            valid_core,
            program,
            endpoint_positions,
            valid_intervals,
            valid_semantics,
        )
        return content


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
