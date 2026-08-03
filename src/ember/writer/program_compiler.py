"""Target-first Program compiler and semantic direction-store decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ember.writer.semantic_program import (
    RMSNorm,
    SemanticProgramError,
    apply_rope,
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


class RoleProgramCrossAttention(torch.nn.Module):
    """Read one physical role history with normalized K and raw V."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 4:
            raise SemanticProgramError("invalid role Program cross-attention")
        self.heads = int(heads)
        self.memory_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_key: torch.Tensor,
        memory: torch.Tensor,
        valid_memory: torch.Tensor,
        *,
        memory_qk_identity: torch.Tensor,
        endpoint_positions: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query_key.ndim != 3
            or memory.ndim != 3
            or query_key.shape[0] != memory.shape[0]
            or query_key.shape[-1] != memory.shape[-1]
            or valid_memory.shape != memory.shape[:2]
            or valid_memory.dtype != torch.bool
            or memory_qk_identity.shape != memory.shape
            or endpoint_positions.shape != memory.shape[:2]
            or endpoint_positions.dtype != torch.long
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target/rank role memory")
        query = split_heads(self.query(query_key), self.heads)
        key = split_heads(
            self.key(self.memory_norm(memory) + memory_qk_identity), self.heads
        )
        query_positions = torch.zeros(
            query_key.shape[:2], dtype=torch.long, device=query_key.device
        )
        query = apply_rope(query, query_positions)
        key = apply_rope(key, endpoint_positions)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            split_heads(memory, self.heads),
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(merge_heads(attended))


class TargetBoundRoleCompiler(torch.nn.Module):
    """Read Core before Program binding, then read A/E/D privately per rank."""

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
        self.role_identity = identity(self.TYPE_COUNT)
        self.target_identity_norm = RMSNorm(width)
        self.rank_identity_norm = RMSNorm(width)
        self.role_identity_norm = RMSNorm(width)
        self.core_norm = RMSNorm(width)
        self.core_reader = CoreCrossAttention(width=width, heads=heads)
        self.program_reader = RoleProgramCrossAttention(width=width, heads=heads)

    def read_target_core(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized target addresses and their raw Core reads."""

        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or not bool(valid_core.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target Core memory")
        target_query = self.target_identity_norm(self.target_identity)[None].expand(
            core.shape[0], -1, -1
        )
        target_core = self.core_reader(
            target_query,
            self.core_norm(core),
            core,
            valid_core,
        )
        return target_query, target_core

    def compile_with_diagnostics(
        self,
        target_query: torch.Tensor,
        target_core: torch.Tensor,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            target_query.ndim != 3
            or target_core.shape != target_query.shape
            or target_query.shape[1] != self.target_count
            or program.ndim != 5
            or program.shape[:2] != target_query.shape[:2]
            or program.shape[3] != self.TYPE_COUNT
            or program.shape[-1] != target_query.shape[-1]
            or endpoint_positions.shape != (program.shape[0], program.shape[2])
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != endpoint_positions.shape
            or valid_intervals.dtype != torch.bool
            or not bool(valid_intervals.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target-bound role compiler memory")
        batch, targets, intervals, roles, width = program.shape
        rank_identity = self.rank_identity_norm(self.rank_identity)
        role_identity = self.role_identity_norm(self.role_identity).to(program.dtype)
        target_address = target_query + self.core_norm(target_core)
        program_query = (
            target_address[:, :, None, None]
            + rank_identity[None, None, :, None]
            + role_identity[None, None, None]
        )
        flat_query = program_query.permute(0, 1, 3, 2, 4).reshape(
            batch * targets * roles, self.rank, width
        )
        flat_program = program.permute(0, 1, 3, 2, 4).reshape(
            batch * targets * roles, intervals, width
        )
        valid_program = valid_intervals[:, None, None].expand(
            batch, targets, roles, intervals
        ).reshape(batch * targets * roles, intervals)
        positions = endpoint_positions[:, None, None].expand(
            batch, targets, roles, intervals
        ).reshape(batch * targets * roles, intervals)
        memory_identity = role_identity[None, None, :, None].expand(
            batch, targets, roles, intervals, width
        ).reshape(batch * targets * roles, intervals, width)
        role_read = self.program_reader(
            flat_query,
            flat_program,
            valid_program,
            memory_qk_identity=memory_identity,
            endpoint_positions=positions,
        ).reshape(batch, targets, roles, self.rank, width).permute(0, 1, 3, 2, 4)
        core_broadcast = target_core[:, :, None].expand(
            batch, targets, self.rank, width
        )
        coordinates = torch.cat(
            (core_broadcast, role_read.reshape(batch, targets, self.rank, roles * width)),
            dim=-1,
        )
        return coordinates, {
            "target_query": target_query,
            "program_query": program_query,
            "core_read": target_core,
            "role_read": role_read,
            "coordinates": coordinates,
        }

    def forward(
        self,
        target_query: torch.Tensor,
        target_core: torch.Tensor,
        program: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
    ) -> torch.Tensor:
        coordinates, _ = self.compile_with_diagnostics(
            target_query,
            target_core,
            program,
            endpoint_positions,
            valid_intervals,
        )
        return coordinates


class SemanticDirectionRouter(torch.nn.Module):
    """Select two fixed semantic stores from a frozen language anchor."""

    def __init__(
        self,
        centers: torch.Tensor,
        *,
        anchor_mean: torch.Tensor,
        top_k: int,
    ) -> None:
        super().__init__()
        if (
            centers.ndim != 2
            or min(centers.shape) <= 0
            or top_k != 2
            or centers.shape[0] <= top_k
            or anchor_mean.shape != (centers.shape[1],)
            or not bool(torch.isfinite(centers).all())
            or not bool(torch.isfinite(anchor_mean).all())
        ):
            raise SemanticProgramError("invalid semantic direction centers")
        normalized = F.normalize(centers.to(torch.float32), dim=-1)
        if not bool(torch.isfinite(normalized).all()):
            raise SemanticProgramError("invalid normalized direction centers")
        self.store_count = int(centers.shape[0])
        self.anchor_width = int(centers.shape[1])
        self.top_k = int(top_k)
        self.register_buffer("centers", normalized.contiguous(), persistent=True)
        self.register_buffer(
            "anchor_mean",
            anchor_mean.to(torch.float32).contiguous(),
            persistent=True,
        )

    def forward(self, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            anchor.ndim != 2
            or anchor.shape[-1] != self.anchor_width
        ):
            raise SemanticProgramError("semantic direction route lost task anchor")
        normalized = F.normalize(
            anchor.to(torch.float32) - self.anchor_mean,
            dim=-1,
        )
        indices = torch.topk(
            normalized @ self.centers.transpose(0, 1),
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        weights = torch.full(
            indices.shape,
            1.0 / self.top_k,
            dtype=anchor.dtype,
            device=anchor.device,
        )
        return indices, weights


class SemanticDirectionStoreHead(torch.nn.Module):
    """Decode coordinates with complete, independently owned factor stores."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        store_count: int,
    ) -> None:
        super().__init__()
        if min(input_width, hidden_width, output_width, store_count) <= 0:
            raise SemanticProgramError("invalid direction-store factor dimensions")
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.output_width = int(output_width)
        self.store_count = int(store_count)
        self.input_weight = torch.nn.Parameter(
            torch.empty(store_count, hidden_width, input_width)
        )
        self.output_weight = torch.nn.Parameter(
            torch.zeros(store_count, output_width, hidden_width)
        )
        for weight in self.input_weight:
            torch.nn.init.kaiming_uniform_(weight, a=5**0.5)

    def forward(
        self,
        value: torch.Tensor,
        store_indices: torch.Tensor,
        store_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            value.ndim != 3
            or value.shape[-1] != self.input_width
            or store_indices.ndim != 2
            or store_indices.shape[0] != value.shape[0]
            or store_weights.shape != store_indices.shape
            or store_indices.dtype != torch.long
        ):
            raise SemanticProgramError("factor head lost semantic direction route")
        selected_input = self.input_weight[store_indices]
        hidden = F.gelu(
            torch.einsum("bri,bkhi->bkrh", value, selected_input)
        )
        selected_output = self.output_weight[store_indices]
        output = torch.einsum(
            "bkrh,bkoh->bkro",
            hidden,
            selected_output,
        )
        return torch.einsum("bk,bkro->bro", store_weights, output)
