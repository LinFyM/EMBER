"""Compile Core and teacher Procedure into addressed spectral LoRA factors."""

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
    """Cross-attend while routing and time affect Q/K, never memory values."""

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
            raise VariableEpisodeInputError("invalid content attention batch")
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
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(_merge_heads(attended))


class CoreTargetReader(torch.nn.Module):
    """Read task-semantic Core content into one slot per policy target."""

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


class ProcedureTargetReader(torch.nn.Module):
    """Read ordered teacher Procedure conditioned on the target's Core."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.core_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.attention = ContentCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=True,
        )

    @staticmethod
    def center(
        procedure: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> torch.Tensor:
        """Mask and time-center Procedure values in FP32."""

        if (
            procedure.ndim != 3
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Procedure centering batch")
        mask = valid_procedure[..., None]
        value = procedure.to(torch.float32)
        count = mask.sum(dim=1, keepdim=True).to(torch.float32)
        mean = (value * mask).sum(dim=1, keepdim=True) / count
        return (value - mean).masked_fill(~mask, 0.0).to(procedure.dtype)

    def forward(
        self,
        routing: torch.Tensor,
        core_targets: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_core = self.core_norm(core_targets)
        centered = self.center(procedure, valid_procedure)
        targets = self.attention(
            routing + normalized_core,
            self.memory_norm(procedure),
            centered,
            valid_procedure,
            positions,
        )
        return targets, normalized_core, centered


class TargetSlotBlock(torch.nn.Module):
    """Coordinate 38 semantic targets before algebraic rank expansion."""

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


def _signed_permutation_bank(
    count: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return cheap deterministic orthogonal coordinate transforms."""

    bank = torch.zeros(count, width, width)
    columns = torch.arange(width)
    for index in range(count):
        rows = torch.randperm(width, generator=generator)
        signs = (
            2
            * torch.randint(
                0,
                2,
                (width,),
                generator=generator,
                dtype=torch.int64,
            )
            - 1
        ).to(bank.dtype)
        bank[index, rows, columns] = signs
    return bank


class TargetSpectralCompiler(torch.nn.Module):
    """Fuse 38 semantic targets, then expand them into 16 addressed ranks."""

    TARGET_COUNT = 38
    RANK = 16

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid target-spectral compiler")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        routing = torch.empty(self.TARGET_COUNT, width)
        routing.normal_(mean=0.0, std=0.02, generator=generator)
        self.target_routing = torch.nn.Parameter(routing)
        self.routing_norm = RMSNorm(width)
        self.core_reader = CoreTargetReader(width=width, heads=heads)
        self.procedure_reader = ProcedureTargetReader(
            width=width,
            heads=heads,
        )
        self.procedure_norm = RMSNorm(width)
        self.modulation = torch.nn.Linear(width, 2 * width, bias=False)
        torch.nn.init.zeros_(self.modulation.weight)
        self.target_block = TargetSlotBlock(width=width, heads=heads)
        self.target_coordinates = torch.nn.Parameter(
            _signed_permutation_bank(self.TARGET_COUNT, width, generator)
        )
        self.rank_coordinates = torch.nn.Parameter(
            _signed_permutation_bank(self.RANK, width, generator)
        )

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
            raise VariableEpisodeInputError("invalid compiler memories")

    def target_slots(
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
        routing = self.routing_norm(self.target_routing)[None].expand(
            core.shape[0],
            -1,
            -1,
        )
        core_targets = self.core_reader(routing, core, valid_core)
        procedure_targets, normalized_core, centered = self.procedure_reader(
            routing,
            core_targets,
            procedure,
            positions,
            valid_procedure,
        )
        gamma, beta = self.modulation(
            self.procedure_norm(procedure_targets)
        ).chunk(2, dim=-1)
        fused = (1.0 + gamma) * normalized_core + beta
        targets = self.target_block(fused, routing)
        return targets, {
            "routing": routing,
            "core_targets": core_targets,
            "procedure_centered": centered,
            "procedure_targets": procedure_targets,
            "adaln_gamma": gamma,
            "adaln_beta": beta,
            "fused_targets": fused,
            "target_slots": targets,
        }

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> torch.Tensor:
        targets, _ = self.target_slots(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
        )
        addressed = torch.einsum(
            "btw,twv->btv",
            targets,
            self.target_coordinates.to(targets.dtype),
        )
        return torch.einsum(
            "btw,rwv->btrv",
            addressed,
            self.rank_coordinates.to(addressed.dtype),
        )
