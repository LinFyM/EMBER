"""Semantic Core compilation and causal Procedure refinement for PI05 Writer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length video-program batch violates its contract."""


class RMSNorm(torch.nn.Module):
    """Small dtype-stable RMS normalization."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0:
            raise VariableEpisodeInputError("RMSNorm width must be positive")
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + self.eps
        ).to(value.dtype)
        return value * scale * self.weight


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply one-dimensional RoPE to ``[B,H,T,D]`` query or key tensors."""

    width = value.shape[-1]
    if width % 2 or positions.shape != (value.shape[0], value.shape[2]):
        raise VariableEpisodeInputError("invalid video ordinal RoPE request")
    inverse_frequency = torch.exp(
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = (
        positions.to(torch.float32)[:, None, :, None]
        * inverse_frequency[None, None, None]
    )
    cosine = torch.cos(angles).to(value.dtype)
    sine = torch.sin(angles).to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)


class CausalProcedureBlock(torch.nn.Module):
    """Pre-norm global causal attention over per-frame interaction content."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid causal Procedure dimensions")
        self.heads = int(heads)
        self.head_width = width // heads
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
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = content.shape
        normalized = self.attention_norm(content)

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch,
                tokens,
                self.heads,
                self.head_width,
            ).transpose(1, 2)

        query = _apply_rope(heads(self.query(normalized)), positions)
        key = _apply_rope(heads(self.key(normalized)), positions)
        value = heads(self.value(content))
        causal = torch.ones(
            tokens,
            tokens,
            dtype=torch.bool,
            device=content.device,
        ).tril()
        allowed = (
            valid_mask[:, None, None, :]
            & causal[None, None, :, :]
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        content = content + self.output(attended)
        content = content + self.ffn(self.ffn_norm(content))
        return content.masked_fill(~valid_mask[..., None], 0.0)


class CausalProcedureEncoder(torch.nn.Module):
    """Keep one causally contextualized Procedure token per sampled frame."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if min(width, heads, blocks) <= 0:
            raise VariableEpisodeInputError("invalid causal Procedure encoder")
        self.blocks = torch.nn.ModuleList(
            CausalProcedureBlock(width, heads) for _ in range(blocks)
        )

    def forward(
        self,
        content: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 3
            or positions.shape != content.shape[:2]
            or positions.dtype != torch.long
            or valid_mask.shape != content.shape[:2]
            or valid_mask.dtype != torch.bool
            or not bool(valid_mask[:, 0].all())
        ):
            raise VariableEpisodeInputError("invalid causal Procedure batch")
        value = content.masked_fill(~valid_mask[..., None], 0.0)
        for block in self.blocks:
            value = block(value, positions, valid_mask)
        return value.masked_fill(~valid_mask[..., None], 0.0)


class CoreCompilerBlock(torch.nn.Module):
    """Compile unordered Semantic Core memory into routed LoRA-slot content."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.cross_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.cross_attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
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

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
        memory: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.cross_norm(content) + routing,
            self.memory_norm(memory),
            memory,
            key_padding_mask=~valid_memory,
            need_weights=False,
        )
        content = content + attended
        addressed = self.self_norm(content) + routing
        attended, _ = self.self_attention(
            addressed,
            addressed,
            content,
            need_weights=False,
        )
        content = content + attended
        return content + self.ffn(self.ffn_norm(content))


class RotaryProcedureCrossAttention(torch.nn.Module):
    """Read ordered Procedure memory while transmitting only raw content values."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid Procedure cross-attention")
        self.heads = int(heads)
        self.head_width = width // heads
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(
        self,
        query_content: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        positions: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, width = query_content.shape
        memory_tokens = memory_key.shape[1]

        def query_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch,
                queries,
                self.heads,
                self.head_width,
            ).transpose(1, 2)

        def memory_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch,
                memory_tokens,
                self.heads,
                self.head_width,
            ).transpose(1, 2)

        query = _apply_rope(
            query_heads(self.query(query_content)),
            torch.zeros(
                batch,
                queries,
                dtype=torch.long,
                device=query_content.device,
            ),
        )
        key = _apply_rope(memory_heads(self.key(memory_key)), positions)
        value = memory_heads(self.value(memory_value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, queries, width)
        return self.output(attended)


class ProcedureRefinerBlock(torch.nn.Module):
    """Produce an independent zero-preserving Procedure delta to Core content."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.cross_attention = RotaryProcedureCrossAttention(width, heads)
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

    def forward(
        self,
        core_content: torch.Tensor,
        routing: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.cross_attention(
            self.query_norm(core_content) + routing,
            self.memory_norm(memory),
            memory,
            positions,
            valid_memory,
        )
        addressed = self.self_norm(delta) + routing
        attended, _ = self.self_attention(
            addressed,
            addressed,
            delta,
            need_weights=False,
        )
        delta = delta + attended
        return delta + self.ffn(self.ffn_norm(delta))


class CoreProcedureLoRACompiler(torch.nn.Module):
    """Compile Core first, then add a directed Procedure refinement."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        core_blocks: int,
        procedure_blocks: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(width, heads, core_blocks, procedure_blocks) <= 0
            or width % heads
        ):
            raise VariableEpisodeInputError("invalid Core/Procedure compiler")
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
        self.core_blocks = torch.nn.ModuleList(
            CoreCompilerBlock(width, heads) for _ in range(core_blocks)
        )
        self.procedure_blocks = torch.nn.ModuleList(
            ProcedureRefinerBlock(width, heads)
            for _ in range(procedure_blocks)
        )
        self.output_norm = RMSNorm(width)

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

    def forward(
        self,
        core_memory: torch.Tensor,
        valid_core: torch.Tensor,
        procedure_memory: torch.Tensor,
        procedure_positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            core_memory.ndim != 3
            or valid_core.shape != core_memory.shape[:2]
            or valid_core.dtype != torch.bool
            or procedure_memory.ndim != 3
            or procedure_positions.shape != procedure_memory.shape[:2]
            or valid_procedure.shape != procedure_memory.shape[:2]
            or valid_procedure.dtype != torch.bool
            or core_memory.shape[0] != procedure_memory.shape[0]
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Core/Procedure compiler memory")
        routing = self.routing_norm(self._routing())[None].expand(
            core_memory.shape[0],
            -1,
            -1,
        )
        core_content = core_memory.new_zeros(
            core_memory.shape[0],
            self.QUERY_COUNT,
            core_memory.shape[-1],
        )
        for block in self.core_blocks:
            core_content = block(
                core_content,
                routing,
                core_memory,
                valid_core,
            )
        delta = core_content.new_zeros(core_content.shape)
        for block in self.procedure_blocks:
            delta = delta + block(
                core_content + delta,
                routing,
                procedure_memory,
                procedure_positions,
                valid_procedure,
            )
        content = self.output_norm(core_content + delta)
        expert_stop = self.EXPERT_LAYERS * self.RANK
        expert = content[:, :expert_stop].reshape(
            core_memory.shape[0],
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
