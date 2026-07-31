"""Semantic Core and causal axial Semantic Program Grid."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SemanticProgramError(ValueError):
    """Raised when variable-length semantic-program tensors violate contract."""


class RMSNorm(torch.nn.Module):
    """Small dtype-stable RMS normalization."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0:
            raise SemanticProgramError("RMSNorm width must be positive")
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + self.eps
        ).to(value.dtype)
        return value * scale * self.weight


def split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = value.shape
    if width % heads:
        raise SemanticProgramError("attention width is not divisible by heads")
    return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * width)


def apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply one-dimensional RoPE to ``[B,H,T,D]`` Q/K tensors."""

    width = value.shape[-1]
    if width % 2 or positions.shape != (value.shape[0], value.shape[2]):
        raise SemanticProgramError("invalid ordinal RoPE request")
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
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def apply_two_axis_rope(
    value: torch.Tensor,
    first_positions: torch.Tensor,
    second_positions: torch.Tensor,
) -> torch.Tensor:
    """Apply independent RoPE halves for two memory-address axes."""

    width = value.shape[-1]
    if width % 4:
        raise SemanticProgramError("two-axis RoPE head width must divide by four")
    first, second = value.split(width // 2, dim=-1)
    return torch.cat(
        (
            apply_rope(first, first_positions),
            apply_rope(second, second_positions),
        ),
        dim=-1,
    )


class RawValueSelfAttention(torch.nn.Module):
    """Self-attend with identities in Q/K and unnormalized content as V."""

    def __init__(self, *, width: int, heads: int, causal: bool) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise SemanticProgramError("invalid raw-value attention dimensions")
        self.heads = int(heads)
        self.causal = bool(causal)
        self.norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        content: torch.Tensor,
        valid: torch.Tensor,
        *,
        positions: torch.Tensor,
        qk_identity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 3
            or valid.shape != content.shape[:2]
            or valid.dtype != torch.bool
            or positions.shape != content.shape[:2]
            or positions.dtype != torch.long
            or qk_identity.shape != content.shape
        ):
            raise SemanticProgramError("invalid raw-value self-attention batch")
        addressed = self.norm(content) + qk_identity
        query = apply_rope(split_heads(self.query(addressed), self.heads), positions)
        key = apply_rope(split_heads(self.key(addressed), self.heads), positions)
        value = split_heads(content, self.heads)
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
        allowed = safe_valid[:, None, None, :]
        if self.causal:
            tokens = content.shape[1]
            causal = torch.ones(
                tokens, tokens, dtype=torch.bool, device=content.device
            ).tril()
            allowed = allowed & causal[None, None]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        output = self.output(merge_heads(attended))
        return output.masked_fill(~valid[..., None], 0.0)


class ContentTransformerBlock(torch.nn.Module):
    """Historical v6-style pre-norm content block with Q/K-only RoPE."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise SemanticProgramError("invalid content Transformer dimensions")
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
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 3
            or positions.shape != content.shape[:2]
            or positions.dtype != torch.long
            or valid.shape != content.shape[:2]
            or valid.dtype != torch.bool
            or not bool(valid.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid content Transformer batch")
        normalized = self.attention_norm(content)
        query = apply_rope(split_heads(self.query(normalized), self.heads), positions)
        key = apply_rope(split_heads(self.key(normalized), self.heads), positions)
        value = split_heads(self.value(content), self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        content = content + self.output(merge_heads(attended))
        content = content + self.ffn(self.ffn_norm(content))
        return content.masked_fill(~valid[..., None], 0.0)


class TaskSelectedSemanticSetFusion(torch.nn.Module):
    """Add task-selected centered frame evidence to a stable mean backbone."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise SemanticProgramError("invalid semantic-set fusion")
        self.heads = int(heads)
        self.head_width = width // heads
        self.query_norm = RMSNorm(width)
        self.evidence_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.mean = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        text_queries: torch.Tensor,
        frame_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            text_queries.ndim != 3
            or frame_evidence.ndim != 4
            or frame_evidence.shape[0] != text_queries.shape[0]
            or frame_evidence.shape[2:] != text_queries.shape[1:]
            or valid_frames.shape != frame_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape != text_queries.shape[:2]
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames.any(dim=1).all())
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid token-aligned frame evidence")
        batch, frames, tokens, width = frame_evidence.shape
        active = valid_frames[:, :, None, None]
        counts = valid_frames.sum(dim=1).to(frame_evidence.dtype)[:, None, None]
        frame_mean = frame_evidence.masked_fill(~active, 0.0).sum(dim=1) / counts
        centered = (frame_evidence - frame_mean[:, None]).masked_fill(~active, 0.0)
        query = split_heads(self.query(self.query_norm(text_queries)), self.heads)
        key = self.key(self.evidence_norm(frame_evidence)).reshape(
            batch, frames, tokens, self.heads, self.head_width
        ).permute(0, 3, 1, 2, 4)
        value = centered.reshape(
            batch, frames, tokens, self.heads, self.head_width
        ).permute(0, 3, 1, 2, 4)
        logits = torch.einsum("bhld,bhtld->bhtl", query, key)
        logits = logits / math.sqrt(self.head_width)
        logits = logits.masked_fill(
            ~valid_frames[:, None, :, None], torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits.to(torch.float32), dim=2).to(logits.dtype)
        attended = torch.einsum("bhtl,bhtld->bhld", weights, value)
        output = self.mean(frame_mean) + self.output(merge_heads(attended))
        output = output.masked_fill(~valid_task_tokens[..., None], 0.0)
        weights = weights.masked_fill(
            ~valid_task_tokens[:, None, None, :], 0.0
        )
        return output, weights


class LanguageSemanticCore(torch.nn.Module):
    """Permutation-invariant mean-backed semantic carrier over task tokens."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if blocks <= 0:
            raise SemanticProgramError("Semantic Core needs language blocks")
        self.semantic_set_fusion = TaskSelectedSemanticSetFusion(
            width=width, heads=heads
        )
        self.blocks = torch.nn.ModuleList(
            ContentTransformerBlock(width=width, heads=heads) for _ in range(blocks)
        )

    def forward(
        self,
        text_queries: torch.Tensor,
        frame_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, weights = self.semantic_set_fusion(
            text_queries, frame_evidence, valid_frames, valid_task_tokens
        )
        positions = torch.arange(
            content.shape[1], dtype=torch.long, device=content.device
        )[None].expand(content.shape[0], -1)
        for block in self.blocks:
            content = block(content, positions, valid_task_tokens)
        return content, weights


class SemanticProgramBlock(torch.nn.Module):
    """Interval-local then column-causal axial reasoning over raw content."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.type_identity = torch.nn.Parameter(torch.empty(2, width))
        torch.nn.init.normal_(self.type_identity, mean=0.0, std=0.02)
        self.local_attention = RawValueSelfAttention(
            width=width, heads=heads, causal=False
        )
        self.temporal_attention = RawValueSelfAttention(
            width=width, heads=heads, causal=True
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def _column_identity(self, columns: int) -> torch.Tensor:
        return torch.cat(
            (
                self.type_identity[:1],
                self.type_identity[1:2].expand(columns - 1, -1),
            ),
            dim=0,
        )

    def forward(
        self,
        content: torch.Tensor,
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        valid_semantics: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 4
            or endpoint_positions.shape != content.shape[:2]
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != content.shape[:2]
            or valid_intervals.dtype != torch.bool
            or valid_semantics.shape != (content.shape[0], content.shape[2])
            or valid_semantics.dtype != torch.bool
            or not bool(valid_intervals.any(dim=1).all())
            or not bool(valid_semantics[:, 0].all())
        ):
            raise SemanticProgramError("invalid axial Semantic Program batch")
        batch, intervals, columns, width = content.shape
        valid_grid = valid_intervals[:, :, None] & valid_semantics[:, None, :]
        identities = self._column_identity(columns).to(content.dtype)

        local_content = content.reshape(batch * intervals, columns, width)
        local_valid = valid_grid.reshape(batch * intervals, columns)
        local_positions = torch.arange(
            columns, dtype=torch.long, device=content.device
        )[None].expand(batch * intervals, -1)
        local_identity = identities[None].expand(batch * intervals, -1, -1)
        local_delta = self.local_attention(
            local_content,
            local_valid,
            positions=local_positions,
            qk_identity=local_identity,
        )
        content = (local_content + local_delta).reshape(
            batch, intervals, columns, width
        )
        content = content.masked_fill(~valid_grid[..., None], 0.0)

        temporal_content = content.permute(0, 2, 1, 3).reshape(
            batch * columns, intervals, width
        )
        temporal_valid = valid_grid.permute(0, 2, 1).reshape(
            batch * columns, intervals
        )
        temporal_positions = endpoint_positions[:, None].expand(
            batch, columns, intervals
        ).reshape(batch * columns, intervals)
        temporal_identity = identities[None, :, None].expand(
            batch, columns, intervals, width
        ).reshape(batch * columns, intervals, width)
        temporal_delta = self.temporal_attention(
            temporal_content,
            temporal_valid,
            positions=temporal_positions,
            qk_identity=temporal_identity,
        )
        content = (temporal_content + temporal_delta).reshape(
            batch, columns, intervals, width
        ).permute(0, 2, 1, 3)
        content = content + self.ffn(self.ffn_norm(content))
        return content.masked_fill(~valid_grid[..., None], 0.0)


class SemanticProgramGrid(torch.nn.Module):
    """Build ``Action + task-grounded change`` intervals and reason axially."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if blocks <= 0:
            raise SemanticProgramError("Semantic Program needs axial blocks")
        self.blocks = torch.nn.ModuleList(
            SemanticProgramBlock(width=width, heads=heads) for _ in range(blocks)
        )

    def forward(
        self,
        grounded_evidence: torch.Tensor,
        action_probe: torch.Tensor,
        frame_positions: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            grounded_evidence.ndim != 4
            or action_probe.shape
            != (*grounded_evidence.shape[:2], grounded_evidence.shape[-1])
            or frame_positions.shape != grounded_evidence.shape[:2]
            or frame_positions.dtype != torch.long
            or valid_frames.shape != grounded_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (grounded_evidence.shape[0], grounded_evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or bool((valid_frames.sum(dim=1) < 2).any())
        ):
            raise SemanticProgramError("invalid Semantic Program evidence")
        valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
        endpoint_positions = frame_positions[:, 1:]
        changes = grounded_evidence[:, 1:] - grounded_evidence[:, :-1]
        program = torch.cat((action_probe[:, :-1, None], changes), dim=2)
        valid_semantics = torch.cat(
            (
                torch.ones(
                    valid_task_tokens.shape[0],
                    1,
                    dtype=torch.bool,
                    device=valid_task_tokens.device,
                ),
                valid_task_tokens,
            ),
            dim=1,
        )
        valid_grid = valid_intervals[:, :, None] & valid_semantics[:, None]
        program = program.masked_fill(~valid_grid[..., None], 0.0)
        for block in self.blocks:
            program = block(
                program,
                endpoint_positions,
                valid_intervals,
                valid_semantics,
            )
        return program, endpoint_positions, valid_intervals, valid_semantics
