"""Unified absolute-state, Action, and change causal Program."""

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
    """Apply independent RoPE halves for interval and semantic-column axes."""

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
    """Self-attend with normalized content and identities in Q/K, raw V."""

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
        return self.output(merge_heads(attended)).masked_fill(
            ~valid[..., None], 0.0
        )


class UnifiedProgramBlock(torch.nn.Module):
    """Reason within each causal interval, then along each semantic column."""

    TYPE_COUNT = 3

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        identity = torch.empty(self.TYPE_COUNT, width)
        identity.normal_(mean=0.0, std=0.02, generator=generator)
        self.type_identity = torch.nn.Parameter(identity)
        self.type_identity_norm = RMSNorm(width)
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

    @staticmethod
    def _task_tokens(columns: int) -> int:
        if columns < 3 or (columns - 1) % 2:
            raise SemanticProgramError("Unified Program column topology changed")
        return (columns - 1) // 2

    def _column_identity(self, columns: int) -> torch.Tensor:
        task_tokens = self._task_tokens(columns)
        identities = self.type_identity_norm(self.type_identity)
        return torch.cat(
            (
                identities[0:1].expand(task_tokens, -1),
                identities[1:2],
                identities[2:3].expand(task_tokens, -1),
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
        ):
            raise SemanticProgramError("invalid unified causal Program batch")
        batch, intervals, columns, width = content.shape
        task_tokens = self._task_tokens(columns)
        if (
            not bool(valid_semantics[:, :task_tokens].any(dim=1).all())
            or not bool(valid_semantics[:, task_tokens].all())
        ):
            raise SemanticProgramError("Unified Program lost absolute or Action value")
        valid_grid = valid_intervals[:, :, None] & valid_semantics[:, None]
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


class UnifiedCausalProgram(torch.nn.Module):
    """Build interval-aligned ``absolute X + Action + outgoing change D``."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        blocks: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise SemanticProgramError("Unified Program needs axial blocks")
        self.blocks = torch.nn.ModuleList(
            UnifiedProgramBlock(
                width=width,
                heads=heads,
                initialization_seed=initialization_seed + index,
            )
            for index in range(blocks)
        )

    def forward(
        self,
        absolute_evidence: torch.Tensor,
        grounded_evidence: torch.Tensor,
        action_probe: torch.Tensor,
        frame_positions: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            absolute_evidence.ndim != 4
            or grounded_evidence.shape != absolute_evidence.shape
            or action_probe.shape
            != (*absolute_evidence.shape[:2], absolute_evidence.shape[-1])
            or frame_positions.shape != absolute_evidence.shape[:2]
            or frame_positions.dtype != torch.long
            or valid_frames.shape != absolute_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (absolute_evidence.shape[0], absolute_evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or bool((valid_frames.sum(dim=1) < 2).any())
        ):
            raise SemanticProgramError("invalid Unified Program evidence")
        valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
        endpoint_positions = frame_positions[:, 1:]
        changes = grounded_evidence[:, 1:] - grounded_evidence[:, :-1]
        program = torch.cat(
            (
                absolute_evidence[:, :-1],
                action_probe[:, :-1, None],
                changes,
            ),
            dim=2,
        )
        valid_semantics = torch.cat(
            (
                valid_task_tokens,
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
