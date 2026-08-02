"""Target-bound, role-preserving Action/Effect/Change causal Program."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SemanticProgramError(ValueError):
    """Raised when variable-length semantic Program tensors violate contract."""


class RMSNorm(torch.nn.Module):
    """Small dtype-stable RMS normalization used only on Q/K paths."""

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
    """Apply one-dimensional RoPE to a Q/K tensor."""

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
    """Apply independent RoPE halves for interval and semantic axes."""

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
    """Contextualize keys with normalized Q/K while keeping physical raw V."""

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


class TargetRoleEvidenceReader(torch.nn.Module):
    """Read raw task-token evidence after target and Action are known."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise SemanticProgramError("invalid target-role evidence reader")
        self.heads = int(heads)
        self.memory_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        address: torch.Tensor,
        memory: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        role_identity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            address.ndim != 4
            or memory.ndim != 4
            or address.shape[0] != memory.shape[0]
            or address.shape[2] != memory.shape[1]
            or address.shape[-1] != memory.shape[-1]
            or valid_task_tokens.shape != (memory.shape[0], memory.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or role_identity.shape != (memory.shape[-1],)
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target-bound role evidence")
        batch, targets, intervals, width = address.shape
        task_tokens = memory.shape[2]
        rows = batch * targets * intervals
        query_content = address + role_identity
        query = split_heads(
            self.query(query_content.reshape(rows, 1, width)), self.heads
        )
        query_positions = torch.zeros(
            rows, 1, dtype=torch.long, device=memory.device
        )
        query = apply_rope(query, query_positions)

        expanded = memory[:, None].expand(
            batch, targets, intervals, task_tokens, width
        ).reshape(rows, task_tokens, width)
        key_content = self.memory_norm(expanded) + role_identity
        key = split_heads(self.key(key_content), self.heads)
        positions = torch.arange(
            task_tokens, dtype=torch.long, device=memory.device
        )[None].expand(rows, -1)
        key = apply_rope(key, positions)
        value = split_heads(expanded, self.heads)
        valid = valid_task_tokens[:, None, None].expand(
            batch, targets, intervals, task_tokens
        ).reshape(rows, task_tokens)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(merge_heads(attended)).reshape(
            batch, targets, intervals, width
        )


class CausalRoleProgramBlock(torch.nn.Module):
    """Contextualize each target-role column without mixing role values."""

    ROLE_COUNT = 3

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.temporal_attention = RawValueSelfAttention(
            width=width, heads=heads, causal=True
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
        endpoint_positions: torch.Tensor,
        valid_intervals: torch.Tensor,
        qk_identity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 5
            or content.shape[3] != self.ROLE_COUNT
            or endpoint_positions.shape != (content.shape[0], content.shape[2])
            or endpoint_positions.dtype != torch.long
            or valid_intervals.shape != endpoint_positions.shape
            or valid_intervals.dtype != torch.bool
            or qk_identity.shape
            != (content.shape[0], content.shape[1], self.ROLE_COUNT, content.shape[-1])
            or not bool(valid_intervals.any(dim=1).all())
        ):
            raise SemanticProgramError("invalid target-role causal Program")
        batch, targets, intervals, roles, width = content.shape
        sequences = content.permute(0, 1, 3, 2, 4).reshape(
            batch * targets * roles, intervals, width
        )
        positions = endpoint_positions[:, None, None].expand(
            batch, targets, roles, intervals
        ).reshape(batch * targets * roles, intervals)
        valid = valid_intervals[:, None, None].expand(
            batch, targets, roles, intervals
        ).reshape(batch * targets * roles, intervals)
        identity = qk_identity[:, :, :, None].expand(
            batch, targets, roles, intervals, width
        ).reshape(batch * targets * roles, intervals, width)
        sequences = sequences + self.temporal_attention(
            sequences,
            valid,
            positions=positions,
            qk_identity=identity,
        )
        sequences = sequences + self.ffn(self.ffn_norm(sequences))
        result = sequences.reshape(batch, targets, roles, intervals, width).permute(
            0, 1, 3, 2, 4
        )
        return result.masked_fill(
            ~valid_intervals[:, None, :, None, None], 0.0
        )


class TargetBoundRoleProgram(torch.nn.Module):
    """Build separate target-bound causal Action, Effect, and Change streams."""

    ROLE_COUNT = 3

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
            raise SemanticProgramError("target-bound Program needs causal blocks")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        identity = torch.empty(self.ROLE_COUNT, width)
        identity.normal_(mean=0.0, std=0.02, generator=generator)
        self.role_identity = torch.nn.Parameter(identity)
        self.role_identity_norm = RMSNorm(width)
        self.core_norm = RMSNorm(width)
        self.action_norm = RMSNorm(width)
        self.evidence_reader = TargetRoleEvidenceReader(width=width, heads=heads)
        self.blocks = torch.nn.ModuleList(
            CausalRoleProgramBlock(width=width, heads=heads) for _ in range(blocks)
        )

    def forward(
        self,
        grounded_evidence: torch.Tensor,
        action_probe: torch.Tensor,
        frame_positions: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        target_query: torch.Tensor,
        target_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            or target_query.ndim != 3
            or target_core.shape != target_query.shape
            or target_query.shape[0] != grounded_evidence.shape[0]
            or target_query.shape[-1] != grounded_evidence.shape[-1]
            or bool((valid_frames.sum(dim=1) < 2).any())
        ):
            raise SemanticProgramError("invalid target-bound Program evidence")
        batch, frames, _tokens, width = grounded_evidence.shape
        targets = target_query.shape[1]
        intervals = frames - 1
        valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
        endpoint_positions = frame_positions[:, 1:]
        effects = grounded_evidence[:, 1:]
        changes = effects - grounded_evidence[:, :-1]
        actions = action_probe[:, :-1]
        roles = self.role_identity_norm(self.role_identity).to(actions.dtype)
        address = (
            target_query[:, :, None]
            + self.core_norm(target_core)[:, :, None]
            + self.action_norm(actions)[:, None]
        )
        effect_read = self.evidence_reader(
            address, effects, valid_task_tokens, roles[1]
        )
        change_read = self.evidence_reader(
            address, changes, valid_task_tokens, roles[2]
        )
        action_value = actions[:, None].expand(batch, targets, intervals, width)
        program = torch.stack((action_value, effect_read, change_read), dim=3)
        program = program.masked_fill(
            ~valid_intervals[:, None, :, None, None], 0.0
        )
        temporal_identity = (
            target_query[:, :, None]
            + self.core_norm(target_core)[:, :, None]
            + roles[None, None]
        )
        for block in self.blocks:
            program = block(
                program,
                endpoint_positions,
                valid_intervals,
                temporal_identity,
            )
        return program, endpoint_positions, valid_intervals
