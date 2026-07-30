"""Semantic Core and uncapped visual-program Procedure for EMBER."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length video-program batch violates its contract."""


class RMSNorm(torch.nn.Module):
    """Small dtype-stable, zero-preserving RMS normalization."""

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
        raise VariableEpisodeInputError("invalid ordinal RoPE request")
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


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = value.shape
    if width % heads:
        raise VariableEpisodeInputError("attention width is not divisible by heads")
    return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * width)


class RoPEContentBlock(torch.nn.Module):
    """Pre-norm content Transformer with ordinal RoPE only in Q/K."""

    def __init__(self, *, width: int, heads: int, causal: bool) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise VariableEpisodeInputError("invalid content Transformer dimensions")
        self.heads = int(heads)
        self.causal = bool(causal)
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
        if (
            content.ndim != 3
            or positions.shape != content.shape[:2]
            or positions.dtype != torch.long
            or valid_mask.shape != content.shape[:2]
            or valid_mask.dtype != torch.bool
        ):
            raise VariableEpisodeInputError("invalid content Transformer batch")
        normalized = self.attention_norm(content)
        query = _apply_rope(
            _split_heads(self.query(normalized), self.heads),
            positions,
        )
        key = _apply_rope(
            _split_heads(self.key(normalized), self.heads),
            positions,
        )
        value = _split_heads(self.value(content), self.heads)
        allowed = valid_mask[:, None, None, :]
        if self.causal:
            tokens = content.shape[1]
            causal = torch.ones(
                tokens,
                tokens,
                dtype=torch.bool,
                device=content.device,
            ).tril()
            allowed = allowed & causal[None, None, :, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        content = content + self.output(_merge_heads(attended))
        content = content + self.ffn(self.ffn_norm(content))
        return content.masked_fill(~valid_mask[..., None], 0.0)


class TaskSelectedSemanticSetFusion(torch.nn.Module):
    """Add task-selected centered frame evidence to a stable mean backbone."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid semantic-set fusion")
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
            raise VariableEpisodeInputError("invalid token-aligned frame evidence")
        batch, frames, tokens, width = frame_evidence.shape
        active = valid_frames[:, :, None, None]
        counts = valid_frames.sum(dim=1).to(frame_evidence.dtype)[:, None, None]
        frame_mean = frame_evidence.masked_fill(~active, 0.0).sum(dim=1)
        frame_mean = frame_mean / counts
        centered = (frame_evidence - frame_mean[:, None]).masked_fill(
            ~active,
            0.0,
        )
        query = _split_heads(
            self.query(self.query_norm(text_queries)),
            self.heads,
        )
        normalized = self.evidence_norm(frame_evidence)
        key = self.key(normalized).reshape(
            batch,
            frames,
            tokens,
            self.heads,
            self.head_width,
        ).permute(0, 3, 1, 2, 4)
        value = centered.reshape(
            batch,
            frames,
            tokens,
            self.heads,
            self.head_width,
        ).permute(0, 3, 1, 2, 4)
        logits = torch.einsum("bhld,bhtld->bhtl", query, key)
        logits = logits / math.sqrt(self.head_width)
        frame_mask = valid_frames[:, None, :, None]
        logits = logits.masked_fill(~frame_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.to(torch.float32), dim=2).to(logits.dtype)
        attended = torch.einsum("bhtl,bhtld->bhld", weights, value)
        output = self.mean(frame_mean) + self.output(_merge_heads(attended))
        output = output.masked_fill(~valid_task_tokens[..., None], 0.0)
        weights = weights.masked_fill(
            ~valid_task_tokens[:, None, None, :],
            0.0,
        )
        return output, weights


class LanguageSemanticCore(torch.nn.Module):
    """Aggregate frame sets, then compose high-level task-token invariants."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        blocks: int,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise VariableEpisodeInputError("Semantic Core needs language blocks")
        self.semantic_set_fusion = TaskSelectedSemanticSetFusion(
            width=width,
            heads=heads,
        )
        self.blocks = torch.nn.ModuleList(
            RoPEContentBlock(width=width, heads=heads, causal=False)
            for _ in range(blocks)
        )

    def forward(
        self,
        text_queries: torch.Tensor,
        frame_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, weights = self.semantic_set_fusion(
            text_queries,
            frame_evidence,
            valid_frames,
            valid_task_tokens,
        )
        positions = torch.arange(
            content.shape[1],
            dtype=torch.long,
            device=content.device,
        )[None].expand(content.shape[0], -1)
        for block in self.blocks:
            content = block(content, positions, valid_task_tokens)
        return content, weights


class TaskGroundedVisualTransitionFusion(torch.nn.Module):
    """Let the native Action probe read uncapped task-grounded visual change."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid visual-transition fusion")
        self.heads = int(heads)
        self.head_width = width // heads
        self.probe_norm = RMSNorm(width)
        self.transition_key_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        action_probe: torch.Tensor,
        grounded_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            action_probe.ndim != 3
            or grounded_evidence.ndim != 4
            or grounded_evidence.shape[:2] != action_probe.shape[:2]
            or grounded_evidence.shape[-1] != action_probe.shape[-1]
            or action_probe.shape[-1] != self.heads * self.head_width
            or valid_frames.shape != action_probe.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (action_probe.shape[0], grounded_evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames[:, 0].all())
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid visual-transition batch")

        batch, frames, task_tokens, width = grounded_evidence.shape
        transition = torch.cat(
            (
                torch.zeros_like(grounded_evidence[:, :1]),
                grounded_evidence[:, 1:] - grounded_evidence[:, :-1],
            ),
            dim=1,
        )
        active = valid_frames[:, :, None] & valid_task_tokens[:, None, :]
        transition = transition.masked_fill(~active[..., None], 0.0)

        query = self.query(self.probe_norm(action_probe)).reshape(
            batch * frames,
            1,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        key = self.key(self.transition_key_norm(transition)).reshape(
            batch * frames,
            task_tokens,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        value = transition.reshape(
            batch * frames,
            task_tokens,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        allowed = (
            valid_task_tokens[:, None, :]
            .expand(-1, frames, -1)
            .reshape(batch * frames, task_tokens)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        residual = self.output(
            attended.transpose(1, 2).reshape(batch, frames, width)
        )
        residual = residual.masked_fill(
            ~valid_frames[..., None],
            0.0,
        )
        fused = (action_probe + residual).masked_fill(
            ~valid_frames[..., None],
            0.0,
        )
        return fused, transition


class CausalProcedureEncoder(torch.nn.Module):
    """Keep one causally contextualized Procedure token per sampled frame."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if blocks <= 0:
            raise VariableEpisodeInputError("invalid causal Procedure encoder")
        self.blocks = torch.nn.ModuleList(
            RoPEContentBlock(width=width, heads=heads, causal=True)
            for _ in range(blocks)
        )

    def forward(
        self,
        content: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(valid_mask[:, 0].all()):
            raise VariableEpisodeInputError("Procedure must begin at frame zero")
        value = content.masked_fill(~valid_mask[..., None], 0.0)
        for block in self.blocks:
            value = block(value, positions, valid_mask)
        return value.masked_fill(~valid_mask[..., None], 0.0)
