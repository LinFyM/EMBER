"""Permutation-invariant mean-backed Semantic Core for the canonical Writer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ember.writer.semantic_program import RMSNorm, apply_rope, merge_heads, split_heads


class SemanticCoreError(ValueError):
    """Raised when token-aligned frame-set evidence violates the Core contract."""


class TaskSelectedSemanticSetFusion(torch.nn.Module):
    """Add task-selected centered frame evidence to an uncentered mean carrier."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise SemanticCoreError("invalid semantic-set fusion dimensions")
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
            raise SemanticCoreError("invalid token-aligned frame-set evidence")
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
        content = self.mean(frame_mean) + self.output(merge_heads(attended))
        content = content.masked_fill(~valid_task_tokens[..., None], 0.0)
        weights = weights.masked_fill(
            ~valid_task_tokens[:, None, None, :], 0.0
        )
        return content, weights


class SemanticTokenBlock(torch.nn.Module):
    """Compose Core slots along task tokens without introducing frame order."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise SemanticCoreError("invalid semantic-token block dimensions")
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
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 3
            or valid_task_tokens.shape != content.shape[:2]
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise SemanticCoreError("invalid semantic-token batch")
        positions = torch.arange(
            content.shape[1], dtype=torch.long, device=content.device
        )[None].expand(content.shape[0], -1)
        normalized = self.attention_norm(content)
        query = apply_rope(split_heads(self.query(normalized), self.heads), positions)
        key = apply_rope(split_heads(self.key(normalized), self.heads), positions)
        value = split_heads(self.value(content), self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_task_tokens[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        content = content + self.output(merge_heads(attended))
        content = content + self.ffn(self.ffn_norm(content))
        return content.masked_fill(~valid_task_tokens[..., None], 0.0)


class MeanBackedSemanticCore(torch.nn.Module):
    """Build stable task-token slots from the unordered set of video frames."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if blocks <= 0:
            raise SemanticCoreError("Semantic Core needs token-axis blocks")
        self.set_fusion = TaskSelectedSemanticSetFusion(width=width, heads=heads)
        self.blocks = torch.nn.ModuleList(
            SemanticTokenBlock(width=width, heads=heads) for _ in range(blocks)
        )

    def forward(
        self,
        text_queries: torch.Tensor,
        frame_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, frame_attention = self.set_fusion(
            text_queries,
            frame_evidence,
            valid_frames,
            valid_task_tokens,
        )
        for block in self.blocks:
            content = block(content, valid_task_tokens)
        return content, frame_attention
