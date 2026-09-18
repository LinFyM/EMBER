"""Repeated full-horizon and adjacent-visual reads on the ordered A backbone."""

from __future__ import annotations

import torch

from ember.writer.temporal import (
    ContentCrossAttention,
    RMSNorm,
    RoPEContentBlock,
    VariableEpisodeInputError,
)


class ProcedureReadBlock(torch.nn.Module):
    """Read native H, then real before/after E, then causal video time and FFN."""

    def __init__(self, *, width: int, expert_width: int, heads: int) -> None:
        super().__init__()
        self.horizon_query_norm = RMSNorm(width)
        self.horizon_memory_norm = RMSNorm(expert_width)
        self.horizon_read = ContentCrossAttention(
            width=width, heads=heads, rotary_keys=True, memory_width=expert_width,
        )
        self.visual_query_norm = RMSNorm(width)
        self.visual_memory_norm = RMSNorm(width)
        self.visual_read = ContentCrossAttention(width=width, heads=heads, rotary_keys=True)
        # Roles address keys only; both endpoint Values retain their real content.
        self.endpoint_identity = torch.nn.Parameter(torch.empty(2, width))
        torch.nn.init.normal_(self.endpoint_identity, std=0.02)
        torch.nn.init.zeros_(self.horizon_read.output.weight)
        torch.nn.init.zeros_(self.visual_read.output.weight)
        self.temporal = RoPEContentBlock(width=width, heads=heads, causal=True)

    def forward(
        self,
        content: torch.Tensor,
        horizon: torch.Tensor,
        visual: torch.Tensor,
        valid_visual: torch.Tensor,
        positions: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, width = content.shape
        flat = content.flatten(0, 1)[:, None]
        horizon_positions = torch.arange(
            horizon.shape[1], device=content.device, dtype=torch.long,
        )[None].expand(batch * frames, -1)
        flat = flat + self.horizon_read(
            self.horizon_query_norm(flat), self.horizon_memory_norm(horizon), horizon,
            valid_frames.flatten()[:, None].expand(-1, horizon.shape[1]),
            horizon_positions,
        )
        tokens = visual.shape[2]
        keys = self.visual_memory_norm(visual) + self.endpoint_identity[None, :, None]
        visual_positions = torch.arange(
            tokens, device=content.device, dtype=torch.long,
        ).repeat(2)[None].expand(batch * frames, -1)
        flat = flat + self.visual_read(
            self.visual_query_norm(flat), keys.flatten(1, 2), visual.flatten(1, 2),
            valid_visual.flatten(1, 2), visual_positions,
        )
        return self.temporal(flat.reshape(batch, frames, width), positions, valid_frames)


class RecurrentProcedureEncoder(torch.nn.Module):
    """Preserve full native H and E for each repeated, independently learned read."""

    def __init__(
        self, *, width: int, expert_width: int, heads: int, blocks: int,
        action_horizon: int = 50,
    ) -> None:
        super().__init__()
        if blocks <= 0 or action_horizon != 50:
            raise VariableEpisodeInputError("invalid recurrent Procedure encoder")
        self.width, self.expert_width = int(width), int(expert_width)
        self.action_horizon = int(action_horizon)
        self.blocks = torch.nn.ModuleList(
            ProcedureReadBlock(width=width, expert_width=expert_width, heads=heads)
            for _ in range(blocks)
        )

    def forward(
        self,
        content: torch.Tensor,
        horizon: torch.Tensor,
        evidence: torch.Tensor,
        positions: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            content.ndim != 3 or content.shape[-1] != self.width
            or horizon.shape != (*content.shape[:2], self.action_horizon, self.expert_width)
            or evidence.ndim != 4 or evidence.shape[:2] != content.shape[:2]
            or evidence.shape[-1] != self.width
            or positions.shape != content.shape[:2] or positions.dtype != torch.long
            or valid_frames.shape != content.shape[:2] or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape != (content.shape[0], evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames[:, 0].all())
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid complete Procedure evidence")
        # Zero storage at a boundary is masked padding, never a repeated/fake frame.
        next_evidence = torch.cat((evidence[:, 1:], torch.zeros_like(evidence[:, :1])), dim=1)
        next_valid = torch.cat((valid_frames[:, 1:], torch.zeros_like(valid_frames[:, :1])), dim=1)
        visual = torch.stack((evidence, next_evidence), dim=2).flatten(0, 1)
        valid_visual = (
            torch.stack((valid_frames, valid_frames & next_valid), dim=2)[..., None]
            & valid_task_tokens[:, None, None]
        ).flatten(0, 1)
        value = content.masked_fill(~valid_frames[..., None], 0.0)
        horizon = horizon.flatten(0, 1)
        for block in self.blocks:
            value = block(value, horizon, visual, valid_visual, positions, valid_frames)
        return value
