"""Semantic Core and shared axial Teacher/Policy Procedure encoding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VariableEpisodeInputError(ValueError):
    """Raised when a variable-length video-program batch violates its contract."""


AxialProcedureOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]


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


class _RoutedContentAttention(torch.nn.Module):
    """Self-attend with routing and positions restricted to Q/K."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or (width // heads) % 2:
            raise VariableEpisodeInputError("invalid routed attention dimensions")
        self.heads = int(heads)
        self.norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
        valid_tokens: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if (
            content.ndim != 3
            or routing.shape != content.shape
            or valid_tokens.shape != content.shape[:2]
            or valid_tokens.dtype != torch.bool
            or (positions is not None and positions.shape != content.shape[:2])
        ):
            raise VariableEpisodeInputError("invalid routed attention batch")
        addressed = self.norm(content) + routing.to(content.dtype)
        query = _split_heads(self.query(addressed), self.heads)
        key = _split_heads(self.key(addressed), self.heads)
        if positions is not None:
            if positions.dtype != torch.long:
                raise VariableEpisodeInputError("Procedure positions must be integral")
            query = _apply_rope(query, positions)
            key = _apply_rope(key, positions)
        value = _split_heads(self.value(content), self.heads)
        allowed = valid_tokens[:, None, None, :]
        if causal:
            tokens = content.shape[1]
            causal_mask = torch.ones(
                tokens,
                tokens,
                dtype=torch.bool,
                device=content.device,
            ).tril()
            allowed = allowed & causal_mask[None, None]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        output = content + self.output(_merge_heads(attended))
        return output.masked_fill(~valid_tokens[..., None], 0.0)


class _AxialProcedureBlock(torch.nn.Module):
    """Apply local slot attention, causal time attention, then one FFN."""

    SLOT_COUNT = 8

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.width = int(width)
        self.local_attention = _RoutedContentAttention(
            width=width,
            heads=heads,
        )
        self.temporal_attention = _RoutedContentAttention(
            width=width,
            heads=heads,
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
        positions: torch.Tensor,
        valid_times: torch.Tensor,
        routing: torch.Tensor,
    ) -> torch.Tensor:
        expected_routing = (content.shape[0], self.SLOT_COUNT, self.width)
        if (
            content.ndim != 4
            or content.shape[2:] != (self.SLOT_COUNT, self.width)
            or positions.shape != content.shape[:2]
            or positions.dtype != torch.long
            or valid_times.shape != content.shape[:2]
            or valid_times.dtype != torch.bool
            or routing.shape != expected_routing
        ):
            raise VariableEpisodeInputError("invalid axial Procedure batch")
        batch, times, slots, width = content.shape
        value = content.masked_fill(~valid_times[..., None, None], 0.0)

        local_content = value.reshape(batch * times, slots, width)
        local_routing = routing[:, None].expand(
            -1, times, -1, -1
        ).reshape(
            batch * times, slots, width
        )
        local_valid = valid_times.reshape(batch * times, 1).expand(-1, slots)
        # Every local token has the same time position, so equal RoPE rotations
        # would cancel in QK; slot and stream identities are the useful routing.
        value = self.local_attention(
            local_content,
            local_routing,
            local_valid,
            positions=None,
            causal=False,
        ).reshape(batch, times, slots, width)

        temporal_content = value.permute(0, 2, 1, 3).reshape(
            batch * slots, times, width
        )
        temporal_routing = routing[:, :, None].expand(
            -1, -1, times, -1
        ).reshape(batch * slots, times, width)
        temporal_positions = positions[:, None].expand(
            -1, slots, -1
        ).reshape(batch * slots, times)
        temporal_valid = valid_times[:, None].expand(
            -1, slots, -1
        ).reshape(batch * slots, times)
        value = self.temporal_attention(
            temporal_content,
            temporal_routing,
            temporal_valid,
            positions=temporal_positions,
            causal=True,
        ).reshape(batch, slots, times, width).permute(0, 2, 1, 3)

        value = value + self.ffn(self.ffn_norm(value))
        return value.masked_fill(~valid_times[..., None, None], 0.0)


class SharedAxialProcedureEncoder(torch.nn.Module):
    """Build separate Teacher and Policy memories with shared axial blocks."""

    SLOT_COUNT = 8

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if (
            min(width, heads, blocks) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid shared axial Procedure")
        self.width = int(width)
        self.teacher_input_norm = RMSNorm(width)
        self.teacher_input = torch.nn.Linear(width, width, bias=False)
        self.policy_input_norm = RMSNorm(width)
        self.policy_input = torch.nn.Linear(width, width, bias=False)
        self.slot_identity = torch.nn.Parameter(
            torch.empty(self.SLOT_COUNT, width)
        )
        self.stream_identity = torch.nn.Parameter(torch.empty(2, width))
        torch.nn.init.normal_(self.slot_identity, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.stream_identity, mean=0.0, std=0.02)
        self.blocks = torch.nn.ModuleList(
            _AxialProcedureBlock(width=width, heads=heads)
            for _ in range(blocks)
        )
        self.confidence_norm = RMSNorm(width)
        self.confidence_head = torch.nn.Linear(width, 1, bias=False)

    def _validate_inputs(
        self,
        teacher_events: torch.Tensor,
        action_probes: torch.Tensor,
        initial_confidence: torch.Tensor,
        frame_positions: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> None:
        if action_probes.ndim != 4:
            raise VariableEpisodeInputError("invalid Policy Procedure inputs")
        batch, frames, slots, width = action_probes.shape
        content_layout = (
            frames >= 2,
            (slots, width) == (self.SLOT_COUNT, self.width),
            teacher_events.shape
            == (batch, frames - 1, self.SLOT_COUNT, self.width),
            initial_confidence.shape == teacher_events.shape[:3],
            initial_confidence.is_floating_point(),
        )
        if not all(content_layout):
            raise VariableEpisodeInputError("invalid shared axial content")
        sequence_layout = (
            frame_positions.shape == action_probes.shape[:2],
            frame_positions.dtype == torch.long,
            valid_frames.shape == action_probes.shape[:2],
            valid_frames.dtype == torch.bool,
            bool(valid_frames[:, 0].all()),
            not bool((valid_frames[:, 1:] & ~valid_frames[:, :-1]).any()),
        )
        if not all(sequence_layout):
            raise VariableEpisodeInputError("invalid shared axial sequence")
        active_intervals = valid_frames[:, 1:] & valid_frames[:, :-1]
        if (
            not bool(active_intervals.any(dim=1).all())
            or bool(
                (
                    (frame_positions[:, 1:] <= frame_positions[:, :-1])
                    & active_intervals
                ).any()
            )
        ):
            raise VariableEpisodeInputError("invalid Procedure frame order")
        confidence_valid = torch.isfinite(initial_confidence)
        confidence_valid &= initial_confidence.ge(0) & initial_confidence.le(1)
        if not bool(confidence_valid.all()):
            raise VariableEpisodeInputError("invalid initial Teacher confidence")
        devices = {
            teacher_events.device,
            action_probes.device,
            initial_confidence.device,
            frame_positions.device,
            valid_frames.device,
        }
        if len(devices) != 1:
            raise VariableEpisodeInputError("shared axial inputs changed device")

    def _routing(
        self,
        batch: int,
        stream: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        routing = self.slot_identity + self.stream_identity[stream]
        return routing.to(dtype=dtype)[None].expand(batch, -1, -1)

    def _encode_stream(
        self,
        content: torch.Tensor,
        positions: torch.Tensor,
        valid_times: torch.Tensor,
        *,
        stream: int,
        norm: torch.nn.Module,
        projection: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = projection(norm(content)).masked_fill(
            ~valid_times[..., None, None],
            0.0,
        )
        memory = projected
        routing = self._routing(content.shape[0], stream, projected.dtype)
        for block in self.blocks:
            memory = block(memory, positions, valid_times, routing)
        return projected, memory

    def forward(
        self,
        teacher_events: torch.Tensor,
        action_probes: torch.Tensor,
        initial_confidence: torch.Tensor,
        frame_positions: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> AxialProcedureOutput:
        """Return separate memories, confidence, positions, masks, and probes."""

        self._validate_inputs(
            teacher_events, action_probes, initial_confidence,
            frame_positions, valid_frames,
        )
        valid_policy = valid_frames
        valid_teacher = valid_frames[:, :-1] & valid_frames[:, 1:]
        policy_positions = 2 * frame_positions
        teacher_positions = frame_positions[:, :-1] + frame_positions[:, 1:]

        teacher_input, teacher_memory = self._encode_stream(
            teacher_events,
            teacher_positions,
            valid_teacher,
            stream=0,
            norm=self.teacher_input_norm,
            projection=self.teacher_input,
        )
        policy_input, policy_memory = self._encode_stream(
            action_probes,
            policy_positions,
            valid_policy,
            stream=1,
            norm=self.policy_input_norm,
            projection=self.policy_input,
        )

        logits = self.confidence_head(self.confidence_norm(teacher_memory))
        coherence = torch.sigmoid(logits.squeeze(-1)).masked_fill(
            ~valid_teacher[..., None], 0.0
        )
        teacher_confidence = torch.where(
            initial_confidence > 0,
            initial_confidence * coherence.to(initial_confidence.dtype),
            torch.zeros_like(initial_confidence),
        ).masked_fill(~valid_teacher[..., None], 0.0)
        diagnostics = {
            "teacher_input": teacher_input,
            "policy_input": policy_input,
            "teacher_coherence": coherence,
        }
        return (
            teacher_memory,
            policy_memory,
            teacher_confidence,
            teacher_positions,
            policy_positions,
            valid_teacher,
            valid_policy,
            diagnostics,
        )
