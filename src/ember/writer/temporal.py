"""Semantic Core, causal action-effect Procedure, and LoRA compilation."""

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


class LanguageSemanticCore(torch.nn.Module):
    """Mean away video time, then compose high-level task-token invariants."""

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
        self.blocks = torch.nn.ModuleList(
            RoPEContentBlock(width=width, heads=heads, causal=False)
            for _ in range(blocks)
        )

    def forward(
        self,
        frame_evidence: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            frame_evidence.ndim != 4
            or valid_frames.shape != frame_evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (frame_evidence.shape[0], frame_evidence.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames.any(dim=1).all())
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Core semantic trajectory")
        active = valid_frames[..., None, None]
        counts = valid_frames.sum(dim=1).to(frame_evidence.dtype)[:, None, None]
        content = frame_evidence.masked_fill(~active, 0.0).sum(dim=1) / counts
        content = content.masked_fill(~valid_task_tokens[..., None], 0.0)
        weights = (
            valid_frames.to(frame_evidence.dtype)[:, :, None]
            / counts[:, :, :1]
        ) * valid_task_tokens[:, None].to(frame_evidence.dtype)
        positions = torch.arange(
            content.shape[1],
            dtype=torch.long,
            device=content.device,
        )[None].expand(content.shape[0], -1)
        for block in self.blocks:
            content = block(content, positions, valid_task_tokens)
        return content, weights


class ActionEffectBinder(torch.nn.Module):
    """Bind each Action anchor to effects, then pool anchors into one event."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid action-effect binder")
        self.heads = int(heads)
        self.head_width = width // heads
        self.probe_norm = RMSNorm(width)
        self.trajectory_norm = RMSNorm(width)
        self.transition_key_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.feature_gate = torch.nn.Linear(width, width, bias=False)
        self.binding_output = torch.nn.Linear(width, width, bias=False)
        self.event_norm = RMSNorm(width)
        self.event_query = torch.nn.Linear(width, width, bias=False)
        self.event_key = torch.nn.Linear(width, width, bias=False)
        self.event_value = torch.nn.Linear(width, width, bias=False)
        self.event_output = torch.nn.Linear(width, width, bias=False)
        torch.nn.init.normal_(self.feature_gate.weight, mean=0.0, std=0.02)

    def _validate_inputs(
        self,
        action_probes: torch.Tensor,
        semantic_trajectory: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> None:
        if (
            action_probes.ndim != 4
            or action_probes.shape[2] != 8
            or semantic_trajectory.ndim != 4
            or semantic_trajectory.shape[:2] != action_probes.shape[:2]
            or semantic_trajectory.shape[-1] != action_probes.shape[-1]
            or action_probes.shape[-1] != self.heads * self.head_width
            or valid_frames.shape != action_probes.shape[:2]
            or valid_frames.dtype != torch.bool
            or valid_task_tokens.shape
            != (action_probes.shape[0], semantic_trajectory.shape[2])
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_frames[:, 0].all())
            or not bool(valid_task_tokens.any(dim=1).all())
            or action_probes.shape[1] < 2
        ):
            raise VariableEpisodeInputError("invalid action-effect batch")

    def _bind_effects(
        self,
        probes: torch.Tensor,
        transition: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, intervals, task_tokens, width = transition.shape
        normalized_probes = self.probe_norm(probes)
        query = self.query(normalized_probes).reshape(
            batch,
            intervals,
            8,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        key = self.key(self.transition_key_norm(transition)).reshape(
            batch,
            intervals,
            task_tokens,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        value = self.value(transition).reshape(
            batch,
            intervals,
            task_tokens,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        logits = torch.einsum("bthkd,bthld->bthkl", query, key)
        logits = logits / math.sqrt(self.head_width)
        allowed = valid_task_tokens[:, None, None, None, :]
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        effect_weights = torch.softmax(
            logits.to(torch.float32),
            dim=-1,
        ).to(logits.dtype)
        action_gate = 1.0 + torch.tanh(
            self.feature_gate(normalized_probes)
        ).reshape(
            batch,
            intervals,
            8,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        bound_heads = torch.einsum(
            "bthkl,bthld,bthkd->bthkd",
            effect_weights,
            value,
            action_gate,
        )
        return self.binding_output(
            bound_heads.permute(0, 1, 3, 2, 4).reshape(
                batch,
                intervals,
                8,
                width,
            )
        )

    def _pool_event(self, bound: torch.Tensor) -> torch.Tensor:
        batch, intervals, anchors, width = bound.shape
        event_query_source = self.event_norm(bound.mean(dim=2, keepdim=True))
        event_memory = self.event_norm(bound)
        event_query = self.event_query(event_query_source).reshape(
            batch,
            intervals,
            1,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        event_key = self.event_key(event_memory).reshape(
            batch,
            intervals,
            anchors,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        event_value = self.event_value(bound).reshape(
            batch,
            intervals,
            anchors,
            self.heads,
            self.head_width,
        ).permute(0, 1, 3, 2, 4)
        event_logits = torch.einsum(
            "bthqd,bthkd->bthqk",
            event_query,
            event_key,
        ) / math.sqrt(self.head_width)
        event_weights = torch.softmax(
            event_logits.to(torch.float32),
            dim=-1,
        ).to(event_logits.dtype)
        event_heads = torch.einsum(
            "bthqk,bthkd->bthqd",
            event_weights,
            event_value,
        )
        return self.event_output(
            event_heads.squeeze(3).reshape(batch, intervals, width)
        )

    def forward(
        self,
        action_probes: torch.Tensor,
        semantic_trajectory: torch.Tensor,
        valid_frames: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            action_probes,
            semantic_trajectory,
            valid_frames,
            valid_task_tokens,
        )
        normalized = self.trajectory_norm(semantic_trajectory)
        transition = normalized[:, 1:] - normalized[:, :-1]
        valid_intervals = valid_frames[:, 1:] & valid_frames[:, :-1]
        active = valid_intervals[:, :, None] & valid_task_tokens[:, None, :]
        transition = transition.masked_fill(~active[..., None], 0.0)
        bound = self._bind_effects(
            action_probes[:, :-1],
            transition,
            valid_task_tokens,
        )
        bound = bound.masked_fill(~valid_intervals[..., None, None], 0.0)
        event = self._pool_event(bound)
        event = event.masked_fill(~valid_intervals[..., None], 0.0)
        return event, transition, valid_intervals


class CausalProcedureEncoder(torch.nn.Module):
    """Keep one causally contextualized Procedure token per frame interval."""

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


class ContentCrossAttention(torch.nn.Module):
    """Cross-attend while routing/position affect only Q/K, never content V."""

    def __init__(self, *, width: int, heads: int, rotary_keys: bool) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid content cross-attention")
        if rotary_keys and (width // heads) % 2:
            raise VariableEpisodeInputError("rotary cross-attention head is odd")
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
        ):
            raise VariableEpisodeInputError("invalid content cross-attention batch")
        query = _split_heads(self.query(query_key), self.heads)
        key = _split_heads(self.key(memory_key), self.heads)
        if self.rotary_keys:
            if memory_positions is None or memory_positions.shape != memory_key.shape[:2]:
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
            raise VariableEpisodeInputError("Core reader received frame positions")
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


class CoreSlotReader(torch.nn.Module):
    """Read language-axis Core content into 320 routed LoRA slots."""

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


class ProcedureSlotReader(torch.nn.Module):
    """Use a Core-conditioned query to read ordered Procedure content."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(width)
        self.memory_norm = RMSNorm(width)
        self.attention = ContentCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=True,
        )

    def forward(
        self,
        routing: torch.Tensor,
        core_slots: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned_query = self.query_norm(routing + core_slots)
        slots = self.attention(
            conditioned_query,
            self.memory_norm(procedure),
            procedure,
            valid_procedure,
            positions,
        )
        return slots, conditioned_query


class PostFusionSlotBlock(torch.nn.Module):
    """Coordinate already fused LoRA slots without reopening either memory path."""

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


class ProcedureContentCompiler(torch.nn.Module):
    """Use Core to interpret Procedure without a Core-only LoRA value path."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid slot-normalized compiler")
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
        self.core_reader = CoreSlotReader(width=width, heads=heads)
        self.procedure_reader = ProcedureSlotReader(width=width, heads=heads)
        self.core_modulation_norm = RMSNorm(width)
        self.core_modulation = torch.nn.Linear(width, width, bias=False)
        self.post_fusion = PostFusionSlotBlock(width=width, heads=heads)
        torch.nn.init.normal_(self.core_modulation.weight, mean=0.0, std=0.02)

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
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or core.shape[0] != procedure.shape[0]
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Core/Procedure compiler memory")

    def fused_slots(
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
        routing = self.routing_norm(self._routing())[None].expand(
            core.shape[0],
            -1,
            -1,
        )
        core_slots = self.core_reader(routing, core, valid_core)
        procedure_slots, conditioned_query = self.procedure_reader(
            routing,
            core_slots,
            procedure,
            positions,
            valid_procedure,
        )
        core_gate = torch.tanh(
            self.core_modulation(self.core_modulation_norm(core_slots))
        )
        modulated_procedure_slots = procedure_slots * (1.0 + core_gate)
        output = self.post_fusion(modulated_procedure_slots, routing)
        return output, {
            "core_slots": core_slots,
            "conditioned_query": conditioned_query,
            "procedure_slots": procedure_slots,
            "core_gate": core_gate,
            "modulated_procedure_slots": modulated_procedure_slots,
            "fused_slots": output,
        }

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content, _ = self.fused_slots(
            core,
            valid_core,
            procedure,
            positions,
            valid_procedure,
        )
        expert_stop = self.EXPERT_LAYERS * self.RANK
        expert = content[:, :expert_stop].reshape(
            core.shape[0],
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
