"""Language Core, visual-transition Procedure, and slot-normalized LoRA fusion."""

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


class TokenAlignedFrameSetAttention(torch.nn.Module):
    """Aggregate video evidence independently for every task-language token."""

    def __init__(self, *, width: int, heads: int, initial_lambda: float) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or not 0.0 < initial_lambda < 1.0
        ):
            raise VariableEpisodeInputError("invalid frame-set attention")
        self.heads = int(heads)
        self.head_width = width // heads
        self.query_norm = RMSNorm(width)
        self.evidence_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        logit = math.log(initial_lambda / (1.0 - initial_lambda))
        self.gate_logits = torch.nn.Parameter(torch.full((heads,), logit))

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
        value = self.value(frame_evidence).reshape(
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
        selected = torch.softmax(logits.to(torch.float32), dim=2).to(logits.dtype)
        counts = valid_frames.sum(dim=1, keepdim=True).to(logits.dtype)
        uniform = valid_frames.to(logits.dtype)[:, None, :, None]
        uniform = uniform / counts[:, :, None, None]
        gate = torch.sigmoid(self.gate_logits).to(logits.dtype)[None, :, None, None]
        weights = (1.0 - gate) * uniform + gate * selected
        attended = torch.einsum("bhtl,bhtld->bhld", weights, value)
        output = self.output(_merge_heads(attended))
        output = output.masked_fill(~valid_task_tokens[..., None], 0.0)
        weights = weights.masked_fill(
            ~valid_task_tokens[:, None, None, :],
            0.0,
        )
        return output, weights


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
    """Aggregate along frames, then compose evidence along task-language tokens."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        blocks: int,
        frame_attention_initial_lambda: float,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise VariableEpisodeInputError("Semantic Core needs language blocks")
        self.frame_attention = TokenAlignedFrameSetAttention(
            width=width,
            heads=heads,
            initial_lambda=frame_attention_initial_lambda,
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
        content, weights = self.frame_attention(
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
    """Let each Action-Expert probe read adjacent task-grounded visual change."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid visual-transition fusion")
        self.heads = int(heads)
        self.head_width = width // heads
        self.probe_norm = RMSNorm(width)
        self.transition_norm = RMSNorm(width)
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
        normalized = self.transition_norm(transition)
        key = self.key(normalized).reshape(
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
    """Read centered ordered Procedure content conditioned on Core slots."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.core_norm = RMSNorm(width)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_core = self.core_norm(core_slots)
        mask = valid_procedure[..., None]
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (procedure * mask).sum(dim=1, keepdim=True) / count
        centered = (procedure - mean).masked_fill(~mask, 0.0)
        slots = self.attention(
            routing + normalized_core,
            self.memory_norm(procedure),
            centered,
            valid_procedure,
            positions,
        )
        return slots, normalized_core, centered


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


class SlotNormalizedCoreProcedureCompiler(torch.nn.Module):
    """Fuse Core and Procedure through centered readout and zero-init AdaLN."""

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
        self.procedure_norm = RMSNorm(width)
        self.modulation = torch.nn.Linear(width, 2 * width, bias=False)
        torch.nn.init.zeros_(self.modulation.weight)
        self.post_fusion = PostFusionSlotBlock(width=width, heads=heads)

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
        procedure_slots, normalized_core, centered = self.procedure_reader(
            routing,
            core_slots,
            procedure,
            positions,
            valid_procedure,
        )
        gamma, beta = self.modulation(
            self.procedure_norm(procedure_slots)
        ).chunk(2, dim=-1)
        fused = (1.0 + gamma) * normalized_core + beta
        output = self.post_fusion(fused, routing)
        return output, {
            "core_slots": core_slots,
            "procedure_centered": centered,
            "procedure_slots": procedure_slots,
            "adaln_gamma": gamma,
            "adaln_beta": beta,
            "fused_slots": fused,
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
