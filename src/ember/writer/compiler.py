"""Compile Core and Teacher–Policy Procedure gaps into routed LoRA slots."""

from __future__ import annotations

import math

import torch

from ember.writer.temporal import (
    RMSNorm,
    VariableEpisodeInputError,
    _apply_rope,
    _merge_heads,
    _split_heads,
)


class ContentCrossAttention(torch.nn.Module):
    """Cross-attend while keeping routing and positions out of value content."""

    def __init__(self, *, width: int, heads: int, rotary_keys: bool) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (rotary_keys and (width // heads) % 2)
        ):
            raise VariableEpisodeInputError("invalid content cross-attention")
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query_key.ndim != 3
            or memory_key.ndim != 3
            or memory_value.shape != memory_key.shape
            or query_key.shape[0] != memory_key.shape[0]
            or query_key.shape[-1] != memory_key.shape[-1]
            or valid_memory.shape != memory_key.shape[:2]
            or valid_memory.dtype != torch.bool
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid content cross-attention batch")
        query = _split_heads(self.query(query_key), self.heads)
        key = _split_heads(self.key(memory_key), self.heads)
        if self.rotary_keys:
            if (
                memory_positions is None
                or memory_positions.shape != memory_key.shape[:2]
                or memory_positions.dtype != torch.long
            ):
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
            raise VariableEpisodeInputError("Core reader received positions")

        value = _split_heads(self.value(memory_value), self.heads)
        logits = torch.matmul(
            query.to(torch.float32),
            key.to(torch.float32).transpose(-1, -2),
        )
        logits = logits / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(
            ~valid_memory[:, None, None, :],
            float("-inf"),
        )
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights.to(value.dtype), value)
        return self.output(_merge_heads(attended)), weights


class CoreSlotReader(torch.nn.Module):
    """Read task Core content into 320 routed LoRA slots."""

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attention(
            routing,
            self.memory_norm(core),
            core,
            valid_core,
        )


class ProcedureSlotReader(torch.nn.Module):
    """Read one owner-specific Procedure memory and expose its weights."""

    SLOT_COUNT = 8

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.memory_norm = RMSNorm(width)
        self.attention = ContentCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory.ndim != 4 or query.ndim != 3:
            raise VariableEpisodeInputError("invalid Procedure slot memory")
        batch, steps, slots, width = memory.shape
        if (
            slots != self.SLOT_COUNT
            or positions.shape != (batch, steps)
            or positions.dtype != torch.long
            or valid_memory.shape != (batch, steps)
            or valid_memory.dtype != torch.bool
            or query.shape[0] != batch
            or query.shape[-1] != width
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Procedure slot memory")
        flat_memory = memory.reshape(batch, steps * slots, width)
        flat_positions = (
            positions[:, :, None]
            .expand(batch, steps, slots)
            .reshape(batch, steps * slots)
        )
        flat_valid = (
            valid_memory[:, :, None]
            .expand(batch, steps, slots)
            .reshape(batch, steps * slots)
        )
        content, weights = self.attention(
            query,
            self.memory_norm(flat_memory),
            flat_memory,
            flat_valid,
            flat_positions,
        )
        return content, weights, flat_valid


class ContentOnlySlotBlock(torch.nn.Module):
    """Coordinate gap content while routing remains confined to Q/K."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.heads = int(heads)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if content.shape != routing.shape or content.ndim != 3:
            raise VariableEpisodeInputError("invalid slot coordination batch")
        addressed = self.attention_norm(content) + routing
        query = _split_heads(self.query(addressed), self.heads)
        key = _split_heads(self.key(addressed), self.heads)
        value = _split_heads(self.value(content), self.heads)
        weights = torch.softmax(
            torch.matmul(
                query.to(torch.float32),
                key.to(torch.float32).transpose(-1, -2),
            )
            / math.sqrt(query.shape[-1]),
            dim=-1,
        )
        attended = torch.matmul(weights.to(value.dtype), value)
        content = content + self.output(_merge_heads(attended))
        content = content + self.ffn(self.ffn_norm(content))
        return self.output_norm(content), weights


class TeacherPolicyGapCompiler(torch.nn.Module):
    """Compile a confidence-authorized Teacher–Policy gap into LoRA content."""

    EXPERT_LAYERS = 18
    RANK = 16
    QUERY_COUNT = EXPERT_LAYERS * RANK + 2 * RANK

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        initialization_seed: int,
        gap_tau: float = 0.5,
    ) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (width // heads) % 2
            or not math.isfinite(gap_tau)
            or gap_tau <= 0.0
        ):
            raise VariableEpisodeInputError("invalid Teacher–Policy gap compiler")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def parameter(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.gap_tau = float(gap_tau)
        self.query_table = parameter(self.QUERY_COUNT)
        self.module_identity = parameter(3)
        self.layer_identity = parameter(self.EXPERT_LAYERS)
        self.rank_identity = parameter(self.RANK)
        self.routing_norm = RMSNorm(width)
        self.core_reader = CoreSlotReader(width=width, heads=heads)
        self.core_query_norm = RMSNorm(width)
        self.teacher_reader = ProcedureSlotReader(width=width, heads=heads)
        self.teacher_query_norm = RMSNorm(width)
        self.policy_reader = ProcedureSlotReader(width=width, heads=heads)
        self.teacher_alignment = torch.nn.Linear(width, width, bias=False)
        self.policy_alignment = torch.nn.Linear(width, width, bias=False)
        self.teacher_alignment_norm = RMSNorm(width)
        self.policy_alignment_norm = RMSNorm(width)
        self.gap_norm = RMSNorm(width)
        self.core_support = torch.nn.Linear(width, width, bias=False)
        self.core_support_norm = RMSNorm(width)
        self.core_gate = torch.nn.Linear(width, width, bias=False)
        self.slot_coordination = ContentOnlySlotBlock(width=width, heads=heads)

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
    def _validate_core(
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> None:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or not bool(valid_core.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Core compiler memory")

    @staticmethod
    def _validate_procedure(
        name: str,
        memory: torch.Tensor,
        positions: torch.Tensor,
        valid_memory: torch.Tensor,
        *,
        batch: int,
        width: int,
    ) -> None:
        if (
            memory.ndim != 4
            or memory.shape[0] != batch
            or memory.shape[2] != ProcedureSlotReader.SLOT_COUNT
            or memory.shape[-1] != width
            or positions.shape != memory.shape[:2]
            or positions.dtype != torch.long
            or valid_memory.shape != memory.shape[:2]
            or valid_memory.dtype != torch.bool
            or not bool(valid_memory.any(dim=1).all())
        ):
            raise VariableEpisodeInputError(
                f"invalid {name} Procedure compiler memory"
            )

    @classmethod
    def _validate_memories(
        cls,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        teacher: torch.Tensor,
        teacher_confidence: torch.Tensor,
        teacher_positions: torch.Tensor,
        valid_teacher: torch.Tensor,
        policy: torch.Tensor,
        policy_positions: torch.Tensor,
        valid_policy: torch.Tensor,
    ) -> None:
        cls._validate_core(core, valid_core)
        batch, _, width = core.shape
        cls._validate_procedure(
            "Teacher",
            teacher,
            teacher_positions,
            valid_teacher,
            batch=batch,
            width=width,
        )
        cls._validate_procedure(
            "Policy",
            policy,
            policy_positions,
            valid_policy,
            batch=batch,
            width=width,
        )
        if teacher_confidence.shape != teacher.shape[:3]:
            raise VariableEpisodeInputError("invalid Teacher confidence shape")
        active_confidence = teacher_confidence.masked_select(
            valid_teacher[:, :, None].expand_as(teacher_confidence)
        )
        if (
            not bool(torch.isfinite(active_confidence).all())
            or bool((active_confidence < 0.0).any())
            or bool((active_confidence > 1.0).any())
        ):
            raise VariableEpisodeInputError("invalid Teacher confidence")

    def _teacher_read(
        self,
        query: torch.Tensor,
        teacher: torch.Tensor,
        teacher_confidence: torch.Tensor,
        teacher_positions: torch.Tensor,
        valid_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, weights, _ = self.teacher_reader(
            query,
            teacher,
            teacher_positions,
            valid_teacher,
        )
        masked_confidence = teacher_confidence.masked_fill(
            ~valid_teacher[:, :, None],
            0.0,
        )
        flat_confidence = masked_confidence.reshape(
            teacher.shape[0],
            teacher.shape[1] * teacher.shape[2],
        ).to(weights.dtype)
        mean_weights = weights.mean(dim=1)
        confidence = torch.einsum(
            "bqm,bm->bq",
            mean_weights,
            flat_confidence,
        )
        return slots, confidence.clamp(0.0, 1.0), weights

    def fused_slots(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        teacher: torch.Tensor,
        teacher_confidence: torch.Tensor,
        teacher_positions: torch.Tensor,
        valid_teacher: torch.Tensor,
        policy: torch.Tensor,
        policy_positions: torch.Tensor,
        valid_policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_memories(
            core,
            valid_core,
            teacher,
            teacher_confidence,
            teacher_positions,
            valid_teacher,
            policy,
            policy_positions,
            valid_policy,
        )
        routing = self.routing_norm(self._routing())[None].expand(
            core.shape[0],
            -1,
            -1,
        )
        core_slots, core_attention = self.core_reader(
            routing,
            core,
            valid_core,
        )
        normalized_core = self.core_query_norm(core_slots)
        teacher_query = routing + normalized_core
        teacher_slots, confidence, teacher_attention = self._teacher_read(
            teacher_query,
            teacher,
            teacher_confidence,
            teacher_positions,
            valid_teacher,
        )
        policy_query = (
            routing
            + normalized_core
            + self.teacher_query_norm(teacher_slots)
        )
        policy_slots, policy_attention, _ = self.policy_reader(
            policy_query,
            policy,
            policy_positions,
            valid_policy,
        )
        aligned_teacher = self.teacher_alignment_norm(
            self.teacher_alignment(teacher_slots)
        )
        aligned_policy = self.policy_alignment_norm(
            self.policy_alignment(policy_slots)
        )
        gap = aligned_teacher - aligned_policy
        gap_rms = gap.to(torch.float32).square().mean(dim=-1).sqrt()
        gap_strength = gap_rms / (gap_rms + self.gap_tau)
        scales = confidence.to(gap_strength.dtype) * gap_strength

        normalized_gap = self.gap_norm(gap)
        core_support = self.core_support_norm(self.core_support(core_slots))
        core_gate = torch.tanh(self.core_gate(normalized_gap))
        assisted_gap = normalized_gap + core_gate * core_support
        content, coordination_attention = self.slot_coordination(
            assisted_gap,
            routing,
        )
        diagnostics = {
            "routing": routing,
            "core_slots": core_slots,
            "core_attention": core_attention,
            "teacher_query": teacher_query,
            "teacher_slots": teacher_slots,
            "teacher_attention": teacher_attention,
            "teacher_confidence": confidence,
            "policy_query": policy_query,
            "policy_slots": policy_slots,
            "policy_attention": policy_attention,
            "aligned_teacher": aligned_teacher,
            "aligned_policy": aligned_policy,
            "gap": gap,
            "gap_rms": gap_rms,
            "gap_strength": gap_strength,
            "adaptation_scale": scales,
            "normalized_gap": normalized_gap,
            "core_support": core_support,
            "core_gate": core_gate,
            "core_assisted_gap": assisted_gap,
            "coordination_attention": coordination_attention,
            "fused_slots": content,
        }
        return content, scales, diagnostics

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        teacher: torch.Tensor,
        teacher_confidence: torch.Tensor,
        teacher_positions: torch.Tensor,
        valid_teacher: torch.Tensor,
        policy: torch.Tensor,
        policy_positions: torch.Tensor,
        valid_policy: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        content, scales, _ = self.fused_slots(
            core,
            valid_core,
            teacher,
            teacher_confidence,
            teacher_positions,
            valid_teacher,
            policy,
            policy_positions,
            valid_policy,
        )
        expert_stop = self.EXPERT_LAYERS * self.RANK
        batch = core.shape[0]
        expert = content[:, :expert_stop].reshape(
            batch,
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        expert_scale = scales[:, :expert_stop].reshape(
            batch,
            self.EXPERT_LAYERS,
            self.RANK,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
            expert_scale,
            scales[:, expert_stop : expert_stop + self.RANK],
            scales[:, -self.RANK :],
        )
