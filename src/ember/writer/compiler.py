"""Compile a semantic Core prior plus an ordered Procedure innovation."""

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


class LearnedCrossAttention(torch.nn.Module):
    """Bias-free learned Q/K/V/O cross-attention."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        rotary_keys: bool,
        remove_uniform_value: bool,
    ) -> None:
        super().__init__()
        if (
            min(width, heads) <= 0
            or width % heads
            or (rotary_keys and (width // heads) % 2)
        ):
            raise VariableEpisodeInputError("invalid learned cross-attention")
        self.heads = int(heads)
        self.rotary_keys = bool(rotary_keys)
        self.remove_uniform_value = bool(remove_uniform_value)
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
            raise VariableEpisodeInputError("invalid learned attention batch")
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
        value_weights = weights
        if self.remove_uniform_value:
            uniform = valid_memory.to(torch.float32)
            uniform = uniform / uniform.sum(dim=1, keepdim=True)
            value_weights = value_weights - uniform[:, None, None, :]
        attended = torch.matmul(value_weights.to(value.dtype), value)
        return self.output(_merge_heads(attended)), weights


class SemanticPriorReader(torch.nn.Module):
    """Route raw semantic Core values into stable public-LoRA slot priors."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.memory_norm = RMSNorm(width)
        self.attention = LearnedCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=False,
            remove_uniform_value=False,
        )
        self.prior_norm = RMSNorm(width)

    def forward(
        self,
        routing: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_prior, weights = self.attention(
            routing,
            self.memory_norm(core),
            core,
            valid_core,
        )
        return self.prior_norm(raw_prior), raw_prior, weights


class ProcedureInnovationReader(torch.nn.Module):
    """Use only the semantic prior to read centered ordered Procedure values."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.memory_norm = RMSNorm(width)
        self.attention = LearnedCrossAttention(
            width=width,
            heads=heads,
            rotary_keys=True,
            remove_uniform_value=True,
        )

    @staticmethod
    def center(
        procedure: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> torch.Tensor:
        """Return an FP32 masked time-centered value cast back to input dtype."""

        if (
            procedure.ndim != 3
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid Procedure centering batch")
        mask = valid_procedure[..., None]
        value = procedure.to(torch.float32)
        count = mask.sum(dim=1, keepdim=True).to(torch.float32)
        mean = (value * mask).sum(dim=1, keepdim=True) / count
        centered = (value - mean).masked_fill(~mask, 0.0)
        return centered.to(procedure.dtype)

    def forward(
        self,
        semantic_prior: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            semantic_prior.ndim != 3
            or procedure.ndim != 3
            or procedure.shape[0] != semantic_prior.shape[0]
            or procedure.shape[-1] != semantic_prior.shape[-1]
            or positions.shape != procedure.shape[:2]
            or positions.dtype != torch.long
        ):
            raise VariableEpisodeInputError("invalid Procedure innovation memory")
        centered = self.center(procedure, valid_procedure)
        innovation, weights = self.attention(
            semantic_prior,
            self.memory_norm(procedure),
            centered,
            valid_procedure,
            positions,
        )
        return innovation, centered, weights


class PriorInnovationSlotBlock(torch.nn.Module):
    """Coordinate formed slots while keeping routing out of value content."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid Prior-Innovation slot block")
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
        self.output_norm = RMSNorm(width)

    def forward(
        self,
        content: torch.Tensor,
        routing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if content.shape != routing.shape or content.ndim != 3:
            raise VariableEpisodeInputError("invalid slot coordination batch")
        addressed = self.attention_norm(content) + routing.to(content.dtype)
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


class PriorInnovationCompiler(torch.nn.Module):
    """Generate LoRA slots from a semantic prior plus ordered innovation."""

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
        if (
            min(width, heads) <= 0
            or width % heads
            or (width // heads) % 2
        ):
            raise VariableEpisodeInputError("invalid Prior-Innovation compiler")
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
        self.core_reader = SemanticPriorReader(width=width, heads=heads)
        self.procedure_reader = ProcedureInnovationReader(
            width=width,
            heads=heads,
        )
        self.slot_block = PriorInnovationSlotBlock(
            width=width,
            heads=heads,
        )

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
            or positions.dtype != torch.long
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or core.shape[0] != procedure.shape[0]
            or core.shape[-1] != procedure.shape[-1]
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
        semantic_prior, raw_core_slots, core_attention = self.core_reader(
            routing,
            core,
            valid_core,
        )
        innovation, centered_procedure, procedure_attention = (
            self.procedure_reader(
                semantic_prior,
                procedure,
                positions,
                valid_procedure,
            )
        )
        prior_plus_innovation = semantic_prior + innovation
        content, slot_attention = self.slot_block(
            prior_plus_innovation,
            routing,
        )
        diagnostics = {
            "routing": routing,
            "raw_core_slots": raw_core_slots,
            "semantic_prior": semantic_prior,
            "core_attention": core_attention,
            "centered_procedure": centered_procedure,
            "procedure_innovation": innovation,
            "procedure_attention": procedure_attention,
            "prior_plus_innovation": prior_plus_innovation,
            "slot_attention": slot_attention,
            "fused_slots": content,
        }
        return content, diagnostics

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
        batch = core.shape[0]
        expert = content[:, :expert_stop].reshape(
            batch,
            self.EXPERT_LAYERS,
            self.RANK,
            -1,
        )
        return (
            expert,
            content[:, expert_stop : expert_stop + self.RANK],
            content[:, -self.RANK :],
        )
