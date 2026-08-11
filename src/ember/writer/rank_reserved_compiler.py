"""Frozen rank-reserved compiler for the active Reward Program deployment."""

from __future__ import annotations

import torch

from ember.writer.condition_update import (
    ConditionUpdateError,
    FixedBalancedCausalConditionFeature,
    ProgramResidualMemory,
    compact_rank2_effective_tangent,
    pivot_preserving_base_factors,
    stable_factor_head_linearization,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.temporal import SlotNormalizedCoreProcedureCompiler


RANK_RESERVED_QV_BASE_RANK = 14
RANK_RESERVED_QV_RESIDUAL_RANK = 2


def compile_rank_reserved_qv_factors(
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    *,
    tangent_a: torch.Tensor | None = None,
    tangent_b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish one native rank14 base plus two physical residual slots.

    The leading dimensions are deliberately opaque.  Online Writer decoding uses
    ``[batch, layer]`` while the compiler-only diagnostic uses the same
    ``[8, 18]`` topology loaded from the immutable v8 cache.
    """

    if (
        base_a.ndim < 2
        or base_b.ndim != base_a.ndim
        or base_a.shape[:-2] != base_b.shape[:-2]
        or base_a.shape[-2] != CompleteLoRAWriter.PUBLIC_LORA_RANK
        or base_b.shape[-1] != CompleteLoRAWriter.PUBLIC_LORA_RANK
        or base_a.dtype != base_b.dtype
        or base_a.device != base_b.device
        or (tangent_a is None) != (tangent_b is None)
    ):
        raise ConditionUpdateError("rank-reserved q/v factor topology changed")
    reduced_a, reduced_b, pivots = pivot_preserving_base_factors(
        base_a,
        base_b,
        keep=RANK_RESERVED_QV_BASE_RANK,
    )
    if tangent_a is None:
        residual_a = reduced_a.new_zeros(
            *reduced_a.shape[:-2],
            RANK_RESERVED_QV_RESIDUAL_RANK,
            reduced_a.shape[-1],
        )
        residual_b = reduced_b.new_zeros(
            *reduced_b.shape[:-2],
            reduced_b.shape[-2],
            RANK_RESERVED_QV_RESIDUAL_RANK,
        )
    else:
        if (
            tangent_a.shape != base_a.shape
            or tangent_b is None
            or tangent_b.shape != base_b.shape
            or tangent_a.device != base_a.device
            or tangent_b.device != base_b.device
        ):
            raise ConditionUpdateError("rank-reserved q/v tangent topology changed")
        residual_a, residual_b = compact_rank2_effective_tangent(
            base_a,
            base_b,
            tangent_a,
            tangent_b,
        )
    return (
        torch.cat((reduced_a, residual_a), dim=-2),
        torch.cat((reduced_b, residual_b), dim=-1),
        pivots,
    )


class FrozenV6RankReservedRewardWriter(torch.nn.Module):
    """Compile frozen-v6 Program motion into one native rank14+2 public LoRA."""

    QV_BASE_RANK = RANK_RESERVED_QV_BASE_RANK
    QV_RESIDUAL_RANK = RANK_RESERVED_QV_RESIDUAL_RANK

    def __init__(
        self,
        base_writer: CompleteLoRAWriter,
        *,
        feature_width: int,
        feature_seed: int,
        enable_program_residual: bool,
    ) -> None:
        super().__init__()
        if (
            base_writer.program_width <= 0
            or base_writer.PUBLIC_LORA_RANK
            != self.QV_BASE_RANK + self.QV_RESIDUAL_RANK
            or not isinstance(enable_program_residual, bool)
        ):
            raise ConditionUpdateError("invalid frozen v6 Writer")
        base_writer.requires_grad_(False).eval()
        self.base_writer = base_writer
        self.condition_feature = FixedBalancedCausalConditionFeature(
            program_width=base_writer.program_width,
            feature_width=feature_width,
            initialization_seed=feature_seed,
        )
        self.program_memory = ProgramResidualMemory(
            feature_width=feature_width,
            program_slots=SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            program_width=base_writer.program_width,
        )
        self.enable_program_residual = enable_program_residual
        self._factor_names: dict[str, tuple[str, ...]] = {}
        self._factor_template_buffers: dict[str, str] = {}
        grouped: dict[str, list[tuple[int | None, str, torch.Tensor]]] = {}
        for item in base_writer.tensor_specs:
            key, layer = base_writer._decoding[item.name]
            grouped.setdefault(key, []).append(
                (
                    layer,
                    item.name,
                    getattr(base_writer, base_writer._template_buffers[item.name]),
                )
            )
        if set(grouped) != set(base_writer.FACTOR_WIDTHS):
            raise ConditionUpdateError("frozen v6 factor layout changed")
        for index, key in enumerate(sorted(grouped)):
            entries = sorted(
                grouped[key],
                key=lambda value: -1 if value[0] is None else value[0],
            )
            if (key.startswith("action_") and len(entries) != 1) or (
                not key.startswith("action_")
                and (
                    len(entries) != base_writer.EXPERT_LAYERS
                    or [entry[0] for entry in entries]
                    != list(range(base_writer.EXPERT_LAYERS))
                )
            ):
                raise ConditionUpdateError("frozen v6 target layout changed")
            template = (
                entries[0][2].detach().clone()
                if len(entries) == 1
                else torch.stack([entry[2] for entry in entries])
            )
            buffer_name = f"rank_reserved_template_{index:02d}"
            self.register_buffer(buffer_name, template, persistent=False)
            self._factor_names[key] = tuple(entry[1] for entry in entries)
            self._factor_template_buffers[key] = buffer_name

    def train(self, mode: bool = True) -> FrozenV6RankReservedRewardWriter:
        super().train(mode)
        self.base_writer.eval()
        self.condition_feature.eval()
        self.program_memory.eval()
        return self

    def _native_factor(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        template = getattr(self, self._factor_template_buffers[key])
        generated = rows.transpose(-1, -2) if key.endswith("_b") else rows
        return generated.to(dtype=template.dtype) + template.unsqueeze(0)

    @staticmethod
    def _slot_sources(slots: torch.Tensor) -> dict[str, torch.Tensor]:
        if slots.ndim != 3:
            raise ConditionUpdateError("rank-reserved Program slots changed")
        expert_stop = (
            CompleteLoRAWriter.EXPERT_LAYERS * CompleteLoRAWriter.PUBLIC_LORA_RANK
        )
        expected = expert_stop + 2 * CompleteLoRAWriter.PUBLIC_LORA_RANK
        if slots.shape[1] != expected:
            raise ConditionUpdateError("rank-reserved Program topology changed")
        return {
            "expert": slots[:, :expert_stop].reshape(
                slots.shape[0],
                CompleteLoRAWriter.EXPERT_LAYERS,
                CompleteLoRAWriter.PUBLIC_LORA_RANK,
                slots.shape[-1],
            ),
            "action_in": slots[
                :, expert_stop : expert_stop + CompleteLoRAWriter.PUBLIC_LORA_RANK
            ],
            "action_out": slots[:, -CompleteLoRAWriter.PUBLIC_LORA_RANK :],
        }

    def _publish_factor(
        self,
        result: dict[str, torch.Tensor],
        key: str,
        value: torch.Tensor,
    ) -> None:
        names = self._factor_names[key]
        batch = value.shape[0]
        if len(names) == 1:
            result[names[0]] = value[0] if batch == 1 else value
            return
        for layer, name in enumerate(names):
            selected = value[:, layer]
            result[name] = selected[0] if batch == 1 else selected

    def _decode_qv(
        self,
        module: str,
        source: torch.Tensor,
        residual: torch.Tensor | None,
        result: dict[str, torch.Tensor],
    ) -> None:
        rows_a, delta_rows_a = stable_factor_head_linearization(
            self.base_writer.factor_heads[f"{module}_a"], source, residual
        )
        rows_b, delta_rows_b = stable_factor_head_linearization(
            self.base_writer.factor_heads[f"{module}_b"], source, residual
        )
        base_a = self._native_factor(f"{module}_a", rows_a)
        base_b = self._native_factor(f"{module}_b", rows_b)
        tangent_a = tangent_b = None
        if residual is not None:
            if delta_rows_a is None or delta_rows_b is None:
                raise ConditionUpdateError("rank-reserved tangent was not decoded")
            tangent_a = delta_rows_a
            tangent_b = delta_rows_b.transpose(-1, -2)
        public_a, public_b, _ = compile_rank_reserved_qv_factors(
            base_a,
            base_b,
            tangent_a=tangent_a,
            tangent_b=tangent_b,
        )
        self._publish_factor(result, f"{module}_a", public_a)
        self._publish_factor(result, f"{module}_b", public_b)

    def _decode_action(
        self,
        module: str,
        source: torch.Tensor,
        residual: torch.Tensor | None,
        result: dict[str, torch.Tensor],
    ) -> None:
        for factor in ("a", "b"):
            key = f"{module}_{factor}"
            rows, tangent = stable_factor_head_linearization(
                self.base_writer.factor_heads[key], source, residual
            )
            value = self._native_factor(key, rows)
            if tangent is not None:
                oriented = tangent.transpose(-1, -2) if factor == "b" else tangent
                value = value + oriented.to(dtype=value.dtype)
            self._publish_factor(result, key, value)

    def _decode_compiled_slots(
        self,
        base_slots: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if residual is not None and residual.shape != base_slots.shape:
            raise ConditionUpdateError("Program residual lost fused-slot topology")
        base_sources = self._slot_sources(base_slots)
        residual_sources = None if residual is None else self._slot_sources(residual)
        result: dict[str, torch.Tensor] = {}
        expert_residual = (
            None if residual_sources is None else residual_sources["expert"]
        )
        self._decode_qv("q", base_sources["expert"], expert_residual, result)
        self._decode_qv("v", base_sources["expert"], expert_residual, result)
        for module in ("action_in", "action_out"):
            self._decode_action(
                module,
                base_sources[module],
                None if residual_sources is None else residual_sources[module],
                result,
            )
        if set(result) != set(self.base_writer.template_state()):
            raise ConditionUpdateError("rank-reserved public LoRA topology changed")
        return result

    def decode_rank_reserved_slots(
        self,
        base_slots: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Decode rank14 base and optional condition-local rank2 Reward motion."""

        if (residual is None) != (not self.enable_program_residual):
            raise ConditionUpdateError("Program residual enablement changed")
        return self._decode_compiled_slots(base_slots, residual)

    def diagnostic_five_arm_slots(
        self,
        base_slots: torch.Tensor,
        residual: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Compile Gate-A references from one condition without extra video forwards."""

        if not self.enable_program_residual or residual.shape != base_slots.shape:
            raise ConditionUpdateError("rank-reserved diagnostic condition changed")
        old_base = self.base_writer.decode_slots(base_slots)
        old_reward = self.base_writer.decode_slots(
            base_slots + residual.to(dtype=base_slots.dtype)
        )
        rank14_base = self._decode_compiled_slots(base_slots, None)
        rank14_reward = self._decode_compiled_slots(base_slots, residual)
        qv_names = {
            name
            for key in ("q_a", "q_b", "v_a", "v_b")
            for name in self._factor_names[key]
        }
        qv_only = {
            name: rank14_reward[name] if name in qv_names else rank14_base[name]
            for name in rank14_base
        }
        return {
            "old_full_rank_base": old_base,
            "old_full_rank_reward": old_reward,
            "rank14_base": rank14_base,
            "rank14_plus2_qv_only": qv_only,
            "rank14_plus2_reward": rank14_reward,
        }

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        evidence = self.base_writer.encode_video_evidence(
            policy,
            frames,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        memories = self.base_writer.build_memories(evidence, frame_indices)
        base_slots = self.base_writer.compile_slots(memories)
        residual = None
        if self.enable_program_residual:
            features = self.condition_feature(evidence, frame_indices)
            residual = self.program_memory(features)
        return self.decode_rank_reserved_slots(base_slots, residual)

    def forward_diagnostic_five_arms(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        offsets: torch.Tensor,
        task_tokens: torch.Tensor,
        task_attention_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode one Gate-A panel once, then compile all numeric references."""

        if not self.enable_program_residual:
            raise ConditionUpdateError("rank-reserved diagnostic requires cycle1")
        evidence = self.base_writer.encode_video_evidence(
            policy,
            frames,
            offsets,
            task_tokens,
            task_attention_mask,
            task_span_mask,
        )
        memories = self.base_writer.build_memories(evidence, frame_indices)
        base_slots = self.base_writer.compile_slots(memories)
        features = self.condition_feature(evidence, frame_indices)
        residual = self.program_memory(features)
        return self.diagnostic_five_arm_slots(base_slots, residual)


def validate_frozen_v6_rank_reserved_writer(
    writer: FrozenV6RankReservedRewardWriter,
    *,
    require_zero_memory: bool = False,
) -> None:
    """Fail closed if the wrapper gains trainable or malformed dynamic state."""

    base_state = writer.base_writer.state_dict()
    if (
        len(base_state) != 600
        or any(parameter.requires_grad for parameter in writer.parameters())
        or writer.program_memory.value.dtype != torch.float32
        or writer.program_memory.value.shape
        != (
            writer.condition_feature.feature_width,
            SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            writer.base_writer.program_width,
        )
        or not isinstance(writer.enable_program_residual, bool)
        or (
            require_zero_memory
            and bool(torch.count_nonzero(writer.program_memory.value))
        )
    ):
        raise WriterModelError("frozen v6 rank-reserved Writer ownership changed")
