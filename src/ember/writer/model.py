"""Canonical V6 layerwise Action-probe conditioned Procedure Writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import weakref

import torch
from safetensors.torch import load_file

from ember.expert_manifold.legacy_v6_architecture import (
    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS,
)
from ember.expert_manifold.legacy_v6_model import (
    CompleteLoRAWriter as NativeV6Writer,
)
from ember.expert_manifold.legacy_v6_model import build_lora_tensor_specs
from ember.writer.errors import WriterModelError
from ember.writer.factor_commitment import (
    FACTOR_FAMILIES,
    DirectJointNativeFactorResidual,
)
from ember.writer.slot_set import PolicyProcedureCommonValueFusion
from ember.writer.temporal import LayerwiseProbeProcedureConditioner
from ember.writer.video_program import LayerwiseActionProbeReader


@dataclass(frozen=True)
class WriterProgramDiagnostics:
    """Layerwise query and native V6 readouts retained for analysis."""

    shared_core_slots: torch.Tensor
    per_video_procedure_slots: torch.Tensor
    shared_procedure_slots: torch.Tensor
    shared_procedure_corrections: torch.Tensor
    per_video_query_conditioners: torch.Tensor
    per_video_query_deltas: torch.Tensor
    shared_probe_value_slots: torch.Tensor
    attention: tuple[torch.Tensor, ...]
    auxiliary_loss: torch.Tensor


@dataclass(frozen=True)
class WriterProgramOutput:
    """Frozen V6 program conditioned by trainable layerwise probe changes."""

    program: torch.Tensor
    diagnostics: WriterProgramDiagnostics
    reference_program: torch.Tensor | None = None
    probe_value_memory: torch.Tensor | None = None
    language_slots: torch.Tensor | None = None


@dataclass(frozen=True)
class WriterConditioningState:
    """Reusable video state before the final layerwise Procedure commitment."""

    shared_core_slots: torch.Tensor
    procedure_memory: torch.Tensor
    procedure_positions: torch.Tensor
    valid_procedure: torch.Tensor
    video_condition_ids: torch.Tensor
    per_video_query_conditioners: torch.Tensor
    language_slots: torch.Tensor


class CompleteLoRAWriter(torch.nn.Module):
    """Condition the strong V6 Procedure reader with layer/rank probe evidence."""

    PUBLIC_LORA_RANK = 16
    PROGRAM_WIDTH = 256
    POLICY_SLOTS = 320

    def __init__(
        self,
        base_writer: NativeV6Writer,
        procedure_set: PolicyProcedureCommonValueFusion,
        layer_probe_reader: LayerwiseActionProbeReader,
        *,
        expert_model: torch.nn.Module,
        initialization_seed: int,
        conditioner_heads: int = 8,
        conditioner_blocks: int = 1,
    ) -> None:
        super().__init__()
        if (
            int(base_writer.program_width) != self.PROGRAM_WIDTH
            or base_writer.PUBLIC_LORA_RANK != self.PUBLIC_LORA_RANK
        ):
            raise WriterModelError("native v6 Writer topology changed")
        self.base_writer = base_writer.requires_grad_(False).eval()
        self.procedure_set = procedure_set.requires_grad_(False).eval()
        self.layer_probe_reader = layer_probe_reader
        self.probe_conditioner = LayerwiseProbeProcedureConditioner(
            heads=conditioner_heads,
            blocks=conditioner_blocks,
        )
        self.query_delta = torch.nn.Linear(
            self.PROGRAM_WIDTH,
            self.PROGRAM_WIDTH,
            bias=False,
        )
        torch.nn.init.zeros_(self.query_delta.weight)
        self.factor_commitment = DirectJointNativeFactorResidual(
            width=self.PROGRAM_WIDTH,
        ).requires_grad_(False)
        object.__setattr__(self, "_expert_model_ref", weakref.ref(expert_model))

    @classmethod
    def from_policy(
        cls,
        *,
        policy: torch.nn.Module,
        template_state: Mapping[str, torch.Tensor],
        writer_config: Mapping[str, Any],
        as139_warm_start_checkpoint: Path,
    ) -> CompleteLoRAWriter:
        """Load the frozen AS139 graph and initialize only its query conditioner."""

        bridge = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
        if bridge is None:
            raise WriterModelError("PI05 policy lost its joint backbone")
        arguments = {
            name: writer_config[name]
            for name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
            if name in writer_config
        }
        base = NativeV6Writer(
            build_lora_tensor_specs(template_state),
            template_state=template_state,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
            **arguments,
        )
        procedure_set = PolicyProcedureCommonValueFusion(width=cls.PROGRAM_WIDTH)
        cls._load_as139_state_(
            base,
            procedure_set,
            as139_warm_start_checkpoint.resolve(),
        )
        layer_probe_reader = LayerwiseActionProbeReader(
            heads=int(writer_config["layer_probe_heads"]),
            initialization_seed=int(writer_config["initialization_seed"]),
        )
        return cls(
            base,
            procedure_set,
            layer_probe_reader,
            expert_model=bridge.gemma_expert.model,
            initialization_seed=int(writer_config["initialization_seed"]),
            conditioner_heads=int(writer_config["conditioner_heads"]),
            conditioner_blocks=int(writer_config["conditioner_blocks"]),
        )

    @staticmethod
    def _load_as139_state_(
        base_writer: NativeV6Writer,
        procedure_set: PolicyProcedureCommonValueFusion,
        checkpoint: Path,
    ) -> None:
        if not checkpoint.is_file() or checkpoint.name != "writer.safetensors":
            raise WriterModelError("missing AS139 Writer warm start")
        state = load_file(str(checkpoint), device="cpu")
        base_prefix = "base_writer."
        set_prefix = "procedure_set."
        expected = {
            *(base_prefix + name for name in base_writer.state_dict()),
            *(set_prefix + name for name in procedure_set.state_dict()),
        }
        if set(state) != expected:
            raise WriterModelError("AS139 Writer warm-start topology changed")
        base_writer.load_state_dict(
            {
                name.removeprefix(base_prefix): value
                for name, value in state.items()
                if name.startswith(base_prefix)
            },
            strict=True,
        )
        procedure_set.load_state_dict(
            {
                name.removeprefix(set_prefix): value
                for name, value in state.items()
                if name.startswith(set_prefix)
            },
            strict=True,
        )

    def train(self, mode: bool = True) -> CompleteLoRAWriter:
        """Keep the inherited AS139 graph frozen under every training mode."""

        super().train(mode)
        self.base_writer.eval()
        self.procedure_set.eval()
        return self

    def load_lpcp_state_(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load sealed LPCP while forcing a fresh commitment initialization."""

        expected_missing = {
            f"factor_commitment.{name}"
            for name in self.factor_commitment.state_dict()
        }
        incompatible = self.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        if missing != expected_missing or incompatible.unexpected_keys:
            raise WriterModelError("LPCP checkpoint topology changed")

    @staticmethod
    def _offsets(
        value: torch.Tensor,
        *,
        final: int,
        name: str,
    ) -> tuple[int, ...]:
        if value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 1:
            raise WriterModelError(f"{name} must be a CPU long tensor")
        offsets = tuple(int(item) for item in value.tolist())
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != final
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise WriterModelError(f"invalid {name}")
        return offsets

    def template_state(self) -> dict[str, torch.Tensor]:
        return self.base_writer.template_state()

    def compile_readouts(
        self,
        shared_core: torch.Tensor,
        per_video_procedure: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        *,
        per_video_query_conditioners: torch.Tensor | None = None,
        per_video_query_deltas: torch.Tensor | None = None,
    ) -> WriterProgramOutput:
        """Aggregate conditioned readouts through the frozen AS139 tail."""

        condition_bounds = self._offsets(
            condition_video_offsets,
            final=per_video_procedure.shape[0],
            name="condition video offsets",
        )
        condition_count = len(condition_bounds) - 1
        if (
            shared_core.shape
            != (condition_count, self.POLICY_SLOTS, self.PROGRAM_WIDTH)
            or per_video_procedure.shape[1:] != (self.POLICY_SLOTS, self.PROGRAM_WIDTH)
            or any(
                right - left not in range(1, 5)
                for left, right in zip(condition_bounds, condition_bounds[1:])
            )
        ):
            raise WriterModelError("invalid cached ordered-Procedure readouts")
        if per_video_query_conditioners is None:
            per_video_query_conditioners = per_video_procedure.new_zeros(
                per_video_procedure.shape
            )
        if per_video_query_deltas is None:
            per_video_query_deltas = per_video_procedure.new_zeros(
                per_video_procedure.shape
            )
        if (
            per_video_query_conditioners.shape != per_video_procedure.shape
            or per_video_query_deltas.shape != per_video_procedure.shape
        ):
            raise WriterModelError("invalid layerwise Procedure query diagnostics")
        shared_procedure, set_diagnostics = self.procedure_set(
            per_video_procedure, condition_video_offsets
        )
        shared_probe_value = torch.stack(
            [
                (
                    attention[..., None]
                    * per_video_query_conditioners[left:right]
                ).sum(dim=0)
                for (left, right), attention in zip(
                    zip(condition_bounds, condition_bounds[1:]),
                    set_diagnostics.attention,
                    strict=True,
                )
            ]
        )
        routing = self.base_writer.compiler.routing(condition_count)
        program, _, _ = self.base_writer.compiler.fuse_readouts(
            routing,
            shared_core,
            shared_procedure,
        )
        return WriterProgramOutput(
            program,
            WriterProgramDiagnostics(
                shared_core_slots=shared_core,
                per_video_procedure_slots=per_video_procedure,
                shared_procedure_slots=shared_procedure,
                shared_procedure_corrections=set_diagnostics.shared_corrections,
                per_video_query_conditioners=per_video_query_conditioners,
                per_video_query_deltas=per_video_query_deltas,
                shared_probe_value_slots=shared_probe_value,
                attention=set_diagnostics.attention,
                auxiliary_loss=set_diagnostics.auxiliary_loss,
            ),
        )

    def compile_conditioning_state(
        self,
        state: WriterConditioningState,
        condition_video_offsets: torch.Tensor,
        *,
        use_query_delta: bool,
    ) -> WriterProgramOutput:
        """Compile one cached context with either LPCP or exact AS139 queries."""

        video_count = int(state.procedure_memory.shape[0])
        expected_slots = (video_count, self.POLICY_SLOTS, self.PROGRAM_WIDTH)
        if (
            state.shared_core_slots.ndim != 3
            or state.shared_core_slots.shape[1:]
            != (self.POLICY_SLOTS, self.PROGRAM_WIDTH)
            or state.procedure_positions.shape != state.procedure_memory.shape[:2]
            or state.valid_procedure.shape != state.procedure_memory.shape[:2]
            or state.valid_procedure.dtype != torch.bool
            or state.video_condition_ids.shape != (video_count,)
            or state.video_condition_ids.dtype != torch.long
            or state.per_video_query_conditioners.shape != expected_slots
            or state.language_slots.shape
            != (
                state.shared_core_slots.shape[0],
                self.POLICY_SLOTS,
                self.PROGRAM_WIDTH,
            )
        ):
            raise WriterModelError("invalid cached layerwise conditioning state")
        compiler = self.base_writer.compiler
        routing = compiler.routing(video_count)
        normalized_core = state.shared_core_slots.index_select(
            0, state.video_condition_ids
        )
        query_deltas = self.query_delta(state.per_video_query_conditioners)
        query_condition: torch.Tensor | None = (
            query_deltas if use_query_delta else None
        )
        if not use_query_delta:
            query_deltas = torch.zeros_like(query_deltas)
        per_video_procedure, _ = compiler.read_procedure_slots(
            routing,
            normalized_core,
            state.procedure_memory,
            state.procedure_positions,
            state.valid_procedure,
            query_condition=query_condition,
        )
        compiled = self.compile_readouts(
            state.shared_core_slots,
            per_video_procedure,
            condition_video_offsets,
            per_video_query_conditioners=state.per_video_query_conditioners,
            per_video_query_deltas=query_deltas,
        )
        if not use_query_delta:
            return compiled
        reference_procedure, _ = compiler.read_procedure_slots(
            routing,
            normalized_core,
            state.procedure_memory,
            state.procedure_positions,
            state.valid_procedure,
            query_condition=None,
        )
        reference = self.compile_readouts(
            state.shared_core_slots,
            reference_procedure,
            condition_video_offsets,
            per_video_query_conditioners=state.per_video_query_conditioners,
        )
        return WriterProgramOutput(
            program=compiled.program,
            diagnostics=compiled.diagnostics,
            reference_program=reference.program,
            probe_value_memory=compiled.diagnostics.shared_probe_value_slots,
            language_slots=state.language_slots,
        )

    def _read_language_slots(
        self,
        evidence: Any,
        condition_bounds: tuple[int, ...],
    ) -> torch.Tensor:
        """Read the stable text-only address once per task condition."""

        compiler = self.base_writer.compiler
        first_video_ids = torch.tensor(
            condition_bounds[:-1],
            dtype=torch.long,
            device=evidence.text_queries.device,
        )
        language_queries = evidence.text_queries.index_select(0, first_video_ids)
        valid_language = evidence.valid_task_tokens.index_select(0, first_video_ids)
        routing = compiler.routing(len(condition_bounds) - 1)
        return compiler.normalize_core_slots(
            compiler.read_core_slots(routing, language_queries, valid_language)
        )

    def encode_conditioning_state(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> WriterConditioningState:
        """Read the joint context once and retain the final commitment inputs."""

        video_bounds = self._offsets(
            video_offsets, final=frames.shape[0], name="video offsets"
        )
        condition_bounds = self._offsets(
            condition_video_offsets,
            final=len(video_bounds) - 1,
            name="condition video offsets",
        )
        cardinalities = tuple(
            right - left for left, right in zip(condition_bounds, condition_bounds[1:])
        )
        condition_counts = torch.tensor(
            cardinalities,
            dtype=torch.long,
            device=language_tokens.device,
        )
        if (
            frame_indices.shape != (frames.shape[0],)
            or frame_indices.dtype != torch.long
            or frame_indices.device != frames.device
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != len(condition_bounds) - 1
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
            or any(count not in range(1, 5) for count in cardinalities)
        ):
            raise WriterModelError("invalid v6 ordered-Procedure common-Value batch")
        video_condition_ids = torch.repeat_interleave(
            torch.arange(
                len(condition_bounds) - 1,
                dtype=torch.long,
                device=language_tokens.device,
            ),
            condition_counts,
        )
        expert_model = self._expert_model_ref()
        if expert_model is None:
            raise WriterModelError("Action Expert owner was released")
        with self.layer_probe_reader.capture(expert_model) as probe_capture:
            with torch.no_grad():
                evidence = self.base_writer.encode_video_evidence(
                    policy,
                    frames,
                    video_offsets,
                    language_tokens.index_select(0, video_condition_ids),
                    language_mask.index_select(0, video_condition_ids),
                    task_span_mask.index_select(0, video_condition_ids),
                )
        layer_rank_probe = probe_capture.result(frames.shape[0])
        query_conditioners = self.probe_conditioner(
            layer_rank_probe,
            frame_indices,
            video_bounds,
        )
        with torch.no_grad():
            memories = self.base_writer.build_memories(evidence, frame_indices)
            compiler = self.base_writer.compiler
            shared_core_slots: list[torch.Tensor | None] = [None] * len(cardinalities)
            for video_count in range(1, 5):
                condition_ids = [
                    condition_id
                    for condition_id, count in enumerate(cardinalities)
                    if count == video_count
                ]
                if not condition_ids:
                    continue
                video_ids = torch.tensor(
                    [
                        video_id
                        for condition_id in condition_ids
                        for video_id in range(
                            condition_bounds[condition_id],
                            condition_bounds[condition_id + 1],
                        )
                    ],
                    dtype=torch.long,
                    device=memories.core.device,
                )
                core = memories.core.index_select(0, video_ids).reshape(
                    len(condition_ids), -1, self.PROGRAM_WIDTH
                )
                valid_core = memories.valid_core.index_select(0, video_ids).reshape(
                    len(condition_ids), -1
                )
                routing = compiler.routing(len(condition_ids))
                core_slots = compiler.read_core_slots(routing, core, valid_core)
                normalized_core = compiler.normalize_core_slots(core_slots)
                for row, condition_id in enumerate(condition_ids):
                    shared_core_slots[condition_id] = normalized_core[row]
            if any(value is None for value in shared_core_slots):
                raise WriterModelError("missing shared Core condition")
            shared_core = torch.stack(
                [value for value in shared_core_slots if value is not None]
            )
            language_slots = self._read_language_slots(evidence, condition_bounds)
        return WriterConditioningState(
            shared_core_slots=shared_core,
            procedure_memory=memories.procedure,
            procedure_positions=memories.positions,
            valid_procedure=memories.valid_procedure,
            video_condition_ids=video_condition_ids,
            per_video_query_conditioners=query_conditioners,
            language_slots=language_slots,
        )

    def encode_program(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
        singleton_video_index: int = 0,
    ) -> WriterProgramOutput:
        """Share frozen Core, then aggregate raw ordered Procedure Value."""

        del singleton_video_index
        state = self.encode_conditioning_state(
            frames,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
        )
        return self.compile_conditioning_state(
            state,
            condition_video_offsets,
            use_query_delta=True,
        )

    def decode_program(
        self,
        program: torch.Tensor,
        *,
        probe_value_memory: torch.Tensor | None = None,
        language_slots: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        baseline = self.base_writer.decode_slots(program)
        if probe_value_memory is None:
            if language_slots is not None:
                raise WriterModelError("incomplete shared native-Value decode")
            return baseline
        if language_slots is None:
            raise WriterModelError("shared native Value lost its language condition")
        rows, _ = self.factor_commitment(
            probe_value_memory,
            language_slots,
        )
        residuals = self._direct_factor_residual_state(rows, program.shape[0])
        if set(residuals) != set(baseline):
            raise WriterModelError("direct native-factor state changed")
        return {
            name: value + residuals[name].to(value.dtype)
            for name, value in baseline.items()
        }

    def _direct_factor_residual_state(
        self,
        rows: Mapping[str, torch.Tensor],
        batch: int,
    ) -> dict[str, torch.Tensor]:
        """Map direct family rows back to the sealed 76-tensor public schema."""

        if set(rows) != set(FACTOR_FAMILIES):
            raise WriterModelError("direct native-factor families changed")
        decoding = getattr(self.base_writer, "_decoding", None)
        tensor_specs = getattr(self.base_writer, "tensor_specs", None)
        if not isinstance(decoding, dict) or tensor_specs is None:
            raise WriterModelError("native V6 factor ownership changed")
        result: dict[str, torch.Tensor] = {}
        for item in tensor_specs:
            family, layer = decoding[item.name]
            source = rows[family]
            if family.startswith(("q_", "v_")):
                if layer is None:
                    raise WriterModelError("expert factor lost its layer")
                source = source[:, layer]
            generated = source.transpose(-1, -2) if item.transpose_output else source
            result[item.name] = generated[0] if batch == 1 else generated
        return result

    def decode_output(self, encoded: WriterProgramOutput) -> dict[str, torch.Tensor]:
        if encoded.probe_value_memory is None or encoded.language_slots is None:
            return self.decode_program(encoded.program)
        return self.decode_program(
            encoded.program,
            probe_value_memory=encoded.probe_value_memory,
            language_slots=encoded.language_slots,
        )

    def forward_training(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
        singleton_video_index: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encode_program(
            frames,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
            singleton_video_index=singleton_video_index,
        )
        return self.decode_output(encoded), encoded.diagnostics.auxiliary_loss

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        generated, _ = self.forward_training(
            frames,
            frame_indices,
            video_offsets,
            condition_video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
            policy=policy,
        )
        return generated
