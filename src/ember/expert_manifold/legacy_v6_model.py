"""Canonical Language-Axial Core + Visual-Transition Procedure PI05 Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.expert_manifold.legacy_v6_architecture import (
    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS,
    validate_writer_dimensions,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    SlotNormalizedCoreProcedureCompiler,
    TaskGroundedVisualTransitionFusion,
)
from ember.writer.video_program import Pi05LanguageAxialEncoder


class WriterModelError(RuntimeError):
    """Raised when the Writer input or sealed LoRA contract is inconsistent."""


@dataclass(frozen=True)
class LoraTensorSpec:
    """One row-oriented output tensor in a PEFT LoRA state."""

    name: str
    module: str
    module_index: int
    factor_index: int
    rank: int
    width: int
    transpose_output: bool


@dataclass(frozen=True)
class WriterVideoEvidence:
    """Per-frame v6 evidence before any temporal ordering is applied."""

    text_queries: torch.Tensor
    frame_evidence: torch.Tensor
    grounded_evidence: torch.Tensor
    interactions: torch.Tensor
    valid_task_tokens: torch.Tensor
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class WriterMemories:
    """Frozen Core and ordered Procedure memories consumed by the compiler."""

    core: torch.Tensor
    valid_core: torch.Tensor
    procedure: torch.Tensor
    positions: torch.Tensor
    valid_procedure: torch.Tensor
    frame_attention: torch.Tensor


def build_lora_tensor_specs(
    state: Mapping[str, torch.Tensor],
) -> tuple[LoraTensorSpec, ...]:
    """Build paired A/B output specifications from a real PEFT state."""

    marker_a = ".lora_A.default.weight"
    marker_b = ".lora_B.default.weight"
    modules: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, value in state.items():
        if name.endswith(marker_a):
            marker, factor = marker_a, "A"
        elif name.endswith(marker_b):
            marker, factor = marker_b, "B"
        else:
            raise WriterModelError(f"non-LoRA tensor in template: {name}")
        if value.ndim != 2:
            raise WriterModelError(f"LoRA tensor is not a matrix: {name}")
        module = name[: -len(marker)]
        modules.setdefault(module, {})[factor] = (name, value)

    if not modules or any(set(pair) != {"A", "B"} for pair in modules.values()):
        raise WriterModelError("every target module must contain one LoRA A/B pair")

    result: list[LoraTensorSpec] = []
    for module_index, module in enumerate(sorted(modules)):
        name_a, value_a = modules[module]["A"]
        name_b, value_b = modules[module]["B"]
        rank, input_width = value_a.shape
        output_width, rank_b = value_b.shape
        if rank <= 0 or rank_b != rank:
            raise WriterModelError(f"LoRA rank differs for {module}")
        result.extend(
            (
                LoraTensorSpec(
                    name=name_a,
                    module=module,
                    module_index=module_index,
                    factor_index=0,
                    rank=rank,
                    width=input_width,
                    transpose_output=False,
                ),
                LoraTensorSpec(
                    name=name_b,
                    module=module,
                    module_index=module_index,
                    factor_index=1,
                    rank=rank,
                    width=output_width,
                    transpose_output=True,
                ),
            )
        )
    return tuple(result)


class FactorHead(torch.nn.Module):
    """Decode one video-conditioned rank-slot state into one LoRA row."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        if min(input_width, hidden_width, output_width) <= 0:
            raise WriterModelError("invalid LoRA factor-head dimensions")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=False),
        )
        torch.nn.init.zeros_(self.network[-1].weight)

    def hidden(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise WriterModelError("factor head lost its rank-slot dimension")
        return self.network[1](self.network[0](value))

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim < 3 or hidden.shape[-1] != self.network[2].in_features:
            raise WriterModelError("factor head hidden state changed shape")
        return self.network[2](hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode_hidden(self.hidden(value))


class CompleteLoRAWriter(torch.nn.Module):
    """Map task language and one raw video to the sealed rank-16 task LoRA."""

    EXPERT_LAYERS = 18
    PUBLIC_LORA_RANK = 16
    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )
    FACTOR_WIDTHS = {
        "q_a": 1024,
        "q_b": 2048,
        "v_a": 1024,
        "v_b": 256,
        "action_in_a": 32,
        "action_in_b": 1024,
        "action_out_a": 1024,
        "action_out_b": 32,
    }

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        expert_width: int,
        program_width: int,
        text_meta_lora_rank: int,
        vl_meta_lora_rank: int,
        action_meta_lora_rank: int,
        patch_grounding_heads: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        semantic_core_heads: int,
        semantic_core_blocks: int,
        procedure_heads: int,
        procedure_blocks: int,
        visual_transition_heads: int,
        fusion_heads: int,
        factor_hidden_width: int,
        initialization_seed: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        constructor_values = locals()
        dimensions = {
            name: constructor_values[name]
            for name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
            if name
            not in {
                "max_frames_per_encoder_call",
                "initialization_seed",
                "activation_checkpointing",
            }
        }
        try:
            validate_writer_dimensions(dimensions)
        except ValueError as error:
            raise WriterModelError("invalid Language-Axial Writer topology") from error
        if not tensor_specs or max_frames_per_encoder_call <= 0:
            raise WriterModelError("invalid Language-Axial Writer topology")
        if set(template_state) != {item.name for item in tensor_specs}:
            raise WriterModelError("Writer LoRA template names changed")
        if (
            len(paligemma_model.layers) != self.EXPERT_LAYERS
            or len(expert_model.layers) != self.EXPERT_LAYERS
        ):
            raise WriterModelError("frozen PI05 expert depth changed")
        if {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}:
            raise WriterModelError("public Writer LoRA rank changed")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.semantic_encoder = Pi05LanguageAxialEncoder(
            paligemma_model=paligemma_model,
            expert_model=expert_model,
            image_width=image_width,
            expert_width=expert_width,
            program_width=program_width,
            text_meta_lora_rank=text_meta_lora_rank,
            vl_meta_lora_rank=vl_meta_lora_rank,
            action_meta_lora_rank=action_meta_lora_rank,
            patch_grounding_heads=patch_grounding_heads,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=initialization_seed,
            activation_checkpointing=activation_checkpointing,
        )
        self.semantic_core = LanguageSemanticCore(
            width=program_width,
            heads=semantic_core_heads,
            blocks=semantic_core_blocks,
        )
        self.visual_transition = TaskGroundedVisualTransitionFusion(
            width=program_width,
            heads=visual_transition_heads,
        )
        self.procedure = CausalProcedureEncoder(
            width=program_width,
            heads=procedure_heads,
            blocks=procedure_blocks,
        )
        self.compiler = SlotNormalizedCoreProcedureCompiler(
            width=program_width,
            heads=fusion_heads,
            initialization_seed=initialization_seed + 1,
        )
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(
                    program_width,
                    factor_hidden_width,
                    width,
                )
                for name, width in self.FACTOR_WIDTHS.items()
            }
        )
        self._register_template_state(tensor_specs, template_state)

    def _register_template_state(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        template_state: Mapping[str, torch.Tensor],
    ) -> None:
        self._template_buffers: dict[str, str] = {}
        self._decoding: dict[str, tuple[str, int | None]] = {}
        observed_heads: dict[str, int] = {}
        observed_layers: set[int] = set()
        for index, item in enumerate(tensor_specs):
            key, layer = self._decode_owner(item)
            observed_heads[key] = item.width
            if layer is not None:
                observed_layers.add(layer)
            value = template_state[item.name].detach().contiguous()
            if item.factor_index == 1 and torch.count_nonzero(value):
                raise WriterModelError("LoRA-B template must begin at physical zero")
            buffer_name = f"template_{index:03d}"
            self.register_buffer(buffer_name, value, persistent=True)
            self._template_buffers[item.name] = buffer_name
            self._decoding[item.name] = (key, layer)
        if (
            observed_heads != self.FACTOR_WIDTHS
            or observed_layers != set(range(self.EXPERT_LAYERS))
        ):
            raise WriterModelError("sealed PI05 LoRA modules changed topology")

    @staticmethod
    def _validated_offsets(
        offsets: torch.Tensor,
        total: int,
    ) -> tuple[int, ...]:
        if (
            offsets.ndim != 1
            or offsets.numel() < 2
            or offsets.dtype != torch.long
            or offsets.device.type != "cpu"
        ):
            raise WriterModelError("Writer video offsets are invalid")
        values = tuple(int(value) for value in offsets.tolist())
        if (
            values[0] != 0
            or values[-1] != total
            or any(right <= left for left, right in zip(values, values[1:]))
        ):
            raise WriterModelError("Writer video offsets are invalid")
        return values

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int | None]:
        factor = "a" if item.factor_index == 0 else "b"
        if item.module.endswith("action_in_proj"):
            return f"action_in_{factor}", None
        if item.module.endswith("action_out_proj"):
            return f"action_out_{factor}", None
        match = self._EXPERT_MODULE.fullmatch(item.module)
        if match is None:
            raise WriterModelError(
                f"unsupported PI05 task-LoRA module: {item.module}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.EXPERT_LAYERS:
            raise WriterModelError("PI05 task-LoRA layer is outside Action Expert")
        return f"{match.group(2)[0]}_{factor}", layer

    def template_state(self) -> dict[str, torch.Tensor]:
        """Return the persistent physical identity used by every generated LoRA."""

        return {
            name: getattr(self, buffer)
            for name, buffer in self._template_buffers.items()
        }

    def _pack_video_program(
        self,
        frame_evidence: torch.Tensor,
        grounded_evidence: torch.Tensor,
        interactions: torch.Tensor,
        frame_indices: torch.Tensor,
        offsets: tuple[int, ...],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            frame_evidence.ndim != 3
            or grounded_evidence.shape != frame_evidence.shape
            or interactions.shape
            != (frame_evidence.shape[0], self.program_width)
            or frame_indices.shape != (frame_evidence.shape[0],)
        ):
            raise WriterModelError("Writer visual-transition evidence changed shape")
        batch = len(offsets) - 1
        lengths = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        maximum = max(lengths)
        packed_evidence = frame_evidence.new_zeros(
            batch,
            maximum,
            frame_evidence.shape[1],
            self.program_width,
        )
        packed_interactions = interactions.new_zeros(
            batch,
            maximum,
            self.program_width,
        )
        packed_grounded = grounded_evidence.new_zeros(
            batch,
            maximum,
            grounded_evidence.shape[1],
            self.program_width,
        )
        positions = torch.zeros(
            batch,
            maximum,
            dtype=torch.long,
            device=interactions.device,
        )
        valid_frames = torch.zeros(
            batch,
            maximum,
            dtype=torch.bool,
            device=interactions.device,
        )
        starts = torch.tensor(
            offsets[:-1],
            dtype=torch.long,
            device=frame_indices.device,
        )
        internal_pairs = torch.ones(
            frame_indices.shape[0] - 1,
            dtype=torch.bool,
            device=frame_indices.device,
        )
        if len(offsets) > 2:
            internal_pairs[
                torch.tensor(
                    offsets[1:-1],
                    dtype=torch.long,
                    device=frame_indices.device,
                )
                - 1
            ] = False
        invalid_ordinals = (
            (frame_indices.index_select(0, starts) != 0).any()
            | (
                (frame_indices[1:] <= frame_indices[:-1])
                & internal_pairs
            ).any()
        )
        if bool(invalid_ordinals):
            raise WriterModelError(
                "sampled frame ordinals must start at zero and increase"
            )
        for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
            length = right - left
            active_positions = frame_indices[left:right]
            packed_evidence[row, :length] = frame_evidence[left:right]
            packed_grounded[row, :length] = grounded_evidence[left:right]
            packed_interactions[row, :length] = interactions[left:right]
            positions[row, :length] = active_positions
            valid_frames[row, :length] = True
        return (
            packed_evidence,
            packed_grounded,
            packed_interactions,
            positions,
            valid_frames,
        )

    def encode_task(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        evidence = self.encode_video_evidence(
            policy,
            frames,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        memories = self.build_memories(evidence, frame_indices)
        return (
            memories.core,
            memories.valid_core,
            memories.procedure,
            memories.positions,
            memories.valid_procedure,
            memories.frame_attention,
        )

    def encode_video_evidence(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> WriterVideoEvidence:
        """Encode frame-local evidence once, before an arm chooses frame order."""

        offsets = self._validated_offsets(video_offsets, frames.shape[0])
        conditions = len(offsets) - 1
        if (
            frames.ndim != 4
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
        ):
            raise WriterModelError("Writer frame-language condition batch changed")
        lengths = torch.tensor(
            [right - left for left, right in zip(offsets, offsets[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            lengths,
        )
        (
            text_queries,
            frame_evidence,
            grounded_evidence,
            interactions,
            valid_task_tokens,
        ) = self.semantic_encoder(
            policy,
            frames,
            condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return WriterVideoEvidence(
            text_queries=text_queries,
            frame_evidence=frame_evidence,
            grounded_evidence=grounded_evidence,
            interactions=interactions,
            valid_task_tokens=valid_task_tokens,
            offsets=offsets,
        )

    @staticmethod
    def _validate_frame_order(
        frame_order: torch.Tensor,
        offsets: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        total = offsets[-1]
        if (
            frame_order.ndim != 1
            or frame_order.shape[0] != total
            or frame_order.dtype != torch.long
            or frame_order.device != device
        ):
            raise WriterModelError("Writer frame order changed shape or device")
        invalid = torch.zeros((), dtype=torch.bool, device=device)
        for left, right in zip(offsets, offsets[1:]):
            observed = frame_order[left:right].sort().values
            expected = torch.arange(left, right, device=device)
            invalid |= (observed != expected).any()
        if bool(invalid):
            raise WriterModelError("Writer frame order crossed video conditions")
        return frame_order

    def build_memories(
        self,
        evidence: WriterVideoEvidence,
        frame_indices: torch.Tensor,
        *,
        frame_order: torch.Tensor | None = None,
    ) -> WriterMemories:
        """Build invariant Core and order-sensitive Procedure for one arm."""

        total = evidence.offsets[-1]
        if (
            frame_indices.ndim != 1
            or frame_indices.shape[0] != total
            or frame_indices.dtype != torch.long
            or frame_indices.device != evidence.frame_evidence.device
        ):
            raise WriterModelError("Writer sampled frame ordinals changed")
        if frame_order is None:
            order = torch.arange(total, device=frame_indices.device)
        else:
            order = self._validate_frame_order(
                frame_order,
                evidence.offsets,
                device=frame_indices.device,
            )
        (
            packed_evidence,
            packed_grounded,
            packed_interactions,
            positions,
            valid_frames,
        ) = self._pack_video_program(
            evidence.frame_evidence.index_select(0, order),
            evidence.grounded_evidence.index_select(0, order),
            evidence.interactions.index_select(0, order),
            frame_indices,
            evidence.offsets,
        )
        core_memory, frame_attention = self.semantic_core(
            evidence.text_queries,
            packed_evidence,
            valid_frames,
            evidence.valid_task_tokens,
        )
        procedure_input, _ = self.visual_transition(
            packed_interactions,
            packed_grounded,
            valid_frames,
            evidence.valid_task_tokens,
        )
        procedure_memory = self.procedure(
            procedure_input,
            positions,
            valid_frames,
        )
        return WriterMemories(
            core=core_memory,
            valid_core=evidence.valid_task_tokens,
            procedure=procedure_memory,
            positions=positions,
            valid_procedure=valid_frames,
            frame_attention=frame_attention,
        )

    def decode_memories(
        self,
        memories: WriterMemories,
    ) -> dict[str, torch.Tensor]:
        """Compile memories through the one canonical Writer decoder."""

        return self.decode_slots(self.compile_slots(memories))

    def compile_slots(
        self,
        memories: WriterMemories,
    ) -> torch.Tensor:
        """Compile one condition into the complete fused policy-slot program."""

        slots, _ = self.compiler.fused_slots(
            memories.core,
            memories.valid_core,
            memories.procedure,
            memories.positions,
            memories.valid_procedure,
        )
        expected = (
            memories.core.shape[0],
            SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            self.program_width,
        )
        if slots.shape != expected:
            raise WriterModelError("Writer fused policy-slot topology changed")
        return slots

    def decode_slots(
        self,
        slots: torch.Tensor,
        *,
        factor_hidden_residuals: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode one fused policy-slot program into the single public LoRA."""

        expected = (
            slots.shape[0] if slots.ndim == 3 else -1,
            SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            self.program_width,
        )
        if slots.ndim != 3 or slots.shape != expected:
            raise WriterModelError("Writer fused policy-slot topology changed")
        expert_stop = self.EXPERT_LAYERS * self.PUBLIC_LORA_RANK
        expert = slots[:, :expert_stop].reshape(
            slots.shape[0],
            self.EXPERT_LAYERS,
            self.PUBLIC_LORA_RANK,
            self.program_width,
        )
        action_in = slots[
            :, expert_stop : expert_stop + self.PUBLIC_LORA_RANK
        ]
        action_out = slots[:, -self.PUBLIC_LORA_RANK :]
        if factor_hidden_residuals is not None and set(factor_hidden_residuals) != set(
            self.factor_heads
        ):
            raise WriterModelError("factor hidden residual families changed")
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, layer = self._decoding[item.name]
            if key.startswith("action_in_"):
                source = action_in
                residual_source = (
                    factor_hidden_residuals[key][
                        :, expert_stop : expert_stop + self.PUBLIC_LORA_RANK
                    ]
                    if factor_hidden_residuals is not None
                    else None
                )
            elif key.startswith("action_out_"):
                source = action_out
                residual_source = (
                    factor_hidden_residuals[key][:, -self.PUBLIC_LORA_RANK :]
                    if factor_hidden_residuals is not None
                    else None
                )
            else:
                if layer is None:
                    raise WriterModelError("expert LoRA output lost its layer")
                source = expert[:, layer]
                residual_source = (
                    factor_hidden_residuals[key]
                    .reshape(
                        slots.shape[0],
                        -1,
                        self.PUBLIC_LORA_RANK,
                        self.program_width,
                    )[:, layer]
                    if factor_hidden_residuals is not None
                    else None
                )
            head = self.factor_heads[key]
            hidden = head.hidden(source)
            if residual_source is not None:
                if residual_source.shape != hidden.shape:
                    raise WriterModelError("factor hidden residual lost slot ownership")
                hidden = hidden + residual_source
            rows = head.decode_hidden(hidden)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if slots.shape[0] == 1 else value
        return result

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
        evidence = self.encode_video_evidence(
            policy,
            frames,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        memories = self.build_memories(evidence, frame_indices)
        return self.decode_memories(memories)
