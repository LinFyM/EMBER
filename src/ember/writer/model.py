"""Canonical Layer-Matched Memory Program Compiler Writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ember.expert_manifold.legacy_v6_model import (
    FactorHead,
    LoraTensorSpec,
    build_lora_tensor_specs,
)
from ember.writer.backbone_memory import (
    LayerMatchedBackboneMemoryEncoder,
    LayerMatchedVideoEncoding,
)
from ember.writer.errors import WriterModelError
from ember.writer.parameter_grid import (
    AddressPreservingVideoSet,
    EXPERT_LAYERS,
    LayerMatchedMemoryProgramCompiler,
    LayerRankMemoryReader,
    MEMORY_WIDTH,
    PARAMETER_GROUPS,
    PROGRAM_WIDTH,
    PUBLIC_RANK,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    TaskGroundedVisualTransitionFusion,
)
from ember.writer.video_program import Pi05LanguageAxialEncoder


@dataclass(frozen=True)
class VideoProgram:
    """Per-video V6 Core and causal Procedure for one frame ordering."""

    core: torch.Tensor
    valid_core: torch.Tensor
    procedure: torch.Tensor
    positions: torch.Tensor
    valid_procedure: torch.Tensor


@dataclass(frozen=True)
class WriterProgramDiagnostics:
    """Stage outputs used to localize the first scientific failure."""

    per_video_core: torch.Tensor
    per_video_procedure: torch.Tensor
    reverse_procedure: torch.Tensor
    shuffled_procedure: torch.Tensor
    natural_parameter_memory: torch.Tensor
    reverse_parameter_memory: torch.Tensor
    shuffled_parameter_memory: torch.Tensor
    directed_parameter_memory: torch.Tensor
    shared_parameter_memory: torch.Tensor
    shared_core: torch.Tensor
    core_fused_grid: torch.Tensor
    compiled_grid: torch.Tensor
    auxiliary_loss: torch.Tensor


@dataclass(frozen=True)
class WriterProgramOutput:
    """One complete parameter-addressed Program before native factor heads."""

    program: torch.Tensor
    diagnostics: WriterProgramDiagnostics


@dataclass(frozen=True)
class EncodedContext:
    """One native content pass plus its ragged ownership."""

    encoding: LayerMatchedVideoEncoding
    frame_indices: torch.Tensor
    video_bounds: tuple[int, ...]
    condition_bounds: tuple[int, ...]
    video_condition_ids: torch.Tensor


class CompleteLoRAWriter(torch.nn.Module):
    """Compile task language and dynamic-K videos into one complete rank16 LoRA."""

    PUBLIC_LORA_RANK = PUBLIC_RANK
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
    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        semantic_encoder: Pi05LanguageAxialEncoder,
        semantic_core: LanguageSemanticCore,
        visual_transition: TaskGroundedVisualTransitionFusion,
        procedure: CausalProcedureEncoder,
        backbone_memory_encoder: LayerMatchedBackboneMemoryEncoder,
        memory_reader: LayerRankMemoryReader,
        video_set: AddressPreservingVideoSet,
        compiler: LayerMatchedMemoryProgramCompiler,
        factor_hidden_width: int,
        initialization_seed: int,
        matching_margin: float,
    ) -> None:
        super().__init__()
        if (
            not tensor_specs
            or factor_hidden_width <= 0
            or matching_margin <= 0
            or {item.rank for item in tensor_specs} != {PUBLIC_RANK}
        ):
            raise WriterModelError("invalid LMMPC Writer topology")
        self.tensor_specs = tensor_specs
        self.semantic_encoder = semantic_encoder
        self.semantic_core = semantic_core
        self.visual_transition = visual_transition
        self.procedure = procedure
        self.backbone_memory_encoder = backbone_memory_encoder
        self.memory_reader = memory_reader
        self.video_set = video_set
        self.compiler = compiler
        self.matching_margin = float(matching_margin)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 0x4D454D)
            memory = torch.empty(PUBLIC_RANK, MEMORY_WIDTH)
            torch.nn.init.normal_(memory, mean=0.0, std=MEMORY_WIDTH**-0.5)
            self.memory_tokens = torch.nn.Parameter(memory)
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(PROGRAM_WIDTH, factor_hidden_width, width)
                for name, width in self.FACTOR_WIDTHS.items()
            }
        )
        self.language_match = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.procedure_match = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.memory_match = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self._register_template_state(template_state)

    @classmethod
    def from_policy(
        cls,
        *,
        policy: torch.nn.Module,
        template_state: Mapping[str, torch.Tensor],
        writer_config: Mapping[str, Any],
    ) -> CompleteLoRAWriter:
        """Build a fresh LMMPC around the frozen source policy topology."""

        bridge = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
        if bridge is None:
            raise WriterModelError("PI05 policy lost its joint backbone")
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        seed = int(writer_config["initialization_seed"])
        semantic = Pi05LanguageAxialEncoder(
            paligemma_model=language_model,
            expert_model=expert_model,
            image_width=int(writer_config["image_width"]),
            expert_width=int(writer_config["expert_width"]),
            program_width=PROGRAM_WIDTH,
            text_meta_lora_rank=int(writer_config["text_meta_lora_rank"]),
            vl_meta_lora_rank=int(writer_config["vl_meta_lora_rank"]),
            action_meta_lora_rank=int(writer_config["action_meta_lora_rank"]),
            patch_grounding_heads=int(writer_config["patch_grounding_heads"]),
            max_frames_per_encoder_call=int(
                writer_config["max_frames_per_encoder_call"]
            ),
            action_horizon=int(writer_config["action_horizon"]),
            padded_action_dim=int(writer_config["padded_action_dim"]),
            initialization_seed=seed,
            activation_checkpointing=bool(
                writer_config["activation_checkpointing"]
            ),
        )
        semantic.vl_meta_lora.requires_grad_(False)
        backbone = LayerMatchedBackboneMemoryEncoder(
            image_width=int(writer_config["image_width"]),
            expert_width=int(writer_config["expert_width"]),
            activation_checkpointing=bool(
                writer_config["activation_checkpointing"]
            ),
        )
        return cls(
            build_lora_tensor_specs(template_state),
            template_state=template_state,
            semantic_encoder=semantic,
            semantic_core=LanguageSemanticCore(
                width=PROGRAM_WIDTH,
                heads=int(writer_config["semantic_core_heads"]),
                blocks=int(writer_config["semantic_core_blocks"]),
            ),
            visual_transition=TaskGroundedVisualTransitionFusion(
                width=PROGRAM_WIDTH,
                heads=int(writer_config["visual_transition_heads"]),
            ),
            procedure=CausalProcedureEncoder(
                width=PROGRAM_WIDTH,
                heads=int(writer_config["procedure_heads"]),
                blocks=int(writer_config["procedure_blocks"]),
            ),
            backbone_memory_encoder=backbone,
            memory_reader=LayerRankMemoryReader(
                heads=int(writer_config["memory_reader_heads"]),
                initialization_seed=seed,
            ),
            video_set=AddressPreservingVideoSet(),
            compiler=LayerMatchedMemoryProgramCompiler(
                heads=int(writer_config["m2p_heads"]),
                blocks=int(writer_config["m2p_blocks"]),
                initialization_seed=seed,
            ),
            factor_hidden_width=int(writer_config["factor_hidden_width"]),
            initialization_seed=seed,
            matching_margin=float(writer_config["matching_margin"]),
        )

    @staticmethod
    def _offsets(
        value: torch.Tensor,
        *,
        final: int,
        maximum_span: int | None,
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
            or (
                maximum_span is not None
                and any(
                    right - left > maximum_span
                    for left, right in zip(offsets, offsets[1:])
                )
            )
        ):
            raise WriterModelError(f"invalid {name}")
        return offsets

    def _register_template_state(
        self, template_state: Mapping[str, torch.Tensor]
    ) -> None:
        if set(template_state) != {item.name for item in self.tensor_specs}:
            raise WriterModelError("Writer LoRA template names changed")
        self._template_buffers: dict[str, str] = {}
        self._decoding: dict[str, tuple[str, int | None]] = {}
        observed_heads: dict[str, int] = {}
        observed_layers: set[int] = set()
        for index, item in enumerate(self.tensor_specs):
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
            or observed_layers != set(range(EXPERT_LAYERS))
        ):
            raise WriterModelError("sealed PI05 LoRA modules changed topology")

    def _decode_owner(self, item: LoraTensorSpec) -> tuple[str, int | None]:
        factor = "a" if item.factor_index == 0 else "b"
        if item.module.endswith("action_in_proj"):
            return f"action_in_{factor}", None
        if item.module.endswith("action_out_proj"):
            return f"action_out_{factor}", None
        match = self._EXPERT_MODULE.fullmatch(item.module)
        if match is None:
            raise WriterModelError(f"unsupported PI05 task-LoRA module: {item.module}")
        layer = int(match.group(1))
        if layer not in range(EXPERT_LAYERS):
            raise WriterModelError("PI05 task-LoRA layer is outside Action Expert")
        return f"{match.group(2)[0]}_{factor}", layer

    def template_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer)
            for name, buffer in self._template_buffers.items()
        }

    @staticmethod
    def _frame_order(
        video_bounds: tuple[int, ...],
        *,
        kind: str,
        device: torch.device,
    ) -> torch.Tensor:
        rows = []
        for left, right in zip(video_bounds[:-1], video_bounds[1:], strict=True):
            length = right - left
            if kind == "natural":
                local = torch.arange(length, device=device)
            elif kind == "reverse":
                local = torch.arange(length - 1, -1, -1, device=device)
            elif kind == "shuffle":
                local = torch.cat(
                    (
                        torch.arange(0, length, 2, device=device),
                        torch.arange(1, length, 2, device=device),
                    )
                )
            else:
                raise WriterModelError("unknown LMMPC frame order")
            rows.append(local + left)
        return torch.cat(rows)

    def _pack_video_program(
        self,
        frame_evidence: torch.Tensor,
        grounded_evidence: torch.Tensor,
        interactions: torch.Tensor,
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
        order: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        videos = len(video_bounds) - 1
        lengths = tuple(
            right - left
            for left, right in zip(video_bounds[:-1], video_bounds[1:])
        )
        maximum = max(lengths)
        tokens = frame_evidence.shape[1]
        evidence = frame_evidence.new_zeros(
            videos, maximum, tokens, PROGRAM_WIDTH
        )
        grounded = grounded_evidence.new_zeros(
            videos, maximum, tokens, PROGRAM_WIDTH
        )
        action = interactions.new_zeros(videos, maximum, PROGRAM_WIDTH)
        positions = torch.zeros(
            videos, maximum, dtype=torch.long, device=frame_indices.device
        )
        valid = torch.zeros(
            videos, maximum, dtype=torch.bool, device=frame_indices.device
        )
        ordered_evidence = frame_evidence.index_select(0, order)
        ordered_grounded = grounded_evidence.index_select(0, order)
        ordered_action = interactions.index_select(0, order)
        for video, (left, right) in enumerate(
            zip(video_bounds[:-1], video_bounds[1:], strict=True)
        ):
            length = right - left
            evidence[video, :length] = ordered_evidence[left:right]
            grounded[video, :length] = ordered_grounded[left:right]
            action[video, :length] = ordered_action[left:right]
            positions[video, :length] = frame_indices[left:right]
            valid[video, :length] = True
        return evidence, grounded, action, positions, valid

    def _build_program(
        self,
        *,
        text_queries: torch.Tensor,
        frame_evidence: torch.Tensor,
        grounded_evidence: torch.Tensor,
        interactions: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        video_condition_ids: torch.Tensor,
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
        order: torch.Tensor,
    ) -> VideoProgram:
        evidence, grounded, action, positions, valid = self._pack_video_program(
            frame_evidence,
            grounded_evidence,
            interactions,
            frame_indices,
            video_bounds,
            order,
        )
        video_queries = text_queries.index_select(0, video_condition_ids)
        video_valid_tokens = valid_task_tokens.index_select(
            0, video_condition_ids
        )
        core, _ = self.semantic_core(
            video_queries,
            evidence,
            valid,
            video_valid_tokens,
        )
        procedure_input, _ = self.visual_transition(
            action,
            grounded,
            valid,
            video_valid_tokens,
        )
        return VideoProgram(
            core=core,
            valid_core=video_valid_tokens,
            procedure=self.procedure(procedure_input, positions, valid),
            positions=positions,
            valid_procedure=valid,
        )

    @staticmethod
    def _last_valid(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        lengths = valid.sum(dim=1).to(torch.long)
        if not bool((lengths > 0).all()):
            raise WriterModelError("empty causal Procedure")
        return value[
            torch.arange(value.shape[0], device=value.device),
            lengths - 1,
        ]

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(value.dtype)
        return (value * valid[..., None]).sum(dim=1) / count

    @staticmethod
    def _condition_mean(
        value: torch.Tensor, condition_bounds: tuple[int, ...]
    ) -> torch.Tensor:
        return torch.stack(
            [
                value[left:right].mean(dim=0)
                for left, right in zip(
                    condition_bounds[:-1], condition_bounds[1:], strict=True
                )
            ]
        )

    def _matching_loss(
        self,
        *,
        language_summary: torch.Tensor,
        video_condition_ids: torch.Tensor,
        condition_bounds: tuple[int, ...],
        natural_procedure: torch.Tensor,
        reverse_procedure: torch.Tensor,
        shuffled_procedure: torch.Tensor,
        natural_memory: torch.Tensor,
        reverse_memory: torch.Tensor,
        shuffled_memory: torch.Tensor,
    ) -> torch.Tensor:
        language = self.language_match(language_summary).index_select(
            0, video_condition_ids
        )

        def representation(
            procedure: torch.Tensor, memory: torch.Tensor
        ) -> torch.Tensor:
            return self.procedure_match(procedure) + self.memory_match(
                memory.mean(dim=(1, 2))
            )

        natural = representation(natural_procedure, natural_memory)
        reverse = representation(reverse_procedure, reverse_memory)
        shuffled = representation(shuffled_procedure, shuffled_memory)
        positive = F.cosine_similarity(language, natural, dim=-1)
        reverse_score = F.cosine_similarity(language, reverse, dim=-1)
        shuffled_score = F.cosine_similarity(language, shuffled, dim=-1)
        ranking = F.relu(self.matching_margin - positive + reverse_score)
        ranking = ranking + F.relu(
            self.matching_margin - positive + shuffled_score
        )
        normalized = F.normalize(natural.to(torch.float32), dim=-1)
        agreement = natural.new_zeros(())
        multi_video = 0
        for left, right in zip(
            condition_bounds[:-1], condition_bounds[1:], strict=True
        ):
            if right - left <= 1:
                continue
            center = F.normalize(
                normalized[left:right].mean(dim=0, keepdim=True), dim=-1
            )
            agreement = agreement + (
                1.0 - (normalized[left:right] * center).sum(dim=-1)
            ).mean().to(agreement.dtype)
            multi_video += 1
        if multi_video:
            agreement = agreement / multi_video
        return ranking.mean() + agreement

    def _encode_context(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        policy: torch.nn.Module,
    ) -> EncodedContext:
        video_bounds = self._offsets(
            video_offsets,
            final=frames.shape[0],
            maximum_span=None,
            name="video offsets",
        )
        condition_bounds = self._offsets(
            condition_video_offsets,
            final=len(video_bounds) - 1,
            maximum_span=4,
            name="condition video offsets",
        )
        conditions = len(condition_bounds) - 1
        if (
            frame_indices.shape != (frames.shape[0],)
            or frame_indices.dtype != torch.long
            or frame_indices.device != frames.device
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
        ):
            raise WriterModelError("invalid LMMPC condition batch")
        cardinalities = torch.tensor(
            [right - left for left, right in zip(condition_bounds, condition_bounds[1:])],
            dtype=torch.long,
            device=language_tokens.device,
        )
        video_condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=language_tokens.device), cardinalities
        )
        video_lengths = torch.tensor(
            [right - left for left, right in zip(video_bounds, video_bounds[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        frame_video_ids = torch.repeat_interleave(
            torch.arange(len(video_bounds) - 1, device=frames.device), video_lengths
        )
        encoding = self.backbone_memory_encoder(
            self.semantic_encoder,
            policy,
            frames,
            video_condition_ids.index_select(0, frame_video_ids),
            language_tokens,
            language_mask,
            task_span_mask,
            self.memory_tokens,
        )
        return EncodedContext(
            encoding=encoding,
            frame_indices=frame_indices,
            video_bounds=video_bounds,
            condition_bounds=condition_bounds,
            video_condition_ids=video_condition_ids,
        )

    def _compile_context(self, context: EncodedContext) -> WriterProgramOutput:
        encoding = context.encoding
        orders = {
            kind: self._frame_order(
                context.video_bounds,
                kind=kind,
                device=context.frame_indices.device,
            )
            for kind in ("natural", "reverse", "shuffle")
        }
        programs = {
            kind: self._build_program(
                text_queries=encoding.text_queries,
                frame_evidence=encoding.frame_evidence,
                grounded_evidence=encoding.grounded_evidence,
                interactions=encoding.interactions,
                valid_task_tokens=encoding.valid_task_tokens,
                video_condition_ids=context.video_condition_ids,
                frame_indices=context.frame_indices,
                video_bounds=context.video_bounds,
                order=order,
            )
            for kind, order in orders.items()
        }
        parameter_memory = {
            kind: self.memory_reader(
                encoding.layer_memory.index_select(0, orders[kind]),
                program.procedure,
                program.valid_procedure,
                context.video_bounds,
            )
            for kind, program in programs.items()
        }
        directed = 0.5 * (
            parameter_memory["natural"] - parameter_memory["reverse"]
        )
        core_summary = self._masked_mean(
            programs["natural"].core, programs["natural"].valid_core
        )
        procedure_summary = self._last_valid(
            programs["natural"].procedure, programs["natural"].valid_procedure
        )
        shared_memory = self.video_set(
            directed,
            core_summary,
            procedure_summary,
            context.condition_bounds,
        )
        shared_core = self._condition_mean(
            programs["natural"].core, context.condition_bounds
        )
        language_summary = self._masked_mean(
            encoding.text_queries, encoding.valid_task_tokens
        )
        core_fused, compiled = self.compiler(
            shared_memory,
            shared_core,
            encoding.valid_task_tokens,
            language_summary,
        )
        summaries = {
            kind: self._last_valid(program.procedure, program.valid_procedure)
            for kind, program in programs.items()
        }
        auxiliary = self._matching_loss(
            language_summary=language_summary,
            video_condition_ids=context.video_condition_ids,
            condition_bounds=context.condition_bounds,
            natural_procedure=summaries["natural"],
            reverse_procedure=summaries["reverse"],
            shuffled_procedure=summaries["shuffle"],
            natural_memory=parameter_memory["natural"],
            reverse_memory=parameter_memory["reverse"],
            shuffled_memory=parameter_memory["shuffle"],
        )
        diagnostics = WriterProgramDiagnostics(
            per_video_core=programs["natural"].core,
            per_video_procedure=programs["natural"].procedure,
            reverse_procedure=programs["reverse"].procedure,
            shuffled_procedure=programs["shuffle"].procedure,
            natural_parameter_memory=parameter_memory["natural"],
            reverse_parameter_memory=parameter_memory["reverse"],
            shuffled_parameter_memory=parameter_memory["shuffle"],
            directed_parameter_memory=directed,
            shared_parameter_memory=shared_memory,
            shared_core=shared_core,
            core_fused_grid=core_fused,
            compiled_grid=compiled,
            auxiliary_loss=auxiliary,
        )
        return WriterProgramOutput(program=compiled, diagnostics=diagnostics)

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
        """Run one content forward, preserve addresses through T/K, and compile."""

        del singleton_video_index
        return self._compile_context(
            self._encode_context(
                frames,
                frame_indices,
                video_offsets,
                condition_video_offsets,
                language_tokens,
                language_mask,
                task_span_mask,
                policy,
            )
        )

    def decode_program(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            program.ndim != 4
            or program.shape[1:]
            != (PARAMETER_GROUPS, PUBLIC_RANK, PROGRAM_WIDTH)
        ):
            raise WriterModelError("LMMPC parameter grid changed shape")
        expert = program[:, 1:-1]
        action_in = program[:, 0]
        action_out = program[:, -1]
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, layer = self._decoding[item.name]
            if key.startswith("action_in_"):
                source = action_in
            elif key.startswith("action_out_"):
                source = action_out
            else:
                if layer is None:
                    raise WriterModelError("expert LoRA output lost its layer")
                source = expert[:, layer]
            rows = self.factor_heads[key](source)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if program.shape[0] == 1 else value
        return result

    def decode_output(self, encoded: WriterProgramOutput) -> dict[str, torch.Tensor]:
        return self.decode_program(encoded.program)

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
