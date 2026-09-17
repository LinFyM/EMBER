"""Unified native-video memory to one complete, identity-initialized task LoRA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.errors import WriterModelError
from ember.writer.temporal import ContinuousParameterDecoder


WRITER_CONSTRUCTOR_KEYS = frozenset({
    "image_width", "expert_width", "program_width", "text_meta_lora_rank",
    "vl_meta_lora_rank", "action_meta_lora_rank", "patch_grounding_heads",
    "max_frames_per_encoder_call", "action_horizon", "padded_action_dim",
    "factor_hidden_width", "initialization_seed", "activation_checkpointing",
    "camera_view", "native_split_layer", "joint_heads", "joint_blocks",
    "decoder_heads", "decoder_blocks",
})


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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise WriterModelError("factor head lost its rank-slot dimension")
        return self.network(value)


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
        factor_hidden_width: int,
        initialization_seed: int,
        activation_checkpointing: bool,
        camera_view: str = "dual",
        native_split_layer: int = 9,
        joint_heads: int = 8,
        joint_blocks: int = 2,
        decoder_heads: int = 8,
        decoder_blocks: int = 2,
    ) -> None:
        super().__init__()
        if (len(tensor_specs) != 76
                or set(template_state) != {item.name for item in tensor_specs}
                or len(paligemma_model.layers) != self.EXPERT_LAYERS
                or len(expert_model.layers) != self.EXPERT_LAYERS
                or {item.rank for item in tensor_specs} != {self.PUBLIC_LORA_RANK}
                or (image_width, expert_width, program_width) != (2048, 1024, 256)
                or (text_meta_lora_rank, vl_meta_lora_rank, action_meta_lora_rank) != (4, 4, 4)
                or (patch_grounding_heads, joint_heads, decoder_heads) != (8, 8, 8)
                or action_horizon != 50 or padded_action_dim != 32
                or native_split_layer != 9 or min(joint_blocks, decoder_blocks) <= 0
                or factor_hidden_width != 216 or camera_view != "dual"):
            raise WriterModelError("invalid unified native-video Writer topology")
        self.tensor_specs = tensor_specs
        self.program_width = int(program_width)
        self.camera_view = camera_view
        from ember.writer.video_program import Pi05UnifiedVideoEncoder

        self.semantic_encoder = Pi05UnifiedVideoEncoder(
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
            camera_view=camera_view,
            native_split_layer=native_split_layer,
            joint_heads=joint_heads,
            joint_blocks=joint_blocks,
        )
        self.compiler = ContinuousParameterDecoder(
            width=program_width, heads=decoder_heads, blocks=decoder_blocks,
            initialization_seed=initialization_seed + 1,
        )
        self.factor_heads = torch.nn.ModuleDict({
            name: FactorHead(program_width, factor_hidden_width, width)
            for name, width in self.FACTOR_WIDTHS.items()
        })
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

    def encode_task(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        frame_parallel_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        parallel = {} if frame_parallel_group is None else {"frame_parallel_group": frame_parallel_group}
        return self.semantic_encoder(
            policy, frames, frame_indices, video_offsets,
            language_tokens, language_mask, task_span_mask, **parallel,
        )

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
        frame_parallel_group=None,
    ) -> dict[str, torch.Tensor]:
        memory, valid_frames, valid_roles, addresses, semantic_tokens = self.encode_task(
            policy, frames, frame_indices, video_offsets,
            language_tokens, language_mask, task_span_mask, frame_parallel_group=frame_parallel_group,
        )
        expert, action_in, action_out = self.compiler(
            memory, valid_frames, valid_roles, addresses, semantic_tokens,
        )
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
            result[item.name] = value[0] if memory.shape[0] == 1 else value
        return result
