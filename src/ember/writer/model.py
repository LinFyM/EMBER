"""Action-Memory generator for a complete task-specific PI05 LoRA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.action_memory import Pi05ActionMemoryEncoder
from ember.writer.temporal import ActionMemoryTemporalEncoder, RMSNorm


class WriterModelError(RuntimeError):
    """Raised when the Writer input or sealed LoRA contract is inconsistent."""


ACTION_MEMORY_WRITER_CONSTRUCTOR_KEYS = frozenset(
    {
        "expert_layers",
        "memory_slots",
        "expert_width",
        "action_code_width",
        "meta_lora_rank",
        "hidden_dim",
        "attention_heads",
        "temporal_blocks",
        "decoder_hidden_dim",
        "frame_microbatch",
        "conditional_linear_bias",
    }
)


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
    """Decode one rank-slot state into one LoRA factor row or column."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        *,
        linear_bias: bool,
    ) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            RMSNorm(input_width),
            torch.nn.Linear(input_width, hidden_width, bias=linear_bias),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=linear_bias),
        )
        # The physical task adapter begins exactly at its identity template.
        torch.nn.init.zeros_(self.network[-1].weight)
        if self.network[-1].bias is not None:
            torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class CompleteLoRAWriter(torch.nn.Module):
    """Generate a complete rank-16 task LoRA from language and one raw video.

    Frozen PaliGemma jointly grounds each sampled frame with the task language.
    Sixteen learned Action-Expert memory tokens then expose all 18 expert-layer
    states.  A variable-length temporal encoder and direct factor heads map
    those content-derived states to the sealed PI05 task-LoRA space.
    """

    _EXPERT_MODULE = re.compile(
        r".*gemma_expert\.model\.layers\.([0-9]+)\.self_attn\.(q_proj|v_proj)$"
    )

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        action_in_projection: torch.nn.Module,
        expert_layers: int,
        memory_slots: int,
        expert_width: int,
        action_code_width: int,
        meta_lora_rank: int,
        hidden_dim: int,
        attention_heads: int,
        temporal_blocks: int,
        decoder_hidden_dim: int,
        frame_microbatch: int,
        conditional_linear_bias: bool,
    ) -> None:
        super().__init__()
        if (
            not tensor_specs
            or set(template_state) != {item.name for item in tensor_specs}
            or min(
                expert_layers,
                memory_slots,
                expert_width,
                action_code_width,
                meta_lora_rank,
                hidden_dim,
                attention_heads,
                temporal_blocks,
                decoder_hidden_dim,
                frame_microbatch,
            )
            <= 0
        ):
            raise WriterModelError("invalid Action-Memory Writer dimensions")
        ranks = {item.rank for item in tensor_specs}
        if ranks != {memory_slots}:
            raise WriterModelError("memory slots must equal the sealed task-LoRA rank")
        self.tensor_specs = tensor_specs
        self.expert_layers = int(expert_layers)
        self.memory_slots = int(memory_slots)
        self.frame_microbatch = int(frame_microbatch)

        self.action_memory = Pi05ActionMemoryEncoder(
            action_in_projection=action_in_projection,
            memory_slots=memory_slots,
            expert_layers=expert_layers,
            expert_width=expert_width,
            action_code_width=action_code_width,
            meta_rank=meta_lora_rank,
        )
        self.task_encoder = ActionMemoryTemporalEncoder(
            input_width=expert_width,
            hidden_width=hidden_dim,
            expert_layers=expert_layers,
            memory_slots=memory_slots,
            attention_heads=attention_heads,
            temporal_blocks=temporal_blocks,
            conditional_linear_bias=conditional_linear_bias,
        )

        expected_heads = {
            "q_a": 1024,
            "q_b": 2048,
            "v_a": 1024,
            "v_b": 256,
            "action_in_a": 32,
            "action_in_b": 1024,
            "action_out_a": 1024,
            "action_out_b": 32,
        }
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: FactorHead(
                    hidden_dim,
                    decoder_hidden_dim,
                    width,
                    linear_bias=conditional_linear_bias,
                )
                for name, width in expected_heads.items()
            }
        )

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
        if observed_heads != expected_heads or observed_layers != set(range(expert_layers)):
            raise WriterModelError("sealed PI05 LoRA modules changed topology")

    @staticmethod
    def _validated_offsets(
        offsets: torch.Tensor, total: int
    ) -> tuple[int, ...]:
        if offsets.ndim != 1 or offsets.numel() < 2:
            raise WriterModelError("Writer video offsets are invalid")
        values = tuple(
            int(value)
            for value in offsets.detach().to(device="cpu", dtype=torch.long).tolist()
        )
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
            raise WriterModelError(f"unsupported PI05 task-LoRA module: {item.module}")
        layer = int(match.group(1))
        if not 0 <= layer < self.expert_layers:
            raise WriterModelError("PI05 task-LoRA layer is outside Action Expert")
        projection = match.group(2)[0]
        return f"{projection}_{factor}", layer

    def _pack_trajectories(
        self,
        states: torch.Tensor,
        frame_indices: torch.Tensor,
        offsets: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(offsets) - 1
        lengths = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        maximum = max(lengths)
        packed = states.new_zeros(
            (
                batch,
                maximum,
                self.expert_layers,
                self.memory_slots,
                states.shape[-1],
            )
        )
        indices = torch.zeros(
            batch, maximum, dtype=torch.long, device=states.device
        )
        mask = torch.zeros(
            batch, maximum, dtype=torch.bool, device=states.device
        )
        for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
            length = right - left
            packed[row, :length] = states[left:right]
            indices[row, :length] = frame_indices[left:right]
            mask[row, :length] = True
        return packed, indices, mask

    def encode_task(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> torch.Tensor:
        offsets = self._validated_offsets(video_offsets, frames.shape[0])
        conditions = len(offsets) - 1
        if (
            frames.ndim != 4
            or frame_indices.ndim != 1
            or frame_indices.shape[0] != frames.shape[0]
            or frame_indices.dtype != torch.long
            or language_tokens.ndim != 2
            or language_tokens.shape[0] != conditions
            or language_mask.shape != language_tokens.shape
        ):
            raise WriterModelError("Writer frame-language condition batch changed")
        lengths = torch.tensor(
            [right - left for left, right in zip(offsets, offsets[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        condition_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device), lengths
        )
        states = self.action_memory(
            policy,
            frames,
            condition_ids,
            language_tokens,
            language_mask,
            frame_microbatch=self.frame_microbatch,
        )
        packed, indices, mask = self._pack_trajectories(
            states, frame_indices, offsets
        )
        return self.task_encoder(packed, indices, mask)

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        *,
        policy: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_task(
            policy,
            frames,
            frame_indices,
            video_offsets,
            language_tokens,
            language_mask,
        )
        early = features[:, :3].mean(dim=1)
        late = features[:, -3:].mean(dim=1)
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            key, layer = self._decoding[item.name]
            if key.startswith("action_in_"):
                source = early
            elif key.startswith("action_out_"):
                source = late
            else:
                if layer is None:
                    raise WriterModelError("expert LoRA output lost its layer")
                source = features[:, layer]
            rows = self.factor_heads[key](source)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            value = generated.to(dtype=template.dtype) + template[None]
            result[item.name] = value[0] if features.shape[0] == 1 else value
        return result
