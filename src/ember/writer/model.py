"""Layer-aware generator for a complete task-specific LoRA state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.writer.temporal import VariableEpisodeTaskEncoder


class WriterModelError(RuntimeError):
    """Raised when the Writer input or LoRA template is inconsistent."""


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


class CompleteLoRAWriter(torch.nn.Module):
    """Generate every A/B tensor from language and full variable-length videos.

    The model places no architectural upper bound on episode count or episode
    length. episode_offsets partitions one concatenated frame-feature tensor
    into a non-empty set of non-empty teaching episodes.
    """

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        vision_feature_dim: int,
        language_feature_dim: int,
        hidden_dim: int,
        attention_heads: int,
        temporal_chunk_size: int,
        chunk_memory_tokens: int,
        episode_memory_tokens: int,
        task_memory_tokens: int,
        decoder_hidden_dim: int,
    ) -> None:
        super().__init__()
        if not tensor_specs or set(template_state) != {item.name for item in tensor_specs}:
            raise WriterModelError("Writer template and tensor specification differ")
        ranks = {item.rank for item in tensor_specs}
        if len(ranks) != 1:
            raise WriterModelError("one Writer cannot mix LoRA ranks")

        self.tensor_specs = tensor_specs
        self.task_encoder = VariableEpisodeTaskEncoder(
            vision_feature_dim=vision_feature_dim,
            language_feature_dim=language_feature_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            temporal_chunk_size=temporal_chunk_size,
            chunk_memory_tokens=chunk_memory_tokens,
            episode_memory_tokens=episode_memory_tokens,
            task_memory_tokens=task_memory_tokens,
        )
        module_count = max(item.module_index for item in tensor_specs) + 1
        rank = next(iter(ranks))
        self.module_embedding = torch.nn.Embedding(module_count, hidden_dim)
        self.factor_embedding = torch.nn.Embedding(2, hidden_dim)
        self.rank_embedding = torch.nn.Embedding(rank, hidden_dim)
        self.parameter_attention = torch.nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True, dropout=0.0
        )
        self.parameter_norm = torch.nn.LayerNorm(hidden_dim)
        self.parameter_ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.heads = torch.nn.ModuleDict()
        for width in sorted({item.width for item in tensor_specs}):
            head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, decoder_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(decoder_hidden_dim, width),
            )
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            self.heads[str(width)] = head

        self._template_buffers: dict[str, str] = {}
        for index, item in enumerate(tensor_specs):
            value = template_state[item.name].detach().to(torch.float32).contiguous()
            if item.factor_index == 1 and torch.count_nonzero(value):
                raise WriterModelError("LoRA-B template must begin at physical zero")
            buffer_name = f"template_{index:03d}"
            self.register_buffer(buffer_name, value, persistent=True)
            self._template_buffers[item.name] = buffer_name

        module_ids: list[int] = []
        factor_ids: list[int] = []
        rank_ids: list[int] = []
        query_slices: list[tuple[int, int]] = []
        cursor = 0
        for item in tensor_specs:
            module_ids.extend([item.module_index] * item.rank)
            factor_ids.extend([item.factor_index] * item.rank)
            rank_ids.extend(range(item.rank))
            query_slices.append((cursor, cursor + item.rank))
            cursor += item.rank
        self.register_buffer(
            "parameter_module_ids", torch.tensor(module_ids), persistent=False
        )
        self.register_buffer(
            "parameter_factor_ids", torch.tensor(factor_ids), persistent=False
        )
        self.register_buffer(
            "parameter_rank_ids", torch.tensor(rank_ids), persistent=False
        )
        self._query_slices = tuple(query_slices)

    def encode_task(
        self,
        language_tokens: torch.Tensor,
        video_features: torch.Tensor,
        episode_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.task_encoder(language_tokens, video_features, episode_offsets)

    def forward(
        self,
        language_tokens: torch.Tensor,
        video_features: torch.Tensor,
        episode_offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        task_memory = self.encode_task(
            language_tokens, video_features, episode_offsets
        )
        queries = (
            self.module_embedding(self.parameter_module_ids)
            + self.factor_embedding(self.parameter_factor_ids)
            + self.rank_embedding(self.parameter_rank_ids)
        )
        normalized_queries = self.parameter_norm(queries)
        normalized_memory = self.parameter_norm(task_memory)
        attended, _ = self.parameter_attention(
            normalized_queries[None],
            normalized_memory[None],
            normalized_memory[None],
            need_weights=False,
        )
        decoded = queries + attended[0]
        decoded = decoded + self.parameter_ffn(decoded)

        result: dict[str, torch.Tensor] = {}
        for item, (start, stop) in zip(
            self.tensor_specs, self._query_slices, strict=True
        ):
            rows = self.heads[str(item.width)](decoded[start:stop])
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            result[item.name] = generated + template
        return result
