"""Bottleneck-free video-conditioned topological decoder for complete LoRA states."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ember.expert_manifold.contract import ExpertManifoldError
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)
from ember.writer.temporal import RMSNorm


@dataclass(frozen=True)
class LoRAChunk:
    tensor_name: str
    factor: str
    target_name: str
    target_ordinal: int
    factor_chunk: int
    valid_width: int


class TopologicalLoRAChunkLayout:
    """Expose full LoRA tensors as chunk × rank × width without compression."""

    def __init__(self, contract: LoRAContract, *, chunk_width: int) -> None:
        if contract.rank <= 0 or chunk_width <= 0:
            raise ExpertManifoldError("invalid topological LoRA chunk layout")
        self.contract = contract
        self.rank = int(contract.rank)
        self.chunk_width = int(chunk_width)
        indexed = {target.name: target for target in contract.targets}
        action = [name for name in indexed if name.endswith(("action_in_proj", "action_out_proj"))]
        policy = [name for name in indexed if name not in action]
        ordered = (*action, *policy)
        if len(ordered) != len(contract.targets) or len(set(ordered)) != len(ordered):
            raise ExpertManifoldError("LoRA target ordering is incomplete")
        chunks: list[LoRAChunk] = []
        for target_ordinal, target_name in enumerate(ordered):
            target = indexed[target_name]
            for factor, width, suffix in (
                ("a", target.in_features, LORA_A_SUFFIX),
                ("b", target.out_features, LORA_B_SUFFIX),
            ):
                tensor_name = target_name + suffix
                count = math.ceil(width / self.chunk_width)
                for factor_chunk in range(count):
                    start = factor_chunk * self.chunk_width
                    chunks.append(
                        LoRAChunk(
                            tensor_name=tensor_name,
                            factor=factor,
                            target_name=target_name,
                            target_ordinal=target_ordinal,
                            factor_chunk=factor_chunk,
                            valid_width=min(self.chunk_width, width - start),
                        )
                    )
        self.chunks = tuple(chunks)
        self.chunk_count = len(self.chunks)
        self.valid_values = self.rank * sum(chunk.valid_width for chunk in self.chunks)
        self.padded_values = self.chunk_count * self.rank * self.chunk_width

    def valid_mask(self) -> torch.Tensor:
        columns = torch.arange(self.chunk_width)
        return torch.stack([columns < chunk.valid_width for chunk in self.chunks])

    def tokenize(
        self,
        state: Mapping[str, torch.Tensor],
        template: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        validate_lora_state(state, self.contract)
        validate_lora_state(template, self.contract)
        rows = []
        for chunk in self.chunks:
            value = state[chunk.tensor_name]
            baseline = template[chunk.tensor_name]
            if chunk.factor == "a":
                rank_first = value - baseline
            else:
                rank_first = value
            if chunk.factor == "b":
                rank_first = rank_first.transpose(0, 1)
            start = chunk.factor_chunk * self.chunk_width
            selected = rank_first[:, start : start + chunk.valid_width]
            rows.append(F.pad(selected, (0, self.chunk_width - chunk.valid_width)))
        result = torch.stack(rows)
        if result.shape != (self.chunk_count, self.rank, self.chunk_width):
            raise ExpertManifoldError("topological LoRA tokenization changed shape")
        return result

    def detokenize(
        self,
        values: torch.Tensor,
        template: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if values.shape[-3:] != (self.chunk_count, self.rank, self.chunk_width):
            raise ExpertManifoldError("topological LoRA values changed shape")
        validate_lora_state(template, self.contract)
        leading = values.shape[:-3]
        grouped: dict[str, list[tuple[int, torch.Tensor]]] = {}
        factors: dict[str, str] = {}
        for ordinal, chunk in enumerate(self.chunks):
            selected = values[..., ordinal, :, : chunk.valid_width]
            grouped.setdefault(chunk.tensor_name, []).append((chunk.factor_chunk, selected))
            factors[chunk.tensor_name] = chunk.factor
        result = {}
        for name, pieces in grouped.items():
            rank_first = torch.cat([value for _, value in sorted(pieces)], dim=-1)
            value = rank_first if factors[name] == "a" else rank_first.transpose(-2, -1)
            baseline = template[name].to(device=value.device, dtype=value.dtype)
            if factors[name] == "a":
                value = value + baseline.reshape(*(1 for _ in leading), *baseline.shape)
            result[name] = value
        expected_leading = tuple(leading)
        for name, value in result.items():
            expected = (*expected_leading, *template[name].shape)
            if tuple(value.shape) != expected:
                raise ExpertManifoldError("topological LoRA reconstruction changed shape")
        return result


class ChunkRankAxialBlock(torch.nn.Module):
    """Exchange information globally over chunks, then over public rank coordinates."""

    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        if min(width, heads, expansion) <= 0 or width % heads:
            raise ExpertManifoldError("invalid chunk-rank axial block")
        self.chunk_norm = RMSNorm(width)
        self.chunk_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True, bias=False
        )
        self.rank_norm = RMSNorm(width)
        self.rank_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True, bias=False
        )
        self.mlp_norm = RMSNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width * expansion, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(width * expansion, width, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, chunks, rank, width = value.shape
        chunk_rows = self.chunk_norm(value).permute(0, 2, 1, 3).reshape(
            batch * rank, chunks, width
        )
        attended, _ = self.chunk_attention(
            chunk_rows, chunk_rows, chunk_rows, need_weights=False
        )
        value = value + attended.reshape(batch, rank, chunks, width).permute(0, 2, 1, 3)
        rank_rows = self.rank_norm(value).reshape(batch * chunks, rank, width)
        attended, _ = self.rank_attention(
            rank_rows, rank_rows, rank_rows, need_weights=False
        )
        value = value + attended.reshape(batch, chunks, rank, width)
        return value + self.mlp(self.mlp_norm(value))


class VideoConditionedTopologicalWriter(torch.nn.Module):
    """Map phase-preserving video innovation directly to one complete LoRA."""

    def __init__(
        self,
        *,
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        phase_slots: int,
        feature_width: int,
        memory_width: int,
        attention_heads: int,
        axial_blocks: int,
        chunk_width: int,
        mlp_expansion: int = 2,
    ) -> None:
        super().__init__()
        dimensions = (
            phase_slots,
            feature_width,
            memory_width,
            attention_heads,
            axial_blocks,
            chunk_width,
            mlp_expansion,
        )
        if any(value <= 0 for value in dimensions) or memory_width % attention_heads:
            raise ExpertManifoldError("invalid video-conditioned topological Writer")
        validate_lora_state(template_state, contract)
        if any(
            name.endswith(LORA_B_SUFFIX) and bool(torch.count_nonzero(value))
            for name, value in template_state.items()
        ):
            raise ExpertManifoldError("topological Writer template LoRA-B must be zero")
        self.layout = TopologicalLoRAChunkLayout(contract, chunk_width=chunk_width)
        self.phase_slots = int(phase_slots)
        self.feature_width = int(feature_width)
        self.memory_width = int(memory_width)
        self.input_projection = torch.nn.Linear(feature_width, memory_width, bias=False)
        self.memory_norm = RMSNorm(memory_width)
        self.phase_keys = torch.nn.Parameter(torch.empty(phase_slots, memory_width))
        self.chunk_queries = torch.nn.Parameter(
            torch.empty(self.layout.chunk_count, memory_width)
        )
        self.rank_queries = torch.nn.Parameter(torch.empty(contract.rank, memory_width))
        self.cross_attention = torch.nn.MultiheadAttention(
            memory_width, attention_heads, batch_first=True, bias=False
        )
        self.blocks = torch.nn.ModuleList(
            ChunkRankAxialBlock(memory_width, attention_heads, mlp_expansion)
            for _ in range(axial_blocks)
        )
        self.output_norm = RMSNorm(memory_width)
        self.output_projection = torch.nn.Linear(memory_width, chunk_width, bias=False)
        self.chunk_log_scale = torch.nn.Linear(memory_width, 1, bias=False)
        torch.nn.init.normal_(self.phase_keys, std=memory_width**-0.5)
        torch.nn.init.normal_(self.chunk_queries, std=memory_width**-0.5)
        torch.nn.init.normal_(self.rank_queries, std=memory_width**-0.5)
        torch.nn.init.zeros_(self.output_projection.weight)
        self._template_buffers: dict[str, str] = {}
        for ordinal, (name, value) in enumerate(template_state.items()):
            buffer = f"template_{ordinal:03d}"
            self.register_buffer(buffer, value.detach().clone(), persistent=True)
            self._template_buffers[name] = buffer
        self.register_buffer("valid_value_mask", self.layout.valid_mask(), persistent=True)

    def template_state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, buffer) for name, buffer in self._template_buffers.items()}

    def tokenize_targets(self, states: Sequence[Mapping[str, torch.Tensor]]) -> torch.Tensor:
        if not states:
            raise ExpertManifoldError("topological Writer target batch is empty")
        return torch.stack(
            [self.layout.tokenize(state, self.template_state()) for state in states]
        )

    def forward_values(self, video_innovation: torch.Tensor) -> torch.Tensor:
        if video_innovation.shape[1:] != (self.phase_slots, self.feature_width):
            raise ExpertManifoldError("video innovation changed phase/feature shape")
        memory = self.input_projection(video_innovation)
        normalized = self.memory_norm(memory)
        keys = normalized + self.phase_keys[None]
        batch = memory.shape[0]
        query = (
            self.chunk_queries[:, None, :] + self.rank_queries[None, :, :]
        ).reshape(1, self.layout.chunk_count * self.layout.rank, self.memory_width)
        query = query.expand(batch, -1, -1)
        value, _ = self.cross_attention(query, keys, memory, need_weights=False)
        value = value.reshape(
            batch, self.layout.chunk_count, self.layout.rank, self.memory_width
        )
        for block in self.blocks:
            value = block(value)
        normalized_output = self.output_norm(value)
        direction = self.output_projection(normalized_output)
        chunk_state = value.mean(dim=2)
        scale = self.chunk_log_scale(chunk_state).clamp(-8.0, 8.0).exp()
        output = direction * scale[:, :, None, :]
        return output.masked_fill(~self.valid_value_mask[None, :, None, :], 0.0)

    def forward(self, video_innovation: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.forward_values(video_innovation)
        return self.layout.detokenize(values, self.template_state())


def topological_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    cosine_weight: float,
    log_scale_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balance raw values, direction, and magnitude without artificial rank targets."""

    if (
        predicted.shape != target.shape
        or predicted.ndim != 4
        or valid_mask.shape != (predicted.shape[1], predicted.shape[3])
        or min(cosine_weight, log_scale_weight) < 0
    ):
        raise ExpertManifoldError("invalid topological reconstruction loss batch")
    mask = valid_mask[None, :, None, :].to(predicted.dtype)
    count = mask.sum(dim=-1).clamp_min(1.0)
    squared = ((predicted - target) * mask).square().sum(dim=-1) / count
    raw = squared.mean()
    left = (predicted * mask).reshape(*predicted.shape[:-1], -1)
    right = (target * mask).reshape(*target.shape[:-1], -1)
    target_norm = torch.linalg.vector_norm(right, dim=-1)
    cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-12)
    active = target_norm > 1e-12
    direction = (1.0 - cosine[active]).mean() if bool(active.any()) else raw.new_zeros(())
    predicted_rms = torch.sqrt((predicted.square() * mask).sum(dim=-1) / count + 1e-24)
    target_rms = torch.sqrt((target.square() * mask).sum(dim=-1) / count + 1e-24)
    log_scale = (predicted_rms.log() - target_rms.log()).square().mean()
    total = raw + cosine_weight * direction + log_scale_weight * log_scale
    return total, {
        "raw_reconstruction": raw.detach(),
        "direction": direction.detach(),
        "log_scale": log_scale.detach(),
    }
