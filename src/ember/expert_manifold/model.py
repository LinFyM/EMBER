"""Analysis-only topology helpers retained for Expert-Manifold evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from ember.expert_manifold.contract import ExpertManifoldError
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)


def phase_centered_causal_memory(memory: torch.Tensor) -> torch.Tensor:
    """Bind dynamic values to ordered prefixes while removing phase-constant DC."""

    if memory.ndim < 2 or memory.shape[-2] < 2:
        raise ExpertManifoldError("causal video memory requires multiple phases")
    centered = memory - memory.mean(dim=-2, keepdim=True)
    phase_count = memory.shape[-2]
    scale = torch.arange(
        1, phase_count + 1, dtype=memory.dtype, device=memory.device
    ).sqrt()
    shape = (1,) * (memory.ndim - 2) + (phase_count, 1)
    return centered.cumsum(dim=-2) / scale.reshape(shape)


@dataclass(frozen=True)
class LoRAChunk:
    tensor_name: str
    factor: str
    target_name: str
    target_ordinal: int
    factor_chunk: int
    valid_width: int


class TopologicalLoRAChunkLayout:
    """Analysis-only exact chunk view of the sealed public LoRA topology."""

    def __init__(self, contract: LoRAContract, *, chunk_width: int) -> None:
        if contract.rank <= 0 or chunk_width <= 0:
            raise ExpertManifoldError("invalid topological LoRA chunk layout")
        self.contract = contract
        self.rank = int(contract.rank)
        self.chunk_width = int(chunk_width)
        indexed = {target.name: target for target in contract.targets}
        action = [
            name
            for name in indexed
            if name.endswith(("action_in_proj", "action_out_proj"))
        ]
        policy = [name for name in indexed if name not in action]
        ordered = (*action, *policy)
        if len(ordered) != len(contract.targets) or len(set(ordered)) != len(ordered):
            raise ExpertManifoldError("LoRA target ordering is incomplete")
        chunks = []
        for target_ordinal, target_name in enumerate(ordered):
            target = indexed[target_name]
            for factor, width, suffix in (
                ("a", target.in_features, LORA_A_SUFFIX),
                ("b", target.out_features, LORA_B_SUFFIX),
            ):
                tensor_name = target_name + suffix
                for factor_chunk in range(math.ceil(width / self.chunk_width)):
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
            rank_first = value - baseline if chunk.factor == "a" else value
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
            grouped.setdefault(chunk.tensor_name, []).append(
                (chunk.factor_chunk, selected)
            )
            factors[chunk.tensor_name] = chunk.factor
        result = {}
        for name, pieces in grouped.items():
            rank_first = torch.cat([value for _, value in sorted(pieces)], dim=-1)
            value = rank_first if factors[name] == "a" else rank_first.transpose(-2, -1)
            baseline = template[name].to(device=value.device, dtype=value.dtype)
            if factors[name] == "a":
                value = value + baseline.reshape(*(1 for _ in leading), *baseline.shape)
            result[name] = value
        for name, value in result.items():
            if tuple(value.shape) != (*leading, *template[name].shape):
                raise ExpertManifoldError(
                    "topological LoRA reconstruction changed shape"
                )
        return result
