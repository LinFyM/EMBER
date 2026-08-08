"""Closed-form causal expert-manifold generation for complete LoRA states."""

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
    """Expose full LoRA tensors as chunk × rank × width without compression."""

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
        if len(ordered) != len(contract.targets) or len(set(ordered)) != len(
            ordered
        ):
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
        expected_leading = tuple(leading)
        for name, value in result.items():
            expected = (*expected_leading, *template[name].shape)
            if tuple(value.shape) != expected:
                raise ExpertManifoldError(
                    "topological LoRA reconstruction changed shape"
                )
        return result


class CausalBarycentricTopologicalWriter(torch.nn.Module):
    """Map one ordered video to one full LoRA through a frozen expert basis."""

    def __init__(
        self,
        *,
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        expert_states: Sequence[Mapping[str, torch.Tensor]],
        task_centroids: torch.Tensor,
        phase_slots: int,
        feature_width: int,
        chunk_width: int,
        ridge: float,
        identity_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if (
            len(expert_states) < 2
            or phase_slots < 2
            or feature_width <= 0
            or chunk_width <= 0
            or ridge <= 0
            or identity_epsilon <= 0
            or task_centroids.shape != (len(expert_states), feature_width)
        ):
            raise ExpertManifoldError("invalid causal barycentric Writer")
        validate_lora_state(template_state, contract)
        if any(
            name.endswith(LORA_B_SUFFIX) and bool(torch.count_nonzero(value))
            for name, value in template_state.items()
        ):
            raise ExpertManifoldError("barycentric Writer template LoRA-B must be zero")
        for state in expert_states:
            validate_lora_state(state, contract)

        self.layout = TopologicalLoRAChunkLayout(contract, chunk_width=chunk_width)
        self.phase_slots = int(phase_slots)
        self.feature_width = int(feature_width)
        self.basis_count = len(expert_states)
        self.ridge = float(ridge)
        self.identity_epsilon = float(identity_epsilon)
        self._template_buffers: dict[str, str] = {}
        for ordinal, (name, value) in enumerate(template_state.items()):
            buffer = f"template_{ordinal:03d}"
            self.register_buffer(
                buffer,
                value.detach().to(device="cpu", dtype=torch.float32).clone(),
                persistent=True,
            )
            self._template_buffers[name] = buffer

        template = self.template_state()
        tokens = torch.stack(
            [
                self.layout.tokenize(
                    {
                        name: value.detach().to(device="cpu", dtype=torch.float32)
                        for name, value in state.items()
                    },
                    template,
                )
                for state in expert_states
            ]
        )
        valid_mask = self.layout.valid_mask()
        mask = valid_mask[None, :, None, :].to(tokens.dtype)
        valid_count = (
            valid_mask.sum(dim=1).to(tokens.dtype)[None] * self.layout.rank
        ).clamp_min(1.0)
        scales = torch.sqrt(
            (tokens.square() * mask).sum(dim=(-2, -1)) / valid_count + 1e-24
        )
        directions = torch.where(
            scales[:, :, None, None] > self.identity_epsilon,
            tokens / scales[:, :, None, None].clamp_min(self.identity_epsilon),
            torch.zeros_like(tokens),
        )

        centroids = task_centroids.detach().to(dtype=torch.float32)
        centroid_norm = torch.linalg.vector_norm(centroids, dim=1, keepdim=True)
        if bool((centroid_norm <= self.identity_epsilon).any()):
            raise ExpertManifoldError("barycentric task centroid is zero")
        centroids = centroids / centroid_norm
        centroid_mean = centroids.mean(dim=0)
        centered = centroids - centroid_mean[None]
        kernel = centered @ centered.T
        kernel.diagonal().add_(self.ridge)
        projection = torch.linalg.solve(kernel, centered)
        if not all(
            bool(torch.isfinite(value).all())
            for value in (directions, scales, centroid_mean, projection)
        ):
            raise ExpertManifoldError("barycentric Writer basis is nonfinite")

        self.register_buffer("valid_value_mask", valid_mask, persistent=True)
        self.register_buffer("expert_directions", directions, persistent=True)
        self.register_buffer(
            "expert_log_scales",
            scales.clamp_min(self.identity_epsilon).log(),
            persistent=True,
        )
        self.register_buffer(
            "chunk_log_scale_min",
            self.expert_log_scales.min(dim=0).values,
            persistent=True,
        )
        self.register_buffer(
            "chunk_log_scale_max",
            self.expert_log_scales.max(dim=0).values,
            persistent=True,
        )
        self.register_buffer("centroid_mean", centroid_mean, persistent=True)
        self.register_buffer("coefficient_projection", projection, persistent=True)

    def template_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer)
            for name, buffer in self._template_buffers.items()
        }

    def causal_representation(self, video_innovation: torch.Tensor) -> torch.Tensor:
        if video_innovation.ndim != 3 or video_innovation.shape[1:] != (
            self.phase_slots,
            self.feature_width,
        ):
            raise ExpertManifoldError("video innovation changed phase/feature shape")
        return phase_centered_causal_memory(video_innovation).mean(dim=1)

    def coefficients(self, video_innovation: torch.Tensor) -> torch.Tensor:
        representation = self.causal_representation(video_innovation).float()

        def solve() -> torch.Tensor:
            norm = torch.linalg.vector_norm(representation, dim=1, keepdim=True)
            query = representation / norm.clamp_min(self.identity_epsilon)
            weights = (query - self.centroid_mean[None]) @ self.coefficient_projection.T
            affine = weights + (
                1.0 - weights.sum(dim=1, keepdim=True)
            ) / self.basis_count
            return torch.where(
                norm <= self.identity_epsilon, torch.zeros_like(affine), affine
            )

        if representation.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                result = solve()
        else:
            result = solve()
        if not bool(torch.isfinite(result).all()):
            raise ExpertManifoldError("barycentric coefficients are nonfinite")
        return result

    def values_from_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != self.basis_count:
            raise ExpertManifoldError("barycentric coefficient shape changed")
        coefficients = coefficients.to(
            device=self.expert_directions.device, dtype=torch.float32
        )

        def reconstruct() -> torch.Tensor:
            direction = torch.einsum(
                "bk,kcrw->bcrw", coefficients, self.expert_directions
            )
            mask = self.valid_value_mask[None, :, None, :].to(direction.dtype)
            valid_count = (
                self.valid_value_mask.sum(dim=1).to(direction.dtype)[None]
                * self.layout.rank
            ).clamp_min(1.0)
            rms = torch.sqrt(
                (direction.square() * mask).sum(dim=(-2, -1))
                / valid_count
                + 1e-24
            )
            direction = torch.where(
                rms[:, :, None, None] > self.identity_epsilon,
                direction / rms[:, :, None, None].clamp_min(self.identity_epsilon),
                torch.zeros_like(direction),
            )
            log_scale = coefficients @ self.expert_log_scales
            log_scale = torch.maximum(
                torch.minimum(log_scale, self.chunk_log_scale_max),
                self.chunk_log_scale_min,
            )
            values = direction * log_scale.exp()[:, :, None, None]
            return values.masked_fill(
                ~self.valid_value_mask[None, :, None, :], 0.0
            )

        if coefficients.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                values = reconstruct()
        else:
            values = reconstruct()
        if not bool(torch.isfinite(values).all()):
            raise ExpertManifoldError("barycentric LoRA values are nonfinite")
        return values

    def forward_values(self, video_innovation: torch.Tensor) -> torch.Tensor:
        return self.values_from_coefficients(self.coefficients(video_innovation))

    def forward(self, video_innovation: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.layout.detokenize(
            self.forward_values(video_innovation), self.template_state()
        )
