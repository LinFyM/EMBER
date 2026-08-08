"""Closed-form causal expert-manifold generation in policy-effective space."""

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


def _psd_sqrt(value: torch.Tensor) -> torch.Tensor:
    symmetric = (value + value.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]) @ eigenvectors.T


def _effective_gram(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact Gram of the expert matrices B_k A_k without materializing them."""

    left = torch.einsum("kor,los->klrs", b, b)
    right = torch.einsum("kri,lsi->klrs", a, a)
    gram = (left * right).sum(dim=(-2, -1))
    return (gram + gram.T) * 0.5


def _energy_subspaces(
    a: torch.Tensor, b: torch.Tensor, basis_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal independent left/right energy subspaces of all expert updates."""

    left = torch.cat(
        tuple(
            expert_b @ _psd_sqrt(expert_a @ expert_a.T)
            for expert_a, expert_b in zip(a, b, strict=True)
        ),
        dim=1,
    )
    right = torch.cat(
        tuple(
            _psd_sqrt(expert_b.T @ expert_b) @ expert_a
            for expert_a, expert_b in zip(a, b, strict=True)
        ),
        dim=0,
    )
    left_u, _, _ = torch.linalg.svd(left, full_matrices=False)
    _, _, right_vh = torch.linalg.svd(right, full_matrices=False)
    return (
        left_u[:, : min(basis_rank, left_u.shape[1])],
        right_vh[: min(basis_rank, right_vh.shape[0])].T,
    )


class HardRoutedPolicyEffectiveWriter(torch.nn.Module):
    """Route one ordered video to one policy-effective expert LoRA."""

    def __init__(
        self,
        *,
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        expert_states: Sequence[Mapping[str, torch.Tensor]],
        task_centroids: torch.Tensor,
        phase_slots: int,
        feature_width: int,
        ridge: float,
        effective_basis_rank: int,
        identity_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if (
            len(expert_states) < 2
            or phase_slots < 2
            or feature_width <= 0
            or ridge <= 0
            or effective_basis_rank < contract.rank
            or identity_epsilon <= 0
            or task_centroids.shape != (len(expert_states), feature_width)
        ):
            raise ExpertManifoldError("invalid hard-routed policy-effective Writer")
        validate_lora_state(template_state, contract)
        if any(
            name.endswith(LORA_B_SUFFIX) and bool(torch.count_nonzero(value))
            for name, value in template_state.items()
        ):
            raise ExpertManifoldError(
                "hard-routed policy-effective Writer template LoRA-B must be zero"
            )
        for state in expert_states:
            validate_lora_state(state, contract)

        self.contract = contract
        self.phase_slots = int(phase_slots)
        self.feature_width = int(feature_width)
        self.basis_count = len(expert_states)
        self.ridge = float(ridge)
        self.effective_basis_rank = int(effective_basis_rank)
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

        centroids = task_centroids.detach().to(device="cpu", dtype=torch.float32)
        centroid_norm = torch.linalg.vector_norm(centroids, dim=1, keepdim=True)
        if bool((centroid_norm <= self.identity_epsilon).any()):
            raise ExpertManifoldError("policy-effective task centroid is zero")
        centroids = centroids / centroid_norm
        centroid_mean = centroids.mean(dim=0)
        centered = centroids - centroid_mean[None]
        kernel = centered @ centered.T
        kernel.diagonal().add_(self.ridge)
        projection = torch.linalg.solve(kernel, centered)
        self.register_buffer("centroid_mean", centroid_mean, persistent=True)
        self.register_buffer("coefficient_projection", projection, persistent=True)

        target_records: list[dict[str, Any]] = []
        for ordinal, target in enumerate(contract.targets):
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            a = torch.stack(
                [
                    state[a_name].detach().to(device="cpu", dtype=torch.float32)
                    for state in expert_states
                ]
            )
            b = torch.stack(
                [
                    state[b_name].detach().to(device="cpu", dtype=torch.float32)
                    for state in expert_states
                ]
            )
            gram = _effective_gram(a, b)
            norms = gram.diag().clamp_min(0).sqrt()
            if bool((norms <= self.identity_epsilon).any()):
                raise ExpertManifoldError("policy expert effective target is zero")
            left, right = _energy_subspaces(a, b, self.effective_basis_rank)
            cores = torch.stack(
                [
                    (left.T @ expert_b) @ (expert_a @ right)
                    for expert_a, expert_b in zip(a, b, strict=True)
                ]
            )
            expert_a_rms = a.square().mean(dim=(-2, -1)).sqrt()
            gauge_a_rms = (
                expert_a_rms.clamp_min(self.identity_epsilon).log().mean().exp()
            )
            values = (gram, norms, left, right, cores, gauge_a_rms)
            if not all(bool(torch.isfinite(value).all()) for value in values):
                raise ExpertManifoldError("policy-effective basis is nonfinite")
            buffers = {}
            for label, value in (
                ("gram", gram),
                ("norms", norms),
                ("left", left),
                ("right", right),
                ("cores", cores),
                ("gauge_a_rms", gauge_a_rms),
            ):
                name = f"target_{ordinal:03d}_{label}"
                self.register_buffer(name, value, persistent=True)
                buffers[label] = name
            target_records.append(
                {
                    "target_name": target.name,
                    "a_name": a_name,
                    "b_name": b_name,
                    "in_features": int(target.in_features),
                    "out_features": int(target.out_features),
                    **buffers,
                }
            )
        self._target_records = tuple(target_records)

        if not all(
            bool(torch.isfinite(value).all())
            for value in (self.centroid_mean, self.coefficient_projection)
        ):
            raise ExpertManifoldError("hard-routed policy-effective basis is nonfinite")

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

    def affine_coefficients(self, video_innovation: torch.Tensor) -> torch.Tensor:
        """Return the sealed affine routing scores for audit, not deployment."""

        representation = self.causal_representation(video_innovation).float()

        def solve() -> torch.Tensor:
            norm = torch.linalg.vector_norm(representation, dim=1, keepdim=True)
            query = representation / norm.clamp_min(self.identity_epsilon)
            weights = (query - self.centroid_mean[None]) @ self.coefficient_projection.T
            affine = (
                weights + (1.0 - weights.sum(dim=1, keepdim=True)) / self.basis_count
            )
            return torch.where(
                norm <= self.identity_epsilon, torch.zeros_like(affine), affine
            )

        if representation.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                result = solve()
        else:
            result = solve()
        if not bool(torch.isfinite(result).all()):
            raise ExpertManifoldError("hard-route affine scores are nonfinite")
        return result

    def coefficients(self, video_innovation: torch.Tensor) -> torch.Tensor:
        """Return deterministic signed-argmax one-hot deployment coefficients."""

        affine = self.affine_coefficients(video_innovation)
        active = affine.abs().sum(dim=1, keepdim=True) > self.identity_epsilon
        selected = affine.argmax(dim=1, keepdim=True)
        hard = torch.zeros_like(affine).scatter_(1, selected, 1.0)
        result = torch.where(active, hard, torch.zeros_like(hard))
        if not bool(torch.isfinite(result).all()):
            raise ExpertManifoldError("hard-routed coefficients are nonfinite")
        return result

    def states_from_coefficients(
        self, coefficients: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if coefficients.ndim != 2 or coefficients.shape[1] != self.basis_count:
            raise ExpertManifoldError("barycentric coefficient shape changed")
        coefficients = coefficients.to(
            device=self.centroid_mean.device, dtype=torch.float32
        )

        def compile_states() -> dict[str, torch.Tensor]:
            nonzero = (
                torch.linalg.vector_norm(coefficients, dim=1) > self.identity_epsilon
            )
            result: dict[str, torch.Tensor] = {}
            for record in self._target_records:
                gram = getattr(self, record["gram"])
                norms = getattr(self, record["norms"])
                left = getattr(self, record["left"])
                right = getattr(self, record["right"])
                cores = getattr(self, record["cores"])
                gauge_a_rms = getattr(self, record["gauge_a_rms"])
                direction_weights = coefficients / norms[None]
                direction_norm = (
                    torch.einsum(
                        "bk,kl,bl->b", direction_weights, gram, direction_weights
                    )
                    .clamp_min(0)
                    .sqrt()
                )
                if bool((nonzero & (direction_norm <= self.identity_epsilon)).any()):
                    raise ExpertManifoldError(
                        "nonzero coefficients cancelled the effective direction"
                    )
                log_scale = coefficients @ norms.log()
                log_scale = log_scale.clamp(norms.log().min(), norms.log().max())
                effective_weights = (
                    direction_weights
                    * (
                        log_scale.exp()
                        / direction_norm.clamp_min(self.identity_epsilon)
                    )[:, None]
                )
                core = torch.einsum("bk,kmn->bmn", effective_weights, cores)
                core_u, singular, core_vh = torch.linalg.svd(core, full_matrices=False)
                public_rank = self.contract.rank
                singular = singular[:, :public_rank]
                column_basis = left[None] @ core_u[:, :, :public_rank]
                row_basis = core_vh[:, :public_rank] @ right.T[None]

                template_a = getattr(self, self._template_buffers[record["a_name"]])
                anchor = template_a[None] @ row_basis.transpose(-2, -1)
                anchor_u, _, anchor_vh = torch.linalg.svd(anchor, full_matrices=False)
                orientation = anchor_u @ anchor_vh
                gauge = gauge_a_rms * math.sqrt(record["in_features"])
                a = gauge * (orientation @ row_basis)
                b = (
                    (column_basis * singular[:, None]) @ orientation.transpose(-2, -1)
                ) / gauge

                template_b = getattr(self, self._template_buffers[record["b_name"]])
                a = torch.where(nonzero[:, None, None], a, template_a[None])
                b = torch.where(nonzero[:, None, None], b, template_b[None])
                result[record["a_name"]] = a
                result[record["b_name"]] = b
            return result

        if coefficients.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                result = compile_states()
        else:
            result = compile_states()
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise ExpertManifoldError("policy-effective LoRA state is nonfinite")
        return result

    def forward(self, video_innovation: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.states_from_coefficients(self.coefficients(video_innovation))
