"""Deterministic robust-span initialization for the G1 free-code oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    NativeOutputBankState,
    NativeTargetChunk,
    NativeVideoReadout,
    TaskLocalNativeFactorOracle,
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    native_output_group_count,
    rms_normalize,
)
from ember.ecp.native_materialization import small_core_balanced_svd
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class CandidateMeasure:
    """One finite candidate measure and its stable centered subspace."""

    values: torch.Tensor
    mean: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    ranks: Mapping[str, int]

    @classmethod
    def build(
        cls, values: torch.Tensor, *, relative_singular_threshold: float
    ) -> "CandidateMeasure":
        if values.ndim < 2 or not 0 < relative_singular_threshold < 1:
            raise ValueError("G1 candidate measure or span threshold changed")
        flat = values.reshape(-1, values.shape[-1]).float()
        if flat.shape[0] <= 1:
            raise ValueError("G1 candidate measure is empty")
        mean = flat.mean(0)
        centered = flat - mean
        scatter = centered.transpose(0, 1) @ centered
        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        singular = eigenvalues.clamp_min(0).sqrt()
        maximum = singular[-1].clamp_min(torch.finfo(singular.dtype).tiny)
        keep = singular > maximum * relative_singular_threshold
        if not torch.any(keep):
            raise ValueError("G1 candidate measure has no stable direction")
        return cls(
            values=flat,
            mean=mean,
            basis=eigenvectors[:, keep],
            eigenvalues=eigenvalues[keep].clamp_min(1e-30),
            ranks={
                f"relative_{threshold:g}": int(
                    (singular > maximum * threshold).sum().item()
                )
                for threshold in (1e-3, 1e-4, 1e-5)
            },
        )

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 2 or vectors.shape[-1] != self.values.shape[-1]:
            raise ValueError("G1 projection vector width changed")
        return (vectors.float() @ self.basis) @ self.basis.transpose(0, 1)

    def signed_probabilities(
        self,
        vectors: torch.Tensor,
        *,
        probability_floor_mass: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map represented directions to positive/negative simplex measures."""

        if not 0 <= probability_floor_mass < 1:
            raise ValueError("G1 signed probability floor changed")
        projected = self.project(vectors)
        coordinates = self.basis.transpose(0, 1) @ projected.transpose(0, 1)
        dual = self.basis @ (coordinates / self.eigenvalues[:, None])
        centered = self.values - self.mean
        weights = (centered @ dual).transpose(0, 1)
        weights = weights - weights.mean(-1, keepdim=True)
        positive = weights.clamp_min(0)
        negative = (-weights).clamp_min(0)
        positive_mass = positive.sum(-1, keepdim=True)
        negative_mass = negative.sum(-1, keepdim=True)
        active = (positive_mass > 0) & (negative_mass > 0)
        uniform = torch.full_like(positive, 1.0 / positive.shape[-1])
        positive = torch.where(
            active,
            positive / positive_mass.clamp_min(1e-30),
            uniform,
        )
        negative = torch.where(
            active,
            negative / negative_mass.clamp_min(1e-30),
            uniform,
        )
        positive = (
            (1.0 - probability_floor_mass) * positive
            + probability_floor_mass * uniform
        )
        negative = (
            (1.0 - probability_floor_mass) * negative
            + probability_floor_mass * uniform
        )
        probabilities = torch.stack((positive, negative), dim=1)
        pooled = torch.einsum("rbn,nd->rbd", probabilities, self.values)
        signed = pooled[:, 0] - pooled[:, 1]
        return probabilities, signed


def cache_native_video_readout(readout: NativeVideoReadout) -> NativeVideoReadout:
    """Capture the frozen Pass-B chunks once without changing chunk semantics."""

    chunks = tuple(readout.chunks())
    next_frame = 0
    for chunk in chunks:
        if chunk.start_frame != next_frame or chunk.frame_count <= 0:
            raise ValueError("cached native chunks changed frame order")
        next_frame += chunk.frame_count
    if next_frame != readout.frame_count:
        raise ValueError("cached native stream ended early")

    def retained_chunks() -> Sequence[NativeTargetChunk]:
        return chunks

    return NativeVideoReadout(
        frame_count=readout.frame_count,
        process=readout.process,
        state_posterior=readout.state_posterior,
        final_outputs=readout.final_outputs,
        chunks=retained_chunks,
    )


def cached_native_bytes(readout: NativeVideoReadout) -> int:
    chunks = tuple(readout.chunks())
    tensors = [value for chunk in chunks for value in (*chunk.inputs, *chunk.outputs)]
    return sum(value.numel() * value.element_size() for value in tensors)


def initialize_oracle_as_carrier(
    *, oracle: TaskLocalNativeFactorOracle, video: NativeVideoReadout
) -> dict[str, Any]:
    """Represent the verified carrier member as an exact zero rank-four residual."""

    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.zero_()
    return {
        "kind": "zero_residual_carrier",
        "candidate_cache_bytes": cached_native_bytes(video),
    }


def _output_bank(
    *, video: NativeVideoReadout, chunks: Sequence[NativeTargetChunk], target: int
) -> torch.Tensor:
    boundary = NativeOutputBankState(final=video.final_outputs[target].detach())
    values = []
    for chunk in chunks:
        values.append(boundary.build(chunk.outputs[target], start_frame=chunk.start_frame))
    if boundary.next_frame != video.frame_count:
        raise ValueError("G1 initialization output bank ended early")
    return torch.cat(values, dim=0)


def _scaled_group_probabilities(
    *,
    measures: Sequence[CandidateMeasure],
    desired: torch.Tensor,
    probability_floor_mass: float,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Preserve desired q-head magnitudes while each head owns a simplex pair."""

    width = measures[0].values.shape[-1]
    if desired.shape[-1] != len(measures) * width:
        raise ValueError("G1 grouped output direction changed")
    base_probabilities = []
    base_signed = []
    for group, measure in enumerate(measures):
        block = desired[:, group * width : (group + 1) * width]
        probabilities, signed = measure.signed_probabilities(
            block, probability_floor_mass=0.0
        )
        base_probabilities.append(probabilities)
        base_signed.append(signed)
    desired_blocks = desired.reshape(desired.shape[0], len(measures), width)
    desired_norms = desired_blocks.float().norm(dim=-1)
    signed_norms = torch.stack(base_signed, dim=1).float().norm(dim=-1)
    feasible = torch.where(
        desired_norms > 1e-12,
        signed_norms / desired_norms.clamp_min(1e-30),
        torch.full_like(desired_norms, torch.inf),
    )
    common = feasible.amin(1)
    common = torch.where(
        torch.isfinite(common), 0.95 * common, torch.zeros_like(common)
    )
    alphas = torch.where(
        desired_norms > 1e-12,
        common[:, None] * desired_norms / signed_norms.clamp_min(1e-30),
        torch.zeros_like(desired_norms),
    ).clamp(0, 1)
    result = []
    realized = []
    for group, (measure, probabilities, signed) in enumerate(
        zip(measures, base_probabilities, base_signed, strict=True)
    ):
        uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        alpha = alphas[:, group, None, None]
        probabilities = alpha * probabilities + (1.0 - alpha) * uniform
        probabilities = (
            (1.0 - probability_floor_mass) * probabilities
            + probability_floor_mass * uniform
        )
        result.append(probabilities)
        realized.append(signed * alphas[:, group, None] * (1.0 - probability_floor_mass))
    return tuple(result), torch.cat(realized, dim=-1)


def initialize_oracle_from_reference(
    *,
    oracle: TaskLocalNativeFactorOracle,
    video: NativeVideoReadout,
    owners: Sequence[TargetOwner],
    contract: LoRAContract,
    reference: Mapping[str, torch.Tensor],
    s_ref: torch.Tensor,
    relative_singular_threshold: float,
    probability_floor_mass: float,
    reference_member: str,
) -> dict[str, Any]:
    """Seed all free logits with a stable bank projection of one verified member."""

    if len(oracle.frame_counts) != 1 or oracle.frame_counts[0] != video.frame_count:
        raise ValueError("G1 reference initialization currently requires K=1")
    chunks = tuple(video.chunks())
    if not chunks or s_ref.shape != (len(owners),):
        raise ValueError("G1 reference initialization authorities changed")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    target_rows = []
    try:
        with torch.no_grad():
            oracle.rank_queries.zero_()
            oracle.event_logits.zero_()
            frame_measure = oracle._frame_log_measure(video).detach()
            for target_index, (target, owner) in enumerate(
                zip(contract.targets, owners, strict=True)
            ):
                inputs = torch.cat(
                    [chunk.inputs[target_index] for chunk in chunks], dim=0
                )
                output_bank = _output_bank(
                    video=video, chunks=chunks, target=target_index
                )
                input_measure = CandidateMeasure.build(
                    inputs,
                    relative_singular_threshold=relative_singular_threshold,
                )
                groups = native_output_group_count(owner)
                grouped_output = output_bank.reshape(
                    *output_bank.shape[:-1],
                    groups,
                    output_bank.shape[-1] // groups,
                ).movedim(-2, 0)
                output_measures = tuple(
                    CandidateMeasure.build(
                        grouped_output[group],
                        relative_singular_threshold=relative_singular_threshold,
                    )
                    for group in range(groups)
                )
                a_name = target.name + LORA_A_SUFFIX
                b_name = target.name + LORA_B_SUFFIX
                reference_a = reference[a_name].float()
                reference_b = reference[b_name].float()
                projected_a = input_measure.project(reference_a)
                output_width = target.out_features // groups
                projected_b = torch.cat(
                    [
                        measure.project(
                            reference_b[
                                group * output_width : (group + 1) * output_width
                            ].transpose(0, 1)
                        ).transpose(0, 1)
                        for group, measure in enumerate(output_measures)
                    ],
                    dim=0,
                )
                canonical_a, canonical_b = small_core_balanced_svd(
                    projected_a, projected_b
                )
                singular = canonical_a.float().square().sum(-1)
                scale_cap = s_ref[target_index].float()
                scales = torch.minimum(
                    singular / math.sqrt(target.in_features * target.out_features),
                    scale_cap,
                )
                a_unit = rms_normalize(canonical_a)
                b_unit = rms_normalize(canonical_b.transpose(0, 1))
                input_probabilities, realized_a = input_measure.signed_probabilities(
                    a_unit,
                    probability_floor_mass=probability_floor_mass,
                )
                if groups == 1:
                    output_probabilities, realized_b = output_measures[
                        0
                    ].signed_probabilities(
                        b_unit,
                        probability_floor_mass=probability_floor_mass,
                    )
                    output_probabilities_by_group = (output_probabilities,)
                else:
                    output_probabilities_by_group, realized_b = (
                        _scaled_group_probabilities(
                            measures=output_measures,
                            desired=b_unit,
                            probability_floor_mass=probability_floor_mass,
                        )
                    )
                input_logits = input_probabilities.clamp_min(1e-30).log().reshape(
                    G1_RESIDUAL_RANK,
                    2,
                    video.frame_count,
                    G1_PROBE_COUNT,
                    ACTION_HORIZON,
                )
                input_logits = input_logits - frame_measure[
                    target_index, :, None, :, None, None
                ]
                oracle.input_logits[target_index].copy_(input_logits)
                output_slice = oracle.output_group_slices[target_index]
                for group, probabilities in enumerate(
                    output_probabilities_by_group
                ):
                    output_logits = probabilities.clamp_min(1e-30).log().reshape(
                        G1_RESIDUAL_RANK,
                        2,
                        video.frame_count,
                        G1_PROBE_COUNT,
                        ACTION_HORIZON,
                        4,
                    )
                    output_logits = output_logits - frame_measure[
                        target_index, :, None, :, None, None, None
                    ]
                    oracle.output_logits[output_slice.start + group].copy_(
                        output_logits
                    )
                ratio = (scales / scale_cap.clamp_min(1e-30)).clamp(0, 1 - 1e-6)
                oracle.scale_logits[target_index].copy_(torch.atanh(ratio))
                target_rows.append(
                    {
                        "target": target.name,
                        "family": owner.family.value,
                        "input_stable_rank": int(input_measure.basis.shape[1]),
                        "output_stable_ranks": [
                            int(measure.basis.shape[1]) for measure in output_measures
                        ],
                        "input_direction_cosine": (
                            torch.nn.functional.cosine_similarity(
                                realized_a.float(), a_unit.float(), dim=-1
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        "output_direction_cosine": (
                            torch.nn.functional.cosine_similarity(
                                realized_b.float(), b_unit.float(), dim=-1
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        "scale_to_s_ref": ratio.detach().cpu().tolist(),
                    }
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return {
        "kind": "robust_reference_projection",
        "reference_member": reference_member,
        "relative_singular_threshold": relative_singular_threshold,
        "probability_floor_mass": probability_floor_mass,
        "candidate_cache_bytes": cached_native_bytes(video),
        "targets": target_rows,
    }
