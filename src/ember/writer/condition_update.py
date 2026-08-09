"""Video-keyed Program residuals with explicit counterfactual-null updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    WriterMemories,
    WriterVideoEvidence,
)
from ember.writer.temporal import SlotNormalizedCoreProcedureCompiler


class ConditionUpdateError(RuntimeError):
    """Raised when the fixed condition-update contract is violated."""


@dataclass(frozen=True)
class CounterfactualNullUpdateSummary:
    """Small-matrix and induced-motion evidence for one full condition update."""

    correct_conditions: int
    negative_conditions: int
    damping: float
    feature_rank: int
    regularized_gram_condition_number: float
    correct_cotangent_rms: float
    predicted_correct_motion_rms: float
    predicted_negative_motion_rms: float
    predicted_negative_to_correct_ratio: float
    value_delta_rms: float


@dataclass(frozen=True)
class ProgramDeltaApplicationSummary:
    """Numerical closure of one optional predicted/observed memory write."""

    observed_motion_rms: float
    predicted_observed_max_abs: float
    predicted_observed_relative_rms: float


class FixedTemporalConditionFeature(torch.nn.Module):
    """Build one zero-preserving, order-sensitive feature from v6 evidence."""

    TEMPORAL_BASIS_COUNT = 4

    def __init__(
        self,
        *,
        program_width: int,
        feature_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(program_width, feature_width) <= 0 or initialization_seed < 0:
            raise ConditionUpdateError("invalid fixed condition-feature dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        projection = torch.empty(
            feature_width,
            self.TEMPORAL_BASIS_COUNT * program_width,
            dtype=torch.float32,
        )
        projection.normal_(generator=generator)
        projection = F.normalize(projection, dim=1, eps=1e-12).contiguous()
        self.program_width = int(program_width)
        self.feature_width = int(feature_width)
        self.initialization_seed = int(initialization_seed)
        # The projection is regenerated from the sealed config seed.  Keeping it
        # non-persistent makes it impossible for a residual checkpoint to cover
        # either this fixed authority or the historical 600-tensor v6 base.
        self.register_buffer("projection", projection, persistent=False)

    @staticmethod
    def _validated_order(
        evidence: WriterVideoEvidence,
        frame_order: torch.Tensor | None,
    ) -> torch.Tensor:
        total = evidence.offsets[-1]
        device = evidence.frame_evidence.device
        if frame_order is None:
            return torch.arange(total, dtype=torch.long, device=device)
        if (
            frame_order.ndim != 1
            or frame_order.shape != (total,)
            or frame_order.dtype != torch.long
            or frame_order.device != device
        ):
            raise ConditionUpdateError("condition feature frame order changed")
        invalid = torch.zeros((), dtype=torch.bool, device=device)
        for left, right in zip(evidence.offsets, evidence.offsets[1:]):
            expected = torch.arange(left, right, dtype=torch.long, device=device)
            invalid |= (frame_order[left:right].sort().values != expected).any()
        if bool(invalid):
            raise ConditionUpdateError("condition feature crossed video boundaries")
        return frame_order

    def forward(
        self,
        evidence: WriterVideoEvidence,
        frame_indices: torch.Tensor,
        *,
        frame_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode actual frame content against fixed sampled-frame ordinals."""

        total = evidence.offsets[-1]
        conditions = len(evidence.offsets) - 1
        if (
            conditions <= 0
            or evidence.frame_evidence.ndim != 3
            or evidence.frame_evidence.shape[0] != total
            or evidence.frame_evidence.shape[-1] != self.program_width
            or evidence.text_queries.shape
            != (
                conditions,
                evidence.frame_evidence.shape[1],
                self.program_width,
            )
            or evidence.valid_task_tokens.shape
            != (conditions, evidence.frame_evidence.shape[1])
            or evidence.valid_task_tokens.dtype != torch.bool
            or frame_indices.shape != (total,)
            or frame_indices.dtype != torch.long
            or frame_indices.device != evidence.frame_evidence.device
        ):
            raise ConditionUpdateError("condition feature evidence topology changed")
        order = self._validated_order(evidence, frame_order)
        starts = torch.tensor(
            evidence.offsets[:-1], dtype=torch.long, device=frame_indices.device
        )
        internal_pairs = torch.ones(
            total - 1, dtype=torch.bool, device=frame_indices.device
        )
        if conditions > 1:
            internal_pairs[
                torch.tensor(
                    evidence.offsets[1:-1],
                    dtype=torch.long,
                    device=frame_indices.device,
                )
                - 1
            ] = False
        invalid = (
            ~evidence.valid_task_tokens.any(dim=1).all()
            | (frame_indices.index_select(0, starts) != 0).any()
            | (((frame_indices[1:] <= frame_indices[:-1]) & internal_pairs).any())
        )
        if bool(invalid):
            raise ConditionUpdateError("condition feature mask or ordinals changed")
        # This fixed key and the manual memory map are an explicit FP32 contract.
        # Both callers intentionally run the surrounding video/Writer graph under
        # BF16 autocast, so disable autocast locally instead of relying on ambient
        # dtype state.
        with torch.autocast(
            device_type=evidence.frame_evidence.device.type,
            enabled=False,
        ):
            ordered_frames = evidence.frame_evidence.index_select(0, order).to(
                dtype=torch.float32
            )
            rows = []
            for condition, (left, right) in enumerate(
                zip(evidence.offsets, evidence.offsets[1:])
            ):
                valid_tokens = evidence.valid_task_tokens[condition]
                ordinals = frame_indices[left:right]
                innovation = (
                    ordered_frames[left:right, valid_tokens]
                    - evidence.text_queries[condition, valid_tokens]
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                ).mean(dim=1)
                if ordinals.numel() == 1:
                    tau = torch.zeros(
                        1,
                        dtype=torch.float32,
                        device=ordinals.device,
                    )
                else:
                    ordinal_values = ordinals.to(dtype=torch.float32)
                    tau = 2.0 * ordinal_values / ordinal_values[-1] - 1.0
                basis = torch.stack(
                    (
                        torch.ones_like(tau),
                        tau,
                        torch.cos(math.pi * tau),
                        torch.sin(math.pi * tau),
                    ),
                    dim=1,
                )
                descriptor = (basis.transpose(0, 1) @ innovation).div_(
                    float(innovation.shape[0])
                )
                rows.append(descriptor.flatten())
            descriptors = torch.stack(rows)
            projected = F.linear(descriptors, self.projection)
            norms = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
            features = projected / norms.clamp_min(
                torch.finfo(projected.dtype).tiny
            )
            features = torch.where(
                norms > 0,
                features,
                torch.zeros_like(features),
            )
        if (
            features.shape != (conditions, self.feature_width)
            or features.dtype != torch.float32
        ):
            raise ConditionUpdateError("condition feature became invalid")
        return features


class ProgramResidualMemory(torch.nn.Module):
    """Read a complete FP32 policy-slot residual from one fixed feature."""

    def __init__(
        self,
        *,
        feature_width: int,
        program_slots: int,
        program_width: int,
    ) -> None:
        super().__init__()
        if min(feature_width, program_slots, program_width) <= 0:
            raise ConditionUpdateError("invalid Program residual dimensions")
        self.register_buffer(
            "value",
            torch.zeros(
                feature_width,
                program_slots,
                program_width,
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != self.value.shape[0]
            or features.device != self.value.device
        ):
            raise ConditionUpdateError("Program residual feature topology changed")
        with torch.autocast(device_type=features.device.type, enabled=False):
            result = torch.matmul(
                features.to(dtype=torch.float32),
                self.value.flatten(1),
            ).reshape(features.shape[0], *self.value.shape[1:])
        return result


class FrozenV6ConditionResidualWriter(torch.nn.Module):
    """Add one video-keyed Program residual before the frozen v6 FactorHeads."""

    def __init__(
        self,
        base_writer: CompleteLoRAWriter,
        *,
        feature_width: int,
        feature_seed: int,
    ) -> None:
        super().__init__()
        if base_writer.program_width <= 0:
            raise ConditionUpdateError("invalid frozen v6 Writer")
        base_writer.requires_grad_(False).eval()
        self.base_writer = base_writer
        self.condition_feature = FixedTemporalConditionFeature(
            program_width=base_writer.program_width,
            feature_width=feature_width,
            initialization_seed=feature_seed,
        )
        self.program_memory = ProgramResidualMemory(
            feature_width=feature_width,
            program_slots=SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            program_width=base_writer.program_width,
        )

    def train(self, mode: bool = True) -> FrozenV6ConditionResidualWriter:
        super().train(mode)
        self.base_writer.eval()
        self.condition_feature.eval()
        self.program_memory.eval()
        return self

    def condition_slots(
        self,
        memories: WriterMemories,
        features: torch.Tensor,
    ) -> torch.Tensor:
        base_slots = self.base_writer.compile_slots(memories)
        residual = self.program_memory(features)
        if residual.shape != base_slots.shape:
            raise ConditionUpdateError("Program residual lost fused-slot topology")
        return base_slots + residual.to(dtype=base_slots.dtype)

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
        evidence = self.base_writer.encode_video_evidence(
            policy,
            frames,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        memories = self.base_writer.build_memories(evidence, frame_indices)
        features = self.condition_feature(evidence, frame_indices)
        return self.base_writer.decode_slots(
            self.condition_slots(memories, features)
        )


def _root_mean_square(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


@torch.no_grad()
def counterfactual_null_program_delta(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    correct_cotangents: torch.Tensor,
    *,
    step_size: float,
    relative_damping: float,
) -> tuple[torch.Tensor, CounterfactualNullUpdateSummary]:
    """Solve the small full-condition Gram and return one FP32 memory write."""

    conditions = correct_features.shape[0] if correct_features.ndim == 2 else 0
    if (
        conditions <= 0
        or negative_features.shape != correct_features.shape
        or correct_cotangents.ndim != 3
        or correct_cotangents.shape[0] != conditions
        or min(correct_cotangents.shape[1:]) <= 0
        or correct_features.device != negative_features.device
        or correct_features.device != correct_cotangents.device
        or not math.isfinite(step_size)
        or step_size <= 0
        or not math.isfinite(relative_damping)
        or relative_damping <= 0
    ):
        raise ConditionUpdateError("invalid counterfactual-null update batch")
    finite = torch.stack(
        (
            torch.isfinite(correct_features).all(),
            torch.isfinite(negative_features).all(),
            torch.isfinite(correct_cotangents).all(),
        )
    ).all()
    if not bool(finite):
        raise ConditionUpdateError("counterfactual-null update contains non-finite values")

    features = torch.cat((correct_features, negative_features), dim=0).to(
        dtype=torch.float32
    )
    small_features = features.to(dtype=torch.float64)
    gram = small_features @ small_features.transpose(0, 1)
    mean_diagonal = gram.diagonal().mean()
    if not bool(torch.isfinite(mean_diagonal)) or float(mean_diagonal) <= 0:
        raise ConditionUpdateError("condition feature Gram has zero energy")
    damping_tensor = float(relative_damping) * mean_diagonal
    regularized = gram + torch.eye(
        gram.shape[0],
        dtype=torch.float64,
        device=gram.device,
    ) * damping_tensor
    try:
        cholesky = torch.linalg.cholesky(regularized)
    except RuntimeError as error:
        raise ConditionUpdateError("condition feature Gram is not positive definite") from error

    # Only the 2N x 2N operator is solved in FP64.  Both the full Program RHS
    # and the roughly 21M-value memory write remain FP32 for throughput.
    inverse = torch.cholesky_inverse(cholesky)
    correct_operator = inverse[:, :conditions].to(dtype=torch.float32)
    cotangent_flat = correct_cotangents.to(dtype=torch.float32).flatten(1)
    coefficients = correct_operator @ cotangent_flat
    delta_flat = -float(step_size) * (features.transpose(0, 1) @ coefficients)
    delta = delta_flat.reshape(
        features.shape[1],
        correct_cotangents.shape[1],
        correct_cotangents.shape[2],
    ).contiguous()

    motion_operator = (gram @ inverse[:, :conditions]).to(dtype=torch.float32)
    predicted = -float(step_size) * (motion_operator @ cotangent_flat)
    correct_motion = predicted[:conditions]
    negative_motion = predicted[conditions:]
    correct_motion_rms = _root_mean_square(correct_motion)
    negative_motion_rms = _root_mean_square(negative_motion)
    eigenvalues = torch.linalg.eigvalsh(gram)
    tolerance = 1e-5 * float(eigenvalues.abs().max())
    rank = int((eigenvalues > tolerance).sum())
    condition = float(torch.linalg.cond(regularized))
    if not bool(torch.isfinite(delta).all()) or not math.isfinite(condition):
        raise ConditionUpdateError("counterfactual-null Program write became invalid")
    return delta, CounterfactualNullUpdateSummary(
        correct_conditions=conditions,
        negative_conditions=conditions,
        damping=float(damping_tensor),
        feature_rank=rank,
        regularized_gram_condition_number=condition,
        correct_cotangent_rms=_root_mean_square(correct_cotangents),
        predicted_correct_motion_rms=correct_motion_rms,
        predicted_negative_motion_rms=negative_motion_rms,
        predicted_negative_to_correct_ratio=(
            negative_motion_rms / correct_motion_rms
            if correct_motion_rms > 0
            else torch.finfo(torch.float32).max
        ),
        value_delta_rms=_root_mean_square(delta),
    )


@torch.no_grad()
def apply_program_residual_delta_(
    memory: ProgramResidualMemory,
    delta: torch.Tensor,
) -> None:
    """Apply one manual FP32 write without optimizer or hidden state."""

    if (
        delta.shape != memory.value.shape
        or delta.device != memory.value.device
    ):
        raise ConditionUpdateError("Program residual delta changed topology")
    memory.value.add_(delta.to(dtype=torch.float32))


@torch.no_grad()
def program_residual_delta_application_evidence(
    memory: ProgramResidualMemory,
    delta: torch.Tensor,
    features: torch.Tensor,
    before: torch.Tensor,
    *,
    predicted: torch.Tensor | None = None,
) -> ProgramDeltaApplicationSummary:
    """Verify a completed write outside the production update timing region."""

    if before.shape != (features.shape[0], *memory.value.shape[1:]):
        raise ConditionUpdateError("Program residual before-read changed topology")
    if predicted is None:
        predicted = torch.matmul(
            features.to(dtype=torch.float32),
            delta.flatten(1).to(dtype=torch.float32),
        ).reshape(features.shape[0], *memory.value.shape[1:])
    elif predicted.shape != before.shape or predicted.dtype != torch.float32:
        raise ConditionUpdateError("Program residual prediction changed topology")
    observed = memory(features) - before
    error = observed - predicted
    predicted_rms = torch.linalg.vector_norm(predicted) / math.sqrt(
        float(predicted.numel())
    )
    error_rms = torch.linalg.vector_norm(error) / math.sqrt(float(error.numel()))
    relative = error_rms / predicted_rms.clamp_min(
        torch.finfo(predicted_rms.dtype).tiny
    )
    if not bool(torch.isfinite(relative)):
        raise ConditionUpdateError("Program residual write evidence became invalid")
    return ProgramDeltaApplicationSummary(
        observed_motion_rms=_root_mean_square(observed),
        predicted_observed_max_abs=float(error.abs().max()),
        predicted_observed_relative_rms=float(relative),
    )


@torch.no_grad()
def apply_program_residual_delta_with_evidence_(
    memory: ProgramResidualMemory,
    delta: torch.Tensor,
    features: torch.Tensor,
) -> ProgramDeltaApplicationSummary:
    """Apply one write and verify it in CPU or one-shot profile oracles."""

    before = memory(features).clone()
    apply_program_residual_delta_(memory, delta)
    return program_residual_delta_application_evidence(
        memory,
        delta,
        features,
        before,
    )


def validate_frozen_v6_residual_writer(
    writer: FrozenV6ConditionResidualWriter,
    *,
    require_zero_memory: bool = False,
) -> None:
    """Fail closed if the wrapper gains trainable or malformed dynamic state."""

    base_state = writer.base_writer.state_dict()
    if (
        len(base_state) != 600
        or any(parameter.requires_grad for parameter in writer.parameters())
        or writer.program_memory.value.dtype != torch.float32
        or writer.program_memory.value.shape
        != (
            writer.condition_feature.feature_width,
            SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            writer.base_writer.program_width,
        )
        or (require_zero_memory and bool(torch.count_nonzero(writer.program_memory.value)))
    ):
        raise WriterModelError("frozen v6 residual Writer ownership changed")
