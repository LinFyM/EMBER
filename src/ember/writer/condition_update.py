"""Video-keyed Program residuals with exact anchored reconciliation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.writer.model import WriterMemories, WriterVideoEvidence


class ConditionUpdateError(RuntimeError):
    """Raised when the fixed condition-update contract is violated."""


@dataclass(frozen=True)
class AnchoredReconciliationUpdateSummary:
    """Small-matrix and induced-motion evidence for one anchored update."""

    correct_conditions: int
    negative_conditions: int
    damping: float
    feature_rank: int
    precision_condition_number: float
    innovation_condition_number: float
    correct_cotangent_rms: float
    predicted_correct_motion_rms: float
    predicted_negative_motion_rms: float
    predicted_negative_to_correct_ratio: float
    blind_predicted_correct_motion_rms: float
    current_motion_to_blind_ratio: float
    reference_correct_rows: int
    reference_motion_rms: float
    blind_reference_motion_rms: float
    reference_to_blind_ratio: float
    reference_rows_improved_fraction: float
    value_delta_rms: float
    assimilated_rows_before: int
    assimilated_rows_after: int


@dataclass(frozen=True)
class _AnchoredLinearSolve:
    """Small feature-space factors shared by the write and diagnostics."""

    features: torch.Tensor
    small_features: torch.Tensor
    gram: torch.Tensor
    damping: torch.Tensor
    gain: torch.Tensor
    blind_cholesky: torch.Tensor
    next_precision: torch.Tensor
    feature_rank: int
    precision_condition_number: float
    innovation_condition_number: float


@dataclass(frozen=True)
class _ReferenceMotionEvidence:
    rows: int
    motion_rms: float
    blind_motion_rms: float
    improved_fraction: float


class ProgramReconciliationState(torch.nn.Module):
    """Training-only sufficient state for exact cumulative anchored ridge."""

    def __init__(self, *, feature_width: int) -> None:
        super().__init__()
        if feature_width <= 0:
            raise ConditionUpdateError("invalid reconciliation feature width")
        self.register_buffer(
            "precision",
            torch.eye(feature_width, dtype=torch.float64),
            persistent=True,
        )
        self.assimilated_rows = 0

    @property
    def feature_width(self) -> int:
        return int(self.precision.shape[0])


@dataclass(frozen=True)
class ProgramDeltaApplicationSummary:
    """Numerical closure of one optional predicted/observed memory write."""

    observed_motion_rms: float
    predicted_observed_max_abs: float
    predicted_observed_relative_rms: float


class FixedBalancedCausalConditionFeature(torch.nn.Module):
    """Build one balanced static/dynamic key from frozen v6 video evidence."""

    BLOCK_COUNT = 2

    def __init__(
        self,
        *,
        program_width: int,
        feature_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(program_width, feature_width) <= 0
            or feature_width % self.BLOCK_COUNT
            or initialization_seed < 0
        ):
            raise ConditionUpdateError("invalid fixed condition-feature dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        block_width = feature_width // self.BLOCK_COUNT
        projection = torch.empty(
            self.BLOCK_COUNT,
            block_width,
            program_width,
            dtype=torch.float32,
        )
        projection.normal_(generator=generator)
        projection = F.normalize(projection, dim=-1, eps=1e-12).contiguous()
        self.program_width = int(program_width)
        self.feature_width = int(feature_width)
        self.block_width = int(block_width)
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
        # Counterfactual frame orders are produced by the sealed schedule owner.
        # Re-sorting every GPU order here added one device synchronization per
        # condition without adding a second trust boundary.
        return frame_order

    @staticmethod
    def _zero_preserving_normalize(value: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        normalized = value / norms.clamp_min(torch.finfo(value.dtype).tiny)
        return torch.where(norms > 0, normalized, torch.zeros_like(normalized))

    def forward(
        self,
        evidence: WriterVideoEvidence,
        frame_indices: torch.Tensor,
        *,
        frame_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode actual frame content in the supplied sampled-frame order."""

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
            descriptor_blocks = []
            for condition, (left, right) in enumerate(
                zip(evidence.offsets, evidence.offsets[1:])
            ):
                valid_tokens = evidence.valid_task_tokens[condition]
                innovation = (
                    ordered_frames[left:right, valid_tokens]
                    - evidence.text_queries[condition, valid_tokens]
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                ).mean(dim=1)
                static = innovation.mean(dim=0)
                centered = innovation - static
                prefix_scale = torch.arange(
                    1,
                    innovation.shape[0] + 1,
                    dtype=torch.float32,
                    device=innovation.device,
                ).sqrt_()
                causal = (centered.cumsum(dim=0) / prefix_scale.unsqueeze(1)).mean(
                    dim=0
                )
                descriptor_blocks.append(torch.stack((static, causal)))
            descriptors = torch.stack(descriptor_blocks)
            projected = torch.einsum("cbw,bhw->cbh", descriptors, self.projection)
            balanced = self._zero_preserving_normalize(projected)
            features = self._zero_preserving_normalize(balanced.flatten(1))
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


def _gelu_derivative(value: torch.Tensor) -> torch.Tensor:
    """Evaluate exact GELU' in FP32, then return the decoder compute dtype."""

    fp32 = value.to(dtype=torch.float32)
    derivative = 0.5 * (1.0 + torch.erf(fp32 / math.sqrt(2.0)))
    derivative = derivative + fp32 * torch.exp(-0.5 * fp32.square()) / math.sqrt(
        2.0 * math.pi
    )
    return derivative.to(dtype=value.dtype)


def stable_factor_head_linearization(
    head: torch.nn.Module,
    source: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode the frozen base and, when requested, its stable analytic JVP."""

    network = getattr(head, "network", None)
    if (
        not isinstance(network, torch.nn.Sequential)
        or len(network) != 3
        or not isinstance(network[0], torch.nn.Linear)
        or not isinstance(network[1], torch.nn.GELU)
        or not isinstance(network[2], torch.nn.Linear)
        or network[0].bias is not None
        or network[2].bias is not None
        or network[1].approximate != "none"
        or source.ndim < 3
        or (residual is not None and residual.shape != source.shape)
    ):
        raise ConditionUpdateError("frozen FactorHead topology changed")
    hidden = network[0](source)
    rows = network[2](network[1](hidden))
    if residual is None:
        return rows, None
    delta_hidden = network[0](residual.to(dtype=source.dtype))
    tangent = network[2](_gelu_derivative(hidden) * delta_hidden)
    return rows, tangent


def deterministic_mgs_column_pivots(
    matrix: torch.Tensor,
    *,
    keep: int,
) -> torch.Tensor:
    """Choose native B columns with deterministic batched MGS pivoting."""

    if matrix.ndim < 2 or not 0 < keep <= matrix.shape[-1]:
        raise ConditionUpdateError("invalid pivot-preserving base matrix")
    rows, columns = matrix.shape[-2:]
    flat = matrix.to(dtype=torch.float32).reshape(-1, rows, columns)
    residual = flat.clone()
    selected = torch.zeros(flat.shape[0], columns, dtype=torch.bool, device=flat.device)
    pivots = []
    for _ in range(keep):
        norms = residual.square().sum(dim=-2)
        scores = norms.masked_fill(selected, -torch.inf)
        pivot = scores.argmax(dim=-1)
        pivots.append(pivot)
        selected.scatter_(1, pivot[:, None], True)
        vector = residual.gather(
            -1,
            pivot[:, None, None].expand(-1, rows, 1),
        ).squeeze(-1)
        unit = vector / torch.linalg.vector_norm(
            vector, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(vector.dtype).tiny)
        coefficients = torch.matmul(unit[:, None], residual).squeeze(1)
        residual = residual - unit[:, :, None] * coefficients[:, None]
    return torch.stack(pivots, dim=-1).reshape(*matrix.shape[:-2], keep)


def pivot_preserving_base_factors(
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    *,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep selected native B columns and solve their least-squares A rows."""

    if (
        base_a.ndim < 2
        or base_b.ndim != base_a.ndim
        or base_a.shape[:-2] != base_b.shape[:-2]
        or base_a.shape[-2] != base_b.shape[-1]
        or not 0 < keep < base_a.shape[-2]
    ):
        raise ConditionUpdateError("invalid native LoRA factor pair")
    pivots = deterministic_mgs_column_pivots(base_b, keep=keep)
    selected_b = torch.gather(
        base_b,
        -1,
        pivots.unsqueeze(-2).expand(*base_b.shape[:-1], keep),
    )
    batch = math.prod(base_a.shape[:-2])
    selected_flat = selected_b.to(dtype=torch.float32).reshape(
        batch, base_b.shape[-2], keep
    )
    base_b_flat = base_b.to(dtype=torch.float32).reshape(
        batch, base_b.shape[-2], base_b.shape[-1]
    )
    coordinates = torch.linalg.lstsq(selected_flat, base_b_flat).solution
    solved_a = torch.matmul(
        coordinates,
        base_a.to(dtype=torch.float32).reshape(
            batch, base_a.shape[-2], base_a.shape[-1]
        ),
    ).reshape(*base_a.shape[:-2], keep, base_a.shape[-1])
    return solved_a.to(dtype=base_a.dtype), selected_b, pivots


def compact_rank2_effective_tangent(
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor the top-2 of T=B0 dA+dB A0 without materializing full T."""

    if (
        base_a.shape != delta_a.shape
        or base_b.shape != delta_b.shape
        or base_a.ndim < 2
        or base_b.ndim != base_a.ndim
        or base_a.shape[:-2] != base_b.shape[:-2]
        or base_a.shape[-2] != base_b.shape[-1]
        or base_a.shape[-2] * 2 > min(base_a.shape[-1], base_b.shape[-2])
    ):
        raise ConditionUpdateError("invalid effective tangent factor pair")
    left = torch.cat(
        (base_b.to(dtype=torch.float32), delta_b.to(dtype=torch.float32)),
        dim=-1,
    )
    right = torch.cat(
        (delta_a.to(dtype=torch.float32), base_a.to(dtype=torch.float32)),
        dim=-2,
    )
    left_q, left_r = torch.linalg.qr(left, mode="reduced")
    right_q, right_r = torch.linalg.qr(right.transpose(-1, -2), mode="reduced")
    core = torch.matmul(left_r, right_r.transpose(-1, -2))
    core_u, singular, core_vh = torch.linalg.svd(core, full_matrices=False)
    scale = singular[..., :2].clamp_min(0).sqrt()
    residual_b = torch.matmul(left_q, core_u[..., :, :2]) * scale.unsqueeze(-2)
    residual_a = scale.unsqueeze(-1) * torch.matmul(
        core_vh[..., :2, :], right_q.transpose(-1, -2)
    )
    return residual_a.to(dtype=base_a.dtype), residual_b.to(dtype=base_b.dtype)


def _root_mean_square(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


def _row_root_mean_square(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32).flatten(1).square().mean(dim=1).sqrt()


def _validate_anchored_reconciliation_batch(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    correct_cotangents: torch.Tensor,
    reconciliation: ProgramReconciliationState,
    reference_correct_features: torch.Tensor | None,
    *,
    step_size: float,
    relative_damping: float,
) -> tuple[int, torch.Tensor]:
    if correct_features.ndim != 2:
        raise ConditionUpdateError("invalid anchored-reconciliation update batch")
    conditions = correct_features.shape[0]
    valid_batch = (
        conditions > 0
        and negative_features.shape == correct_features.shape
        and correct_cotangents.ndim == 3
        and correct_cotangents.shape[0] == conditions
        and min(correct_cotangents.shape[1:]) > 0
        and correct_features.device == negative_features.device
        and correct_features.device == correct_cotangents.device
    )
    valid_state = (
        reconciliation.precision.shape
        == (correct_features.shape[1], correct_features.shape[1])
        and reconciliation.precision.dtype == torch.float64
        and reconciliation.precision.device == correct_features.device
        and type(reconciliation.assimilated_rows) is int
        and reconciliation.assimilated_rows >= 0
    )
    valid_scalars = (
        math.isfinite(step_size)
        and step_size > 0
        and math.isfinite(relative_damping)
        and relative_damping > 0
    )
    if not (valid_batch and valid_state and valid_scalars):
        raise ConditionUpdateError("invalid anchored-reconciliation update batch")
    if reference_correct_features is None:
        reference_correct_features = correct_features.new_empty(
            (0, correct_features.shape[1])
        )
    if (
        reference_correct_features.ndim != 2
        or reference_correct_features.shape[1:] != correct_features.shape[1:]
        or reference_correct_features.device != correct_features.device
    ):
        raise ConditionUpdateError("invalid reconciliation reference features")
    finite_values = (
        correct_features,
        negative_features,
        correct_cotangents,
        reconciliation.precision,
        reference_correct_features,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise ConditionUpdateError("anchored reconciliation contains non-finite values")
    return conditions, reference_correct_features


def _anchored_linear_solve(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    reconciliation: ProgramReconciliationState,
    *,
    relative_damping: float,
) -> _AnchoredLinearSolve:
    features = torch.cat((correct_features, negative_features), dim=0).to(
        dtype=torch.float32
    )
    small_features = features.to(dtype=torch.float64)
    gram = small_features @ small_features.transpose(0, 1)
    mean_diagonal = gram.diagonal().mean()
    if not bool(torch.isfinite(mean_diagonal)) or float(mean_diagonal) <= 0:
        raise ConditionUpdateError("condition feature Gram has zero energy")
    damping = float(relative_damping) * mean_diagonal
    identity = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
    try:
        precision_cholesky = torch.linalg.cholesky(reconciliation.precision)
        precision_solve = torch.cholesky_solve(
            small_features.transpose(0, 1), precision_cholesky
        )
        innovation = damping * identity + small_features @ precision_solve
        innovation_cholesky = torch.linalg.cholesky(innovation)
        blind_cholesky = torch.linalg.cholesky(gram + identity * damping)
    except RuntimeError as error:
        raise ConditionUpdateError(
            "anchored reconciliation feature solve is not positive definite"
        ) from error
    gain = torch.cholesky_solve(
        precision_solve.transpose(0, 1), innovation_cholesky
    ).transpose(0, 1)
    next_precision = (
        reconciliation.precision
        + small_features.transpose(0, 1) @ small_features / damping
    ).contiguous()
    eigenvalues = torch.linalg.eigvalsh(gram)
    tolerance = 1e-5 * float(eigenvalues.abs().max())
    precision_condition = float(torch.linalg.cond(reconciliation.precision))
    innovation_condition = float(torch.linalg.cond(innovation))
    if (
        not bool(torch.isfinite(next_precision).all())
        or not math.isfinite(precision_condition)
        or not math.isfinite(innovation_condition)
    ):
        raise ConditionUpdateError("anchored reconciliation solve became invalid")
    return _AnchoredLinearSolve(
        features=features,
        small_features=small_features,
        gram=gram,
        damping=damping,
        gain=gain,
        blind_cholesky=blind_cholesky,
        next_precision=next_precision,
        feature_rank=int((eigenvalues > tolerance).sum()),
        precision_condition_number=precision_condition,
        innovation_condition_number=innovation_condition,
    )


def _reference_motion_evidence(
    reference_features: torch.Tensor,
    solve: _AnchoredLinearSolve,
    correct_gain: torch.Tensor,
    cotangent_flat: torch.Tensor,
    *,
    step_size: float,
) -> _ReferenceMotionEvidence:
    rows = int(reference_features.shape[0])
    if not rows:
        return _ReferenceMotionEvidence(0, 0.0, 0.0, 1.0)
    reference_small = reference_features.to(dtype=torch.float64)
    reference_motion = -float(step_size) * (
        (reference_small @ correct_gain).to(dtype=torch.float32) @ cotangent_flat
    )
    blind_gain = torch.cholesky_solve(
        solve.small_features, solve.blind_cholesky
    ).transpose(0, 1)[:, : correct_gain.shape[1]]
    blind_motion = -float(step_size) * (
        (reference_small @ blind_gain).to(dtype=torch.float32) @ cotangent_flat
    )
    reference_rows = _row_root_mean_square(reference_motion)
    blind_rows = _row_root_mean_square(blind_motion)
    return _ReferenceMotionEvidence(
        rows=rows,
        motion_rms=_root_mean_square(reference_motion),
        blind_motion_rms=_root_mean_square(blind_motion),
        improved_fraction=float(
            (reference_rows < blind_rows).to(dtype=torch.float32).mean()
        ),
    )


@torch.no_grad()
def anchored_reconciliation_program_delta(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    correct_cotangents: torch.Tensor,
    reconciliation: ProgramReconciliationState,
    *,
    step_size: float,
    relative_damping: float,
    reference_correct_features: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    AnchoredReconciliationUpdateSummary,
]:
    """Return one exact recursive anchored-ridge write and next precision."""
    conditions, reference_correct_features = _validate_anchored_reconciliation_batch(
        correct_features,
        negative_features,
        correct_cotangents,
        reconciliation,
        reference_correct_features,
        step_size=step_size,
        relative_damping=relative_damping,
    )
    solve = _anchored_linear_solve(
        correct_features,
        negative_features,
        reconciliation,
        relative_damping=relative_damping,
    )
    # Only feature-space solves use FP64; the 21M-value Program write stays FP32.
    correct_gain = solve.gain[:, :conditions].to(dtype=torch.float32)
    cotangent_flat = correct_cotangents.to(dtype=torch.float32).flatten(1)
    delta_flat = -float(step_size) * (correct_gain @ cotangent_flat)
    delta = delta_flat.reshape(
        solve.features.shape[1],
        correct_cotangents.shape[1],
        correct_cotangents.shape[2],
    ).contiguous()
    motion_operator = (solve.small_features @ solve.gain[:, :conditions]).to(
        dtype=torch.float32
    )
    predicted = -float(step_size) * (motion_operator @ cotangent_flat)
    correct_motion = predicted[:conditions]
    negative_motion = predicted[conditions:]
    correct_motion_rms = _root_mean_square(correct_motion)
    negative_motion_rms = _root_mean_square(negative_motion)
    blind_operator = torch.cholesky_solve(
        solve.gram[:, :conditions], solve.blind_cholesky
    ).to(dtype=torch.float32)
    blind_current = -float(step_size) * (blind_operator @ cotangent_flat)
    blind_correct_rms = _root_mean_square(blind_current[:conditions])
    reference = _reference_motion_evidence(
        reference_correct_features,
        solve,
        solve.gain[:, :conditions],
        cotangent_flat,
        step_size=step_size,
    )
    if not bool(torch.isfinite(delta).all()):
        raise ConditionUpdateError("anchored Program write became invalid")
    return (
        delta,
        solve.next_precision,
        AnchoredReconciliationUpdateSummary(
            correct_conditions=conditions,
            negative_conditions=conditions,
            damping=float(solve.damping),
            feature_rank=solve.feature_rank,
            precision_condition_number=solve.precision_condition_number,
            innovation_condition_number=solve.innovation_condition_number,
            correct_cotangent_rms=_root_mean_square(correct_cotangents),
            predicted_correct_motion_rms=correct_motion_rms,
            predicted_negative_motion_rms=negative_motion_rms,
            predicted_negative_to_correct_ratio=(
                negative_motion_rms / correct_motion_rms
                if correct_motion_rms > 0
                else torch.finfo(torch.float32).max
            ),
            blind_predicted_correct_motion_rms=blind_correct_rms,
            current_motion_to_blind_ratio=(
                correct_motion_rms / blind_correct_rms if blind_correct_rms > 0 else 1.0
            ),
            reference_correct_rows=reference.rows,
            reference_motion_rms=reference.motion_rms,
            blind_reference_motion_rms=reference.blind_motion_rms,
            reference_to_blind_ratio=(
                reference.motion_rms / reference.blind_motion_rms
                if reference.blind_motion_rms > 0
                else 0.0
            ),
            reference_rows_improved_fraction=reference.improved_fraction,
            value_delta_rms=_root_mean_square(delta),
            assimilated_rows_before=reconciliation.assimilated_rows,
            assimilated_rows_after=(
                reconciliation.assimilated_rows + solve.features.shape[0]
            ),
        ),
    )


@torch.no_grad()
def apply_anchored_reconciliation_update_(
    memory: ProgramResidualMemory,
    reconciliation: ProgramReconciliationState,
    delta: torch.Tensor,
    next_precision: torch.Tensor,
    *,
    assimilated_rows_after: int,
) -> None:
    """Commit one prevalidated FP32 Program write and FP64 precision update."""

    if (
        delta.shape != memory.value.shape
        or delta.device != memory.value.device
        or next_precision.shape != reconciliation.precision.shape
        or next_precision.dtype != torch.float64
        or next_precision.device != reconciliation.precision.device
        or type(assimilated_rows_after) is not int
        or assimilated_rows_after <= reconciliation.assimilated_rows
    ):
        raise ConditionUpdateError("anchored Program update changed topology")
    memory.value.add_(delta.to(dtype=torch.float32))
    reconciliation.precision.copy_(next_precision)
    reconciliation.assimilated_rows = assimilated_rows_after


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
