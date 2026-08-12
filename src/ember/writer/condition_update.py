"""Policy-innovation-keyed Program residuals with blind full48 updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    WriterMemories,
)
from ember.writer.policy_innovation import FrozenPi05VideoInnovationEncoder
from ember.writer.temporal import SlotNormalizedCoreProcedureCompiler


class ConditionUpdateError(RuntimeError):
    """Raised when the fixed condition-update contract is violated."""


@dataclass(frozen=True)
class SuccessKeyNullspaceUpdateSummary:
    """Small-matrix and induced-motion evidence for one SKNC full48 write."""

    correct_conditions: int
    negative_conditions: int
    current_protected_conditions: int
    unprotected_correct_conditions: int
    anchor_constraint_rows: int
    anchor_rank: int
    original_feature_rank: int
    projected_feature_rank: int
    damping: float
    active_regularized_gram_condition_number: float
    correct_cotangent_rms: float
    predicted_unprotected_correct_motion_rms: float
    predicted_protected_correct_motion_rms: float
    predicted_negative_motion_rms: float
    predicted_protected_to_unprotected_ratio: float
    predicted_negative_to_unprotected_ratio: float
    predicted_anchor_motion_rms: float
    predicted_anchor_motion_max_abs: float
    unprotected_projected_feature_energy_ratio_median: float
    value_delta_rms: float


@dataclass(frozen=True)
class ProgramDeltaApplicationSummary:
    """Numerical closure of one optional predicted/observed memory write."""

    observed_motion_rms: float
    predicted_observed_max_abs: float
    predicted_observed_relative_rms: float


class PolicyInnovationGoalCausalConditionFeature(torch.nn.Module):
    """Build one goal/causal key from phase-aligned frozen-policy innovations."""

    BLOCK_COUNT = 2

    def __init__(
        self,
        *,
        innovation_width: int,
        feature_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(innovation_width, feature_width) <= 0
            or feature_width % self.BLOCK_COUNT
            or initialization_seed < 0
        ):
            raise ConditionUpdateError("invalid policy-innovation key dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        block_width = feature_width // self.BLOCK_COUNT
        projection = torch.empty(
            self.BLOCK_COUNT,
            block_width,
            innovation_width,
            dtype=torch.float32,
        )
        projection.normal_(generator=generator)
        projection = F.normalize(projection, dim=-1, eps=1e-12).contiguous()
        self.innovation_width = int(innovation_width)
        self.feature_width = int(feature_width)
        self.block_width = int(block_width)
        self.initialization_seed = int(initialization_seed)
        # Regenerate this fixed authority from config rather than allowing a
        # residual checkpoint to own either the key or the historical v6 base.
        self.register_buffer("projection", projection, persistent=False)

    @staticmethod
    def _validated_order(
        innovations: torch.Tensor,
        phase_order: torch.Tensor | None,
    ) -> torch.Tensor:
        phases = innovations.shape[1]
        device = innovations.device
        if phase_order is None:
            return torch.arange(phases, dtype=torch.long, device=device)
        if (
            phase_order.ndim != 1
            or phase_order.shape != (phases,)
            or phase_order.dtype != torch.long
            or phase_order.device != device
            or not torch.equal(
                phase_order.sort().values,
                torch.arange(phases, dtype=torch.long, device=device),
            )
        ):
            raise ConditionUpdateError("policy-innovation phase order changed")
        return phase_order

    @staticmethod
    def _zero_preserving_normalize(value: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        normalized = value / norms.clamp_min(torch.finfo(value.dtype).tiny)
        return torch.where(norms > 0, normalized, torch.zeros_like(normalized))

    def forward(
        self,
        innovations: torch.Tensor,
        *,
        phase_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ordered `[conditions, phases, innovation_width]` values."""

        if (
            innovations.ndim != 3
            or min(innovations.shape[:2]) <= 0
            or innovations.shape[1] < 4
            or innovations.shape[1] % 4
            or innovations.shape[2] != self.innovation_width
            or not innovations.is_floating_point()
        ):
            raise ConditionUpdateError("policy-innovation key topology changed")
        order = self._validated_order(innovations, phase_order)
        with torch.autocast(
            device_type=innovations.device.type,
            enabled=False,
        ):
            ordered = innovations.index_select(1, order).to(dtype=torch.float32)
            quartile = ordered.shape[1] // 4
            whole = ordered.mean(dim=1)
            terminal = ordered[:, -quartile:].mean(dim=1)
            goal = terminal - whole
            centered = ordered - whole.unsqueeze(1)
            prefix_scale = torch.arange(
                1,
                ordered.shape[1] + 1,
                dtype=torch.float32,
                device=ordered.device,
            ).sqrt_()
            causal = (
                centered.cumsum(dim=1) / prefix_scale[None, :, None]
            ).mean(dim=1)
            descriptors = torch.stack((goal, causal), dim=1)
            projected = torch.einsum("cbw,bhw->cbh", descriptors, self.projection)
            balanced = self._zero_preserving_normalize(projected)
            features = self._zero_preserving_normalize(balanced.flatten(1))
        if (
            features.shape != (innovations.shape[0], self.feature_width)
            or features.dtype != torch.float32
            or not bool(torch.isfinite(features).all())
        ):
            raise ConditionUpdateError("policy-innovation key became invalid")
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
        innovation_width: int = 3072,
        phase_slots: int = 16,
        max_frames_per_encoder_call: int = 32,
        image_width: int = 2048,
        expert_width: int = 1024,
        action_horizon: int = 50,
        padded_action_dim: int = 32,
        innovation_seed: int = 7,
    ) -> None:
        super().__init__()
        if base_writer.program_width <= 0:
            raise ConditionUpdateError("invalid frozen v6 Writer")
        base_writer.requires_grad_(False).eval()
        self.base_writer = base_writer
        self.policy_innovation = FrozenPi05VideoInnovationEncoder(
            image_width=image_width,
            expert_width=expert_width,
            feature_width=innovation_width,
            phase_slots=phase_slots,
            max_frames_per_encoder_call=max_frames_per_encoder_call,
            action_horizon=action_horizon,
            padded_action_dim=padded_action_dim,
            initialization_seed=innovation_seed,
        )
        self.condition_feature = PolicyInnovationGoalCausalConditionFeature(
            innovation_width=innovation_width,
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
        self.policy_innovation.eval()
        self.condition_feature.eval()
        self.program_memory.eval()
        return self

    def condition_features(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        frame_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one or more raw videos through the frozen policy key owner."""

        offsets = self.base_writer._validated_offsets(video_offsets, frames.shape[0])
        conditions = len(offsets) - 1
        lengths = torch.tensor(
            [right - left for left, right in zip(offsets, offsets[1:])],
            dtype=torch.long,
            device=frames.device,
        )
        frame_video_ids = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device), lengths
        )
        ordered_frames = frames
        if frame_order is not None:
            order = self.base_writer._validate_frame_order(
                frame_order, offsets, device=frames.device
            )
            ordered_frames = frames.index_select(0, order)
        innovations = self.policy_innovation(
            policy,
            ordered_frames,
            frame_video_ids,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return self.condition_feature(innovations)

    def paired_condition_features(
        self,
        policy: torch.nn.Module,
        correct_frames: torch.Tensor,
        correct_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        negative_frames: torch.Tensor | None = None,
        negative_offsets: torch.Tensor | None = None,
        frame_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode correct and counterfactual keys without duplicate frame forwards."""

        correct_bounds = self.base_writer._validated_offsets(
            correct_offsets, correct_frames.shape[0]
        )
        wrong_video = negative_frames is not None or negative_offsets is not None
        if (
            len(correct_bounds) != 2
            or language_tokens.shape[0] != 1
            or language_mask.shape[0] != 1
            or task_span_mask.shape[0] != 1
            or wrong_video == (frame_order is not None)
            or (negative_frames is None) != (negative_offsets is None)
        ):
            raise ConditionUpdateError("paired policy-innovation ownership changed")
        if wrong_video:
            assert negative_frames is not None and negative_offsets is not None
            negative_bounds = self.base_writer._validated_offsets(
                negative_offsets, negative_frames.shape[0]
            )
            if len(negative_bounds) != 2:
                raise ConditionUpdateError("wrong-video policy key is not one-shot")
            lengths = torch.tensor(
                (correct_frames.shape[0], negative_frames.shape[0]),
                dtype=torch.long,
                device=correct_frames.device,
            )
            frames = torch.cat((correct_frames, negative_frames), dim=0)
            frame_video_ids = torch.repeat_interleave(
                torch.arange(2, device=correct_frames.device), lengths
            )
            offsets = torch.tensor(
                (0, correct_frames.shape[0], frames.shape[0]),
                dtype=torch.long,
                device=correct_offsets.device,
            )
            innovations, counts = self.policy_innovation.frame_innovations(
                policy,
                frames,
                frame_video_ids,
                offsets,
                language_tokens.repeat(2, 1),
                language_mask.repeat(2, 1),
                task_span_mask.repeat(2, 1),
            )
            aligned = self.policy_innovation.align_phases(innovations, counts)
            features = self.condition_feature(aligned)
            return features[:1], features[1:]

        assert frame_order is not None
        order = self.base_writer._validate_frame_order(
            frame_order, correct_bounds, device=correct_frames.device
        )
        frame_video_ids = torch.zeros(
            correct_frames.shape[0], dtype=torch.long, device=correct_frames.device
        )
        innovations, counts = self.policy_innovation.frame_innovations(
            policy,
            correct_frames,
            frame_video_ids,
            correct_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        correct_aligned = self.policy_innovation.align_phases(innovations, counts)
        negative_aligned = self.policy_innovation.align_phases(
            innovations.index_select(0, order), counts
        )
        return (
            self.condition_feature(correct_aligned),
            self.condition_feature(negative_aligned),
        )

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
        features = self.condition_features(
            policy,
            frames,
            video_offsets,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return self.base_writer.decode_slots(self.condition_slots(memories, features))


def _root_mean_square(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float32).square().mean().sqrt())


def _numerical_rank(eigenvalues: torch.Tensor, dimension: int) -> int:
    maximum = float(eigenvalues.abs().max()) if eigenvalues.numel() else 0.0
    if maximum == 0:
        return 0
    tolerance = (
        max(1, dimension)
        * torch.finfo(eigenvalues.dtype).eps
        * maximum
    )
    return int((eigenvalues > tolerance).sum())


def _motion_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else torch.finfo(torch.float32).max


def _constraint_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    previous_tf32 = None
    if left.is_cuda:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(
            left.to(dtype=torch.float32), right.to(dtype=torch.float32)
        )
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@torch.no_grad()
def success_key_constraint_motion(
    features: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Read stored FP32 equality geometry without TF32 diagnostic roundoff."""

    return _constraint_matmul(features, delta.flatten(1)).reshape(
        features.shape[0], *delta.shape[1:]
    )


@torch.no_grad()
def success_key_nullspace_program_delta(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    correct_cotangents: torch.Tensor,
    anchor_features: torch.Tensor,
    current_protected_mask: torch.Tensor,
    *,
    step_size: float,
    relative_damping: float,
) -> tuple[torch.Tensor, SuccessKeyNullspaceUpdateSummary]:
    """Solve the blind full48 objective inside certified success-key nullspace."""

    conditions = correct_features.shape[0] if correct_features.ndim == 2 else 0
    if (
        conditions <= 0
        or negative_features.shape != correct_features.shape
        or correct_cotangents.ndim != 3
        or correct_cotangents.shape[0] != conditions
        or min(correct_cotangents.shape[1:]) <= 0
        or correct_features.device != negative_features.device
        or correct_features.device != correct_cotangents.device
        or anchor_features.ndim != 2
        or anchor_features.shape[1:] != correct_features.shape[1:]
        or anchor_features.device != correct_features.device
        or current_protected_mask.shape != (conditions,)
        or current_protected_mask.dtype != torch.bool
        or current_protected_mask.device != correct_features.device
        or not math.isfinite(step_size)
        or step_size <= 0
        or not math.isfinite(relative_damping)
        or relative_damping <= 0
    ):
        raise ConditionUpdateError("invalid success-key nullspace update batch")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            correct_features,
            negative_features,
            correct_cotangents,
            anchor_features,
        )
    ):
        raise ConditionUpdateError(
            "success-key nullspace update contains non-finite values"
        )

    features = torch.cat((correct_features, negative_features), dim=0).to(
        dtype=torch.float32
    )
    small_features = features.to(dtype=torch.float64)
    original_gram = small_features @ small_features.transpose(0, 1)
    original_rank = _numerical_rank(
        torch.linalg.eigvalsh(original_gram), small_features.shape[1]
    )

    anchor_rank = 0
    basis64 = small_features.new_empty((small_features.shape[1], 0))
    anchor_pinv64 = small_features.new_empty((small_features.shape[1], 0))
    if anchor_features.shape[0]:
        anchors64 = anchor_features.to(dtype=torch.float64)
        anchor_u, singular_values, vh = torch.linalg.svd(
            anchors64, full_matrices=False
        )
        maximum = float(singular_values.max()) if singular_values.numel() else 0.0
        tolerance = (
            max(anchors64.shape)
            * torch.finfo(torch.float64).eps
            * maximum
        )
        anchor_rank = int((singular_values > tolerance).sum())
        if anchor_rank <= 0:
            raise ConditionUpdateError("success-key anchors have zero numerical rank")
        basis64 = vh[:anchor_rank].transpose(0, 1).contiguous()
        anchor_pinv64 = (
            basis64 / singular_values[:anchor_rank]
        ) @ anchor_u[:, :anchor_rank].transpose(0, 1)
        projected64 = small_features - (small_features @ basis64) @ basis64.T
    else:
        if bool(current_protected_mask.any()):
            raise ConditionUpdateError("protected correct rows lack success-key anchors")
        # Preserve the sealed PICK-GC arithmetic path exactly when no anchor exists.
        projected64 = small_features

    projected = projected64.to(dtype=torch.float32)
    if bool(current_protected_mask.any()):
        protected_projected = projected64[:conditions][current_protected_mask]
        protected_original = small_features[:conditions][current_protected_mask]
        scale = protected_original.square().sum(dim=1).sqrt().clamp_min(
            torch.finfo(torch.float64).tiny
        )
        residual = protected_projected.square().sum(dim=1).sqrt() / scale
        tolerance = (
            64
            * max(anchor_features.shape)
            * torch.finfo(torch.float64).eps
        )
        if bool((residual > tolerance).any()):
            raise ConditionUpdateError(
                "current protected rows are absent from the success-key span"
            )

    gram = projected64 @ projected64.transpose(0, 1)
    mean_diagonal = gram.diagonal().mean()
    if not bool(torch.isfinite(mean_diagonal)) or float(mean_diagonal) < 0:
        raise ConditionUpdateError("condition feature Gram has invalid energy")
    if float(mean_diagonal) == 0:
        delta = torch.zeros(
            features.shape[1],
            correct_cotangents.shape[1],
            correct_cotangents.shape[2],
            dtype=torch.float32,
            device=features.device,
        )
        projected_eigenvalues = torch.zeros(
            features.shape[0], dtype=torch.float64, device=features.device
        )
        damping = 0.0
        active_condition = 1.0
    else:
        damping_tensor = float(relative_damping) * mean_diagonal
        damping = float(damping_tensor)
        regularized = (
            gram
            + torch.eye(
                gram.shape[0],
                dtype=torch.float64,
                device=gram.device,
            )
            * damping_tensor
        )
        try:
            cholesky = torch.linalg.cholesky(regularized)
        except RuntimeError as error:
            raise ConditionUpdateError(
                "condition feature Gram is not positive definite"
            ) from error

        # Only the 48x48 solve is FP64. The complete 21M-value RHS and write stay FP32.
        inverse = torch.cholesky_inverse(cholesky)
        correct_operator = inverse[:, :conditions].to(dtype=torch.float32)
        cotangent_flat = correct_cotangents.to(dtype=torch.float32).flatten(1)
        coefficients = correct_operator @ cotangent_flat
        delta_flat = -float(step_size) * (
            projected.transpose(0, 1) @ coefficients
        )
        if anchor_rank:
            basis = basis64.to(dtype=torch.float32)
            delta_flat.sub_(basis @ (basis.transpose(0, 1) @ delta_flat))
            anchor_residual = _constraint_matmul(anchor_features, delta_flat)
            delta_flat.sub_(
                _constraint_matmul(
                    anchor_pinv64.to(dtype=torch.float32), anchor_residual
                )
            )
        delta = delta_flat.reshape(
            features.shape[1],
            correct_cotangents.shape[1],
            correct_cotangents.shape[2],
        ).contiguous()
        projected_eigenvalues = torch.linalg.eigvalsh(gram)
        positive = projected_eigenvalues[
            projected_eigenvalues
            > (
                max(gram.shape)
                * torch.finfo(torch.float64).eps
                * projected_eigenvalues.abs().max()
            )
        ]
        active_condition = (
            float((positive.max() + damping_tensor) / (positive.min() + damping_tensor))
            if positive.numel()
            else 1.0
        )

    delta_flat = delta.flatten(1)
    predicted = success_key_constraint_motion(features, delta).flatten(1)
    correct_motion = predicted[:conditions]
    negative_motion = predicted[conditions:]
    protected_motion = correct_motion[current_protected_mask]
    unprotected_mask = ~current_protected_mask
    unprotected_motion = correct_motion[unprotected_mask]
    protected_motion_rms = (
        _root_mean_square(protected_motion) if protected_motion.numel() else 0.0
    )
    unprotected_motion_rms = (
        _root_mean_square(unprotected_motion) if unprotected_motion.numel() else 0.0
    )
    negative_motion_rms = _root_mean_square(negative_motion)
    anchor_motion = success_key_constraint_motion(anchor_features, delta).flatten(1)
    anchor_motion_rms = (
        _root_mean_square(anchor_motion) if anchor_motion.numel() else 0.0
    )
    anchor_motion_max_abs = (
        float(anchor_motion.abs().max()) if anchor_motion.numel() else 0.0
    )
    original_correct_energy = small_features[:conditions].square().sum(dim=1)
    projected_correct_energy = projected64[:conditions].square().sum(dim=1)
    unprotected_energy_ratio = (
        projected_correct_energy[unprotected_mask]
        / original_correct_energy[unprotected_mask].clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    energy_median = (
        float(unprotected_energy_ratio.median())
        if unprotected_energy_ratio.numel()
        else 1.0
    )
    projected_rank = _numerical_rank(
        projected_eigenvalues, projected64.shape[1]
    )
    if not bool(torch.isfinite(delta).all()) or not math.isfinite(active_condition):
        raise ConditionUpdateError("success-key nullspace Program write became invalid")
    return delta, SuccessKeyNullspaceUpdateSummary(
        correct_conditions=conditions,
        negative_conditions=conditions,
        current_protected_conditions=int(current_protected_mask.sum()),
        unprotected_correct_conditions=int(unprotected_mask.sum()),
        anchor_constraint_rows=int(anchor_features.shape[0]),
        anchor_rank=anchor_rank,
        original_feature_rank=original_rank,
        projected_feature_rank=projected_rank,
        damping=float(damping),
        active_regularized_gram_condition_number=active_condition,
        correct_cotangent_rms=_root_mean_square(correct_cotangents),
        predicted_unprotected_correct_motion_rms=unprotected_motion_rms,
        predicted_protected_correct_motion_rms=protected_motion_rms,
        predicted_negative_motion_rms=negative_motion_rms,
        predicted_protected_to_unprotected_ratio=_motion_ratio(
            protected_motion_rms, unprotected_motion_rms
        ),
        predicted_negative_to_unprotected_ratio=_motion_ratio(
            negative_motion_rms, unprotected_motion_rms
        ),
        predicted_anchor_motion_rms=anchor_motion_rms,
        predicted_anchor_motion_max_abs=anchor_motion_max_abs,
        unprotected_projected_feature_energy_ratio_median=energy_median,
        value_delta_rms=_root_mean_square(delta),
    )


@torch.no_grad()
def apply_program_residual_delta_(
    memory: ProgramResidualMemory,
    delta: torch.Tensor,
) -> None:
    """Apply one manual FP32 write without optimizer or hidden state."""

    if delta.shape != memory.value.shape or delta.device != memory.value.device:
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
    projection_shape = (
        PolicyInnovationGoalCausalConditionFeature.BLOCK_COUNT,
        writer.condition_feature.block_width,
        writer.policy_innovation.feature_width,
    )
    fixed_state_is_empty = (
        not writer.policy_innovation.state_dict()
        and not writer.condition_feature.state_dict()
    )
    if (
        len(base_state) != 600
        or any(parameter.requires_grad for parameter in writer.parameters())
        or not fixed_state_is_empty
        or writer.policy_innovation.feature_width
        != writer.condition_feature.innovation_width
        or writer.condition_feature.projection.shape != projection_shape
        or writer.condition_feature.projection.dtype != torch.float32
        or writer.policy_innovation.fixed_suffix_noise.dtype != torch.float32
        or writer.policy_innovation.fixed_suffix_noise.shape
        != (
            writer.policy_innovation.action_horizon,
            writer.policy_innovation.padded_action_dim,
        )
        or writer.program_memory.value.dtype != torch.float32
        or writer.program_memory.value.shape
        != (
            writer.condition_feature.feature_width,
            SlotNormalizedCoreProcedureCompiler.QUERY_COUNT,
            writer.base_writer.program_width,
        )
        or (
            require_zero_memory
            and bool(torch.count_nonzero(writer.program_memory.value))
        )
    ):
        raise WriterModelError("frozen v6 residual Writer ownership changed")
