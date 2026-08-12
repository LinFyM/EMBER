"""Policy-innovation-keyed Program residuals with paired-video joint credit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.expert_manifold.legacy_v6_model import (
    CompleteLoRAWriter,
    WriterModelError,
    WriterMemories,
)
from ember.writer.policy_innovation import FrozenPi05VideoInnovationEncoder
from ember.writer.temporal import SlotNormalizedCoreProcedureCompiler


class ConditionUpdateError(RuntimeError):
    """Raised when the fixed condition-update contract is violated."""


@dataclass(frozen=True)
class ProgramDeltaApplicationSummary:
    """Numerical closure of one optional predicted/observed memory write."""

    observed_motion_rms: float
    predicted_observed_max_abs: float
    predicted_observed_relative_rms: float


@dataclass(frozen=True)
class PairedVideoJointUpdateSummary:
    """Evidence for one weighted paired-video joint Program solve."""

    task_count: int
    views_per_task: int
    correct_conditions: int
    negative_conditions: int
    row_count: int
    positive_feature_rank: int
    original_feature_rank: int
    regularized_gram_condition_number: float
    damping: float
    correct_cotangent_rms: float
    primary_directional_derivative: float
    companion_directional_derivative: float
    joint_directional_derivative: float
    negative_to_correct_motion_ratio: float
    primary_motion_rms: float
    companion_motion_rms: float
    negative_motion_rms: float
    value_delta_rms: float


class PolicyInnovationGoalCausalConditionFeature(torch.nn.Module):
    """Build one magnitude-gated causal/goal key from policy innovations."""

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
            goal_block, causal_block = balanced.unbind(dim=1)
            odd = self._zero_preserving_normalize(
                goal_block.abs() * causal_block
            )
            even = self._zero_preserving_normalize(goal_block * causal_block)
            features = self._zero_preserving_normalize(
                torch.cat((odd, even), dim=1)
            )
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
    return torch.matmul(left.to(dtype=torch.float32), right.to(dtype=torch.float32))


@torch.no_grad()
def program_delta_condition_motion(
    features: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Read the induced Program motion with the runtime's native matmul mode."""

    return _constraint_matmul(features, delta.flatten(1)).reshape(
        features.shape[0], *delta.shape[1:]
    )


@torch.no_grad()
def paired_video_joint_program_delta(
    correct_features: torch.Tensor,
    negative_features: torch.Tensor,
    correct_cotangents: torch.Tensor,
    *,
    task_count: int,
    view_weights: torch.Tensor,
    step_size: float,
    relative_damping: float,
) -> tuple[torch.Tensor, PairedVideoJointUpdateSummary, torch.Tensor]:
    """Solve the symmetric paired-video functional objective in one shared map."""

    conditions = correct_features.shape[0] if correct_features.ndim == 2 else 0
    views = conditions // task_count if task_count > 0 else 0
    if (
        task_count <= 0
        or views != 2
        or conditions != 2 * task_count
        or negative_features.shape != correct_features.shape
        or correct_cotangents.ndim != 3
        or correct_cotangents.shape[0] != conditions
        or min(correct_cotangents.shape[1:]) <= 0
        or correct_features.device != negative_features.device
        or correct_features.device != correct_cotangents.device
        or view_weights.shape != (conditions,)
        or view_weights.dtype != torch.float32
        or view_weights.device != correct_features.device
        or not math.isfinite(step_size)
        or step_size <= 0
        or not math.isfinite(relative_damping)
        or relative_damping <= 0
    ):
        raise ConditionUpdateError("invalid paired-video joint update batch")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            correct_features,
            negative_features,
            correct_cotangents,
            view_weights,
        )
    ):
        raise ConditionUpdateError("paired-video joint update contains non-finite values")
    expected_weights = torch.full_like(view_weights, 0.5)
    if not torch.equal(view_weights, expected_weights):
        raise ConditionUpdateError("paired-video view weights changed")

    features = torch.cat((correct_features, negative_features), dim=0).to(
        dtype=torch.float32
    )
    row_weights = torch.cat((view_weights, view_weights), dim=0)
    square_root_weights = row_weights.sqrt()
    weighted_features64 = (
        features.to(dtype=torch.float64)
        * square_root_weights.to(dtype=torch.float64).unsqueeze(1)
    )
    gram = weighted_features64 @ weighted_features64.transpose(0, 1)
    feature_energy = features.to(dtype=torch.float64).square().sum(dim=1)
    damping_tensor = float(relative_damping) * (
        (row_weights.to(dtype=torch.float64) * feature_energy).sum()
        / row_weights.to(dtype=torch.float64).sum()
    )
    if not bool(torch.isfinite(damping_tensor)) or float(damping_tensor) <= 0:
        raise ConditionUpdateError("paired-video joint damping became invalid")
    regularized = gram + torch.eye(
        gram.shape[0], dtype=torch.float64, device=gram.device
    ) * damping_tensor
    try:
        cholesky = torch.linalg.cholesky(regularized)
    except RuntimeError as error:
        raise ConditionUpdateError(
            "paired-video regularized Gram is not positive definite"
        ) from error

    rhs = torch.zeros(
        features.shape[0],
        correct_cotangents[0].numel(),
        dtype=torch.float32,
        device=features.device,
    )
    rhs[:conditions] = correct_cotangents.to(dtype=torch.float32).flatten(1)
    rhs.mul_(square_root_weights.unsqueeze(1))
    # Keep only the 96x96 inverse in FP64. The 96x81,920 cotangent RHS and
    # complete condition-to-Program write remain FP32.
    inverse = torch.cholesky_inverse(cholesky).to(dtype=torch.float32)
    coefficients = inverse @ rhs
    delta_flat = -float(step_size) * (
        weighted_features64.to(dtype=torch.float32).transpose(0, 1) @ coefficients
    )
    delta = delta_flat.reshape(
        features.shape[1],
        correct_cotangents.shape[1],
        correct_cotangents.shape[2],
    ).contiguous()

    motion = program_delta_condition_motion(features, delta)
    correct_motion = motion[:conditions]
    negative_motion = motion[conditions:]
    primary_motion = correct_motion[:task_count]
    companion_motion = correct_motion[task_count:]
    cotangents = correct_cotangents.to(dtype=torch.float32)
    primary_derivative = float((cotangents[:task_count] * primary_motion).sum())
    companion_derivative = float((cotangents[task_count:] * companion_motion).sum())
    joint_derivative = 0.5 * (primary_derivative + companion_derivative)
    correct_rms = _root_mean_square(correct_motion)
    negative_rms = _root_mean_square(negative_motion)
    positive_gram = weighted_features64[:conditions] @ weighted_features64[
        :conditions
    ].transpose(0, 1)
    positive_eigenvalues = torch.linalg.eigvalsh(positive_gram)
    eigenvalues = torch.linalg.eigvalsh(gram)
    regularized_eigenvalues = torch.linalg.eigvalsh(regularized)
    active_condition = float(
        regularized_eigenvalues.max() / regularized_eigenvalues.min()
    )
    summary = PairedVideoJointUpdateSummary(
        task_count=task_count,
        views_per_task=views,
        correct_conditions=conditions,
        negative_conditions=conditions,
        row_count=2 * conditions,
        positive_feature_rank=_numerical_rank(
            positive_eigenvalues, features.shape[1]
        ),
        original_feature_rank=_numerical_rank(eigenvalues, features.shape[1]),
        regularized_gram_condition_number=active_condition,
        damping=float(damping_tensor),
        correct_cotangent_rms=_root_mean_square(cotangents),
        primary_directional_derivative=primary_derivative,
        companion_directional_derivative=companion_derivative,
        joint_directional_derivative=joint_derivative,
        negative_to_correct_motion_ratio=_motion_ratio(negative_rms, correct_rms),
        primary_motion_rms=_root_mean_square(primary_motion),
        companion_motion_rms=_root_mean_square(companion_motion),
        negative_motion_rms=negative_rms,
        value_delta_rms=_root_mean_square(delta),
    )
    if not all(
        math.isfinite(float(value))
        for value in (
            *summary.__dict__.values(),
        )
        if isinstance(value, float)
    ):
        raise ConditionUpdateError("paired-video joint update became non-finite")
    return delta, summary, motion


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
