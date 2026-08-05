"""Fixed condition features and explicit program-value kernel updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class ConditionKernelError(RuntimeError):
    """Raised when the sealed condition-kernel contract is violated."""


CONDITION_AUTHORITY_TENSORS = frozenset(
    {"task_center", "task_frequencies", "video_frequencies"}
)


def load_condition_authority(path: str) -> dict[str, torch.Tensor]:
    """Load the tiny sealed condition address authority on CPU."""

    tensors = load_file(path, device="cpu")
    if set(tensors) != CONDITION_AUTHORITY_TENSORS:
        raise ConditionKernelError("condition address authority tensor set changed")
    task_center = tensors["task_center"]
    task_frequencies = tensors["task_frequencies"]
    video_frequencies = tensors["video_frequencies"]
    if (
        task_center.shape != (2048,)
        or task_frequencies.shape != (16, 2048)
        or video_frequencies.shape != (16, 512)
        or any(value.dtype != torch.float32 for value in tensors.values())
        or not all(bool(torch.isfinite(value).all()) for value in tensors.values())
    ):
        raise ConditionKernelError("condition address authority values changed")
    return {name: value.contiguous() for name, value in tensors.items()}


@dataclass(frozen=True)
class KernelUpdateSummary:
    """Small diagnostics for one exact condition-kernel value update."""

    damping: float
    gram_condition_number: float
    feature_rank: int
    cotangent_rms: float
    predicted_update_rms: float
    applied_scale: float


@dataclass(frozen=True)
class ProgramApplicationSummary:
    """Numerical agreement between one predicted and observed Program write."""

    observed_update_rms: float
    predicted_observed_max_abs: float
    predicted_observed_relative_rms: float


class FactorizedConditionFeature(torch.nn.Module):
    """Map frozen task/video descriptors to one fixed product-kernel feature."""

    def __init__(
        self,
        *,
        task_center: torch.Tensor,
        task_frequencies: torch.Tensor,
        video_frequencies: torch.Tensor,
    ) -> None:
        super().__init__()
        if (
            task_center.ndim != 1
            or task_frequencies.ndim != 2
            or video_frequencies.ndim != 2
            or task_frequencies.shape[1] != task_center.numel()
            or task_frequencies.shape[0] <= 0
            or video_frequencies.shape[0] <= 0
        ):
            raise ConditionKernelError("invalid fixed condition-feature authority")
        self.register_buffer(
            "task_center",
            task_center.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "task_frequencies",
            task_frequencies.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "video_frequencies",
            video_frequencies.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )

    @property
    def feature_width(self) -> int:
        return 4 * int(self.task_frequencies.shape[0]) * int(
            self.video_frequencies.shape[0]
        )

    @staticmethod
    def _rff(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        phase = F.linear(value.to(torch.float32), frequencies)
        result = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return result / math.sqrt(float(frequencies.shape[0]))

    def forward(
        self,
        task_descriptor: torch.Tensor,
        video_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_descriptor.ndim != 2
            or video_descriptor.ndim != 2
            or task_descriptor.shape[0] != video_descriptor.shape[0]
            or task_descriptor.shape[1] != self.task_center.numel()
            or video_descriptor.shape[1] != self.video_frequencies.shape[1]
        ):
            raise ConditionKernelError("condition descriptors changed shape")
        task = F.normalize(
            task_descriptor.to(torch.float32) - self.task_center,
            dim=-1,
            eps=1e-12,
        )
        video = F.normalize(video_descriptor.to(torch.float32), dim=-1, eps=1e-12)
        task_feature = self._rff(task, self.task_frequencies)
        video_feature = self._rff(video, self.video_frequencies)
        product = torch.einsum(
            "bt,bv->btv", task_feature, video_feature
        ).flatten(1)
        feature = F.normalize(product, dim=-1, eps=1e-12)
        if (
            feature.shape
            != (task_descriptor.shape[0], self.feature_width)
            or not bool(torch.isfinite(feature).all())
        ):
            raise ConditionKernelError("condition feature became invalid")
        return feature


class ProgramValueMemory(torch.nn.Module):
    """Linearly read a complete policy program from fixed condition features."""

    def __init__(
        self,
        *,
        feature_width: int,
        program_slots: int,
        program_width: int,
        initialization_seed: int,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if (
            min(feature_width, program_slots, program_width) <= 0
            or initialization_seed < 0
            or initialization_std <= 0
        ):
            raise ConditionKernelError("invalid Program Value Memory dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        value = torch.empty(feature_width, program_slots, program_width)
        value.normal_(mean=0.0, std=initialization_std, generator=generator)
        self.value = torch.nn.Parameter(value)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[1] != self.value.shape[0]:
            raise ConditionKernelError("Program Value Memory feature changed shape")
        program = torch.einsum(
            "bf,fsw->bsw", feature.to(self.value.dtype), self.value
        )
        if not bool(torch.isfinite(program).all()):
            raise ConditionKernelError("Program Value Memory produced non-finite output")
        return program


@torch.no_grad()
def kernel_corrected_value_delta(
    features: torch.Tensor,
    cotangents: torch.Tensor,
    *,
    step_size: float,
    relative_damping: float = 0.01,
    induced_update_rms_cap: float | None = None,
) -> tuple[torch.Tensor, KernelUpdateSummary]:
    """Return the exact regularized function-space update for one full task set."""

    if (
        features.ndim != 2
        or cotangents.ndim != 3
        or features.shape[0] != cotangents.shape[0]
        or features.shape[0] <= 0
        or features.shape[1] <= 0
        or cotangents.shape[1] <= 0
        or cotangents.shape[2] <= 0
        or step_size <= 0
        or relative_damping <= 0
        or (
            induced_update_rms_cap is not None
            and induced_update_rms_cap <= 0
        )
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(cotangents).all())
    ):
        raise ConditionKernelError("invalid condition-kernel update batch")

    # The scientific correction is the small full-task Gram solve.  Keep that
    # solve in FP64, then perform the 84M-value write in FP32: materializing an
    # FP64 Program Memory delta would double both bandwidth and peak memory
    # without changing the solved coefficients.
    phi = features.to(torch.float64)
    gram = phi @ phi.transpose(0, 1)
    mean_diagonal = gram.diagonal().mean()
    damping = float(relative_damping) * mean_diagonal
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * damping
    try:
        cholesky = torch.linalg.cholesky(regularized)
    except RuntimeError as error:
        raise ConditionKernelError("condition Gram is not positive definite") from error
    flat = cotangents.to(torch.float64).flatten(1)
    coefficients = torch.cholesky_solve(flat, cholesky)
    phi_write = features.to(torch.float32)
    coefficient_write = coefficients.to(torch.float32)
    delta = -float(step_size) * torch.einsum(
        "tf,td->fd", phi_write, coefficient_write
    )
    delta = delta.reshape(
        features.shape[1], cotangents.shape[1], cotangents.shape[2]
    )
    predicted = torch.einsum("tf,fsw->tsw", phi_write, delta)
    predicted_rms = predicted.square().mean().sqrt()
    applied_scale = 1.0
    if (
        induced_update_rms_cap is not None
        and float(predicted_rms) > induced_update_rms_cap
    ):
        applied_scale = float(induced_update_rms_cap) / float(predicted_rms)
        delta.mul_(applied_scale)
        predicted_rms.mul_(applied_scale)
    eigenvalues = torch.linalg.eigvalsh(gram)
    tolerance = torch.finfo(eigenvalues.dtype).eps * max(gram.shape) * float(
        eigenvalues.abs().max()
    )
    feature_rank = int((eigenvalues > tolerance).sum())
    condition = torch.linalg.cond(regularized)
    result = delta.to(dtype=cotangents.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ConditionKernelError("condition-kernel update became non-finite")
    return result, KernelUpdateSummary(
        damping=float(damping),
        gram_condition_number=float(condition),
        feature_rank=feature_rank,
        cotangent_rms=float(cotangents.to(torch.float64).square().mean().sqrt()),
        predicted_update_rms=float(predicted_rms),
        applied_scale=applied_scale,
    )


@torch.no_grad()
def apply_program_value_delta(
    memory: ProgramValueMemory,
    delta: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, ProgramApplicationSummary]:
    """Apply one value update and measure its induced Program change."""

    if (
        delta.shape != memory.value.shape
        or features.ndim != 2
        or features.shape[1] != memory.value.shape[0]
        or not bool(torch.isfinite(delta).all())
        or not bool(torch.isfinite(features).all())
    ):
        raise ConditionKernelError("Program Value Memory delta changed shape")
    before = memory(features).detach().clone()
    predicted = torch.einsum(
        "tf,fsw->tsw", features.to(delta.dtype), delta
    )
    memory.value.add_(delta.to(device=memory.value.device, dtype=memory.value.dtype))
    observed = memory(features) - before
    error = observed - predicted.to(observed.dtype)
    error_rms = error.square().mean().sqrt()
    predicted_rms = predicted.square().mean().sqrt()
    relative = error_rms / predicted_rms.clamp_min(
        torch.finfo(predicted_rms.dtype).tiny
    )
    if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(relative)):
        raise ConditionKernelError("Program Value Memory observation became invalid")
    return observed, ProgramApplicationSummary(
        observed_update_rms=float(observed.square().mean().sqrt()),
        predicted_observed_max_abs=float(error.abs().max()),
        predicted_observed_relative_rms=float(relative),
    )
