"""Low-dimensional bank-adaptive functional coordinates for Native-Factor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class SketchCrossImage:
    """Centered native/key cross-images without a native covariance matrix."""

    mean: torch.Tensor
    key_images: torch.Tensor
    total_mass: torch.Tensor
    event_mass: torch.Tensor


@dataclass(frozen=True)
class BankAdaptiveBasis:
    """One prefix of a task-independent nested projection of the current bank."""

    vectors: torch.Tensor
    singular_values: torch.Tensor
    requested_rank: int
    retained_rank: int
    mode: str


@dataclass(frozen=True)
class SketchedFunctionalStatistics:
    """Functional sufficient statistics retained only in a small native basis."""

    mean: torch.Tensor
    basis: torch.Tensor
    covariance: torch.Tensor
    replay_covariances: torch.Tensor
    replay_images: torch.Tensor
    key_images: torch.Tensor
    total_mass: torch.Tensor
    projection_singular_values: torch.Tensor


@dataclass(frozen=True)
class FunctionalTargetQueries:
    """Task-local native queries used only by the sketch capacity oracle."""

    native: torch.Tensor
    reduced: torch.Tensor
    linear_recovery: torch.Tensor


class StreamingSketchCrossImage:
    """Accumulate centered event native/key cross-images over arbitrary chunks."""

    def __init__(
        self,
        *,
        native_width: int,
        key_width: int,
        events: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        leading_block_size: int = 4,
    ) -> None:
        if min(native_width, key_width, events, leading_block_size) <= 0 or dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise BankConditioningError("invalid functional-sketch topology")
        self.native_width = int(native_width)
        self.key_width = int(key_width)
        self.events = int(events)
        self.dtype = dtype
        self.leading_block_size = int(leading_block_size)
        self.total_mass = torch.zeros((), device=device, dtype=dtype)
        self.first = torch.zeros(native_width, device=device, dtype=dtype)
        self.event_mass = torch.zeros(events, device=device, dtype=dtype)
        self.event_value = torch.zeros(
            events, native_width, device=device, dtype=dtype
        )
        self.event_key = torch.zeros(events, key_width, device=device, dtype=dtype)
        self.event_cross = torch.zeros(
            events, native_width, key_width, device=device, dtype=dtype
        )
        self.candidate_count = 0
        self._pending: tuple[torch.Tensor, ...] | None = None

    def _consume(
        self,
        values: torch.Tensor,
        base_mass: torch.Tensor,
        event_mass: torch.Tensor,
        event_keys: torch.Tensor,
    ) -> None:
        value = values.reshape(-1, self.native_width)
        base = base_mass.reshape(-1)
        event = event_mass.reshape(self.events, -1)
        key = event_keys.reshape(self.events, -1, self.key_width)
        self.total_mass += base.sum()
        self.first += torch.einsum("n,nd->d", base, value)
        self.event_mass += event.sum(-1)
        self.event_value += torch.einsum("en,nd->ed", event, value)
        self.event_key += torch.einsum("en,enw->ew", event, key)
        self.event_cross += torch.einsum("en,nd,enw->edw", event, value, key)
        self.candidate_count += int(value.shape[0])

    def add(
        self,
        values: torch.Tensor,
        base_mass: torch.Tensor,
        event_mass: torch.Tensor,
        event_keys: torch.Tensor,
    ) -> None:
        candidate_shape = values.shape[:-1]
        if (
            values.ndim < 2
            or values.shape[-1] != self.native_width
            or base_mass.shape != candidate_shape
            or event_mass.shape != (self.events, *candidate_shape)
            or event_keys.shape
            != (self.events, *candidate_shape, self.key_width)
            or values.numel() <= 0
        ):
            raise BankConditioningError("functional-sketch candidate axes changed")
        value = values.detach().to(dtype=self.dtype).reshape(-1, self.native_width)
        base = base_mass.detach().to(dtype=self.dtype).reshape(-1)
        event = event_mass.detach().to(dtype=self.dtype).reshape(self.events, -1)
        key = event_keys.detach().to(dtype=self.dtype).reshape(
            self.events, -1, self.key_width
        )
        if (
            torch.any(base < 0)
            or torch.any(event < 0)
            or not bool(torch.isfinite(value).all())
            or not bool(torch.isfinite(base).all())
            or not bool(torch.isfinite(event).all())
            or not bool(torch.isfinite(key).all())
        ):
            raise BankConditioningError("functional-sketch stream is invalid")
        value = value.reshape(*values.shape[:-1], self.native_width)
        base = base.reshape(base_mass.shape)
        event = event.reshape(self.events, *event_mass.shape[1:])
        key = key.reshape(self.events, *event_keys.shape[1:])
        if self._pending is not None:
            old_value, old_base, old_event, old_key = self._pending
            value = torch.cat((old_value, value), dim=0)
            base = torch.cat((old_base, base), dim=0)
            event = torch.cat((old_event, event), dim=1)
            key = torch.cat((old_key, key), dim=1)
            self._pending = None
        stop = value.shape[0] - value.shape[0] % self.leading_block_size
        for start in range(0, stop, self.leading_block_size):
            end = start + self.leading_block_size
            self._consume(
                value[start:end],
                base[start:end],
                event[:, start:end],
                key[:, start:end],
            )
        if stop < value.shape[0]:
            self._pending = (
                value[stop:],
                base[stop:],
                event[:, stop:],
                key[:, stop:],
            )

    def finalize(self) -> SketchCrossImage:
        if self._pending is not None:
            self._consume(*self._pending)
            self._pending = None
        if (
            self.candidate_count <= 1
            or float(self.total_mass) <= 0.0
            or torch.any(self.event_mass <= 0)
        ):
            raise BankConditioningError("functional-sketch stream is empty")
        mean = self.first / self.total_mass
        event_value = self.event_value / self.event_mass[:, None]
        event_key = self.event_key / self.event_mass[:, None]
        key_images = self.event_cross / self.event_mass[:, None, None]
        key_images -= torch.einsum("ed,ew->edw", event_value, event_key)
        if not bool(torch.isfinite(key_images).all()):
            raise BankConditioningError("functional-sketch cross-image is non-finite")
        return SketchCrossImage(
            mean=mean,
            key_images=key_images,
            total_mass=self.total_mass,
            event_mass=self.event_mass,
        )


def fixed_nested_projection(
    *,
    events: int,
    key_width: int,
    maximum_rank: int,
    mode: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create one sealed projection whose column prefixes define the rank curve."""

    if (
        min(events, key_width, maximum_rank) <= 0
        or mode not in {"global", "per_event"}
        or seed < 0
        or dtype not in (torch.float32, torch.float64)
    ):
        raise BankConditioningError("invalid nested projection contract")
    rows = events * key_width
    if maximum_rank > rows:
        raise BankConditioningError("nested projection exceeds its source space")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if mode == "global":
        raw = torch.randn(rows, maximum_rank, generator=generator, dtype=torch.float64)
        projection, triangular = torch.linalg.qr(raw, mode="reduced")
        signs = torch.sign(torch.diagonal(triangular)).clamp_min(0).mul(2).sub(1)
        projection = projection * signs
    else:
        per_event = (maximum_rank + events - 1) // events
        if per_event > key_width:
            raise BankConditioningError("per-event projection exceeds key width")
        local = []
        for _ in range(events):
            raw = torch.randn(
                key_width, per_event, generator=generator, dtype=torch.float64
            )
            basis, triangular = torch.linalg.qr(raw, mode="reduced")
            signs = torch.sign(torch.diagonal(triangular)).clamp_min(0).mul(2).sub(1)
            local.append(basis * signs)
        projection = torch.zeros(rows, maximum_rank, dtype=torch.float64)
        for column in range(maximum_rank):
            event = column % events
            local_column = column // events
            start = event * key_width
            projection[start : start + key_width, column] = local[event][
                :, local_column
            ]
    return projection.to(device=device, dtype=dtype)


def bank_adaptive_basis(
    cross_image: SketchCrossImage,
    projection: torch.Tensor,
    *,
    requested_rank: int,
    mode: str,
    relative_singular_floor: float = 1e-6,
) -> BankAdaptiveBasis:
    """Project the current native/key image, then retain one nested SVD prefix."""

    events, native_width, key_width = cross_image.key_images.shape
    if (
        projection.shape[0] != events * key_width
        or not 0 < requested_rank <= projection.shape[1]
        or mode not in {"global", "per_event"}
        or not 0.0 < relative_singular_floor < 1.0
    ):
        raise BankConditioningError("bank-adaptive basis contract changed")
    flattened = cross_image.key_images.permute(1, 0, 2).reshape(
        native_width, events * key_width
    )
    projected = flattened @ projection
    vectors, singular, _ = torch.linalg.svd(projected, full_matrices=False)
    maximum = singular[0].clamp_min(torch.finfo(singular.dtype).tiny)
    supported = int((singular > maximum * float(relative_singular_floor)).sum())
    retained = min(int(requested_rank), supported, vectors.shape[1])
    if retained <= 0:
        raise BankConditioningError("bank-adaptive basis retained no direction")
    return BankAdaptiveBasis(
        vectors=vectors[:, :retained].detach(),
        singular_values=singular.detach(),
        requested_rank=int(requested_rank),
        retained_rank=retained,
        mode=mode,
    )


class StreamingProjectedFunctionalStatistics:
    """Accumulate C0 and rank-specific Cr after the bank basis is known."""

    def __init__(
        self,
        basis: BankAdaptiveBasis,
        *,
        ranks: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        leading_block_size: int = 4,
    ) -> None:
        if min(ranks, leading_block_size) <= 0 or dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise BankConditioningError("invalid projected-statistics topology")
        self.basis = basis
        self.ranks = int(ranks)
        self.dtype = dtype
        self.leading_block_size = int(leading_block_size)
        width = basis.retained_rank
        self.total_mass = torch.zeros((), device=device, dtype=dtype)
        self.first = torch.zeros(width, device=device, dtype=dtype)
        self.second = torch.zeros(width, width, device=device, dtype=dtype)
        self.replay_mass = torch.zeros(ranks, device=device, dtype=dtype)
        self.replay_first = torch.zeros(ranks, width, device=device, dtype=dtype)
        self.replay_native_first = torch.zeros(
            ranks, basis.vectors.shape[0], device=device, dtype=dtype
        )
        self.replay_second = torch.zeros(
            ranks, width, width, device=device, dtype=dtype
        )
        self.replay_cross = torch.zeros(
            ranks, basis.vectors.shape[0], width, device=device, dtype=dtype
        )
        self.candidate_count = 0
        self._pending: tuple[torch.Tensor, ...] | None = None

    def _consume(
        self,
        values: torch.Tensor,
        base_mass: torch.Tensor,
        replay_mass: torch.Tensor,
    ) -> None:
        value = values.reshape(-1, self.basis.vectors.shape[0])
        projected = value @ self.basis.vectors.to(value)
        base = base_mass.reshape(-1)
        replay = replay_mass.reshape(self.ranks, -1)
        self.total_mass += base.sum()
        self.first += torch.einsum("n,nd->d", base, projected)
        self.second += projected.T @ (base[:, None] * projected)
        self.replay_mass += replay.sum(-1)
        self.replay_first += torch.einsum("rn,nd->rd", replay, projected)
        self.replay_native_first += torch.einsum("rn,nd->rd", replay, value)
        self.replay_second += torch.einsum(
            "rn,nd,nf->rdf", replay, projected, projected
        )
        self.replay_cross += torch.einsum(
            "rn,nd,ns->rds", replay, value, projected
        )
        self.candidate_count += int(projected.shape[0])

    def add(
        self,
        values: torch.Tensor,
        base_mass: torch.Tensor,
        replay_mass: torch.Tensor,
    ) -> None:
        candidate_shape = values.shape[:-1]
        if (
            values.ndim < 2
            or values.shape[-1] != self.basis.vectors.shape[0]
            or base_mass.shape != candidate_shape
            or replay_mass.shape != (self.ranks, *candidate_shape)
            or values.numel() <= 0
        ):
            raise BankConditioningError("projected-statistics candidate axes changed")
        value = values.detach().to(dtype=self.dtype)
        projected = (
            value.reshape(-1, self.basis.vectors.shape[0])
            @ self.basis.vectors.to(value)
        )
        base = base_mass.detach().to(dtype=self.dtype)
        replay = replay_mass.detach().to(dtype=self.dtype)
        if (
            torch.any(base < 0)
            or torch.any(replay < 0)
            or not bool(torch.isfinite(projected).all())
        ):
            raise BankConditioningError("projected-statistics stream is invalid")
        if self._pending is not None:
            old_value, old_base, old_replay = self._pending
            value = torch.cat((old_value, value), dim=0)
            base = torch.cat((old_base, base), dim=0)
            replay = torch.cat((old_replay, replay), dim=1)
            self._pending = None
        stop = value.shape[0] - value.shape[0] % self.leading_block_size
        for start in range(0, stop, self.leading_block_size):
            end = start + self.leading_block_size
            self._consume(
                value[start:end], base[start:end], replay[:, start:end]
            )
        if stop < value.shape[0]:
            self._pending = (
                value[stop:], base[stop:], replay[:, stop:]
            )

    def finalize(self, cross_image: SketchCrossImage) -> SketchedFunctionalStatistics:
        if self._pending is not None:
            self._consume(*self._pending)
            self._pending = None
        if (
            self.candidate_count <= 1
            or float(self.total_mass) <= 0.0
            or torch.any(self.replay_mass <= 0)
            or cross_image.mean.shape[0] != self.basis.vectors.shape[0]
        ):
            raise BankConditioningError("projected-statistics stream is empty")
        mean = self.first / self.total_mass
        covariance = self.second / self.total_mass - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        replay_mean = self.replay_first / self.replay_mass[:, None]
        replay_native_mean = self.replay_native_first / self.replay_mass[:, None]
        replay_covariance = self.replay_second / self.replay_mass[:, None, None]
        replay_covariance -= torch.einsum("rd,re->rde", replay_mean, replay_mean)
        replay_covariance = 0.5 * (
            replay_covariance + replay_covariance.transpose(-1, -2)
        )
        replay_images = self.replay_cross / self.replay_mass[:, None, None]
        replay_images -= torch.einsum(
            "rd,rs->rds", replay_native_mean, replay_mean
        )
        key_images = torch.einsum(
            "ds,edm->esm",
            self.basis.vectors.to(cross_image.key_images),
            cross_image.key_images,
        )
        if not all(
            bool(torch.isfinite(value).all())
            for value in (covariance, replay_covariance, replay_images, key_images)
        ):
            raise BankConditioningError("projected statistics are non-finite")
        return SketchedFunctionalStatistics(
            mean=cross_image.mean,
            basis=self.basis.vectors,
            covariance=covariance,
            replay_covariances=replay_covariance,
            replay_images=replay_images,
            key_images=key_images,
            total_mass=self.total_mass,
            projection_singular_values=self.basis.singular_values,
        )


def _symmetric_pseudoinverse(
    matrix: torch.Tensor, *, relative_floor: float
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    maximum = eigenvalues[-1].clamp_min(torch.finfo(matrix.dtype).tiny)
    keep = eigenvalues > maximum * float(relative_floor)
    if int(keep.sum()) <= 0:
        raise BankConditioningError("sketched covariance retained no direction")
    safe = torch.where(keep, eigenvalues, torch.ones_like(eigenvalues))
    return (eigenvectors * (safe.reciprocal() * keep)) @ eigenvectors.T


def functional_target_queries(
    desired: torch.Tensor,
    statistics: SketchedFunctionalStatistics,
    *,
    relative_floor: float = 1e-6,
) -> FunctionalTargetQueries:
    """Solve a task-local target in the sketched functional image.

    This is the S1/S2-free-query positive control. It consumes a privileged
    desired factor and must never be used by the shared deployment forward.
    """

    ranks = statistics.replay_images.shape[0]
    native_width, reduced_width = statistics.basis.shape
    if (
        desired.ndim < 2
        or desired.shape[-2:] != (ranks, native_width)
        or statistics.replay_covariances.shape
        != (ranks, reduced_width, reduced_width)
        or statistics.replay_images.shape
        != (ranks, native_width, reduced_width)
        or not 0.0 < relative_floor < 1.0
    ):
        raise BankConditioningError("functional target-query contract changed")
    reduced_queries = []
    recoveries = []
    for rank in range(ranks):
        image = statistics.replay_images[rank]
        target = desired[..., rank, :].to(image)
        gram_inverse = _symmetric_pseudoinverse(
            image.T @ image, relative_floor=relative_floor
        )
        reduced = (target @ image) @ gram_inverse
        response = reduced @ image.T
        recovery = torch.nn.functional.cosine_similarity(
            response.float(), target.float(), dim=-1
        )
        reduced_queries.append(reduced)
        recoveries.append(recovery)
    reduced = torch.stack(reduced_queries, dim=-2)
    native = reduced @ statistics.basis.T
    return FunctionalTargetQueries(
        native=native.to(desired),
        reduced=reduced.to(desired),
        linear_recovery=torch.stack(recoveries, dim=-1),
    )
