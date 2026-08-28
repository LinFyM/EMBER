"""Current-bank functional coordinates for Program/native signed pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.operator import (
    BankConditioningError,
    BankStatistics,
)


@dataclass(frozen=True)
class FunctionalBankStatistics:
    """Detached native and key moments under one base and rank measures."""

    mean: torch.Tensor
    covariance: torch.Tensor
    replay_covariances: torch.Tensor
    key_images: torch.Tensor
    total_mass: torch.Tensor


@dataclass(frozen=True)
class FunctionalPolarQueries:
    """Unbounded projected queries plus compact numerical diagnostics."""

    queries: torch.Tensor
    metrics: torch.Tensor


class StreamingFunctionalBankStatistics:
    """Accumulate C0, rank-specific Cr, and event native/key images."""

    def __init__(
        self,
        *,
        native_width: int,
        key_width: int,
        events: int,
        ranks: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if min(native_width, key_width, events, ranks) <= 0:
            raise BankConditioningError("invalid functional-statistics topology")
        self.native_width = int(native_width)
        self.key_width = int(key_width)
        self.events = int(events)
        self.ranks = int(ranks)
        self.dtype = dtype
        self.total_mass = torch.zeros((), device=device, dtype=dtype)
        self.first = torch.zeros(native_width, device=device, dtype=dtype)
        self.second = torch.zeros(
            native_width, native_width, device=device, dtype=dtype
        )
        self.replay_mass = torch.zeros(ranks, device=device, dtype=dtype)
        self.replay_first = torch.zeros(
            ranks, native_width, device=device, dtype=dtype
        )
        self.replay_second = torch.zeros(
            ranks, native_width, native_width, device=device, dtype=dtype
        )
        self.event_mass = torch.zeros(events, device=device, dtype=dtype)
        self.event_key = torch.zeros(
            events, key_width, device=device, dtype=dtype
        )
        self.event_cross = torch.zeros(
            events, native_width, key_width, device=device, dtype=dtype
        )
        self.candidate_count = 0

    def add(
        self,
        values: torch.Tensor,
        base_mass: torch.Tensor,
        replay_mass: torch.Tensor,
        event_mass: torch.Tensor,
        event_keys: torch.Tensor,
    ) -> None:
        candidate_shape = values.shape[:-1]
        if (
            values.shape[-1] != self.native_width
            or base_mass.shape != candidate_shape
            or replay_mass.shape != (self.ranks, *candidate_shape)
            or event_mass.shape != (self.events, *candidate_shape)
            or event_keys.shape
            != (self.events, *candidate_shape, self.key_width)
            or values.numel() <= 0
        ):
            raise BankConditioningError("functional-statistics candidate axes changed")
        value = values.detach().to(dtype=self.dtype).reshape(-1, self.native_width)
        base = base_mass.detach().to(dtype=self.dtype).reshape(-1)
        replay = replay_mass.detach().to(dtype=self.dtype).reshape(self.ranks, -1)
        event = event_mass.detach().to(dtype=self.dtype).reshape(self.events, -1)
        key = event_keys.detach().to(dtype=self.dtype).reshape(
            self.events, -1, self.key_width
        )
        if (
            torch.any(base < 0)
            or torch.any(replay < 0)
            or torch.any(event < 0)
            or not bool(torch.isfinite(value).all())
            or not bool(torch.isfinite(key).all())
        ):
            raise BankConditioningError("functional statistics are invalid")
        self.total_mass += base.sum()
        self.first += torch.einsum("n,nd->d", base, value)
        self.second += value.T @ (base[:, None] * value)
        self.replay_mass += replay.sum(-1)
        self.replay_first += torch.einsum("rn,nd->rd", replay, value)
        self.replay_second += torch.einsum(
            "rn,nd,nf->rdf", replay, value, value
        )
        self.event_mass += event.sum(-1)
        self.event_key += torch.einsum("en,enw->ew", event, key)
        self.event_cross += torch.einsum("en,nd,enw->edw", event, value, key)
        self.candidate_count += int(value.shape[0])

    def finalize(self) -> FunctionalBankStatistics:
        if (
            self.candidate_count <= 1
            or float(self.total_mass) <= 0.0
            or torch.any(self.replay_mass <= 0)
            or torch.any(self.event_mass <= 0)
        ):
            raise BankConditioningError("functional-statistics stream is empty")
        mean = self.first / self.total_mass
        covariance = self.second / self.total_mass - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        replay_mean = self.replay_first / self.replay_mass[:, None]
        replay_covariance = self.replay_second / self.replay_mass[:, None, None]
        replay_covariance -= torch.einsum(
            "rd,re->rde", replay_mean, replay_mean
        )
        replay_covariance = 0.5 * (
            replay_covariance + replay_covariance.transpose(-1, -2)
        )
        # Event measures are normalized to the same mass as the base measure.
        key_images = self.event_cross / self.total_mass
        key_images -= torch.einsum(
            "d,ew->edw", mean, self.event_key / self.total_mass
        )
        if not all(
            bool(torch.isfinite(value).all())
            for value in (covariance, replay_covariance, key_images)
        ):
            raise BankConditioningError("functional statistics are non-finite")
        return FunctionalBankStatistics(
            mean=mean,
            covariance=covariance,
            replay_covariances=replay_covariance,
            key_images=key_images,
            total_mass=self.total_mass,
        )


class StreamingCenteredAnchor:
    """Accumulate differentiable native anchors against detached bank moments."""

    def __init__(
        self,
        statistics: FunctionalBankStatistics,
        *,
        query_shape: tuple[int, ...],
    ) -> None:
        if not query_shape or min(query_shape) <= 0:
            raise BankConditioningError("invalid centered-anchor topology")
        self.statistics = statistics
        self.query_shape = tuple(map(int, query_shape))
        self.anchor = statistics.mean.new_zeros(
            *self.query_shape, statistics.mean.numel()
        )
        self.total_mass = statistics.mean.new_zeros(())
        self.candidate_count = 0

    def add(
        self,
        values: torch.Tensor,
        mass: torch.Tensor,
        compatibility: torch.Tensor,
    ) -> None:
        if (
            values.shape[:-1] != mass.shape
            or values.shape[-1] != self.statistics.mean.numel()
            or compatibility.shape != (*self.query_shape, *mass.shape)
        ):
            raise BankConditioningError("centered-anchor candidate axes changed")
        value = values.detach().to(self.statistics.mean).reshape(
            -1, self.statistics.mean.numel()
        )
        centered = value - self.statistics.mean
        weight = mass.detach().to(self.statistics.mean).reshape(-1)
        score = compatibility.to(self.statistics.mean).reshape(-1, weight.numel())
        self.anchor += torch.einsum(
            "n,nd,qn->qd", weight, centered, score
        ).reshape_as(self.anchor)
        self.total_mass += weight.sum()
        self.candidate_count += int(weight.numel())

    def finalize(self) -> BankStatistics:
        if self.candidate_count <= 1 or float(self.total_mass.detach()) <= 0.0:
            raise BankConditioningError("centered-anchor stream is empty")
        anchor = self.anchor / self.total_mass
        if not bool(torch.isfinite(anchor).all()):
            raise BankConditioningError("centered anchor is non-finite")
        return BankStatistics(
            mean=self.statistics.mean,
            covariance=self.statistics.covariance,
            anchor=anchor,
            total_mass=self.total_mass,
        )


def functional_polar_queries(
    raw_queries: torch.Tensor,
    event_weights: torch.Tensor,
    statistics: FunctionalBankStatistics,
    *,
    covariance_floor: float,
    image_floor: float,
) -> FunctionalPolarQueries:
    """Map Program queries through the polar gauge of the actual B0/B1 operator."""

    if (
        raw_queries.ndim != 4
        or raw_queries.shape[:2] != event_weights.shape
        or raw_queries.shape[2] != 2
        or raw_queries.shape[-1] != statistics.key_images.shape[-1]
        or statistics.key_images.shape[0] != event_weights.shape[1]
        or statistics.replay_covariances.shape[0] != event_weights.shape[0]
        or not 0.0 < covariance_floor < 1.0
        or not 0.0 < image_floor < 1.0
    ):
        raise BankConditioningError("functional-polar query contract changed")

    covariance = statistics.covariance.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance_maximum = eigenvalues[-1].clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    covariance_keep = eigenvalues > covariance_maximum * covariance_floor
    if not torch.any(covariance_keep):
        raise BankConditioningError("functional polar retained no covariance")
    basis = eigenvectors[:, covariance_keep]
    images = statistics.key_images.double()
    coordinates = torch.einsum("di,edw->eiw", basis, images)
    dual_images = torch.einsum(
        "di,eiw->edw",
        basis,
        coordinates / eigenvalues[covariance_keep][None, :, None],
    )

    functional = torch.einsum(
        "rdf,efw->redw",
        statistics.replay_covariances.double(),
        dual_images,
    )
    image = (
        functional * event_weights.detach().double()[:, :, None, None]
    ).movedim(1, 2).flatten(2)
    _, singular, right = torch.linalg.svd(image, full_matrices=False)
    maximum = singular[:, :1].clamp_min(torch.finfo(singular.dtype).tiny)
    keep = singular > maximum * image_floor
    if torch.any(keep.sum(-1) <= 0):
        raise BankConditioningError("functional polar retained no image")
    safe_singular = torch.where(keep, singular, torch.ones_like(singular))
    inverse = safe_singular.reciprocal() * keep
    raw = raw_queries.permute(0, 2, 1, 3).flatten(2).double()
    coordinates = torch.einsum("rbm,rkm->rbk", raw, right)
    coefficient = torch.einsum(
        "rbk,rk,rkm->rbm", coordinates, inverse, right
    )
    replay = torch.einsum("rdm,rbm->rbd", image, coefficient)
    polar_basis = torch.einsum("rdm,rkm->rdk", image, right)
    polar_basis = polar_basis * inverse[:, None]
    polar_target = torch.einsum(
        "rbk,rdk->rbd", coordinates * keep[:, None], polar_basis
    )
    residual = (replay - polar_target).norm(dim=-1) / polar_target.norm(
        dim=-1
    ).clamp_min(1e-30)
    retained_ratio = torch.where(
        keep, singular / maximum, torch.full_like(singular, torch.inf)
    )
    metrics = torch.stack(
        (
            keep.sum(-1).min(),
            retained_ratio.amin(),
            residual.detach().amax(),
            covariance_keep.sum(),
        )
    ).to(raw_queries)
    projected = coefficient.reshape(
        raw_queries.shape[0], 2, raw_queries.shape[1], raw_queries.shape[-1]
    ).permute(0, 2, 1, 3)
    return FunctionalPolarQueries(
        queries=projected.to(raw_queries), metrics=metrics
    )


def bound_functional_queries(
    query_groups: Sequence[torch.Tensor], *, score_bound: float
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Apply one conservative scale per target/rank across all groups."""

    rows = tuple(query_groups)
    if not rows or not 0.0 < score_bound <= 0.01:
        raise BankConditioningError("functional trust-region contract changed")
    ranks = rows[0].shape[0]
    if any(row.ndim != 4 or row.shape[0] != ranks for row in rows):
        raise BankConditioningError("functional query groups changed")
    maximum = torch.stack(
        tuple(row.float().norm(dim=-1).flatten(1).amax(-1) for row in rows)
    ).amax(0)
    scale = maximum.detach().clamp_min(1e-30).reciprocal() * float(score_bound)
    return tuple(row * scale[:, None, None, None] for row in rows), scale


def normalize_replay_queries(
    query_groups: Sequence[torch.Tensor],
    statistics: Sequence[FunctionalBankStatistics],
    *,
    score_rms: float,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Use one B1 covariance-score scale per target/rank across all groups."""

    queries = tuple(query_groups)
    banks = tuple(statistics)
    if (
        not queries
        or len(queries) != len(banks)
        or not 0.0 < score_rms <= 0.1
    ):
        raise BankConditioningError("replay normalization contract changed")
    rms = torch.stack(
        tuple(
            torch.einsum(
                "rbd,de,rbe->rb",
                query.double(),
                bank.covariance.double(),
                query.double(),
            )
            .clamp_min(0)
            .sqrt()
            for query, bank in zip(queries, banks, strict=True)
        )
    )
    gain = float(score_rms) / rms.detach().amax(dim=(0, 2)).clamp_min(1e-30)
    return (
        tuple(query * gain.to(query)[:, None, None] for query in queries),
        gain.to(queries[0]),
    )
