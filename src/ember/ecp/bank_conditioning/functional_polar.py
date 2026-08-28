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


@dataclass(frozen=True)
class _PolarBatch:
    raw_rows: tuple[torch.Tensor, ...]
    statistics_rows: tuple[FunctionalBankStatistics, ...]
    reference: torch.Tensor
    functional: torch.Tensor
    weights: torch.Tensor
    covariance_keep: torch.Tensor
    compute_dtype: torch.dtype


@dataclass(frozen=True)
class _PolarSpectrum:
    singular: torch.Tensor
    right_rows: tuple[torch.Tensor, ...]
    maximum: torch.Tensor
    keep: torch.Tensor
    inverse: torch.Tensor
    matrices_per_batch: int


def _economy_svd_right(
    matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return singular values/right vectors through a stable thin QR."""

    rows, columns = matrices.shape[-2:]
    if rows >= 2 * columns:
        _, triangular = torch.linalg.qr(matrices, mode="reduced")
        _, singular, right = torch.linalg.svd(
            triangular, full_matrices=False
        )
        return singular, right
    if columns >= 2 * rows:
        basis, triangular = torch.linalg.qr(
            matrices.transpose(-1, -2), mode="reduced"
        )
        _, singular, local_right = torch.linalg.svd(
            triangular.transpose(-1, -2), full_matrices=False
        )
        right = local_right @ basis.transpose(-1, -2)
        return singular, right
    _, singular, right = torch.linalg.svd(matrices, full_matrices=False)
    return singular, right


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
        leading_block_size: int = 4,
    ) -> None:
        if min(native_width, key_width, events, ranks, leading_block_size) <= 0:
            raise BankConditioningError("invalid functional-statistics topology")
        self.native_width = int(native_width)
        self.key_width = int(key_width)
        self.events = int(events)
        self.ranks = int(ranks)
        self.dtype = dtype
        self.leading_block_size = int(leading_block_size)
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
        self._pending: tuple[torch.Tensor, ...] | None = None

    def _consume(
        self,
        value: torch.Tensor,
        base: torch.Tensor,
        replay: torch.Tensor,
        event: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        value = value.reshape(-1, self.native_width)
        base = base.reshape(-1)
        replay = replay.reshape(self.ranks, -1)
        event = event.reshape(self.events, -1)
        key = key.reshape(self.events, -1, self.key_width)
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
        value = values.detach().to(dtype=self.dtype)
        base = base_mass.detach().to(dtype=self.dtype)
        replay = replay_mass.detach().to(dtype=self.dtype)
        event = event_mass.detach().to(dtype=self.dtype)
        key = event_keys.detach().to(dtype=self.dtype)
        if (
            torch.any(base < 0)
            or torch.any(replay < 0)
            or torch.any(event < 0)
            or not bool(torch.isfinite(value).all())
            or not bool(torch.isfinite(key).all())
        ):
            raise BankConditioningError("functional statistics are invalid")
        if self._pending is not None:
            old_value, old_base, old_replay, old_event, old_key = self._pending
            value = torch.cat((old_value, value), dim=0)
            base = torch.cat((old_base, base), dim=0)
            replay = torch.cat((old_replay, replay), dim=1)
            event = torch.cat((old_event, event), dim=1)
            key = torch.cat((old_key, key), dim=1)
            self._pending = None
        stop = value.shape[0] - value.shape[0] % self.leading_block_size
        for start in range(0, stop, self.leading_block_size):
            end = start + self.leading_block_size
            self._consume(
                value[start:end],
                base[start:end],
                replay[:, start:end],
                event[:, start:end],
                key[:, start:end],
            )
        if stop < value.shape[0]:
            self._pending = (
                value[stop:],
                base[stop:],
                replay[:, stop:],
                event[:, stop:],
                key[:, stop:],
            )

    def finalize(self) -> FunctionalBankStatistics:
        if self._pending is not None:
            self._consume(*self._pending)
            self._pending = None
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
        leading_block_size: int = 4,
    ) -> None:
        if (
            not query_shape
            or min(query_shape) <= 0
            or leading_block_size <= 0
        ):
            raise BankConditioningError("invalid centered-anchor topology")
        self.statistics = statistics
        self.query_shape = tuple(map(int, query_shape))
        self.leading_block_size = int(leading_block_size)
        self.anchor = statistics.mean.new_zeros(
            *self.query_shape, statistics.mean.numel()
        )
        self.total_mass = statistics.mean.new_zeros(())
        self.candidate_count = 0
        self._pending: tuple[torch.Tensor, ...] | None = None

    def _consume(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        score: torch.Tensor,
    ) -> None:
        value = value.reshape(-1, self.statistics.mean.numel())
        centered = value - self.statistics.mean
        weight = weight.reshape(-1)
        score = score.reshape(-1, weight.numel())
        self.anchor = self.anchor + torch.einsum(
            "n,nd,qn->qd", weight, centered, score
        ).reshape_as(self.anchor)
        self.total_mass = self.total_mass + weight.sum()
        self.candidate_count += int(weight.numel())

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
        value = values.detach().to(self.statistics.mean)
        weight = mass.detach().to(self.statistics.mean)
        score = compatibility.to(self.statistics.mean)
        if self._pending is not None:
            old_value, old_weight, old_score = self._pending
            value = torch.cat((old_value, value), dim=0)
            weight = torch.cat((old_weight, weight), dim=0)
            score = torch.cat((old_score, score), dim=len(self.query_shape))
            self._pending = None
        stop = value.shape[0] - value.shape[0] % self.leading_block_size
        for start in range(0, stop, self.leading_block_size):
            end = start + self.leading_block_size
            selection = (slice(None),) * len(self.query_shape) + (
                slice(start, end),
            )
            self._consume(
                value[start:end], weight[start:end], score[selection]
            )
        if stop < value.shape[0]:
            selection = (slice(None),) * len(self.query_shape) + (
                slice(stop, None),
            )
            self._pending = (value[stop:], weight[stop:], score[selection])

    def finalize(self) -> BankStatistics:
        if self._pending is not None:
            self._consume(*self._pending)
            self._pending = None
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

    return batched_functional_polar_queries(
        (raw_queries,),
        (event_weights,),
        (statistics,),
        covariance_floor=covariance_floor,
        image_floor=image_floor,
    )[0]


def _validate_polar_batch(
    raw_rows: tuple[torch.Tensor, ...],
    weight_rows: tuple[torch.Tensor, ...],
    statistics_rows: tuple[FunctionalBankStatistics, ...],
    *,
    covariance_floor: float,
    image_floor: float,
    mode: str,
    maximum_svd_elements: int,
) -> tuple[torch.Tensor, int, int, torch.dtype]:
    if not raw_rows or not (
        len(raw_rows) == len(weight_rows) == len(statistics_rows)
    ):
        raise BankConditioningError("functional-polar batch is empty")
    reference = raw_rows[0]
    native_width = statistics_rows[0].mean.numel()
    key_width = statistics_rows[0].key_images.shape[-1]
    compute_dtype = statistics_rows[0].covariance.dtype
    valid_scalar_contract = (
        mode in {"global", "per_event"}
        and maximum_svd_elements > 0
        and 0.0 < covariance_floor < 1.0
        and 0.0 < image_floor < 1.0
        and compute_dtype in (torch.float32, torch.float64)
    )
    if (
        reference.ndim != 4
        or reference.shape[:2] != weight_rows[0].shape
        or reference.shape[2] != 2
        or reference.shape[-1] != key_width
        or not valid_scalar_contract
    ):
        raise BankConditioningError("functional-polar query contract changed")
    expected_keys = (reference.shape[1], native_width, key_width)
    expected_replay = (reference.shape[0], native_width, native_width)
    for raw, weights, statistics in zip(
        raw_rows, weight_rows, statistics_rows, strict=True
    ):
        if (
            raw.shape != reference.shape
            or weights.shape != weight_rows[0].shape
            or statistics.mean.numel() != native_width
            or statistics.covariance.shape != (native_width, native_width)
            or statistics.covariance.dtype != compute_dtype
            or statistics.key_images.shape != expected_keys
            or statistics.replay_covariances.shape != expected_replay
        ):
            raise BankConditioningError(
                "functional-polar batch contains incompatible native groups"
            )
    return reference, native_width, key_width, compute_dtype


def _prepare_polar_batch(
    raw_rows: tuple[torch.Tensor, ...],
    weight_rows: tuple[torch.Tensor, ...],
    statistics_rows: tuple[FunctionalBankStatistics, ...],
    *,
    covariance_floor: float,
    image_floor: float,
    mode: str,
    maximum_svd_elements: int,
) -> _PolarBatch:
    reference, _, _, compute_dtype = _validate_polar_batch(
        raw_rows,
        weight_rows,
        statistics_rows,
        covariance_floor=covariance_floor,
        image_floor=image_floor,
        mode=mode,
        maximum_svd_elements=maximum_svd_elements,
    )
    with torch.no_grad():
        covariance = torch.stack(
            tuple(row.covariance.detach() for row in statistics_rows)
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        maximum = eigenvalues[:, -1].clamp_min(torch.finfo(compute_dtype).tiny)
        keep = eigenvalues > maximum[:, None] * covariance_floor
        if torch.any(keep.sum(-1) <= 0):
            raise BankConditioningError("functional polar retained no covariance")
        safe = torch.where(keep, eigenvalues, torch.ones_like(eigenvalues))
        coordinates = torch.einsum(
            "gdi,gedw->geiw",
            eigenvectors,
            torch.stack(tuple(row.key_images.detach() for row in statistics_rows)),
        )
        dual_images = torch.einsum(
            "gdi,geiw->gedw",
            eigenvectors,
            coordinates * (safe.reciprocal() * keep)[:, None, :, None],
        )
        replay = torch.stack(
            tuple(row.replay_covariances.detach() for row in statistics_rows)
        )
        functional = torch.einsum("grdf,gefw->gredw", replay, dual_images)
        weights = torch.stack(weight_rows).detach().to(dtype=compute_dtype)
    return _PolarBatch(
        raw_rows=raw_rows,
        statistics_rows=statistics_rows,
        reference=reference,
        functional=functional,
        weights=weights,
        covariance_keep=keep,
        compute_dtype=compute_dtype,
    )


def _flatten_polar_batch(
    batch: _PolarBatch, *, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.stack(batch.raw_rows).to(dtype=batch.compute_dtype)
    if mode == "global":
        image = (batch.functional * batch.weights[:, :, :, None, None]).permute(
            0, 1, 3, 2, 4
        )
        return image.flatten(3).flatten(0, 1), raw.permute(
            0, 1, 3, 2, 4
        ).flatten(3).flatten(0, 1)
    return batch.functional.flatten(0, 2), raw.flatten(0, 2)


def _polar_spectrum(
    flat_image: torch.Tensor,
    *,
    groups: int,
    ranks: int,
    events: int,
    mode: str,
    image_floor: float,
    maximum_svd_elements: int,
) -> _PolarSpectrum:
    matrix_elements = flat_image.shape[-2] * flat_image.shape[-1]
    matrices_per_batch = max(1, maximum_svd_elements // max(matrix_elements, 1))
    singular_rows = []
    right_rows = []
    for start in range(0, flat_image.shape[0], matrices_per_batch):
        stop = min(start + matrices_per_batch, flat_image.shape[0])
        with torch.no_grad():
            singular, right = _economy_svd_right(flat_image[start:stop])
        singular_rows.append(singular)
        right_rows.append(right)
    singular = torch.cat(singular_rows)
    if mode == "global":
        maximum = singular[:, :1]
    else:
        maximum = singular[:, 0].reshape(groups, ranks, events).amax(
            -1, keepdim=True
        )
        maximum = maximum.expand(-1, -1, events).reshape(-1, 1)
    maximum = maximum.clamp_min(torch.finfo(singular.dtype).tiny)
    keep = singular > maximum * image_floor
    if torch.any(keep.sum(-1) <= 0):
        raise BankConditioningError("functional polar retained no image")
    safe = torch.where(keep, singular, torch.ones_like(singular))
    return _PolarSpectrum(
        singular=singular,
        right_rows=tuple(right_rows),
        maximum=maximum,
        keep=keep,
        inverse=safe.reciprocal() * keep,
        matrices_per_batch=matrices_per_batch,
    )


def _project_polar_batch(
    flat_image: torch.Tensor,
    flat_raw: torch.Tensor,
    spectrum: _PolarSpectrum,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = []
    residuals = []
    starts = range(0, flat_image.shape[0], spectrum.matrices_per_batch)
    for right, start in zip(spectrum.right_rows, starts, strict=True):
        stop = min(start + spectrum.matrices_per_batch, flat_image.shape[0])
        image = flat_image[start:stop]
        inverse = spectrum.inverse[start:stop]
        keep = spectrum.keep[start:stop]
        coordinates = torch.einsum("nbm,nkm->nbk", flat_raw[start:stop], right)
        coefficient = torch.einsum(
            "nbk,nk,nkm->nbm", coordinates, inverse, right
        )
        coefficients.append(coefficient)
        with torch.no_grad():
            replay = torch.einsum("ndm,nbm->nbd", image, coefficient.detach())
            basis = torch.einsum("ndm,nkm->ndk", image, right) * inverse[:, None]
            target = torch.einsum(
                "nbk,ndk->nbd", coordinates.detach() * keep[:, None], basis
            )
            residuals.append(
                ((replay - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-30))
                .amax(-1)
            )
    return torch.cat(coefficients), torch.cat(residuals)


def _format_polar_results(
    batch: _PolarBatch,
    spectrum: _PolarSpectrum,
    coefficient: torch.Tensor,
    residual: torch.Tensor,
    *,
    mode: str,
) -> tuple[FunctionalPolarQueries, ...]:
    groups = len(batch.raw_rows)
    ranks, events = batch.reference.shape[:2]
    ratio = torch.where(
        spectrum.keep,
        spectrum.singular / spectrum.maximum,
        torch.full_like(spectrum.singular, torch.inf),
    )
    if mode == "global":
        count = spectrum.keep.sum(-1).reshape(groups, ranks)
        retained = ratio.amin(-1).reshape(groups, ranks)
        error = residual.reshape(groups, ranks)
        projected = coefficient.reshape(
            groups, ranks, 2, events, batch.reference.shape[-1]
        ).permute(0, 1, 3, 2, 4)
    else:
        count = spectrum.keep.sum(-1).reshape(groups, ranks, events).amin(-1)
        retained = ratio.amin(-1).reshape(groups, ranks, events).amin(-1)
        error = residual.reshape(groups, ranks, events).amax(-1)
        projected = coefficient.reshape(
            groups, ranks, events, 2, batch.reference.shape[-1]
        )
    metrics = torch.stack(
        (
            count.amin(-1),
            retained.amin(-1),
            error.amax(-1),
            batch.covariance_keep.sum(-1),
        ),
        dim=-1,
    )
    return tuple(
        FunctionalPolarQueries(
            queries=projected[index].to(batch.raw_rows[index]),
            metrics=metrics[index].to(batch.raw_rows[index]),
        )
        for index in range(groups)
    )


def batched_functional_polar_queries(
    raw_query_groups: Sequence[torch.Tensor],
    event_weight_groups: Sequence[torch.Tensor],
    statistics_groups: Sequence[FunctionalBankStatistics],
    *,
    covariance_floor: float,
    image_floor: float,
    mode: str = "global",
    maximum_svd_elements: int = 64 * 1024 * 1024,
) -> tuple[FunctionalPolarQueries, ...]:
    """Polarize same-shaped native groups with bounded batched decompositions."""

    batch = _prepare_polar_batch(
        tuple(raw_query_groups),
        tuple(event_weight_groups),
        tuple(statistics_groups),
        covariance_floor=covariance_floor,
        image_floor=image_floor,
        mode=mode,
        maximum_svd_elements=maximum_svd_elements,
    )
    flat_image, flat_raw = _flatten_polar_batch(batch, mode=mode)
    ranks, events = batch.reference.shape[:2]
    spectrum = _polar_spectrum(
        flat_image,
        groups=len(batch.raw_rows),
        ranks=ranks,
        events=events,
        mode=mode,
        image_floor=image_floor,
        maximum_svd_elements=maximum_svd_elements,
    )
    coefficient, residual = _project_polar_batch(flat_image, flat_raw, spectrum)
    return _format_polar_results(
        batch, spectrum, coefficient, residual, mode=mode
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
                query.to(bank.covariance),
                bank.covariance,
                query.to(bank.covariance),
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
