"""Streaming sufficient statistics and spectral solve for Native-Factor banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


class BankConditioningError(RuntimeError):
    """Raised when a native-bank operator crosses its fixed contract."""


@dataclass(frozen=True)
class BankStatistics:
    mean: torch.Tensor
    covariance: torch.Tensor
    anchor: torch.Tensor
    total_mass: torch.Tensor


@dataclass(frozen=True)
class SpectralBankQuery:
    query: torch.Tensor
    retained_rank: int
    eigenvalue_floor: float
    retained_condition: float
    relative_residual_maximum: float
    anchor_projection_minimum: float
    retained_trace_fraction: float


class StreamingBankStatistics:
    """Accumulate one unit-mass bank without retaining its candidates.

    Native values and the base measure are frozen evidence.  Compatibility may
    carry gradients from the shared Program/candidate anchor scorer.  The
    centered anchor is recovered as E[v g] - E[v] E[g], so the mean does not
    need to be known while chunks are streamed.
    """

    def __init__(
        self,
        *,
        width: int,
        query_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if width <= 0 or not query_shape or min(query_shape) <= 0:
            raise BankConditioningError("invalid bank-statistics topology")
        self.width = int(width)
        self.query_shape = tuple(map(int, query_shape))
        self.dtype = dtype
        self.total_mass = torch.zeros((), device=device, dtype=dtype)
        self.first_moment = torch.zeros(width, device=device, dtype=dtype)
        self.second_moment = torch.zeros(
            width, width, device=device, dtype=dtype
        )
        self.compatibility_mass = torch.zeros(
            self.query_shape, device=device, dtype=dtype
        )
        self.native_anchor = torch.zeros(
            *self.query_shape, width, device=device, dtype=dtype
        )
        self.candidate_count = 0

    def add(
        self,
        values: torch.Tensor,
        mass: torch.Tensor,
        compatibility: torch.Tensor,
    ) -> None:
        if (
            values.ndim < 2
            or values.shape[-1] != self.width
            or mass.shape != values.shape[:-1]
            or compatibility.shape
            != (*self.query_shape, *mass.shape)
            or values.numel() <= 0
        ):
            raise BankConditioningError("bank-statistics candidate axes changed")
        flat_values = values.detach().to(dtype=self.dtype).reshape(-1, self.width)
        flat_mass = mass.detach().to(dtype=self.dtype).reshape(-1)
        flat_compatibility = compatibility.to(dtype=self.dtype).reshape(
            -1, flat_mass.numel()
        )
        if (
            torch.any(flat_mass < 0)
            or not bool(torch.isfinite(flat_values).all())
            or not bool(torch.isfinite(flat_mass).all())
            or not bool(torch.isfinite(flat_compatibility).all())
        ):
            raise BankConditioningError("bank-statistics received invalid values")
        self.total_mass = self.total_mass + flat_mass.sum()
        self.first_moment = self.first_moment + torch.einsum(
            "n,nd->d", flat_mass, flat_values
        )
        self.second_moment = self.second_moment + flat_values.T @ (
            flat_mass[:, None] * flat_values
        )
        self.compatibility_mass = self.compatibility_mass + torch.einsum(
            "n,qn->q", flat_mass, flat_compatibility
        ).reshape(self.query_shape)
        self.native_anchor = self.native_anchor + torch.einsum(
            "n,nd,qn->qd", flat_mass, flat_values, flat_compatibility
        ).reshape(*self.query_shape, self.width)
        self.candidate_count += int(flat_mass.numel())

    def finalize(self) -> BankStatistics:
        if (
            self.candidate_count <= 1
            or not bool(torch.isfinite(self.total_mass))
            or float(self.total_mass.detach()) <= 0.0
        ):
            raise BankConditioningError("bank-statistics stream is empty")
        mean = self.first_moment / self.total_mass
        covariance = self.second_moment / self.total_mass - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        anchor = self.native_anchor / self.total_mass
        anchor = anchor - mean * (
            self.compatibility_mass / self.total_mass
        )[..., None]
        if not bool(torch.isfinite(covariance).all()) or not bool(
            torch.isfinite(anchor).all()
        ):
            raise BankConditioningError("bank-statistics finalize is non-finite")
        return BankStatistics(
            mean=mean,
            covariance=covariance,
            anchor=anchor,
            total_mass=self.total_mass,
        )


def materialized_bank_statistics(
    values: torch.Tensor,
    mass: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> BankStatistics:
    query_ndim = compatibility.ndim - mass.ndim
    if query_ndim <= 0:
        raise BankConditioningError("materialized bank lost query axes")
    accumulator = StreamingBankStatistics(
        width=values.shape[-1],
        query_shape=tuple(compatibility.shape[:query_ndim]),
        device=values.device,
        dtype=dtype,
    )
    accumulator.add(values, mass, compatibility)
    return accumulator.finalize()


def spectral_bank_query(
    statistics: BankStatistics,
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> SpectralBankQuery:
    """Invert the retained covariance subspace without ridge shrinkage.

    The floor is the squared 1e-3 singular-value authority already used by the
    successful Native-Factor capacity solve.  A ridge at the same value changes
    high-gain functional directions; truncation removes only unsupported modes.
    """

    return batched_spectral_bank_query(
        (statistics,), relative_eigenvalue_floor=relative_eigenvalue_floor
    )[0]


def batched_spectral_bank_query(
    statistics: Sequence[BankStatistics],
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> tuple[SpectralBankQuery, ...]:
    """Solve same-width banks in one batched GPU eigensystem."""

    rows = tuple(statistics)
    if not rows or not 0.0 < relative_eigenvalue_floor < 1.0:
        raise BankConditioningError("invalid batched spectral solve contract")
    covariance_shape = rows[0].covariance.shape
    anchor_shape = rows[0].anchor.shape
    if any(
        row.covariance.shape != covariance_shape or row.anchor.shape != anchor_shape
        for row in rows
    ):
        raise BankConditioningError("batched spectral banks changed shape")
    compute_dtype = rows[0].covariance.dtype
    if compute_dtype not in (torch.float32, torch.float64):
        raise BankConditioningError("spectral bank precision changed")
    covariance = torch.stack(
        [row.covariance.detach().to(dtype=compute_dtype) for row in rows]
    )
    anchor = torch.stack(
        [row.anchor.to(dtype=compute_dtype) for row in rows]
    )
    if (
        covariance.ndim != 3
        or covariance.shape[1] != covariance.shape[2]
        or anchor.shape[-1] != covariance.shape[1]
    ):
        raise BankConditioningError("spectral bank solve shape changed")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    maximum = eigenvalues[:, -1].clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    floor = maximum * float(relative_eigenvalue_floor)
    keep = eigenvalues > floor[:, None]
    if torch.any(keep.sum(-1) <= 0):
        raise BankConditioningError("spectral bank solve retained no direction")
    flat_anchor = anchor.flatten(1, -2)
    coordinates = torch.einsum("bqd,bdr->bqr", flat_anchor, eigenvectors)
    coordinates = coordinates * keep[:, None]
    safe_eigenvalues = torch.where(keep, eigenvalues, torch.ones_like(eigenvalues))
    inverse = safe_eigenvalues.reciprocal() * keep
    query = torch.einsum(
        "bqr,bdr->bqd", coordinates * inverse[:, None], eigenvectors
    )
    projected_anchor = torch.einsum("bqr,bdr->bqd", coordinates, eigenvectors)
    replayed = torch.einsum("bqd,bde->bqe", query, covariance)
    residual = (replayed - projected_anchor).norm(dim=-1) / projected_anchor.norm(
        dim=-1
    ).clamp_min(1e-30)
    projection = projected_anchor.norm(dim=-1) / flat_anchor.norm(dim=-1).clamp_min(1e-30)
    positive_eigenvalues = eigenvalues.clamp_min(0)
    retained_trace = (positive_eigenvalues * keep).sum(-1) / positive_eigenvalues.sum(
        -1
    ).clamp_min(1e-30)
    if not bool(torch.isfinite(query).all()) or not bool(
        torch.isfinite(residual).all()
    ):
        raise BankConditioningError("spectral bank solve is non-finite")
    results = []
    for index, row in enumerate(rows):
        retained = eigenvalues[index, keep[index]]
        results.append(
            SpectralBankQuery(
                query=query[index].reshape_as(row.anchor).to(dtype=row.anchor.dtype),
                retained_rank=int(keep[index].sum()),
                eigenvalue_floor=float(floor[index]),
                retained_condition=float(retained[-1] / retained[0]),
                relative_residual_maximum=float(residual[index].detach().max()),
                anchor_projection_minimum=float(projection[index].detach().min()),
                retained_trace_fraction=float(retained_trace[index]),
            )
        )
    return tuple(results)


def bounded_relative_group_gain(score_maximum: torch.Tensor) -> torch.Tensor:
    """Preserve native output-group amplitudes with one common rank scale."""

    if score_maximum.ndim < 2 or score_maximum.shape[0] <= 0:
        raise BankConditioningError("output group gain axes changed")
    magnitude = score_maximum.clamp_min(0)
    denominator = magnitude.amax(0, keepdim=True).clamp_min(1e-12)
    gain = magnitude / denominator
    if not bool(torch.isfinite(gain).all()) or torch.any(gain > 1.0 + 1e-6):
        raise BankConditioningError("output group gain left its bounded contract")
    return gain.to(dtype=score_maximum.dtype)


class StreamingSignedPool:
    """Exact positive/negative online softmax under an explicit base measure.

    Capacity probes use one antithetic query.  The shared compiler can instead
    provide two independently solved branch queries.  A query-specific logit
    bias carries event-conditioned measures without copying native values.
    """

    def __init__(
        self,
        query: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        explicit_branches: bool = False,
        trusted_positive_measure: bool = False,
        trusted_finite_bias: bool = False,
        canonical_block_candidates: int | None = None,
    ):
        if (
            query.ndim < 2
            or query.shape[-1] <= 0
            or (
                canonical_block_candidates is not None
                and canonical_block_candidates <= 0
            )
        ):
            raise BankConditioningError("signed-pool query shape changed")
        if explicit_branches:
            if query.ndim < 3 or query.shape[-2] != 2:
                raise BankConditioningError("signed-pool explicit branches changed")
            self.query_shape = tuple(query.shape[:-2])
            branch_query = query
        else:
            self.query_shape = tuple(query.shape[:-1])
            branch_query = torch.stack((query, -query), dim=-2)
        self.width = int(query.shape[-1])
        self.query = branch_query.to(dtype=dtype)
        self.trusted_positive_measure = bool(trusted_positive_measure)
        self.trusted_finite_bias = bool(trusted_finite_bias)
        self.canonical_block_candidates = (
            None
            if canonical_block_candidates is None
            else int(canonical_block_candidates)
        )
        self.maximum = torch.full(
            (*self.query_shape, 2),
            -torch.inf,
            device=query.device,
            dtype=dtype,
        )
        self.normalizer = torch.zeros_like(self.maximum)
        self.weighted_sum = torch.zeros(
            *self.query_shape,
            2,
            self.width,
            device=query.device,
            dtype=dtype,
        )
        self.candidate_count = 0
        self._pending_values = torch.empty(
            0, self.width, device=query.device, dtype=dtype
        )
        self._pending_mass = torch.empty(0, device=query.device, dtype=dtype)
        self._pending_bias: torch.Tensor | None = None
        self._uses_bias: bool | None = None

    def _accumulated_state(
        self,
        flat_values: torch.Tensor,
        flat_mass: torch.Tensor,
        bias: torch.Tensor | None,
        maximum: torch.Tensor,
        normalizer: torch.Tensor,
        weighted_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = self.query.reshape(-1, 2, self.width) @ flat_values.T
        score = score.reshape(*self.query_shape, 2, flat_values.shape[0])
        log_mass = flat_mass.log().reshape(
            *((1,) * len(self.query_shape)), flat_values.shape[0]
        )
        logits = score + log_mass[..., None, :]
        if bias is not None:
            logits = logits + bias
        chunk_maximum = logits.amax(-1)
        next_maximum = torch.maximum(maximum, chunk_maximum)
        old_scale = torch.exp(maximum - next_maximum)
        weights = torch.exp(logits - next_maximum[..., None])
        next_weighted_sum = weighted_sum * old_scale[..., None] + torch.einsum(
            "...bn,nd->...bd", weights, flat_values
        )
        next_normalizer = normalizer * old_scale + weights.sum(-1)
        return next_maximum, next_normalizer, next_weighted_sum

    def _accumulate(
        self,
        flat_values: torch.Tensor,
        flat_mass: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        self.maximum, self.normalizer, self.weighted_sum = self._accumulated_state(
            flat_values,
            flat_mass,
            bias,
            self.maximum,
            self.normalizer,
            self.weighted_sum,
        )
        self.candidate_count += int(flat_values.shape[0])

    def add(
        self,
        values: torch.Tensor,
        mass: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> None:
        if (
            values.ndim < 2
            or values.shape[-1] != self.width
            or mass.shape != values.shape[:-1]
        ):
            raise BankConditioningError("signed-pool candidate axes changed")
        flat_values = values.detach().to(self.query).reshape(-1, self.width)
        flat_mass = mass.detach().to(self.query).reshape(-1)
        if not self.trusted_positive_measure and (
            torch.any(flat_mass <= 0) or not bool(torch.isfinite(flat_mass).all())
        ):
            raise BankConditioningError("signed-pool measure is not positive")
        bias = None
        if logit_bias is not None:
            shared_shape = (*self.query_shape, *mass.shape)
            branch_shape = (*self.query_shape, 2, *mass.shape)
            if logit_bias.shape == shared_shape:
                bias = logit_bias.to(self.query).reshape(
                    *self.query_shape, 1, flat_values.shape[0]
                ).expand(*self.query_shape, 2, flat_values.shape[0])
            elif logit_bias.shape == branch_shape:
                bias = logit_bias.to(self.query).reshape(
                    *self.query_shape, 2, flat_values.shape[0]
                )
            else:
                raise BankConditioningError("signed-pool logit bias axes changed")
            if not self.trusted_finite_bias and not bool(torch.isfinite(bias).all()):
                raise BankConditioningError("signed-pool logit bias is non-finite")
        uses_bias = bias is not None
        if self._uses_bias is None:
            self._uses_bias = uses_bias
        elif self._uses_bias != uses_bias:
            raise BankConditioningError("signed-pool bias mode changed mid-stream")
        if self.canonical_block_candidates is None:
            self._accumulate(flat_values, flat_mass, bias)
            return
        flat_values = torch.cat((self._pending_values, flat_values))
        flat_mass = torch.cat((self._pending_mass, flat_mass))
        if bias is not None:
            if self._pending_bias is not None:
                bias = torch.cat((self._pending_bias, bias), dim=-1)
        elif self._pending_bias is not None:
            raise BankConditioningError("signed-pool pending bias state changed")
        block = self.canonical_block_candidates
        complete = (flat_values.shape[0] // block) * block
        for start in range(0, complete, block):
            self._accumulate(
                flat_values[start : start + block],
                flat_mass[start : start + block],
                None if bias is None else bias[..., start : start + block],
            )
        if complete == flat_values.shape[0]:
            self._pending_values = self._pending_values.new_empty((0, self.width))
            self._pending_mass = self._pending_mass.new_empty((0,))
            self._pending_bias = None
        else:
            self._pending_values = flat_values[complete:].clone()
            self._pending_mass = flat_mass[complete:].clone()
            self._pending_bias = (
                None if bias is None else bias[..., complete:].clone()
            )

    def signed_mean(self) -> torch.Tensor:
        if self._pending_values.shape[0] > 0:
            self._accumulate(
                self._pending_values, self._pending_mass, self._pending_bias
            )
            self._pending_values = self._pending_values.new_empty((0, self.width))
            self._pending_mass = self._pending_mass.new_empty((0,))
            self._pending_bias = None
        if self.candidate_count <= 0 or (
            not self.trusted_positive_measure and torch.any(self.normalizer <= 0)
        ):
            raise BankConditioningError("signed-pool stream is empty")
        mean = self.weighted_sum / self.normalizer[..., None]
        return mean[..., 0, :] - mean[..., 1, :]


def materialized_signed_pool(
    query: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
    explicit_branches: bool = False,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    accumulator = StreamingSignedPool(
        query, dtype=dtype, explicit_branches=explicit_branches
    )
    accumulator.add(values, mass, logit_bias)
    return accumulator.signed_mean()
