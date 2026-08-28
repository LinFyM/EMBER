"""Global native-bank covariance and primal-to-dual conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.contracts import ACTION_HORIZON
from ember.ecp.native_factors import G1_PROBE_COUNT, OUTPUT_BANK_TYPES


def native_candidate_mass(
    frame_mass: torch.Tensor, *, output: bool
) -> torch.Tensor:
    """Lift one normalized frame measure onto the true native candidate axes."""

    if frame_mass.ndim != 1 or frame_mass.numel() <= 0:
        raise BankConditioningError("native frame measure changed")
    shape = (frame_mass.shape[0], G1_PROBE_COUNT, ACTION_HORIZON)
    if output:
        shape = (*shape, len(OUTPUT_BANK_TYPES))
    return frame_mass.reshape(frame_mass.shape[0], *((1,) * (len(shape) - 1))).expand(
        shape
    )


@dataclass(frozen=True)
class NativeCovarianceStatistics:
    mean: torch.Tensor
    covariance: torch.Tensor
    total_mass: torch.Tensor
    candidate_count: int


class StreamingNativeCovariance:
    """Stable weighted covariance without materializing the candidate bank."""

    def __init__(
        self,
        *,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        canonical_block_candidates: int | None = None,
    ) -> None:
        if (
            width <= 0
            or dtype not in (torch.float32, torch.float64)
            or (
                canonical_block_candidates is not None
                and canonical_block_candidates <= 0
            )
        ):
            raise BankConditioningError("invalid native covariance topology")
        self.width = int(width)
        self.dtype = dtype
        self.canonical_block_candidates = (
            None
            if canonical_block_candidates is None
            else int(canonical_block_candidates)
        )
        self.total_mass = torch.zeros((), device=device, dtype=dtype)
        self.mean = torch.zeros(width, device=device, dtype=dtype)
        self.centered_sum = torch.zeros(width, width, device=device, dtype=dtype)
        self.candidate_count = 0
        self._pending_values = torch.empty(0, width, device=device, dtype=dtype)
        self._pending_weights = torch.empty(0, device=device, dtype=dtype)

    def _add_block(self, flat: torch.Tensor, weights: torch.Tensor) -> None:
        chunk_mass = weights.sum()

        chunk_mean = torch.einsum("n,nd->d", weights, flat) / chunk_mass
        centered = flat - chunk_mean
        chunk_sum = centered.T @ (weights[:, None] * centered)
        if self.candidate_count == 0:
            self.mean = chunk_mean
            self.centered_sum = chunk_sum
            self.total_mass = chunk_mass
        else:
            combined = self.total_mass + chunk_mass
            delta = chunk_mean - self.mean
            correction = self.total_mass * chunk_mass / combined
            self.centered_sum = (
                self.centered_sum
                + chunk_sum
                + correction * torch.outer(delta, delta)
            )
            self.mean = self.mean + delta * (chunk_mass / combined)
            self.total_mass = combined
        self.candidate_count += int(flat.shape[0])

    def add(self, values: torch.Tensor, mass: torch.Tensor) -> None:
        if (
            values.ndim < 2
            or values.shape[-1] != self.width
            or mass.shape != values.shape[:-1]
            or values.numel() <= 0
        ):
            raise BankConditioningError("native covariance candidate axes changed")
        flat = values.detach().to(dtype=self.dtype).reshape(-1, self.width)
        weights = mass.detach().to(dtype=self.dtype).reshape(-1)

        # The quadrature owner constructs a finite, strictly-positive measure.
        # Do not turn every target/chunk into a host synchronization by
        # rechecking those scalar facts here; the retained F0 checks the joined
        # compiler output once after all banks have been consumed.
        if self.canonical_block_candidates is None:
            self._add_block(flat, weights)
            return
        flat = torch.cat((self._pending_values, flat))
        weights = torch.cat((self._pending_weights, weights))
        block = self.canonical_block_candidates
        complete = (flat.shape[0] // block) * block
        for start in range(0, complete, block):
            self._add_block(
                flat[start : start + block], weights[start : start + block]
            )
        if complete == flat.shape[0]:
            self._pending_values = self._pending_values.new_empty((0, self.width))
            self._pending_weights = self._pending_weights.new_empty((0,))
        else:
            # Own only the remainder; retaining a view would pin the entire
            # just-consumed native chunk for every target until the next read.
            self._pending_values = flat[complete:].clone()
            self._pending_weights = weights[complete:].clone()

    def finalize(self) -> NativeCovarianceStatistics:
        if self._pending_values.shape[0] > 0:
            self._add_block(self._pending_values, self._pending_weights)
            self._pending_values = self._pending_values.new_empty((0, self.width))
            self._pending_weights = self._pending_weights.new_empty((0,))
        if self.candidate_count <= 1:
            raise BankConditioningError("native covariance stream is empty")
        covariance = self.centered_sum / self.total_mass
        covariance = 0.5 * (covariance + covariance.T)
        return NativeCovarianceStatistics(
            mean=self.mean,
            covariance=covariance,
            total_mass=self.total_mass,
            candidate_count=self.candidate_count,
        )


@dataclass(frozen=True)
class SpectralNativeCovariance:
    """Detached retained eigenspace that maps primal rows to native duals."""

    basis: torch.Tensor
    eigenvalues: torch.Tensor
    native_width: int
    retained_rank: int
    eigenvalue_floor: torch.Tensor
    retained_condition: torch.Tensor
    retained_trace_fraction: torch.Tensor

    def dual_and_score_rms(
        self, primal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if primal.ndim != 2 or primal.shape[-1] != self.native_width:
            raise BankConditioningError("native primal width changed")
        basis = self.basis.to(primal)
        eigenvalues = self.eigenvalues.to(primal)
        coordinates = primal.float() @ basis.float()
        query = (coordinates / eigenvalues.float()[None]) @ basis.float().T
        score_rms = (
            coordinates.square() / eigenvalues.float()[None]
        ).sum(-1).clamp_min(0).sqrt()
        projected = coordinates @ basis.float().T
        projection = projected.norm(dim=-1) / primal.float().norm(
            dim=-1
        ).clamp_min(1e-30)
        return query.to(primal), score_rms.to(primal), projection.to(primal)


def batched_spectral_native_covariances(
    statistics: Sequence[NativeCovarianceStatistics],
    *,
    relative_eigenvalue_floor: float,
) -> tuple[SpectralNativeCovariance, ...]:
    """Factor same-width covariances in one IEEE eigensystem."""

    rows = tuple(statistics)
    if not rows or not 0.0 < relative_eigenvalue_floor < 1.0:
        raise BankConditioningError("invalid native covariance solve contract")
    shape = rows[0].covariance.shape
    dtype = rows[0].covariance.dtype
    if (
        len(shape) != 2
        or shape[0] != shape[1]
        or dtype not in (torch.float32, torch.float64)
        or any(row.covariance.shape != shape for row in rows)
    ):
        raise BankConditioningError("batched native covariance shape changed")
    covariance = torch.stack(tuple(row.covariance.detach() for row in rows))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    maximum = eigenvalues[:, -1].clamp_min(torch.finfo(dtype).tiny)
    floors = maximum * float(relative_eigenvalue_floor)
    keep = eigenvalues > floors[:, None]
    if torch.any(keep.sum(-1) <= 0):
        raise BankConditioningError("native covariance retained no direction")
    positive = eigenvalues.clamp_min(0)
    results = []
    for index in range(len(rows)):
        mask = keep[index]
        retained = eigenvalues[index, mask]
        trace = retained.sum() / positive[index].sum().clamp_min(1e-30)
        results.append(
            SpectralNativeCovariance(
                basis=eigenvectors[index][:, mask],
                eigenvalues=retained,
                native_width=shape[0],
                retained_rank=retained.shape[0],
                eigenvalue_floor=floors[index].detach(),
                retained_condition=(retained[-1] / retained[0]).detach(),
                retained_trace_fraction=trace.detach(),
            )
        )
    return tuple(results)
