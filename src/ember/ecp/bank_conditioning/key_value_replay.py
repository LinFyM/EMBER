"""Differentiable key geometry and exact real-value signed replay for PNBTT."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class DifferentiableKeyMoments:
    """One event-indexed current-bank key coordinate system."""

    mean: torch.Tensor
    covariance: torch.Tensor
    cholesky: torch.Tensor
    normalized_mass: torch.Tensor
    ridge: float


@dataclass(frozen=True)
class KeyValueSignedPoolResult:
    """Exact antithetic real-value transport and compact diagnostics."""

    direction: torch.Tensor
    score_rms: torch.Tensor
    positive_maximum_weight: torch.Tensor
    negative_maximum_weight: torch.Tensor


def safe_rms_normalize(
    value: torch.Tensor, *, epsilon: float = 1e-4
) -> torch.Tensor:
    """Unit-RMS away from zero with a finite, useful derivative at zero."""

    if value.ndim < 1 or epsilon <= 0.0:
        raise BankConditioningError("invalid safe query normalization")
    rms = (
        value.float().square().mean(-1, keepdim=True) + float(epsilon) ** 2
    ).sqrt()
    return value / rms.to(value)


def differentiable_key_moments(
    keys: torch.Tensor,
    event_mass: torch.Tensor,
    *,
    ridge: float,
) -> DifferentiableKeyMoments:
    """Build differentiable event-key covariance without detaching keys or mass."""

    if (
        keys.ndim != 3
        or event_mass.ndim != 3
        or event_mass.shape[0] != keys.shape[0]
        or event_mass.shape[2] != keys.shape[1]
        or keys.shape[1] <= 1
        or keys.shape[2] <= 0
        or ridge <= 0.0
    ):
        raise BankConditioningError("PNBTT key-moment topology changed")
    if torch.any(event_mass < 0) or not bool(torch.isfinite(keys).all()) or not bool(
        torch.isfinite(event_mass).all()
    ):
        raise BankConditioningError("PNBTT key moments received invalid evidence")

    compute_keys = keys.float()
    compute_mass = event_mass.float()
    total = compute_mass.sum(-1, keepdim=True)
    if torch.any(total <= 0):
        raise BankConditioningError("PNBTT event measure is empty")
    normalized = compute_mass / total
    mean = torch.einsum("sen,snm->sem", normalized, compute_keys)
    centered = compute_keys[:, None] - mean[:, :, None]
    covariance = torch.einsum(
        "sen,senm,senp->semp", normalized, centered, centered
    )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    identity = torch.eye(
        compute_keys.shape[-1], device=keys.device, dtype=compute_keys.dtype
    )
    regularized = covariance + float(ridge) * identity
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    if torch.any(info != 0) or not bool(torch.isfinite(cholesky).all()):
        raise BankConditioningError("PNBTT differentiable key whitening failed")
    return DifferentiableKeyMoments(
        mean=mean,
        covariance=covariance,
        cholesky=cholesky,
        normalized_mass=normalized,
        ridge=float(ridge),
    )


def whiten_queries(
    queries: torch.Tensor, moments: DifferentiableKeyMoments
) -> torch.Tensor:
    """Apply the Cholesky whitening convention ``L^-T q`` event-wise."""

    if (
        queries.ndim != 4
        or queries.shape[0] != moments.mean.shape[0]
        or queries.shape[2:] != moments.mean.shape[1:]
    ):
        raise BankConditioningError("PNBTT query/key whitening axes changed")
    right_hand_side = queries.float().permute(0, 2, 3, 1)
    whitened = torch.linalg.solve_triangular(
        moments.cholesky.transpose(-1, -2),
        right_hand_side,
        upper=True,
    ).permute(0, 3, 1, 2)
    if not bool(torch.isfinite(whitened).all()):
        raise BankConditioningError("PNBTT whitened query is non-finite")
    return whitened.to(queries)


def _online_signed_means(
    scores: torch.Tensor,
    values: torch.Tensor,
    normalized_mass: torch.Tensor,
    *,
    temperature: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scopes, ranks, events, candidates = scores.shape
    width = values.shape[-1]
    maximum = scores.new_full((scopes, ranks, events, 2), -torch.inf)
    normalizer = scores.new_zeros((scopes, ranks, events, 2))
    weighted_sum = scores.new_zeros((scopes, ranks, events, 2, width))
    tiny = torch.finfo(scores.dtype).tiny
    log_mass = normalized_mass.clamp_min(tiny).log()
    branch_scores = torch.stack((scores, -scores), dim=3) / temperature[
        :, None, None, None, None
    ]

    for start in range(0, candidates, chunk_size):
        stop = min(start + chunk_size, candidates)
        logits = branch_scores[..., start:stop] + log_mass[
            :, None, :, None, start:stop
        ]
        chunk_maximum = logits.amax(-1)
        next_maximum = torch.maximum(maximum, chunk_maximum)
        old_scale = torch.exp(maximum - next_maximum)
        weights = torch.exp(logits - next_maximum[..., None])
        weighted_sum = weighted_sum * old_scale[..., None] + torch.einsum(
            "srebn,snd->srebd", weights, values[:, start:stop]
        )
        normalizer = normalizer * old_scale + weights.sum(-1)
        maximum = next_maximum
    if torch.any(normalizer <= 0) or not bool(torch.isfinite(normalizer).all()):
        raise BankConditioningError("PNBTT signed replay normalization failed")
    means = weighted_sum / normalizer[..., None]
    return means, normalizer


def signed_key_value_pool(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    moments: DifferentiableKeyMoments,
    whitened_queries: torch.Tensor,
    temperature: torch.Tensor,
    score_epsilon: float,
    chunk_size: int,
) -> KeyValueSignedPoolResult:
    """Pool real native values with antithetic weights scored in key space."""

    if (
        keys.ndim != 3
        or values.ndim != 3
        or keys.shape[:2] != values.shape[:2]
        or whitened_queries.ndim != 4
        or whitened_queries.shape[0] != keys.shape[0]
        or whitened_queries.shape[2:] != moments.mean.shape[1:]
        or temperature.shape != (keys.shape[0],)
        or bool(torch.any(temperature <= 0.0))
        or score_epsilon <= 0.0
        or chunk_size <= 0
    ):
        raise BankConditioningError("PNBTT key/value replay topology changed")
    if not bool(torch.isfinite(values).all()):
        raise BankConditioningError("PNBTT native values are non-finite")

    centered_keys = keys.float()[:, None] - moments.mean[:, :, None]
    raw_scores = torch.einsum(
        "srem,senm->sren", whitened_queries.float(), centered_keys
    )
    score_square = torch.einsum(
        "sen,sren->sre", moments.normalized_mass, raw_scores.square()
    )
    score_rms = score_square.clamp_min(0).sqrt()
    score_scale = (score_square.clamp_min(0) + float(score_epsilon) ** 2).sqrt()
    scores = raw_scores / score_scale[..., None]
    means, normalizer = _online_signed_means(
        scores,
        values.detach().float(),
        moments.normalized_mass,
        temperature=temperature.float(),
        chunk_size=int(chunk_size),
    )
    direction = means[..., 0, :] - means[..., 1, :]
    if not bool(torch.isfinite(direction).all()):
        raise BankConditioningError("PNBTT real-value direction is non-finite")
    maximum_weight = normalizer.reciprocal()
    return KeyValueSignedPoolResult(
        direction=direction.to(keys),
        score_rms=score_rms.to(keys),
        positive_maximum_weight=maximum_weight[..., 0].to(keys),
        negative_maximum_weight=maximum_weight[..., 1].to(keys),
    )
