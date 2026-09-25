"""Correlated Gaussian score after observing five executed gripper signs."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm


def correlated_sign_score(mean, signs, *, rho: float = .8, sigma: float = .1,
                          threshold: float = -1. + 2. / (2. + 1e-6),
                          seed: int = 20260925):
    """Return joint sign probability, its mean score, and a tail upper bound.

    The conditional four-dimensional CDF retains the full temporal covariance.
    The bound uses a union bound and remains informative when a CDF rounds to 1.
    """
    mean = np.asarray(mean, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    if (mean.shape != (5,) or signs.shape != (5,)
            or not np.all(np.isfinite(mean)) or not np.all(np.isin(signs, (-1., 1.)))
            or not 0 <= rho < 1 or sigma <= 0):
        raise ValueError("invalid five-step gripper mean, signs or covariance")
    times = np.arange(5)
    correlation = rho ** np.abs(times[:, None] - times[None, :])
    x = signs * (mean - threshold) / sigma
    covariance = correlation * signs[:, None] * signs[None, :]

    def cdf(boundary, matrix):
        return float(multivariate_normal.cdf(
            boundary, mean=np.zeros(len(boundary)), cov=matrix, maxpts=200000,
            abseps=1e-10, releps=1e-8, rng=np.random.default_rng(seed)))

    probability = cdf(x, covariance)
    if probability <= 0 or not np.isfinite(probability):
        raise ValueError("registered sign probability is numerically unresolved")
    score = np.empty(5, dtype=np.float64)
    for j in range(5):
        keep = np.arange(5) != j
        conditional_boundary = x[keep] - covariance[keep, j] * x[j]
        conditional_covariance = (covariance[np.ix_(keep, keep)]
                                  - np.outer(covariance[keep, j], covariance[keep, j]))
        score[j] = (signs[j] / sigma * norm.pdf(x[j])
                    * cdf(conditional_boundary, conditional_covariance) / probability)
    flip_union_bound = float(norm.sf(x).sum())
    lower_probability = max(0., 1. - flip_union_bound)
    coordinate_bound = (float(np.max(norm.pdf(x))) / (sigma * lower_probability)
                        if lower_probability > 0 else None)
    return probability, score, flip_union_bound, coordinate_bound
