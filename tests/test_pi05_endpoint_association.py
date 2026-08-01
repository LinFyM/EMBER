from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ember.writer.endpoint_association import (
    PERMUTATION_SEED_LABEL,
    _average_ranks,
    _correlation_record,
    _pearson,
    _permutation_seed,
    _spearman,
)
from ember.writer.endpoint_provenance import SEALED_PANEL_PAYLOAD_SHA256


def test_average_ranks_and_correlations_handle_ties_deterministically() -> None:
    values = np.asarray([7.0, 1.0, 1.0, 4.0], dtype=np.float64)
    assert _average_ranks(values).tolist() == [4.0, 1.5, 1.5, 3.0]
    assert _pearson(values, values) == pytest.approx(1.0)
    assert _spearman(values, values) == pytest.approx(1.0)
    assert _spearman(values, -values) == pytest.approx(-1.0)


def test_constant_correlation_is_explicitly_undefined() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    constant = np.ones(3, dtype=np.float64)
    assert _pearson(values, constant) is None
    assert _spearman(values, constant) is None
    assert _correlation_record(values, constant) == {
        "count": 3,
        "pearson": None,
        "spearman": None,
    }


def test_permutation_seed_matches_the_preregistered_big_endian_digest() -> None:
    expected = int.from_bytes(
        hashlib.sha256(
            f"{SEALED_PANEL_PAYLOAD_SHA256}:{PERMUTATION_SEED_LABEL}".encode()
        ).digest(),
        byteorder="big",
        signed=False,
    )
    assert _permutation_seed() == expected
