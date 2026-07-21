from __future__ import annotations

from ember.pi05_source_corpus import active_source_ids, excluded_source_ids


def test_reviewed_source_overlap_partition_is_exact() -> None:
    assert excluded_source_ids() == (
        8,
        9,
        10,
        20,
        25,
        27,
        30,
        31,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        77,
    )
    active = active_source_ids()
    assert len(active) == 71
    assert set(active).isdisjoint(excluded_source_ids())
    assert set(active) | set(excluded_source_ids()) == set(range(90))
