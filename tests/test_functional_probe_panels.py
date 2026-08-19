import pytest

from ember.functional_adaptation.probe_panels import (
    FunctionalProbePanel,
    panel_for_visit,
    selected_probe_rows,
)


def test_probe_rows_use_only_requested_episodes_and_are_repeatable() -> None:
    episodes = {0: tuple(range(0, 5)), 1: tuple(range(5, 10)), 2: (10, 11)}

    first = selected_probe_rows(
        episodes,
        demo_indices=(0, 1),
        panel_count=2,
        batch_size=3,
        seed=17,
    )
    second = selected_probe_rows(
        episodes,
        demo_indices=(0, 1),
        panel_count=2,
        batch_size=3,
        seed=17,
    )

    assert first == second
    assert len({row for panel in first for row in panel}) == 6
    assert all(row < 10 for panel in first for row in panel)


def test_probe_rows_reject_an_oversized_panel() -> None:
    with pytest.raises(ValueError):
        selected_probe_rows(
            {0: (1, 2)},
            demo_indices=(0,),
            panel_count=2,
            batch_size=2,
            seed=3,
        )


def test_task_visits_rotate_fixed_panels_without_resampling() -> None:
    panels = tuple(
        FunctionalProbePanel(batch={}, target=object(), policy_seed=index)
        for index in range(3)
    )

    observed = [panel_for_visit(panels, visit)[0].policy_seed for visit in range(7)]

    assert observed == [0, 1, 2, 0, 1, 2, 0]
