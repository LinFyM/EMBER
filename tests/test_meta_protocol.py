from pathlib import Path

from ember.meta_protocol import (
    load_meta_protocol,
    meta_task_split,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs/libero90_nonheld_meta_v1/protocol.json"


def test_nonheld_meta_protocol_uses_all_and_only_the_audited_71_tasks() -> None:
    protocol = load_meta_protocol(PROTOCOL)
    assert len(protocol["active_source_task_ids"]) == 71
    assert len(protocol["excluded_target_overlap_task_ids"]) == 19
    assert set(protocol["active_source_task_ids"]).isdisjoint(
        protocol["excluded_target_overlap_task_ids"]
    )
    assert protocol["information_wall"][
        "meta_privileged_values_never_become_deployment_inputs"
    ]


def test_default_meta_split_is_task_disjoint_and_result_independent() -> None:
    protocol = load_meta_protocol(PROTOCOL)
    split = meta_task_split(protocol)
    assert split.held_out_fold == 0
    assert len(split.train) == 56
    assert len(split.validation) == 15
    assert {task.task_id for task in split.train}.isdisjoint(
        task.task_id for task in split.validation
    )
    assert {task.task_id for task in split.train + split.validation} == set(
        protocol["active_source_task_ids"]
    )


def test_rotating_held_out_fold_covers_each_meta_task_once() -> None:
    protocol = load_meta_protocol(PROTOCOL)
    held_out = [
        task.task_id
        for fold in range(5)
        for task in meta_task_split(protocol, fold).validation
    ]
    assert len(held_out) == 71
    assert len(set(held_out)) == 71
