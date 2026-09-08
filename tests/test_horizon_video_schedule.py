from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from ember.expert_manifold.video_schedule import condition_demo_index
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_target_data import SUITE_ORDER
from ember.writer import evaluation
from ember.writer.materialization import (
    BANK_KIND, BANK_SCHEMA, condition_id, paired_video_sets, planned_episodes, selection_contract,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = (1, 3, 11, 13, 23, 26, 31, 32)
SOURCE = {key: f"/test/source/{key}" for key in ("source_run", "checkpoint", "model_path")}
GIT = {"branch": "", "commit": "a" * 40, "dirty_paths": [],
       "authority_ref": "origin/main", "authority_contains_commit": True}


def selection(**overrides):
    values = dict(role="validation", task_ids=TASKS, cardinality=1, arm="correct",
                  mode="per_init_ordinal", seed=20260907, init_state_ids=tuple(range(50)),
                  video_pool=tuple(range(50)))
    return selection_contract(**(values | overrides))


def test_full_round_actual_teacher_identities_match_canonical_correct_and_other():
    correct, other = selection(), selection(arm="same_task_other")
    all_conditions = set()
    for task in TASKS:
        left, right = planned_episodes(correct, task), planned_episodes(other, task)
        for rows in (left, right):
            assert Counter(row["teacher_demo_indices"][0] for row in rows) == Counter(range(50))
            assert len({row["condition_id"] for row in rows}) == 50
        for state, (a, b) in enumerate(zip(left, right, strict=True)):
            assert a["teacher_demo_indices"] != b["teacher_demo_indices"]
            for arm, row in (("correct", a), ("same_task_other", b)):
                expected = condition_demo_index(20260907, SUITE_ORDER[task // 10], task % 10,
                    state, condition=arm, demo_count=50, sampling_mode="without_replacement")
                assert row["teacher_demo_indices"] == [expected]
            all_conditions.add((task, a["teacher_demo_indices"][0]))
    assert len(all_conditions) == 400


@pytest.mark.parametrize("workers", [1, 2, 3, 6])
def test_schedule_survives_shards_execution_order_and_resumed_cursors(workers):
    selected = selection()
    keys = [(task, state) for task in TASKS for state in range(50)]
    expected = {(task, state): paired_video_sets(selected, task, state) for task, state in keys}
    shards = [keys[worker::workers] for worker in range(workers)]
    random.Random(17).shuffle(shards)
    reconstructed = {}
    for shard in shards:
        # A saved cursor resumes this shard without reseeding a per-worker stream.
        cut = len(shard) // 3
        for portion in (shard[:cut], shard[cut:]):
            for task, state in reversed(portion):
                reconstructed[task, state] = paired_video_sets(copy.deepcopy(selected), task, state)
    assert reconstructed == expected
    assert {key: paired_video_sets(selection(), *key) for key in reversed(keys)} == expected


def manifest():
    selected = selection()
    authority = json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())
    rows = []
    for task in authority["tasks"]:
        if task["global_task_id"] in TASKS:
            source = ROOT / "data/datasets" / authority["dataset"]["revision"] / task["hdf5"]["relative_path"]
            rows.append({key: task[key] for key in ("global_task_id", "suite", "task_id", "language", "split_role")} |
                {"teacher_source": {"path": str(source), "bytes": task["hdf5"]["bytes"]},
                 "episodes": planned_episodes(selected, task["global_task_id"])})
    return {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "status": "sealed",
            "evaluation_role": "validation", "arm": "correct", "selection": selected,
            "tasks": rows, "source": SOURCE, "materialization_git": GIT,
            "asset_root": str(ROOT), "single_complete_rank16": True}


def test_full_manifest_scope_checks_all_400_actual_episode_assignments():
    value = manifest()
    keys = tuple((row["suite"], row["task_id"]) for row in value["tasks"])
    evaluation._inspect_scope(value, SOURCE, keys, "validation", {key: tuple(range(50)) for key in keys}, True)
    assert sum(len(row["episodes"]) for row in value["tasks"]) == 400


@pytest.mark.parametrize("damage", ["duplicate", "missing", "permuted", "old_with_replacement", "false_label"])
def test_invalid_actual_manifest_schedule_rejected_before_checkpoint_or_rollout(tmp_path, monkeypatch, damage):
    value = manifest()
    episodes = value["tasks"][0]["episodes"]
    if damage == "duplicate":
        episodes[1]["teacher_demo_indices"] = episodes[0]["teacher_demo_indices"]
        episodes[1]["condition_id"] = "invented_unique_name_does_not_fix_video_identity"
    elif damage == "missing":
        episodes.pop()
    elif damage == "permuted":
        a, b = episodes[:2]
        a["teacher_demo_indices"], b["teacher_demo_indices"] = b["teacher_demo_indices"], a["teacher_demo_indices"]
    elif damage == "old_with_replacement":
        for task in value["tasks"]:
            for row in task["episodes"]:
                rng = random.Random(20260907 + 1_000_003 * task["global_task_id"] + 7_919 * row["init_state_id"])
                video = rng.sample(list(range(50)), 1)
                row.update(teacher_demo_indices=video, condition_id=condition_id(task["global_task_id"], video))
    else:
        value["selection"]["without_replacement"] = False
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(evaluation, "inspect_writer_checkpoint", lambda *_: pytest.fail("bad schedule reached checkpoint loading"))
    with pytest.raises(Pi05EvaluationError):
        evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE,
            task_keys=tuple((row["suite"], row["task_id"]) for row in value["tasks"]),
            evaluation_role="validation", require_formal=True)


def test_finite_pool_diagnostics_have_four_unique_videos_and_reject_five_rows():
    args = dict(role="development_train", task_ids=(0,), init_state_ids=tuple(range(32, 36)), video_pool=(46, 47, 48, 49))
    correct, other = selection(**args), selection(**args, arm="same_task_other")
    left, right = planned_episodes(correct, 0), planned_episodes(other, 0)
    assert Counter(row["teacher_demo_indices"][0] for row in left) == Counter(range(46, 50))
    assert Counter(row["teacher_demo_indices"][0] for row in right) == Counter(range(46, 50))
    assert all(a["teacher_demo_indices"] != b["teacher_demo_indices"] for a, b in zip(left, right))
    with pytest.raises(ValueError, match="finite-pool"):
        selection(**(args | {"init_state_ids": tuple(range(32, 37))}))
    fixed = selection(**args, mode="fixed_per_task")
    assert fixed["without_replacement"] is False
    assert fixed["without_replacement_scope"] == "fixed_diagnostic_video_reuse"


def test_formal_scope_rejects_partial_round_and_validation_rejects_restricted_pool():
    with pytest.raises(ValueError, match="all 50"):
        selection(video_pool=tuple(range(49)))
    value = manifest()
    value["selection"] = selection(init_state_ids=tuple(range(10)))
    for task in value["tasks"]:
        task["episodes"] = planned_episodes(value["selection"], task["global_task_id"])
    with pytest.raises(ValueError, match="50 init states"):
        evaluation._validate_round(value["selection"], value["tasks"], True)
