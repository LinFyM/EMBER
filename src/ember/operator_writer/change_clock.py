"""The bounded fresh T change-clock calculation and its registered artifacts."""

from __future__ import annotations

from pathlib import Path


MODE = "T_change_clock"
TASK = "operator_change_clock_learning_20260930"
ROOT = Path("/data1/user/ymdai/ember_runs/operator_change_clock_learning_20260930")
SPEC_NAME = "change_clock_spec.json"
CAPTURE_NAME = "change_clock_capture.json"
ERASE_RULE = "erase=(memory@key)*(-expm1(-vector_norm(delta_hbar,dim=-1)/sqrt(1024)))[None,:]; value=ungated"
TRAINING_GIT = {"commit": "517bc8d42780fabf8f3faa7d707651f99b8beed0", "branch": "",
                "dirty_paths": [], "pushed_ref": "origin/codex/operator-change-clock"}
TRAINING_SPEC_PATH = Path("/data1/user/ymdai/projects/EMBER-change-clock-learning-v2-formal"
                          "/configs/operator_read_write_v1/change_clock_spec.json")
CONTINUATION_TASK = "operator_change_clock_continuation450_20260930"
CONTINUATION_ROOT = ROOT.with_name(CONTINUATION_TASK)
CONTINUATION_SPEC_NAME = "change_clock_continuation450_spec.json"
CONTINUATION_CAPTURE_NAME = "change_clock_continuation450_capture.json"
CONTINUATION_CHECKPOINTS = (360, 450)
PARENT_CHECKPOINT = (ROOT / MODE / "train/attempts/resume180_gpu02_world1"
                     / "checkpoints/macro_00000270")


def expected_spec(base: dict) -> dict:
    execution = {**base["execution"], "modes": [MODE], "world_sizes": [1, 2, 3, 4]}
    execution.pop("world_size")
    return {
        **base,
        "task": TASK,
        "design": "docs/designs/operator_read_write_learning_design.md#31",
        "run_root": str(ROOT),
        "operator": {**base["operator"], "erase_rule": ERASE_RULE},
        "execution": execution,
        "budget": {"new_gpu_hours_hard": 8, "peak_new_gib": 24},
    }


def expected_continuation_spec(base: dict, events: dict) -> dict:
    """Same operator and absolute LR; one bounded extension of the sealed event stream."""
    return {**base, "task": CONTINUATION_TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#32",
            "run_root": str(CONTINUATION_ROOT), "events": events,
            "execution": {**base["execution"], "updates_per_mode": 450,
                          "queries_per_mode": 450 * 112,
                          "checkpoints": list(CONTINUATION_CHECKPOINTS),
                          "only_selected_checkpoint": 450},
            "evaluation": {**base["evaluation"], "bank_macro": 450},
            "continuation": {"parent_run_root": str(ROOT), "parent_arm": MODE,
                             "parent_macro": 270, "parent_training_git": TRAINING_GIT["commit"],
                             "parent_event_schema": base["events"]["schema_version"],
                             "sampler_migration": "teacher_pool_0_29_is_visit_index_not_demo_pool"},
            "budget": {"new_gpu_hours_expected": [4.1, 4.8],
                       "new_gpu_hours_hard": 6, "peak_new_gib": 24}}
