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
