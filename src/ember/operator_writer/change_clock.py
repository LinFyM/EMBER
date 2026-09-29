"""The bounded fresh T change-clock calculation and its registered artifacts."""

from __future__ import annotations

from pathlib import Path


MODE = "T_change_clock"
TASK = "operator_change_clock_learning_20260930"
ROOT = Path("/data1/user/ymdai/ember_runs/operator_change_clock_learning_20260930")
SPEC_NAME = "change_clock_spec.json"
CAPTURE_NAME = "change_clock_capture.json"
ERASE_RULE = "erase=(memory@key)*(-expm1(-vector_norm(delta_hbar,dim=-1)/sqrt(1024)))[None,:]; value=ungated"


def expected_spec(base: dict) -> dict:
    return {
        **base,
        "task": TASK,
        "design": "docs/designs/operator_read_write_learning_design.md#31",
        "run_root": str(ROOT),
        "operator": {**base["operator"], "erase_rule": ERASE_RULE},
        "execution": {**base["execution"], "modes": [MODE]},
        "budget": {"new_gpu_hours_hard": 8, "peak_new_gib": 24},
    }
