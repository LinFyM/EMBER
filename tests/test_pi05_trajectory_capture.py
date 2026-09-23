from pathlib import Path

import torch

from ember.pi05_eval.trajectory_capture import (
    capture_level,
    initialize_capture,
    record_replan,
    save_capture,
)


def _slot(level: str) -> dict:
    slot = {
        "init_state_id": 3,
        "steps": 7,
        "policy_noise_seeds": [11],
    }
    initialize_capture(slot, level)
    record_replan(
        slot,
        {"observation.state": torch.arange(8, dtype=torch.float32)},
        {"image": torch.ones((1, 3, 4, 4)), "token": torch.ones((1, 2), dtype=torch.long)},
        torch.zeros((1, 50, 7)),
        torch.ones((5, 7)),
    )
    return slot


def test_compact_capture_keeps_state_action_and_only_selected_images(tmp_path: Path) -> None:
    task = {"suite": "libero_goal", "task_id": 4}
    capture = {
        "trajectory_root": str(tmp_path),
        "mode": "compact",
        "full_conditions": [
            {"suite": "libero_goal", "task_id": 4, "init_state_id": 3}
        ],
    }
    assert capture_level(capture, task, 2) == "compact"
    assert capture_level(capture, task, 3) == "full"

    compact_slot = _slot("compact")
    compact_slot["init_state_id"] = 2
    full_slot = _slot("full")
    compact_record = save_capture(capture, task, compact_slot, success=False)
    full_record = save_capture(capture, task, full_slot, success=True)
    compact = torch.load(compact_record["path"], map_location="cpu", weights_only=False)
    full = torch.load(full_record["path"], map_location="cpu", weights_only=False)

    assert compact["schema_version"] == full["schema_version"] == "ember_pi05_diagnostic_trajectory_v2"
    assert compact["capture_level"] == "compact"
    assert full["capture_level"] == "full"
    assert "observations" not in compact
    assert len(compact["states"]) == len(compact["action_chunks"]) == 1
    assert len(compact["executed_action_prefixes"]) == 1
    assert compact["replan_steps"] == (7,)
    assert len(full["states"]) == len(full["action_chunks"]) == len(full["observations"]) == 1
    assert full["replan_predicates"] == (None,)
    assert tuple(compact["states"][0].shape) == (8,)


def test_legacy_capture_payload_remains_v1(tmp_path: Path) -> None:
    task = {"suite": "libero_goal", "task_id": 4}
    capture = {"trajectory_root": str(tmp_path)}
    record = save_capture(capture, task, _slot("full"), success=True)
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "ember_pi05_occupancy_trajectory_v1"
    assert "observations" in payload
    assert "states" not in payload
