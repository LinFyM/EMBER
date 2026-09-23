"""Optional trajectory capture used by sealed PI05 diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError


def capture_level(
    capture: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    init_state_id: int,
) -> str | None:
    if capture is None:
        return None
    mode = str(capture.get("mode", "full"))
    if mode == "full":
        return "full"
    if mode != "compact":
        raise Pi05EvaluationError("unsupported diagnostic occupancy capture mode")
    key = (str(task["suite"]), int(task["task_id"]), int(init_state_id))
    selected = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in capture.get("full_conditions", ())
    }
    return "full" if key in selected else "compact"


def initialize_capture(slot: dict[str, Any], level: str | None) -> None:
    if level is None:
        return
    slot["occupancy_capture_level"] = level
    slot["replay_states"] = []
    slot["replay_action_chunks"] = []
    slot["replay_executed_prefixes"] = []
    slot["replay_replan_steps"] = []
    if level == "full":
        slot["replay_observations"] = []
        slot["replay_replan_predicates"] = []


def record_replan(
    slot: dict[str, Any], raw_input: Mapping[str, Any], processed: Mapping[str, Any],
    chunk: Any, executed_prefix: Any | None = None,
) -> None:
    if "replay_action_chunks" not in slot:
        return
    import torch

    slot["replay_states"].append(
        raw_input["observation.state"].detach().to(device="cpu").contiguous()
    )
    slot["replay_action_chunks"].append(chunk.detach().to(device="cpu").contiguous())
    slot["replay_replan_steps"].append(int(slot["steps"]))
    if executed_prefix is not None:
        slot["replay_executed_prefixes"].append(torch.as_tensor(executed_prefix).clone())
    if "replay_observations" in slot:
        slot["replay_replan_predicates"].append(
            list(slot["stage_predicate_last"]) if "stage_predicate_last" in slot else None
        )
        slot["replay_observations"].append(
            {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in processed.items()
                if isinstance(value, torch.Tensor)
            }
        )


def save_capture(
    capture: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    slot: Mapping[str, Any],
    *,
    success: bool,
) -> dict[str, Any] | None:
    if capture is None:
        return None
    import torch

    root = Path(str(capture["trajectory_root"]))
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        f"{task['suite']}_task_{int(task['task_id']):02d}_"
        f"state_{int(slot['init_state_id']):03d}.pt"
    )
    level = str(slot["occupancy_capture_level"])
    common = {
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "init_state_id": int(slot["init_state_id"]),
        "success": bool(success),
        "steps": int(slot["steps"]),
        "policy_noise_seeds": tuple(slot["policy_noise_seeds"]),
        "action_chunks": tuple(slot["replay_action_chunks"]),
        "executed_action_prefixes": tuple(slot["replay_executed_prefixes"]),
        "replan_steps": tuple(slot["replay_replan_steps"]),
    }
    if "mode" not in capture:
        payload = {
            "schema_version": "ember_pi05_occupancy_trajectory_v1",
            **common,
            "observations": tuple(slot["replay_observations"]),
        }
    else:
        payload = {
            "schema_version": "ember_pi05_diagnostic_trajectory_v2",
            "capture_level": level,
            **common,
            "states": tuple(slot["replay_states"]),
        }
        if level == "full":
            payload["observations"] = tuple(slot["replay_observations"])
            payload["replan_predicates"] = tuple(slot["replay_replan_predicates"])
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "replans": len(slot["replay_action_chunks"]),
        "capture_level": level,
    }
