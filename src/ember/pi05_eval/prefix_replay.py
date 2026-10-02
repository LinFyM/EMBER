"""Read-only validation of archived frozen-prefix episode evidence."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from ember.pi05_assets import Pi05EvaluationError

ROW_SCHEMA = "ember_frozen_prefix_episode_v1"

def validate_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if contract.get("approach_channel_intervention") is not None:
        from ember.pi05_eval.approach_channel_replay import validate_channel_row

        validate_channel_row(row, contract)
        return
    intervention = contract["frozen_prefix_intervention"]
    prefix = row.get("frozen_prefix_intervention")
    if not isinstance(prefix, Mapping) or prefix.get("schema_version") != ROW_SCHEMA:
        raise Pi05EvaluationError("frozen-prefix row evidence missing")
    key = f"{row['suite']}:{int(row['task_id'])}:{int(row['init_state_id'])}"
    reference = intervention["references"].get(key)
    cut = int(intervention["cut_control_steps"])
    terminal = bool(reference["success"] and int(reference["steps"]) <= cut) if reference else None
    trace = prefix.get("trace", {})
    path = Path(str(trace.get("path", "")))
    if (reference is None or prefix.get("reference") != reference
            or prefix.get("anchor") != intervention["anchor"]
            or prefix.get("follower") != intervention["follower"]
            or prefix.get("cut_control_steps") != cut
            or prefix.get("actual_prefix_steps") != min(cut, int(reference["steps"]))
            or prefix.get("prefix_terminal") is not terminal
            or prefix.get("follower_start_replan_index") != min(cut, int(reference["steps"])) // 5
            or (terminal and (row["steps"] != reference["steps"] or row["success"] is not True))
            or (not terminal and int(row["steps"]) <= cut)
            or not path.is_relative_to(Path(intervention["trace_root"]))
            or not path.is_file() or path.stat().st_size != int(trace.get("bytes", -1))
            or int(trace.get("steps", -1)) != int(row["steps"])):
        raise Pi05EvaluationError("frozen-prefix row contract changed")
