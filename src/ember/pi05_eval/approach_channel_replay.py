"""Read-only validation of archived approach-channel episode evidence."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping
from ember.pi05_assets import Pi05EvaluationError

ROW_SCHEMA = "ember_approach_channel_episode_v1"

def _channel_assets_valid(prefix: Mapping[str, Any], intervention: Mapping[str, Any],
                          steps: int) -> bool:
    trace, contacts = prefix.get("trace", {}), prefix.get("contacts", {})
    for asset in (trace, contacts):
        path = Path(str(asset.get("path", "")))
        if (not path.is_relative_to(Path(intervention["trace_root"]))
                or not path.is_file() or path.stat().st_size != int(asset.get("bytes", -1))):
            return False
    return int(trace.get("steps", -1)) == steps and int(contacts.get("samples", -1)) == steps + 1

def _channel_timing_valid(prefix: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    actual = int(prefix.get("actual_prefix_steps", -1))
    steps = int(row["steps"])
    terminal = prefix.get("prefix_terminal")
    if type(terminal) is not bool or not 1 <= actual <= 25:
        return False
    if terminal:
        return (row["success"] is True and steps == actual
                and prefix.get("follower_start_replan_index") == math.ceil(actual / 5))
    return actual == 25 and steps > 25 and prefix.get("follower_start_replan_index") == 5

def validate_channel_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    intervention = contract["approach_channel_intervention"]
    prefix = row.get("approach_channel_intervention")
    if not isinstance(prefix, Mapping) or prefix.get("schema_version") != ROW_SCHEMA:
        raise Pi05EvaluationError("approach-channel row evidence missing")
    key = f"{row['suite']}:{int(row['task_id'])}:{int(row['init_state_id'])}"
    expected_refs = {donor: intervention["donor_references"][donor].get(key)
                     for donor in ("B", "C_correct")}
    if (any(value is None for value in expected_refs.values())
            or prefix.get("donor_references") != expected_refs
            or prefix.get("prefix") != intervention["prefix"]
            or prefix.get("follower") != intervention["follower"]
            or prefix.get("channel_donors") != intervention["channel_donors"]
            or not _channel_timing_valid(prefix, row)
            or not _channel_assets_valid(prefix, intervention, int(row["steps"]))):
        raise Pi05EvaluationError("approach-channel row contract changed")
