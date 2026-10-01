"""Read-only validation of archived readout intervention evidence."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from ember.pi05_assets import Pi05EvaluationError

ROW_SCHEMA = "ember_readout_realization_episode_v1"

def validate_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    intervention = contract["readout_realization_intervention"]
    info = row.get("readout_realization_intervention")
    if (not isinstance(info, Mapping) or info.get("schema_version") != ROW_SCHEMA
            or info.get("group") != intervention["group"]
            or info.get("original_condition_id") != row["horizon_writer_lora"]["condition_id"]):
        raise Pi05EvaluationError("readout row intervention identity changed")
    for key, count in (("trace", int(row["steps"])), ("contacts", int(row["steps"]) + 1)):
        record = info[key]
        path = Path(record["path"])
        if (not path.is_relative_to(Path(intervention["trace_root"]))
                or not path.is_file() or path.stat().st_size != int(record["bytes"])
                or int(record["steps" if key == "trace" else "samples"]) != count):
            raise Pi05EvaluationError("readout row continuous trace changed")
