"""Configuration contract shared by OCPB training and materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.ecp.stage1_outcome import OUTCOME_COORDINATES
from ember.ecp.stage1_training import REPO_ROOT
from ember.pi05_source_checkpoint import read_json


RUN_SCHEMA = "ember_ecp_stage1_outcome_binding_run_v11"
STAGE = "stage1_outcome_binding_v11"


def outcome_repo_authority(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def outcome_asset_authority(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else asset_root / path


def load_outcome_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    outcome = config.get("outcome_calibration", {})
    optimization = config.get("optimization", {})
    profile = config.get("profile_defaults", {})
    if (
        config.get("schema_version") != "ember_ecp_stage1_outcome_binding_v11"
        or config.get("status") != "active_stage1_outcome_binding"
        or tuple(outcome.get("coordinate_sequence", ()))
        != (
            "program_binding",
            "compiler_binding",
            "program_binding",
            "compiler_binding",
        )
        or set(outcome.get("sigma", {})) != set(OUTCOME_COORDINATES)
        or set(outcome.get("surrogate_weight", {})) != set(OUTCOME_COORDINATES)
        or int(outcome.get("lanes_per_arm", -1)) != 2
        or int(optimization.get("world_size", -1)) != 6
        or int(optimization.get("task_count", -1)) != 19
        or int(profile.get("world_size", -1)) != 1
        or config.get("information_wall", {}).get(
            "held5_action_or_reward_reads"
        )
        != 0
        or config.get("information_wall", {}).get(
            "validation_action_or_reward_reads"
        )
        != 0
        or config.get("information_wall", {}).get(
            "test_action_or_reward_reads"
        )
        != 0
    ):
        raise ValueError("unsupported ECP Stage 1 outcome contract")
    return config
