"""Configuration contract shared by OCPB training and materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.ecp.stage1_training import REPO_ROOT
from ember.pi05_source_checkpoint import read_json


RUN_SCHEMA = "ember_ecp_stage1_outcome_binding_run_v12"
STAGE = "stage1_outcome_binding_v12"


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
        config.get("schema_version") != "ember_ecp_stage1_outcome_binding_v12"
        or config.get("status") != "active_stage1_outcome_binding"
        or tuple(outcome.get("coordinate_sequence", ()))
        != ("compiler_binding",)
        or set(outcome.get("sigma", {})) != {"compiler_binding"}
        or set(outcome.get("surrogate_weight", {})) != {"compiler_binding"}
        or int(outcome.get("credit_macro_offset", -1)) != 1
        or int(outcome.get("lanes_per_arm", -1)) != 2
        or int(optimization.get("world_size", -1)) != 6
        or int(optimization.get("task_count", -1)) != 19
        or int(optimization.get("total_macros", -1)) != 1
        or int(profile.get("world_size", -1)) != 1
        or int(profile.get("total_macros", -1)) != 1
        or config.get("initialization", {}).get("stage")
        != "stage1_outcome_binding_v11"
        or config.get("initialization", {}).get("run_contract_schema")
        != "ember_ecp_stage1_outcome_binding_run_v11"
        or int(config.get("initialization", {}).get("checkpoint_macro", -1))
        != 1
        or config.get("initialization", {}).get(
            "restore_optimizer_and_rank_rng_in_formal"
        )
        is not True
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
