"""Canonical configuration and initialization authority for ECP Stage 1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_ecp_stage1_fixed_compiler_free_program_run_v21"
STAGE = "stage1_fixed_compiler_free_program_reachability_v21"


def stage1_repo_authority(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def stage1_asset_authority(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else asset_root / path


def load_stage1_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if (
        config.get("schema_version")
        != "ember_ecp_stage1_fixed_compiler_free_program_v21"
        or config.get("status")
        != "active_stage1_fixed_compiler_free_program_reachability"
        or config.get("model", {}).get("hard_rank_partition") is not False
        or config.get("model", {}).get("query_to_output_shortcut") is not False
        or "query_content_modulation" not in config.get("model", {})
        or tuple(config.get("policy_support", {}).get("channels", ()))
        != (
            "successful_expert_minus_source",
            "successful_shared_minus_source",
            "learner_expert_minus_source",
            "learner_policy_minus_source",
            "learner_shared_minus_source",
        )
        or int(config.get("policy_support", {}).get("horizon_basis", -1)) != 4
        or config.get("policy_support", {}).get(
            "target_local_activation_effect_panels_required"
        )
        is not True
        or config.get("objective", {}).get("support_preservation")
        != "baseline_relative_response_barrier"
        or float(
            config.get("objective", {}).get("activation_effect_distillation_weight", -1)
        )
        <= 0
        or float(config.get("objective", {}).get("action_policy_loss_weight", -1)) <= 0
        or config.get("objective", {}).get("action_supervision_weights")
        != {
            "successful": 1.0,
            "verified_successful_learner": 1.0,
            "failed_learner": 0.0,
        }
        or config.get("objective", {}).get("policy_flow_time_sampling_scheme")
        != "task_logical_batch_keyed_independent_beta15_time_v2"
        or config.get("objective", {}).get("policy_flow_noise_sampling_scheme")
        != "task_logical_batch_keyed_independent_gaussian_v2"
        or int(
            config.get("optimization", {}).get("functional_policy_microbatch_size", -1)
        )
        != 1
        or float(config.get("model", {}).get("replacement_head_init_multiplier", -1))
        != 0.1
        or float(config.get("model", {}).get("selector_max_angle_radians", -1))
        != math.pi / 2.0
        or "process-value-only bounded rank-one retraction"
        not in config.get("model", {}).get("full_process_surface", "")
        or config.get("information_wall", {}).get("validation_action_or_reward_reads")
        != 0
        or config.get("information_wall", {}).get("test_action_or_reward_reads") != 0
        or config.get("information_wall", {}).get("held5_action_or_reward_reads") != 0
        or config.get("training_ownership")
        != {
            "free_program_trainable": True,
            "visible_program_trainable": False,
            "policy_teacher_trainable": False,
            "compiler_trainable": False,
            "source_policy_trainable": False,
            "observer_trainable": False,
        }
        or any(
            float(config.get("objective", {}).get("weights", {}).get(name, -1)) != 0.0
            for name in (
                "member_effective_update",
                "member_canonical_factor",
                "consensus_canonical_factor",
            )
        )
        or float(
            config.get("objective", {})
            .get("weights", {})
            .get("consensus_effective_update", -1)
        )
        != 1.0
        or config.get("initialization", {}).get("stage")
        != "stage1_program_locked_compiler_identification_v20"
        or config.get("initialization", {}).get("run_contract_schema")
        != "ember_ecp_stage1_program_locked_compiler_run_v20"
        or int(config.get("initialization", {}).get("checkpoint_macro", -1)) != 114
        or config.get("initialization", {}).get("load_model_weights_only") is not True
        or config.get("initialization", {}).get("fresh_optimizer") is not True
        or int(config.get("optimization", {}).get("visits_per_fit_task", -1)) != 12
        or int(config.get("optimization", {}).get("total_task_visits", -1)) != 228
        or int(config.get("optimization", {}).get("optimizer_updates", -1)) != 38
        or tuple(config.get("optimization", {}).get("checkpoint_task_visits", ()))
        != (228,)
        or tuple(config.get("optimization", {}).get("stage_stop_task_visits", ()))
        != (228,)
        or config.get("free_program_oracle", {}).get("scope")
        != "fit19 task-local privileged reachability diagnostic only"
        or tuple(config.get("free_program_oracle", {}).get("optimized_fields", ()))
        != ("process", "uncertainty")
        or tuple(config.get("free_program_oracle", {}).get("fixed_fields", ()))
        != ("language", "scene", "presence")
        or float(config.get("free_program_oracle", {}).get("process_delta_scale", -1))
        != 2.0
        or float(
            config.get("free_program_oracle", {}).get("uncertainty_log_scale_bound", -1)
        )
        != 2.0
        or config.get("free_program_oracle", {}).get("task_id_route") is not False
        or int(config.get("free_program_oracle", {}).get("held_program_parameters", -1))
        != 0
    ):
        raise ValueError("unsupported ECP Stage 1 free-Program oracle contract")
    return config


def load_stage1_initialization(
    *,
    checkpoint: Path,
    model: torch.nn.Module,
    device: torch.device,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    weights = checkpoint / "ecp.safetensors"
    run_contract = read_json(checkpoint.parent.parent / "run_contract.json")
    expected_stage = str(initialization["stage"])
    expected_schema = str(initialization["run_contract_schema"])
    expected_macro = int(initialization["checkpoint_macro"])
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != expected_stage
        or manifest.get("run_contract_schema") != expected_schema
        or int(manifest.get("next_macro", -1)) != expected_macro
        or checkpoint_macro(checkpoint) != expected_macro
        or int(manifest.get("world_size", -1)) != 6
        or run_contract.get("schema_version") != expected_schema
        or run_contract.get("stage") != expected_stage
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
    ):
        raise ValueError("ECP Stage 1 initialization authority changed")
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "weights": str(weights.resolve()),
        "weights_bytes": weights.stat().st_size,
        "training_commit": str(run_contract["git"]["commit"]),
        "stage": expected_stage,
        "run_contract_schema": expected_schema,
        "checkpoint_macro": expected_macro,
        "model_weights_only": True,
        "fresh_optimizer": True,
    }
