"""Canonical configuration and initialization authority for ECP Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_ecp_stage1_single_surface_absolute_compiler_run_v23"
STAGE = "stage1_single_surface_absolute_compiler_v23"


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
        != "ember_ecp_stage1_single_surface_absolute_compiler_v23"
        or config.get("status")
        != "active_stage1_single_surface_absolute_compiler"
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
        or int(config.get("model", {}).get("compiler_tokens", -1)) != 380
        or "direct absolute complete rank16 LoRA"
        not in config.get("model", {}).get("full_process_surface", "")
        or "family-specific A/B heads"
        not in config.get("model", {}).get("absolute_factor_heads", "")
        or config.get("model", {}).get("single_surface_prior_full") is not True
        or config.get("model", {}).get("exact_template_output_bypass") is not False
        or any(
            name in config.get("model", {})
            for name in (
                "rank_selector",
                "replacement_normalization",
                "replacement_head_init_multiplier",
                "selector_max_angle_radians",
            )
        )
        or config.get("information_wall", {}).get("validation_action_or_reward_reads")
        != 0
        or config.get("information_wall", {}).get("test_action_or_reward_reads") != 0
        or config.get("information_wall", {}).get("held5_action_or_reward_reads") != 0
        or config.get("training_ownership")
        != {
            "visible_program_trainable": False,
            "policy_teacher_trainable": False,
            "compiler_trainable": True,
            "source_policy_trainable": False,
            "observer_trainable": False,
        }
        or any(
            float(config.get("objective", {}).get("weights", {}).get(name, -1)) != 0.0
            for name in (
                "member_effective_update",
                "consensus_effective_update",
                "member_canonical_factor",
                "consensus_canonical_factor",
            )
        )
        or float(config.get("objective", {}).get("prior_shared_response_weight", -1))
        <= 0.0
        or config.get("initialization", {}).get("stage")
        != "stage1_policy_support_v6"
        or config.get("initialization", {}).get("run_contract_schema")
        != "ember_ecp_stage1_policy_support_run_v6"
        or int(config.get("initialization", {}).get("checkpoint_macro", -1)) != 228
        or config.get("initialization", {}).get("load_model_weights_only") is not True
        or config.get("initialization", {}).get("fresh_optimizer") is not True
        or int(config.get("optimization", {}).get("visits_per_fit_task", -1)) != 6
        or int(config.get("optimization", {}).get("total_task_visits", -1)) != 114
        or int(config.get("optimization", {}).get("optimizer_updates", -1)) != 19
        or tuple(config.get("optimization", {}).get("checkpoint_task_visits", ()))
        != (114,)
        or tuple(config.get("optimization", {}).get("stage_stop_task_visits", ()))
        != (114,)
        or config.get("fixed_program_coordinate", {}).get("scope")
        != "fit19 privileged q_pi coordinates captured once and frozen for compiler-only identification"
        or int(config.get("fixed_program_coordinate", {}).get("video_visit", -1))
        != 12099
        or tuple(config.get("fixed_program_coordinate", {}).get("optimized_fields", ()))
        != ()
        or config.get("fixed_program_coordinate", {}).get("task_id_route") is not False
        or config.get("fixed_program_coordinate", {}).get("checkpoint_state") is not False
    ):
        raise ValueError("unsupported ECP Stage 1 single-surface compiler contract")
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
