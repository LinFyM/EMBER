"""Canonical configuration and initialization authority for ECP Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = "ember_ecp_stage1_mapping_diverse_compiler_oracle_v1"
RUN_SCHEMA = "ember_ecp_stage1_mapping_diverse_compiler_oracle_run_v1"
STAGE = "stage1_mapping_diverse_compiler_oracle"


def stage1_repo_authority(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def stage1_asset_authority(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else asset_root / path


def _valid_model_contract(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    forbidden = {
        "rank_selector",
        "replacement_normalization",
        "replacement_head_init_multiplier",
        "selector_max_angle_radians",
    }
    return all(
        (
            model.get("hard_rank_partition") is False,
            model.get("query_to_output_shortcut") is False,
            "query_content_modulation" in model,
            int(model.get("compiler_tokens", -1)) == 380,
            int(model.get("static_tokens", -1)) == 76,
            int(model.get("process_tokens", -1)) == 304,
            "direct absolute complete rank16 LoRA"
            in model.get("full_process_surface", ""),
            "target-local" in model.get("absolute_factor_heads", ""),
            model.get("static_process_local_reads") is True,
            model.get("continuous_static_process_fusion") is True,
            model.get("target_local_factor_heads") is True,
            model.get("single_surface_prior_full") is True,
            model.get("exact_template_output_bypass") is False,
            forbidden.isdisjoint(model),
        )
    )


def _valid_objective_contract(config: Mapping[str, Any]) -> bool:
    objective = config.get("objective", {})
    support = config.get("policy_support", {})
    diagnostic = (
        "member_effective_update",
        "consensus_effective_update",
        "member_canonical_factor",
        "consensus_canonical_factor",
    )
    return all(
        (
            tuple(support.get("channels", ()))
            == (
                "successful_expert_minus_source",
                "successful_shared_minus_source",
                "learner_expert_minus_source",
                "learner_policy_minus_source",
                "learner_shared_minus_source",
            ),
            int(support.get("horizon_basis", -1)) == 4,
            support.get("target_local_activation_effect_panels_required") is True,
            objective.get("support_preservation")
            == "baseline_relative_response_barrier",
            float(objective.get("activation_effect_distillation_weight", -1)) > 0,
            float(objective.get("action_policy_loss_weight", -1)) > 0,
            objective.get("action_supervision_weights")
            == {
                "successful": 1.0,
                "verified_successful_learner": 1.0,
                "failed_learner": 0.0,
            },
            objective.get("policy_flow_time_sampling_scheme")
            == "task_logical_batch_keyed_independent_beta15_time_v2",
            objective.get("policy_flow_noise_sampling_scheme")
            == "task_logical_batch_keyed_independent_gaussian_v2",
            all(
                float(objective.get("weights", {}).get(name, -1)) == 0
                for name in diagnostic
            ),
            float(objective.get("prior_shared_response_weight", -1)) > 0,
        )
    )


def _valid_run_contract(config: Mapping[str, Any]) -> bool:
    initialization = config.get("initialization", {})
    optimization = config.get("optimization", {})
    calibration = config.get("prior_calibration", {})
    data = config.get("data", {})
    structured = config.get("structured_calibration", {})
    environment = config.get("environment", {})
    return all(
        (
            initialization.get("stage")
            == "stage1_single_surface_absolute_compiler_v23",
            initialization.get("run_contract_schema")
            == "ember_ecp_stage1_single_surface_absolute_compiler_run_v23",
            int(initialization.get("checkpoint_macro", -1)) == 114,
            initialization.get("load_model_weights_only") is True,
            initialization.get("fresh_optimizer") is True,
            initialization.get("migrate_family_heads_to_targets") is True,
            float(initialization.get("prior_head_relative_ridge", -1)) > 0,
            int(optimization.get("functional_policy_microbatch_size", -1)) == 1,
            int(optimization.get("visits_per_fit_task", -1)) == 12,
            int(optimization.get("total_task_visits", -1)) == 1080,
            tuple(optimization.get("checkpoint_task_visits", ())) == (540, 1080),
            tuple(optimization.get("stage_stop_task_visits", ())) == (540, 1080),
            tuple(optimization.get("allowed_world_sizes", ())) == (1, 2, 3, 4, 5, 6),
            tuple(config.get("roles", {}).get("fit_task_ordinals", ()))
            == tuple(range(90)),
            tuple(config.get("roles", {}).get("held_task_ordinals", ()))
            == tuple(range(90, 95)),
            calibration.get("scope")
            == (
                "fit90 initial privileged q_pi coordinates used once for "
                "minimum-change prior calibration"
            ),
            int(calibration.get("video_visit", -1)) == 12099,
            calibration.get("task_id_route") is False,
            calibration.get("retained_program_table") is False,
            data.get("task_namespace") == ["source90", "target40"],
            int(data.get("fit_mappings", -1)) == 90,
            int(data.get("held_mappings", -1)) == 5,
            int(data.get("successful_members", -1)) == 118,
            structured.get("status")
            == "required_after_task_visit_540_before_held5_materialization",
            int(structured.get("after_task_visits", -1)) == 540,
            int(structured.get("task_count", -1)) == 90,
            int(structured.get("profile_task_count", -1)) == 1,
            structured.get("task_weight") == "equal",
            structured.get("global_16d_estimator") is False,
            "event-owner q_pi" in structured.get("surface", ""),
            "compiler joint" in structured.get("surface", ""),
            float(structured.get("relative_factor_sigma", -1)) > 0,
            int(structured.get("minimum_active_owners", -1)) == 38,
            int(structured.get("lanes_per_arm", -1)) == 2,
            float(structured.get("success_weight", -1)) > 0,
            float(structured.get("progress_weight", -1)) >= 0,
            float(structured.get("success_efficiency_weight", -1)) >= 0,
            float(structured.get("outcome_leaf_weight", -1)) > 0,
            float(structured.get("dense_anchor_weight", -1)) > 0,
            int(environment.get("render_resolution", -1)) == 256,
            int(environment.get("dummy_settling_steps", -1)) == 10,
            environment.get("dummy_action") == [0, 0, 0, 0, 0, 0, -1],
            int(environment.get("action_execution_horizon", -1)) == 5,
            int(environment.get("num_inference_steps", -1)) == 10,
            str(config.get("authorities", {}).get("libero_assets_root", "")).startswith(
                "data/simulation/ember_assets/datasets/libero-assets/"
            ),
        )
    )


def load_stage1_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    information_wall = config.get("information_wall", {})
    valid = all(
        (
            config.get("schema_version") == CONFIG_SCHEMA,
            config.get("status") == "active_mapping_diverse_compiler_oracle",
            _valid_model_contract(config),
            _valid_objective_contract(config),
            _valid_run_contract(config),
            information_wall.get("validation_action_or_reward_reads") == 0,
            information_wall.get("test_action_or_reward_reads") == 0,
            information_wall.get("held5_action_or_reward_reads") == 0,
            config.get("training_ownership")
            == {
                "visible_program_trainable": False,
                "policy_teacher_trainable": True,
                "compiler_trainable": True,
                "source_policy_trainable": False,
                "observer_trainable": False,
            },
        )
    )
    if not valid:
        raise ValueError("unsupported ECP Stage 1 mapping-diverse compiler contract")
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
    source_state = load_file(str(weights), device=str(device))
    target_state = model.state_dict()
    compatible = 0
    source_factor_keys: set[str] = set()
    for name, value in source_state.items():
        target = target_state.get(name)
        if target is not None and target.shape == value.shape:
            target_state[name] = value
            compatible += 1
        elif name.startswith("compiler.factor_a.") or name.startswith(
            "compiler.factor_b."
        ):
            source_factor_keys.add(name)
        else:
            raise ValueError(f"unmigrated v23 initialization tensor: {name}")
    migrated_heads = 0
    for owner in model.compiler.owners:
        target_key = model.compiler.owner_head_key(owner)
        family = owner.family.value
        for side in ("a", "b"):
            source_name = f"compiler.factor_{side}.{family}.weight"
            target_name = f"compiler.factor_{side}.{target_key}.weight"
            if (
                source_name not in source_state
                or target_name not in target_state
                or source_state[source_name].shape != target_state[target_name].shape
            ):
                raise ValueError("v23 family head cannot initialize v24 target head")
            target_state[target_name] = source_state[source_name]
            migrated_heads += 1
    expected_source_factor_keys = {
        f"compiler.factor_{side}.{family}.weight"
        for side in ("a", "b")
        for family in ("q", "v", "action_in", "action_out")
    }
    if source_factor_keys != expected_source_factor_keys:
        raise ValueError("v23 family head authority changed")
    model.load_state_dict(target_state, strict=True)
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
        "shape_compatible_tensors": compatible,
        "migrated_target_factor_heads": migrated_heads,
        "new_static_process_parameters_initialized_fresh": True,
    }
