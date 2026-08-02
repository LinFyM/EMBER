"""Sealed configuration loading and validation for PI05 AS-Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, sha256_file
from ember.writer.architecture import expected_writer_contract
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_BOUND_ROLE_CONFIG_SCHEMA = (
    "ember_pi05_target_bound_role_program_as_writer_v1"
)
TARGET_BOUND_ROLE_CONFIG_OVERLAY_SCHEMA = (
    "ember_pi05_target_bound_role_program_recipe_overlay_v1"
)
AS_WRITER_CONFIG_SCHEMA = "ember_pi05_contextual_value_dual_read_full24_as_writer_v1"
AS_WRITER_CONFIG_OVERLAY_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_full24_as_writer_recipe_overlay_v1"
)
AS_WRITER_SERIAL4_CONFIG_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_serial4_exposurematched_as_writer_v1"
)
AS_WRITER_SERIAL4_CONFIG_OVERLAY_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_serial4_exposurematched_recipe_overlay_v1"
)
AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_cycle_normalized_as_writer_v2"
)
AS_WRITER_CYCLE_NORMALIZED_CONFIG_OVERLAY_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_cycle_normalized_recipe_overlay_v2"
)
AS_WRITER_CONFIG_SCHEMAS = (
    TARGET_BOUND_ROLE_CONFIG_SCHEMA,
    AS_WRITER_CONFIG_SCHEMA,
    AS_WRITER_SERIAL4_CONFIG_SCHEMA,
    AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA,
)
AS_WRITER_STAGES = ("development", "final")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    """Return the sealed data stage, preserving old development artifacts."""

    stage = str(config.get("sealed_stage", "development"))
    if stage not in AS_WRITER_STAGES:
        raise WriterModelError("unsupported PI05 AS-Writer stage")
    return stage


def writer_split_roles(config: Mapping[str, Any]) -> tuple[str, ...]:
    if writer_stage(config) == "development":
        return ("train",)
    return ("train", "validation")


def _validate_authorities(config: Mapping[str, Any]) -> None:
    authorities = config.get("authorities", {})
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(authorities) != required:
        raise WriterModelError("AS-Writer authority set changed")
    for name, authority in authorities.items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if (
            not artifact.is_file()
            or sha256_file(artifact) != authority.get("sha256")
        ):
            raise WriterModelError(
                f"sealed AS-Writer authority changed: {name}"
            )


def _validate_protocol(config: Mapping[str, Any]) -> None:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = target.get("summary", {}).get("roles", {})
    if (
        target.get("schema_version")
        != "ember_pi05_target_data_manifest_v1"
        or int(target.get("summary", {}).get("tasks", -1)) != 40
        or int(target.get("summary", {}).get("episodes", -1)) != 2000
        or {
            name: len(roles.get(name, []))
            for name in ("train", "validation", "test")
        }
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise WriterModelError(
            "AS-Writer target-data authority is not sealed 24/8/8"
        )
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if (
        lora.source_base_config_sha256
        != config["authorities"]["source_base_config"]["sha256"]
    ):
        raise WriterModelError(
            "AS-Writer LoRA and source-base authorities disagree"
        )
    writer = config.get("writer", {})
    if (
        writer.get("frame_stride") != 5
        or int(writer.get("max_frames_per_encoder_call", 0)) <= 0
    ):
        raise WriterModelError(
            "sealed target-bound-role Writer dimensions changed"
        )
    expected = expected_writer_contract(writer)
    if writer != expected:
        missing = sorted(set(expected) - set(writer))
        extra = sorted(set(writer) - set(expected))
        changed = sorted(
            key
            for key in set(writer) & set(expected)
            if writer[key] != expected[key]
        )
        raise WriterModelError(
            "target-bound-role AS-Writer architecture changed; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    common = {
        "writer_input": (
            "task language plus exactly one raw action-hidden teacher video "
            "at inference"
        ),
        "writer_forbidden_inputs": [
            "action",
            "proprio",
            "state",
            "reward",
            "terminal",
            "task_id",
            "filename",
            "hidden_normalization",
            "policy_outcome",
        ],
        "action_owner": "frozen functional behavior loss only",
        "test_actions_read": 0,
        "test_video_values_read": 0,
    }
    if writer_stage(config) == "development":
        expected = {
            **common,
            "development_action_split_roles": ["train"],
            "development_video_split_roles": ["train"],
            "validation_actions_read_by_training_optimizer": 0,
            "validation_action_queries_per_checkpoint_monitor": 512,
            "validation_action_gradient": False,
        }
    else:
        expected = {
            **common,
            "final_action_split_roles": ["train", "validation"],
            "final_video_split_roles": ["train", "validation"],
        }
    if config.get("information_wall") != expected:
        raise WriterModelError("AS-Writer information wall changed")

    data = config.get("data", {})
    update_topology = str(
        config.get("conditioning_training", {}).get("update_topology", "")
    )
    teacher_video_sampling = (
        "per_task_cycle_visit_deterministic_single_same_task_video_in_"
        "no_replacement_cycles"
        if update_topology in {
            "serial4_exposure_matched_six_phase_task_cycle",
            "cycle_normalized_randomized_group4_six_phase_task_cycle",
        }
        else (
            "per_task_macro_visit_deterministic_single_same_task_video_in_"
            "no_replacement_cycles"
        )
    )
    required = {
        "task_count": 24 if writer_stage(config) == "development" else 32,
        "demo_indices": [0, 49],
        "episodes_per_task": 50,
        "teacher_video_sampling": teacher_video_sampling,
        "action_query_sampling": (
            "task-balanced deterministic no-replacement episode cycles with "
            "per-visit exact normalized-progress strata permutation and "
            "deterministic within-stratum jitter"
        ),
        "video_action_pairing": (
            "one task-video LoRA conditions the complete task-local action batch"
        ),
        "writer_generation_reuse": (
            "generate one task-video LoRA once then reuse it across the "
            "complete task-local action batch"
        ),
    }
    if any(data.get(name) != value for name, value in required.items()):
        raise WriterModelError("AS-Writer sampling contract changed")


def _conditioning_common(*, task_query_keyed: bool) -> dict[str, Any]:
    common: dict[str, Any] = {
        "writer_language_contract": (
            "correct_task_language_state_free_teacher_action_suffix"
        ),
        "policy_language_contract": "correct_action_query_task_language",
        "teacher_videos_per_task_visit": 1,
        "action_video_assignment": "all_actions_share_single_video_lora",
        "logical_pair_batch": "per_task_action_batch",
        "policy_noise_contract": (
            "one independent policy flow noise and time draw per action query"
        ),
        "single_video_gradient_direction_diagnostic": (
            "fixed_countsketch_32_per_task_per_semantic_frontend_core_program_"
            "compiler_factor_block"
        ),
        "normal_loss_weight": 1.0,
    }
    if task_query_keyed:
        common["policy_noise_contract"] = (
            "one task-query-keyed stateless policy flow noise and time draw "
            "per action query"
        )
        common["policy_randomness_scheme"] = (
            "task_query_keyed_stateless_policy_cpu_cuda_v2"
        )
    return common


def _full24_conditioning(
    common: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "update_topology": "task_complete_all_tasks",
        "action_query_batch_owner": (
            "six sequential task-pure physical action batches per rank per "
            "macro optimizer update"
        ),
        "task_assignment": (
            "every macro optimizer update covers all 24 tasks exactly once "
            "globally with six cost-balanced long-first tasks per rank"
        ),
        "tasks_per_rank_per_optimizer_update": 6,
        "global_tasks_per_optimizer_update": 24,
        "pair_loss_reduction": (
            "mean_within_task_then_equal_mean_over_24_tasks"
        ),
        "task_loss_scale_before_backward": (
            "per_task_unscaled_then_exact_raw_full24_mean"
        ),
        "ddp_gradient_sync": (
            "none_during_task_gradients_then_bounded_parameter_chunk_"
            "allgathers_for_exact_raw_full24_mean_and_read_only_grams"
        ),
        "gradient_composition": (
            "exact_raw_equal_weight_full24_mean_without_projection"
        ),
        "optimizer_steps_per_macro_update": 1,
        "checkpoint_boundary": "complete_macro_optimizer_update_only",
        **common,
    }


def _serial4_conditioning(common: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": (
            "raw_serial4_exposure_matched_single_video_multi_action_"
            "positive_functional_loss"
        ),
        "update_topology": "serial4_exposure_matched_six_phase_task_cycle",
        "action_query_batch_owner": (
            "one task-pure physical action batch per rank per optimizer update"
        ),
        "task_assignment": (
            "six optimizer phases reuse one full24 cost-balanced rank rotation; "
            "each phase selects one long-first task per rank and all six phases "
            "cover every task exactly once"
        ),
        "tasks_per_rank_per_optimizer_update": 1,
        "global_tasks_per_optimizer_update": 4,
        "optimizer_updates_per_task_cycle": 6,
        "scheduler_updates_per_task_cycle": 1,
        "task_visit_axis": "zero_based_task_cycle_floor_optimizer_update_div_6",
        "pair_loss_reduction": (
            "mean_within_task_then_equal_raw_mean_over_selected_4_tasks"
        ),
        "task_loss_scale_before_backward": (
            "per_task_unscaled_then_exact_raw_selected4_mean"
        ),
        "ddp_gradient_sync": (
            "none_during_task_gradients_then_bounded_parameter_chunk_"
            "allgathers_for_exact_raw_selected4_mean_and_read_only_4x4_grams"
        ),
        "gradient_composition": (
            "exact_raw_equal_weight_selected4_mean_without_projection"
        ),
        "optimizer_steps_per_task_cycle": 6,
        "scheduler_step_cadence": (
            "once_after_each_six_optimizer_update_task_cycle"
        ),
        "checkpoint_boundary": "complete_optimizer_update_only",
        **common,
    }


def _randomized_group4_conditioning(
    common: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": (
            "cycle_normalized_randomized_group4_single_video_multi_action_"
            "positive_functional_loss"
        ),
        "update_topology": (
            "cycle_normalized_randomized_group4_six_phase_task_cycle"
        ),
        "action_query_batch_owner": (
            "one task-pure physical action batch per rank per optimizer update"
        ),
        "task_assignment": (
            "six randomized Latin phases each select four tasks without cost "
            "input; every task appears once per cycle and phase-balanced over "
            "six-cycle superblocks with sealed complementary tail"
        ),
        "tasks_per_rank_per_optimizer_update": 1,
        "global_tasks_per_optimizer_update": 4,
        "optimizer_updates_per_task_cycle": 6,
        "scheduler_updates_per_task_cycle": 1,
        "task_visit_axis": "zero_based_task_cycle_floor_optimizer_update_div_6",
        "pair_loss_reduction": (
            "mean_within_task_then_equal_raw_mean_over_selected_4_tasks"
        ),
        "task_loss_scale_before_backward": (
            "per_task_unscaled_then_exact_raw_selected4_mean"
        ),
        "ddp_gradient_sync": (
            "none_during_task_gradients_then_bounded_parameter_chunk_"
            "allgathers_for_exact_raw_selected4_mean_and_read_only_4x4_grams"
        ),
        "gradient_composition": (
            "exact_raw_equal_weight_selected4_mean_without_projection"
        ),
        "optimizer_steps_per_task_cycle": 6,
        "scheduler_step_cadence": (
            "once_after_each_six_optimizer_update_task_cycle"
        ),
        "checkpoint_boundary": "complete_optimizer_update_only",
        **common,
    }


def _validate_conditioning_training(config: Mapping[str, Any]) -> None:
    value = config.get("conditioning_training", {})
    legacy_common = _conditioning_common(task_query_keyed=False)
    task_query_common = _conditioning_common(task_query_keyed=True)
    full24 = _full24_conditioning(
        legacy_common,
        method=(
            "raw_task_complete_single_video_multi_action_"
            "positive_functional_loss"
        ),
    )
    task_query_raw = _full24_conditioning(
        task_query_common,
        method=(
            "task_query_keyed_raw_task_complete_single_video_multi_action_"
            "positive_functional_loss"
        ),
    )
    serial4 = _serial4_conditioning(legacy_common)
    randomized_group4 = _randomized_group4_conditioning(task_query_common)
    if config.get("schema_version") in {
        AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA,
        TARGET_BOUND_ROLE_CONFIG_SCHEMA,
    }:
        expected = (
            randomized_group4
            if value.get("update_topology")
            == "cycle_normalized_randomized_group4_six_phase_task_cycle"
            else (
                task_query_raw
                if value.get("policy_randomness_scheme")
                == "task_query_keyed_stateless_policy_cpu_cuda_v2"
                else full24
            )
        )
    else:
        expected = (
            serial4
            if value.get("update_topology")
            == "serial4_exposure_matched_six_phase_task_cycle"
            else full24
        )
    if value != expected:
        raise WriterModelError("AS-Writer conditioning contract changed")


def _validate_cycle_normalized_optimization(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        not in {
            AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA,
            TARGET_BOUND_ROLE_CONFIG_SCHEMA,
        }
        or "cycle_normalization" not in config.get("optimization", {})
    ):
        return
    training = config["conditioning_training"]
    optimizer = config.get("optimization", {}).get("optimizer", {})
    scheduler = config.get("optimization", {}).get("scheduler", {})
    normalization = config.get("optimization", {}).get("cycle_normalization", {})
    group4 = (
        training.get("update_topology")
        == "cycle_normalized_randomized_group4_six_phase_task_cycle"
    )
    expected_divisor = 6 if group4 else 1
    expected_betas = (
        [0.9825931938526898, 0.9914875553891529]
        if group4
        else [0.9, 0.95]
    )
    expected_mode = (
        "cycle_normalized_randomized_group4"
        if group4
        else "task_query_keyed_raw_reference"
    )
    invalid = (
        optimizer.get("name") != "AdamW"
        or optimizer.get("betas") != expected_betas
        or optimizer.get("eps") != 1e-8
        or optimizer.get("weight_decay") != 1e-4
        or optimizer.get("gradient_clip_norm") != 1.0
        or normalization.get("mode") != expected_mode
        or normalization.get("optimizer_updates_per_task_cycle")
        != expected_divisor
        or normalization.get("lr_divisor") != expected_divisor
        or normalization.get("reference_betas") != [0.9, 0.95]
        or normalization.get("applied_betas") != expected_betas
        or normalization.get("reference_weight_decay") != 1e-4
        or normalization.get("scheduler_updates_per_task_cycle") != 1
        or scheduler.get("optimizer_updates_per_scheduler_step")
        != expected_divisor
        or config.get("optimization", {}).get("optimizer_diagnostics")
        != "per_owned_block_moment_parameter_and_actual_update_l2"
    )
    if invalid:
        raise WriterModelError("cycle-normalized optimizer contract changed")


def _validate_formal_schedule(config: Mapping[str, Any]) -> None:
    """Reject formal runs that would silently compress the sealed LR schedule."""

    formal = config.get("formal_run", {})
    scheduler = config.get("optimization", {}).get("scheduler", {})
    training = config.get("conditioning_training", {})
    updates_per_cycle = int(
        training.get("optimizer_updates_per_task_cycle", 1)
    )
    total_updates = int(formal.get("total_steps", 0))
    selected_stop = int(formal.get("selected_stop_step", 0))
    decay_steps = int(scheduler.get("decay_steps", 0))
    try:
        stage_stops = tuple(int(value) for value in formal["stage_stop_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise WriterModelError(
            "formal AS-Writer runtime would auto-scale or truncate its "
            "sealed scheduler"
        ) from error
    if (
        updates_per_cycle <= 0
        or total_updates <= 0
        or total_updates % updates_per_cycle
        or total_updates // updates_per_cycle < decay_steps
        or selected_stop <= 0
        or selected_stop > total_updates
        or selected_stop % updates_per_cycle
        or not stage_stops
        or stage_stops != tuple(sorted(set(stage_stops)))
        or stage_stops[-1] != total_updates
        or selected_stop not in stage_stops
        or any(
            stop <= 0
            or stop > total_updates
            or stop % updates_per_cycle
            for stop in stage_stops
        )
    ):
        raise WriterModelError(
            "formal AS-Writer runtime would auto-scale or truncate its "
            "sealed scheduler"
        )


def _load_recipe_overlay(
    config: Mapping[str, Any], *, overlay_schema: str
) -> dict[str, Any]:
    required = {
        "schema_version",
        "base_config",
        "base_sha256",
        "replace",
    }
    replacements = config.get("replace")
    allowed_replacements = {
        "data",
        "conditioning_training",
        "optimization",
        "profile_defaults",
        "profile_evidence",
        "formal_run",
    }
    base_path = (REPO_ROOT / str(config.get("base_config", ""))).resolve()
    if (
        set(config) != required
        or not isinstance(replacements, dict)
        or not set(replacements).issubset(allowed_replacements)
        or not base_path.is_relative_to(REPO_ROOT.resolve())
        or not base_path.is_file()
        or sha256_file(base_path) != config.get("base_sha256")
    ):
        raise WriterModelError("invalid AS-Writer recipe overlay")
    base = read_json(base_path)
    base.update({name: value for name, value in replacements.items()})
    if overlay_schema == AS_WRITER_SERIAL4_CONFIG_OVERLAY_SCHEMA:
        base["schema_version"] = AS_WRITER_SERIAL4_CONFIG_SCHEMA
    elif overlay_schema == AS_WRITER_CYCLE_NORMALIZED_CONFIG_OVERLAY_SCHEMA:
        base["schema_version"] = AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA
    elif overlay_schema == TARGET_BOUND_ROLE_CONFIG_OVERLAY_SCHEMA:
        base["schema_version"] = TARGET_BOUND_ROLE_CONFIG_SCHEMA
    base["_config_derivation"] = {
        "overlay_schema": overlay_schema,
        "base_config": str(base_path.relative_to(REPO_ROOT)),
        "base_sha256": config["base_sha256"],
    }
    return base


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    schema = config.get("schema_version")
    if schema in {
        TARGET_BOUND_ROLE_CONFIG_OVERLAY_SCHEMA,
        AS_WRITER_CONFIG_OVERLAY_SCHEMA,
        AS_WRITER_SERIAL4_CONFIG_OVERLAY_SCHEMA,
        AS_WRITER_CYCLE_NORMALIZED_CONFIG_OVERLAY_SCHEMA,
    }:
        config = _load_recipe_overlay(config, overlay_schema=str(schema))
    if config.get("schema_version") not in AS_WRITER_CONFIG_SCHEMAS:
        raise WriterModelError("unsupported PI05 AS-Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    _validate_conditioning_training(config)
    _validate_cycle_normalized_optimization(config)
    _validate_formal_schedule(config)
    return config
