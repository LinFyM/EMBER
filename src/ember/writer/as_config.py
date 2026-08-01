"""Sealed configuration loading and validation for PI05 AS-Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, sha256_file
from ember.writer.architecture import expected_writer_contract
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = (
    "ember_pi05_amplitude_preserving_dual_read_full24_as_writer_v1"
)
AS_WRITER_CONFIG_OVERLAY_SCHEMA = (
    "ember_pi05_amplitude_preserving_dual_read_full24_recipe_overlay_v1"
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
            "sealed AP-ADR Writer dimensions changed"
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
            "AP-ADR AS-Writer architecture changed; "
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
    teacher_video_sampling = (
        "per_task_macro_visit_deterministic_single_same_task_video_in_"
        "no_replacement_cycles"
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


def _validate_conditioning_training(config: Mapping[str, Any]) -> None:
    value = config.get("conditioning_training", {})
    common = {
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
    full24 = {
        "method": (
            "raw_task_complete_single_video_multi_action_"
            "positive_functional_loss"
        ),
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
    if value != full24:
        raise WriterModelError("AS-Writer conditioning contract changed")


def _load_recipe_overlay(config: Mapping[str, Any]) -> dict[str, Any]:
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
    base["schema_version"] = AS_WRITER_CONFIG_SCHEMA
    base["_config_derivation"] = {
        "overlay_schema": AS_WRITER_CONFIG_OVERLAY_SCHEMA,
        "base_config": str(base_path.relative_to(REPO_ROOT)),
        "base_sha256": config["base_sha256"],
    }
    return base


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    schema = config.get("schema_version")
    if schema == AS_WRITER_CONFIG_OVERLAY_SCHEMA:
        config = _load_recipe_overlay(config)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported PI05 AS-Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    _validate_conditioning_training(config)
    return config
