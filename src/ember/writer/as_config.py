"""Sealed configuration loading for the canonical condition-kernel Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, sha256_file
from ember.writer.architecture import expected_writer_contract
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
CONDITION_KERNEL_CONFIG_SCHEMA = (
    "ember_pi05_factorized_condition_kernel_program_memory_as_writer_v1"
)
AS_WRITER_STAGES = ("development", "final")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    stage = str(config.get("sealed_stage", "development"))
    if stage not in AS_WRITER_STAGES:
        raise WriterModelError("unsupported PI05 AS-Writer stage")
    return stage


def writer_split_roles(config: Mapping[str, Any]) -> tuple[str, ...]:
    return ("train",) if writer_stage(config) == "development" else (
        "train",
        "validation",
    )


def resolve_mode_config(
    config: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    if mode not in {"profile", "formal"}:
        raise WriterModelError("unsupported PI05 AS-Writer runtime mode")
    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    if mode == "profile":
        profile_seed = config.get("profile_evidence", {}).get(
            "profile_teacher_video_seed"
        )
        if profile_seed is not None:
            resolved["data"]["teacher_video_seed"] = int(profile_seed)
    return resolved


def _validate_authorities(config: Mapping[str, Any]) -> None:
    authorities = config.get("authorities", {})
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
        "condition_address",
    }
    if set(authorities) != required:
        raise WriterModelError("condition-kernel authority set changed")
    for name, authority in authorities.items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if (
            not artifact.is_file()
            or sha256_file(artifact) != authority.get("sha256")
        ):
            raise WriterModelError(
                f"sealed condition-kernel authority changed: {name}"
            )


def _validate_protocol(config: Mapping[str, Any]) -> None:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = target.get("summary", {}).get("roles", {})
    if (
        target.get("schema_version")
        != "ember_pi05_target_data_manifest_v1"
        or int(target.get("summary", {}).get("tasks", -1)) != 40
        or int(target.get("summary", {}).get("episodes", -1)) != 2000
        or {name: len(roles.get(name, [])) for name in ("train", "validation", "test")}
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise WriterModelError("condition-kernel split authority changed")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if (
        lora.source_base_config_sha256
        != config["authorities"]["source_base_config"]["sha256"]
    ):
        raise WriterModelError("condition-kernel source and LoRA disagree")
    writer = config.get("writer", {})
    if (
        writer.get("frame_stride") != 5
        or int(writer.get("max_frames_per_encoder_call", 0)) <= 0
    ):
        raise WriterModelError("condition-kernel frame contract changed")
    expected = expected_writer_contract(writer)
    if writer != expected:
        raise WriterModelError("condition-kernel Writer architecture changed")


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
    expected = (
        {
            **common,
            "development_action_split_roles": ["train"],
            "development_video_split_roles": ["train"],
            "validation_actions_read_by_training_optimizer": 0,
            "validation_action_queries_per_checkpoint_monitor": 512,
            "validation_action_gradient": False,
        }
        if writer_stage(config) == "development"
        else {
            **common,
            "final_action_split_roles": ["train", "validation"],
            "final_video_split_roles": ["train", "validation"],
        }
    )
    if config.get("information_wall") != expected:
        raise WriterModelError("condition-kernel information wall changed")


def _validate_data(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    required = {
        "task_count": 24 if writer_stage(config) == "development" else 32,
        "episodes_per_task": 50,
        "demo_indices": [0, 49],
        "teacher_video_sampling": (
            "per_task_macro_visit_deterministic_single_same_task_video_in_"
            "no_replacement_cycles"
        ),
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
        raise WriterModelError("condition-kernel data schedule changed")


def _validate_training(config: Mapping[str, Any]) -> None:
    training = config.get("conditioning_training", {})
    profile_world = int(config.get("profile_defaults", {}).get("expected_world_size", 0))
    formal_world = int(config.get("formal_run", {}).get("expected_world_size", 0))
    tasks_per_rank = 24 // formal_world if formal_world in {4, 6} else -1
    required = {
        "method": (
            "factorized_condition_kernel_program_memory_task_query_keyed_"
            "independent_b20_functional_cotangent"
        ),
        "update_topology": "task_complete_all_tasks",
        "tasks_per_rank_per_optimizer_update": tasks_per_rank,
        "global_tasks_per_optimizer_update": 24,
        "teacher_videos_per_task_visit": 1,
        "action_video_assignment": "all_actions_share_single_video_lora",
        "pair_loss_reduction": "mean_within_task_then_equal_mean_over_24_tasks",
        "program_memory_update": (
            "full24_regularized_condition_gram_solve_relative_damping_0.01"
        ),
        "program_memory_optimizer": "none",
        "factor_decoder_train_through_macro": 50,
        "factor_decoder_frozen_after_macro": 50,
        "policy_randomness_scheme": "task_query_keyed_stateless_policy_cpu_cuda_v2",
        "policy_flow_time_sampling_scheme": "task_query_keyed_independent_beta15_time_v1",
        "policy_flow_noise_sampling_scheme": "task_query_keyed_independent_gaussian_v1",
        "checkpoint_boundary": "complete_full24_kernel_update_only",
        "normal_loss_weight": 1,
    }
    if (
        profile_world != formal_world
        or tasks_per_rank <= 0
        or any(training.get(name) != value for name, value in required.items())
    ):
        raise WriterModelError("condition-kernel training contract changed")


def _validate_optimization(config: Mapping[str, Any]) -> None:
    optimization = config.get("optimization", {})
    decoder = optimization.get("factor_decoder_optimizer", {})
    memory = optimization.get("program_memory_update", {})
    scheduler = optimization.get("scheduler", {})
    if (
        decoder
        != {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1,
        }
        or memory.get("relative_damping") != 0.01
        or float(memory.get("step_size", 0)) <= 0
        or float(memory.get("induced_program_rms_cap", 0)) <= 0
        or scheduler.get("kind") != "cosine_decay_with_warmup"
        or scheduler.get("step_axis") != "completed_full24_task_cycle"
        or int(optimization.get("functional_policy_microbatch_size", 0)) <= 0
    ):
        raise WriterModelError("condition-kernel optimization contract changed")


def _validate_schedule(config: Mapping[str, Any]) -> None:
    formal = config.get("formal_run", {})
    profile = config.get("profile_defaults", {})
    if (
        int(formal.get("total_steps", 0)) != 200
        or formal.get("checkpoint_steps") != [50, 100, 150, 200]
        or formal.get("stage_stop_steps") != [50, 100, 150, 200]
        or int(formal.get("selected_stop_step", 0)) != 200
        or int(profile.get("total_steps", 0)) != 3
        or profile.get("checkpoint_steps") != [1, 2, 3]
    ):
        raise WriterModelError("condition-kernel formal schedule changed")
    formal_seed = config.get("profile_evidence", {}).get(
        "formal_teacher_video_seed_after_profile_seal"
    )
    if (
        formal.get("status") == "sealed"
        and formal_seed is not None
        and int(config.get("data", {}).get("teacher_video_seed", -1))
        != int(formal_seed)
    ):
        raise WriterModelError("formal config retained its profile video seed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != CONDITION_KERNEL_CONFIG_SCHEMA:
        raise WriterModelError("unsupported PI05 AS-Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    _validate_data(config)
    _validate_training(config)
    _validate_optimization(config)
    _validate_schedule(config)
    return config
