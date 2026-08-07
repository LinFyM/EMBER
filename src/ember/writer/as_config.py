"""Configuration for the K4 phase-aligned Language-Axial Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import expected_writer_contract
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
K4_PHASE_ALIGNED_CONFIG_SCHEMA = (
    "ember_pi05_k4_phase_aligned_language_axial_semantic_procedure_as_writer_v1"
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


def resolve_mode_config(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"profile", "formal"}:
        raise WriterModelError("unsupported PI05 AS-Writer runtime mode")
    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    if mode == "profile":
        seed = config.get("profile_evidence", {}).get("profile_teacher_video_seed")
        if seed is not None:
            resolved["data"]["teacher_video_seed"] = int(seed)
    return resolved


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
        raise WriterModelError("K4 phase-aligned authority set changed")
    for name, authority in authorities.items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if not artifact.is_file():
            raise WriterModelError(f"missing K4 phase-aligned authority: {name}")


def _validate_protocol(config: Mapping[str, Any]) -> None:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = target.get("summary", {}).get("roles", {})
    if (
        target.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(target.get("summary", {}).get("tasks", -1)) != 40
        or int(target.get("summary", {}).get("episodes", -1)) != 2000
        or {name: len(roles.get(name, [])) for name in ("train", "validation", "test")}
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise WriterModelError("K4 phase-aligned split authority changed")
    writer = config.get("writer", {})
    if writer.get("frame_stride") != 5 or int(
        writer.get("max_frames_per_encoder_call", 0)
    ) <= 0:
        raise WriterModelError("K4 phase-aligned frame contract changed")
    if writer != expected_writer_contract(writer):
        raise WriterModelError("K4 phase-aligned Writer architecture changed")


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    common = {
        "writer_input": (
            "task language plus exactly four raw action-hidden same-task "
            "teacher videos jointly generating one LoRA"
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
            "validation_action_queries_per_checkpoint_monitor": 0,
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
        raise WriterModelError("K4 phase-aligned information wall changed")


def _validate_data(config: Mapping[str, Any]) -> None:
    required = {
        "task_count": 24 if writer_stage(config) == "development" else 32,
        "episodes_per_task": 50,
        "demo_indices": [0, 49],
        "teacher_video_sampling": (
            "per_task_macro_visit_deterministic_four_unique_same_task_videos"
        ),
        "action_query_sampling": (
            "task-balanced deterministic no-replacement episode cycles with "
            "per-visit exact normalized-progress strata permutation and "
            "deterministic within-stratum jitter"
        ),
        "video_action_pairing": (
            "action batch fixed first then four teacher episodes selected from "
            "its exact complement"
        ),
        "writer_generation_reuse": (
            "generate one joint K4 LoRA once then reuse it across the complete "
            "task-local action batch"
        ),
    }
    data = config.get("data", {})
    if any(data.get(name) != value for name, value in required.items()):
        raise WriterModelError("K4 phase-aligned data schedule changed")


def _validate_training(config: Mapping[str, Any]) -> None:
    training = config.get("conditioning_training", {})
    profile_world = int(config.get("profile_defaults", {}).get("expected_world_size", 0))
    formal_world = int(config.get("formal_run", {}).get("expected_world_size", 0))
    tasks_per_rank = 24 // formal_world if formal_world in {4, 6} else -1
    required = {
        "method": "k4_phase_aligned_language_axial_semantic_procedure_rawfull24",
        "update_topology": "task_complete_all_tasks",
        "tasks_per_rank_per_optimizer_update": tasks_per_rank,
        "global_tasks_per_optimizer_update": 24,
        "teacher_videos_per_task_visit": 4,
        "action_video_assignment": "all_actions_share_one_joint_k4_lora",
        "pair_loss_reduction": "mean_within_task_then_equal_mean_over_24_tasks",
        "policy_randomness_scheme": "task_query_keyed_stateless_policy_cpu_cuda_v2",
        "policy_flow_time_sampling_scheme": "task_query_keyed_independent_beta15_time_v1",
        "policy_flow_noise_sampling_scheme": "task_query_keyed_independent_gaussian_v1",
        "checkpoint_boundary": "complete_full24_end_to_end_update_only",
        "normal_loss_weight": 1,
    }
    if (
        profile_world != formal_world
        or tasks_per_rank <= 0
        or any(training.get(name) != value for name, value in required.items())
    ):
        raise WriterModelError("K4 phase-aligned training contract changed")


def _validate_optimization(config: Mapping[str, Any]) -> None:
    optimization = config.get("optimization", {})
    optimizer = optimization.get("optimizer", {})
    scheduler = optimization.get("scheduler", {})
    if (
        optimizer
        != {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1,
        }
        or scheduler.get("kind") != "cosine_decay_with_warmup"
        or scheduler.get("step_axis") != "completed_full24_task_cycle"
        or int(optimization.get("functional_policy_microbatch_size", 0)) <= 0
    ):
        raise WriterModelError("K4 phase-aligned optimization contract changed")


def _validate_schedule(config: Mapping[str, Any]) -> None:
    formal = config.get("formal_run", {})
    profile = config.get("profile_defaults", {})
    if (
        int(formal.get("total_steps", 0)) != 400
        or formal.get("checkpoint_steps") != "every:25"
        or formal.get("stage_stop_steps") != [50, 100, 150, 200, 250, 300, 350, 400]
        or int(formal.get("selected_stop_step", 0)) != 200
        or int(profile.get("total_steps", 0)) != 400
        or profile.get("checkpoint_steps") != [1, 2, 3, 400]
        or profile.get("optimizer_clock")
        != "formal_400_step_scheduler_with_early_stop_at_3"
    ):
        raise WriterModelError("K4 phase-aligned schedule changed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != K4_PHASE_ALIGNED_CONFIG_SCHEMA:
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
