"""Frozen configuration checks for the four conditional-compilation diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ember.pi05_source_checkpoint import read_json
from ember.writer.learning_data import CONDITIONAL_EVENT_SCHEMA, query_allocation
from ember.writer.runtime import require_architecture_identity


CONFIG_SCHEMA = "ember_conditional_compilation_diagnostics_config_v1"
UPDATE_VERSION = "conditional_compilation_diagnostics_v1"
EXPERIMENT = "conditional_compilation_diagnostics_20260923"
SPEC_PATH = "configs/conditional_compilation_diagnostics_v1/experiment_spec.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
PARAMETERIZATIONS = {
    "A_direct16": "direct_lora",
    "B_language": "language_writer",
    "C_video_fm": "video_writer",
    "D_video_aux": "video_writer",
}


def _validate_data_and_model(config: dict[str, Any], spec: dict[str, Any],
                             parent: dict[str, Any], parameterization: str) -> None:
    sampling, opt = spec["sampling"], spec["optimization"]
    expected_data = {
        "version": CONDITIONAL_EVENT_SCHEMA, "event_schema_version": CONDITIONAL_EVENT_SCHEMA,
        "protocol": spec["protocol"]["parent"], "task_ids": spec["protocol"]["fit28"],
        "extra_meta_tasks": [], "video_demos": sampling["fit_video_demos"],
        "action_demos": sampling["fit_action_demos"],
        "diagnostic_action_demos": sampling["reserved_demo_indices"],
        "held_video_demos": sampling["reserved_demo_indices"],
        "frame_stride": sampling["frame_stride"], "include_last_frame": sampling["include_last_frame"],
        "maximum_updates": opt["updates"], "tasks_per_update": sampling["tasks_per_update"],
        "conditions_per_task": 1, "cardinalities": [1],
        "queries_per_task": sampling["main_queries_per_condition"],
        "teaching_queries_per_task": sampling["extra_queries_per_condition"],
        "action_start_offset": sampling["action_start_offset"], "query_alignment": sampling["query_alignment"],
        "seed": sampling["model_seed"], "sampler_seed": sampling["sampler_seed"],
        "teacher_video_seed": sampling["teacher_video_seed"], "teaching_seed": sampling["teaching_seed"],
        "grouping": "baseline", "teaching_episode": sampling["extra_episode_relation"],
    }
    if any(config["data"].get(key) != value for key, value in expected_data.items()):
        raise ValueError("conditional compilation task or event contract changed")
    if config["source"] != parent["source"] or config["model"] != parent["model"]:
        raise ValueError("conditional compilation source or Writer topology changed")
    try:
        require_architecture_identity(config["model"])
    except ValueError as error:
        raise ValueError("conditional compilation base Writer configuration changed") from error
    if config["observer"] != {**parent["observer"], "route": parameterization}:
        raise ValueError("conditional compilation deployment input route changed")


def _validate_objective(config: dict[str, Any], spec: dict[str, Any],
                        arm: dict[str, Any], parameterization: str) -> None:
    arm_id = arm["id"]
    sampling, opt = spec["sampling"], spec["optimization"]
    expected_optimization = {
        "loss": arm["objective"], "joint_train_all_writer_modules": arm_id in {"C_video_fm", "D_video_aux"},
        "normalizer": 1.0, "teaching_weight": sampling["extra_group_weight"],
        "teaching_prefix_steps": 5, "teaching_flow_time": 1,
        "seed": opt["seed"], "lr": opt["lr"], "betas": opt["betas"], "eps": opt["eps"],
        "weight_decay": opt["weight_decay"], "grad_clip": opt["grad_clip"],
        "warmup_updates": opt["warmup_updates"], "decay_updates": opt["decay_updates"],
        "decay_lr": opt["decay_lr"], "tail_start_update": opt["tail_start_update"],
        "tail_end_update": opt["tail_end_update"], "tail_final_ratio": opt["tail_final_ratio"],
    }
    if config["optimization"] != expected_optimization:
        raise ValueError("conditional compilation optimizer or objective changed")
    expected_experiment = {"kind": EXPERIMENT, "arm_id": arm_id,
                           "parameterization": parameterization, "objective": arm["objective"],
                           "extra_endpoint_prefix": arm_id == "D_video_aux"}
    if config["experiment"] != expected_experiment:
        raise ValueError("conditional compilation arm definition changed")


def _validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("status") not in {"pending", "complete"}:
        raise ValueError("conditional profile registration has an unknown state")
    if profile["status"] == "pending":
        return
    batches = profile.get("policy_microbatches")
    measurements = (profile.get("mean_update_seconds"), profile.get("peak_reserved_gib"))
    if (type(profile.get("world_size")) is not int or not 1 <= profile["world_size"] <= 6
            or not isinstance(profile.get("reference"), str) or not profile["reference"]
            or not isinstance(batches, list) or len(batches) != profile["world_size"]
            or any(type(value) is not int or value <= 0 for value in batches)
            or type(profile.get("validated_updates")) is not int or profile["validated_updates"] <= 0
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                   for value in measurements)):
        raise ValueError("completed profile registration is incomplete or malformed")


def _validate_execution(config: dict[str, Any], spec: dict[str, Any]) -> None:
    opt = spec["optimization"]
    evidence = config["evidence"]
    if (evidence.get("checkpoint_updates") != list(range(opt["checkpoint_interval"],
                                                          opt["updates"] + 1, opt["checkpoint_interval"]))
            or evidence.get("evaluation_updates") != opt["evaluation_updates"]
            or evidence.get("supervised_validation", {}).get("optimizer_updates") != []):
        raise ValueError("conditional compilation checkpoint or action-read schedule changed")
    _validate_profile(evidence.get("profile_registration", {}))
    if any(type(value) is not int or value <= 0 for value in config["runtime"].values()):
        raise ValueError("conditional compilation physical batches and cache budget must be positive")
    if config.get("training_control") or config.get("continuation") or config.get("phase_continuation"):
        raise ValueError("conditional compilation is fresh with a fixed 1260-update window")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Require an arm to match the sealed scientific spec and current base recipe."""
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("study_spec") != SPEC_PATH:
        raise ValueError("conditional compilation config schema or study spec changed")
    spec = read_json(REPO_ROOT / SPEC_PATH)
    if spec.get("study_id") != EXPERIMENT:
        raise ValueError("conditional compilation config points at a different study spec")
    arms = {arm["id"]: arm for arm in spec["arms"]}
    arm_id = config.get("experiment", {}).get("arm_id")
    if arm_id not in PARAMETERIZATIONS or arm_id not in arms:
        raise ValueError("conditional compilation arm is not in the registered specification")
    parent = read_json(REPO_ROOT / "configs/libero_24_8_8_coverage_v1/writer.json")
    parameterization = PARAMETERIZATIONS[arm_id]
    if (config.get("status") != "registered_video_teaching_learning"
            or config.get("update_version") != UPDATE_VERSION
            or config.get("execution_precision") != "native_bf16_writer_fm_fp32_lora"):
        raise ValueError("conditional compilation run identity changed")
    _validate_data_and_model(config, spec, parent, parameterization)
    _validate_objective(config, spec, arms[arm_id], parameterization)
    _validate_execution(config, spec)
    query_allocation(config["data"], 0)
    return config
