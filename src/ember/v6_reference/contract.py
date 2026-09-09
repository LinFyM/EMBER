"""Explicit scope of the historical v6 model under the current lawful protocol."""

from __future__ import annotations

from typing import Any, Mapping

from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority
from ember.v6_reference.architecture import (
    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS, WRITER_DIMENSION_CONTRACT,
)


HISTORICAL_COMMIT = "bad9a961ff864d593cf0eb5b05d93930cd2280c1"
CONFIG_SCHEMA = "ember_exploratory_v6_matched_config_v1"
RUN_SCHEMA = "ember_exploratory_v6_matched_run_v1"
STAGE = "exploratory_v6_matched_supervised"
TRAINING_SCHEMA = "ember_exploratory_v6_training_state_v1"
UPDATE_VERSION = "exploratory_v6_matched_supervised_fm_v1"
PRECISION = "legacy_v6_bf16_autocast_current_fm_microbatch"
BANK_KIND = "exploratory_v6_lora_bank"
BANK_SCHEMA = "ember_exploratory_v6_lora_bank_v1"
ADAPTER_SCHEMA = "ember_exploratory_v6_materialized_adapter_v1"
EVALUATION_SCHEMA = "ember_exploratory_v6_eval_adapter_v1"
EPISODE_SCHEMA = "ember_exploratory_v6_episode_v1"
PROBE_SEED = 7 + 0x5A17
TRAIN_TASKS = (0, 2, 4, 5, 7, 9, 12, 14, 15, 16, 18, 19,
               20, 21, 22, 25, 28, 29, 34, 35, 36, 37, 38, 39)


def is_reference(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == CONFIG_SCHEMA


def execution_authority(state: Mapping[str, Any]) -> bool:
    """Exploration is pinned to the clean pushed detached reference branch."""
    return bool(state.get("branch") == ""
                and state.get("authority_ref") == "origin/codex/v6-causal-reference"
                and git_state_is_clean_pushed_or_frozen_authority(state))


def _require_fields(value, expected, label):
    if {key: value.get(key) for key in expected} != expected:
        raise ValueError(f"exploratory v6 {label} contract changed")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model, data, opt = config["model"], config["data"], config["optimization"]
    _require_fields(config, {"schema_version": CONFIG_SCHEMA, "historical_model_commit": HISTORICAL_COMMIT,
                            "update_version": UPDATE_VERSION, "execution_precision": PRECISION}, "method")
    _require_fields(model, {**WRITER_DIMENSION_CONTRACT, "initialization_seed": 7,
                            "activation_checkpointing": True}, "historical model")
    if set(model) != LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS or int(model["max_frames_per_encoder_call"]) <= 0:
        raise ValueError("exploratory v6 model constructor or frame chunk changed")
    _require_fields(data, {"extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
                          "queries_per_task": 64, "tasks_per_update": 4, "cardinalities": [1], "seed": 7,
                          "task_ids": list(TRAIN_TASKS), "version": "train24_supervised_suite_rng_cross_episode_k1_v2",
                          "video_demos": list(range(16)), "action_demos": list(range(16, 42)),
                          "diagnostic_action_demos": list(range(42, 46)),
                          "held_video_demos": list(range(46, 50))}, "matched sampling")
    _require_fields(opt, {"seed": 7, "lr": 3e-5, "betas": [0.9, 0.95], "eps": 1e-8,
                         "weight_decay": 1e-4, "grad_clip": 1.0, "warmup_updates": 8,
                         "loss": "supervised_fm", "normalizer": 1.0,
                         "joint_train_all_writer_modules": True}, "matched optimization")
    if "rl" in config or "trust_scales" in opt:
        raise ValueError("exploratory v6 is positive supervised FM only")
    _require_fields(config["observer"], {"probe_seed": PROBE_SEED, "flow_time": 1,
                    "meta_groups": ["text", "vl", "action"], "meta_rank": 4,
                    "prefix_cache": "forbidden_trainable_text_vl_meta",
                    "response": "mean_50_final_suffix_hidden_then_shared_bias_free_1024_to_256"}, "observer")
    if set(config["runtime"]) != {"policy_microbatch"} or int(config["runtime"]["policy_microbatch"]) <= 0:
        raise ValueError("exploratory v6 execution uses physical FM chunks and no prefix cache")
    evidence = config["evidence"]
    _require_fields(evidence, {"checkpoint_updates": [200, 400], "final_checkpoint_selection": False,
                              "shuffled_reversed_use": False, "test_use": False}, "evidence")
    _require_fields(evidence["supervised_validation"], {
        "task_ids": list(TRAIN_TASKS), "optimizer_updates": [200, 400], "queries_per_task": 128,
        "teacher_video_pool": list(range(46, 50)), "action_demos": list(range(42, 46)),
        "K": 1, "seed": 20260908, "gradients": False, "checkpoint_selection": False,
    }, "fixed held diagnostic")
    return dict(config)



def schemas(config: Mapping[str, Any]) -> tuple[str, str, str]:
    if not is_reference(config):
        raise ValueError("v6 schema lookup received another method")
    return RUN_SCHEMA, STAGE, TRAINING_SCHEMA


def method_metadata(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": "exploratory_v6_matched_current_protocol", "historical_model_commit": HISTORICAL_COMMIT,
        "model_config": run["model_config"], "observer": run["config"]["observer"],
        "execution_precision": PRECISION, "training_stage": STAGE, "training_objective": "supervised_fm",
        "update_version": UPDATE_VERSION, "scientific_qualification": False,
        "checkpoint_state": "complete fresh v6 Writer including Text/VL/Action Meta and original public probe",
        "frame_stride": 5, "include_last_frame": True, "camera": "agentview_rotated_180",
        "native_response_shape": [50, 1024], "interaction_reduction": "fixed_horizon_mean_before_projection",
        "visual_token_gradient": "through_trainable_vl_meta", "frame_attention": "two_causal_procedure_blocks",
        "compiler_slots": 320, "factor_heads": "eight_family_side_heads_shared_across_layer_and_rank",
        "execution_rank": 16, "generated_tensor_count": 76, "macro_cursor": "optimizer_updates",
    }


def validate_selection(selection):
    if (selection.get("K") != 1 or selection.get("arm") != "correct"
            or selection.get("mode") != "per_init_ordinal"):
        raise ValueError("initial v6 causal reference uses only registered correct K1 evidence")
    if selection["evaluation_role"] == "validation":
        valid = (selection["task_ids"] == [1, 3, 11, 13, 23, 26, 31, 32]
                 and selection["init_state_ids"] == list(range(50))
                 and selection["video_pool"] == list(range(50)))
    else:
        valid = (selection["evaluation_role"] == "development_train"
                 and selection["task_ids"] == list(TRAIN_TASKS)
                 and selection["init_state_ids"] == list(range(32, 36))
                 and selection["video_pool"] == list(range(46, 50)))
    if not valid or selection.get("seed") != 20260907:
        raise ValueError("v6 closed-loop evidence requires the fixed paired validation400 or held-video train96 panel")
