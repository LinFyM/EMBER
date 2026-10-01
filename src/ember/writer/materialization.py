"""Read-only contracts, pairing and provenance for sealed video Writer banks.

Historical training and compilation are retired from main. Use each run's
frozen commit, and check checkpoint_retirement.json / payload_retirement.json
before attempting replay; a weights-only archive is not an exact-resume state.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET, paired_condition_demo_indices, reference_demo_indices,
)
from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.data import teacher_camera_names
from ember.writer.learning_data import EVENT_SCHEMA
from ember.writer.conditional_contract import (
    CONFIG_SCHEMA as CONDITIONAL_CONFIG_SCHEMA, validate_config as _conditional_config,
)
from ember.writer.relational_contract import (
    CONFIG_SCHEMA as RELATIONAL_CONFIG_SCHEMA, validate_config as _relational_config,
)
from ember.writer.language_content_contract import (
    CONFIG_SCHEMA as LANGUAGE_CONTENT_CONFIG_SCHEMA, validate_config as _language_content_config,
)
from ember.writer.learned_initial_content_contract import (
    CONFIG_SCHEMA as INITIAL_CONTENT_CONFIG_SCHEMA, validate_config as _initial_content_config,
)
from ember.writer.video_controls import CONTROL_ARMS, require_control_selection, video_task_id
from ember.writer.runtime import require_architecture_identity

CONFIG_SCHEMA = "ember_video_teaching_writer_config_v1"
RUN_SCHEMA = "ember_video_teaching_writer_run_v1"
STAGE = "video_teaching_writer_fresh"
TRAINING_SCHEMA = "ember_video_teaching_training_state_v1"
UPDATE_VERSION = "video_teaching_twelve_condition_joint_meta_v2"

def observer_mode_contract(model: dict[str, Any]) -> dict[str, str]:
    """Bind the ordered full-horizon and adjacent-content native reads."""
    require_architecture_identity(model)
    patches = 512 if model["camera_view"] == "dual" else 256
    return {"camera_view": model["camera_view"],
            "native_inputs": f"full{patches}_patch_content_and_repeated_full50_H_adjacent_E_reads",
            "horizon_read": "repeated_content_position_attention_over_all_50_raw_H_values",
            "video_order": "causal_RoPE_with_real_frame_positions_and_ordered_adjacent_roles"}

BANK_SCHEMA = "ember_video_writer_lora_bank_v1"

BANK_KIND = "horizon_writer_lora_bank"

ADAPTER_SCHEMA = "ember_video_writer_materialized_adapter_v1"

TRAIN_DIAGNOSTIC_INIT_STATE_IDS = tuple(range(32, 36))

VIDEO_SCHEDULE = "expert_manifold_canonical_permutation_v1"

DEFAULT_SELECTION_SEED = 20260907

REPO_ROOT = Path(__file__).resolve().parents[3]

def frozen_authority(state: Mapping[str, Any]) -> bool:
    return state.get("branch") == "" and git_state_is_clean_pushed_or_frozen_authority(state)

def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}

def source_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("source_run", "checkpoint", "model_path")
    return all(left.get(key) and right.get(key) and
               Path(left[key]).resolve() == Path(right[key]).resolve() for key in keys)

def inspect_writer_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check formal supervised authority with metadata-only trainer tensor loading."""
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    run_path = checkpoint.parent.parent / "run_contract.json"
    run, manifest = read_json(run_path), read_json(checkpoint / "checkpoint_manifest.json")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"supervised Writer checkpoint file changed: {name}")
    # Inspect scalar provenance without reading optimizer tensor payloads.
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True, weights_only=True)
    if run["config"]["data"]["maximum_updates"] is not None and macro > run["config"]["data"]["maximum_updates"]:
        raise ValueError("checkpoint exceeds the registered training budget")
    observer = observer_mode_contract(run.get("model_config", {}))
    if run["model_config"] != run.get("config", {}).get("model"):
        raise ValueError("checkpoint and run configuration disagree on the video Writer architecture")
    world_size = int(manifest.get("world_size", 0))
    expected = {"ecp.safetensors", "trainer_state.pt", *(f"rank_{rank:02d}_state.pt" for rank in range(world_size))}
    config = run["config"]
    data = config.get("data", {})
    conditional = config.get("schema_version") in {
        CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA, LANGUAGE_CONTENT_CONFIG_SCHEMA,
        INITIAL_CONTENT_CONFIG_SCHEMA}
    if conditional:
        {RELATIONAL_CONFIG_SCHEMA: _relational_config,
         CONDITIONAL_CONFIG_SCHEMA: _conditional_config,
         LANGUAGE_CONTENT_CONFIG_SCHEMA: _language_content_config,
         INITIAL_CONTENT_CONFIG_SCHEMA: _initial_content_config}[config["schema_version"]](config)
        if config["schema_version"] in {LANGUAGE_CONTENT_CONFIG_SCHEMA, INITIAL_CONTENT_CONFIG_SCHEMA} and macro > 630:
            raise ValueError("language-content bank cannot use a checkpoint after macro630")
        parameterization = config["experiment"]["parameterization"]
        observer = {**observer, "route": parameterization}
        identities = (
            (run, {"schema_version": RUN_SCHEMA, "stage": STAGE, "mode": "formal"}),
            (config, {"schema_version": config["schema_version"], "update_version": config["update_version"],
                      "execution_precision": "native_bf16_writer_fm_fp32_lora"}),
            (config.get("optimization", {}), {"loss": config["experiment"]["objective"]}),
            (config.get("observer", {}), observer),
            (data, {"version": data["event_schema_version"], "action_start_offset": 1,
                    "query_alignment": "post_action_observation_future_control_v1"}),
            (manifest, {"schema_version": ECP_CHECKPOINT_SCHEMA, "stage": STAGE,
                        "run_contract_schema": RUN_SCHEMA, "next_macro": macro}),
        )
    else:
        identities = (
            (run, {"schema_version": RUN_SCHEMA, "stage": STAGE, "mode": "formal"}),
            (config, {"schema_version": CONFIG_SCHEMA, "update_version": UPDATE_VERSION,
                      "execution_precision": "native_bf16_writer_fm_fp32_lora"}),
            (config.get("optimization", {}), {"loss": "main_fm_plus_video_teaching"}),
            (config.get("observer", {}), observer),
            (data, {"version": EVENT_SCHEMA, "action_start_offset": 1,
                    "query_alignment": "post_action_observation_future_control_v1"}),
            (manifest, {"schema_version": ECP_CHECKPOINT_SCHEMA, "stage": STAGE,
                        "run_contract_schema": RUN_SCHEMA, "next_macro": macro}),
        )
    if (macro <= 0 or not 1 <= world_size <= 6
            or any(value.get(key) != wanted for value, fields in identities for key, wanted in fields.items())
            or {"local_field_supervision", "correction_supervision", "spatial_supervision"} & config.keys()
            or type(data.get("action_start_offset")) is not int
            or not frozen_authority(run.get("git", {})) or set(manifest.get("files", {})) != expected):
        raise ValueError("materialization requires a complete formal supervised Writer checkpoint")
    training = trainer.get("training_state", {})
    data_version = run.get("config", {}).get("data", {}).get("version")
    if (trainer.get("schema_version") != ECP_CHECKPOINT_SCHEMA or trainer.get("stage") != STAGE
            or trainer.get("next_macro") != macro or not data_version):
        raise ValueError("supervised Writer training state or optimizer-update cursor changed")
    expected_training = {"schema_version": TRAINING_SCHEMA, "updates": macro,
                         "update_version": config["update_version"], "data_version": data_version}
    if run.get("support_slot_credit") is not None:
        raise ValueError("retired support-slot diagnostic requires its recorded frozen runtime")
    if training != expected_training:
        raise ValueError("supervised Writer training state or optimizer-update cursor changed")
    return run, {"path": str(checkpoint), "macro": macro,
                 "weights": file_record(checkpoint / "ecp.safetensors"),
                 "manifest": file_record(checkpoint / "checkpoint_manifest.json"),
                 "run_contract": file_record(run_path), "training_commit": run["git"]["commit"]}

def _fixed_video_selection(fixed_videos, *, mode, tasks, cardinality, pool):
    fixed = {str(key): sorted(map(int, value)) for key, value in (fixed_videos or {}).items()}
    if fixed and (mode != "fixed_per_task" or set(fixed) != set(map(str, tasks)) or
                  any(len(value) != cardinality or len(set(value)) != cardinality or not set(value) <= set(pool)
                      for value in fixed.values())):
        raise ValueError("fixed diagnostic videos must provide one distinct K-set for every task")
    return fixed

def selection_contract(
    *, role: str, task_ids: Sequence[int], cardinality: int, arm: str, mode: str,
    seed: int, init_state_ids: Sequence[int], video_pool: Sequence[int],
    fixed_videos: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    tasks, states, pool = tuple(task_ids), tuple(init_state_ids), tuple(video_pool)
    if (role not in {"development_train", "nonheld_meta", "validation", "test"} or cardinality not in (1, 2, 4)
            or arm not in {"correct", "same_task_other", *CONTROL_ARMS} or mode not in {"fixed_per_task", "per_init_ordinal"}
            or not tasks or len(set(tasks)) != len(tasks) or seed < 0
            or not states or tuple(sorted(set(states))) != states or not set(states) <= set(range(50))
            or len(set(pool)) != len(pool) or not set(pool) <= set(range(50)) or len(pool) < cardinality):
        raise ValueError("invalid explicit task/condition selection")
    fixed = _fixed_video_selection(fixed_videos, mode=mode, tasks=tasks, cardinality=cardinality, pool=pool)
    if role == "validation" and mode != "per_init_ordinal":
        raise ValueError("validation banks use canonical per-init video schedules; fixed sets are train diagnostics")
    if role == "validation" and states != tuple(range(len(states))):
        raise ValueError("validation init states must retain the canonical zero-based prefix")
    if role == "validation" and sorted(pool) != list(range(50)):
        raise ValueError("validation schedules require all 50 teacher videos")
    origin = 32 if role == "development_train" and set(states) <= set(TRAIN_DIAGNOSTIC_INIT_STATE_IDS) else 0
    restricted = sorted(pool) != list(range(50))
    reserved_seen = (role == "development_train" and mode == "per_init_ordinal"
                     and states == tuple(range(4)) and tuple(sorted(pool)) == tuple(range(46, 50)))
    if mode == "per_init_ordinal" and restricted and (
            cardinality != 1 or min(states) < origin or max(states) - origin >= len(pool)):
        raise ValueError("finite-pool K1 diagnostics need one distinct allowed video per init state; use states32..35 for four videos")
    selection = {"evaluation_role": role, "task_ids": list(tasks), "K": cardinality, "arm": arm,
            "mode": mode, "seed": seed, "init_state_ids": list(states), "video_pool": sorted(pool),
            "fixed_videos": fixed, "outcome_dependence": False, "gradient_use": False,
            "without_replacement": mode == "per_init_ordinal",
            "schedule": "conditional_compilation_reserved_seen_v1" if reserved_seen else VIDEO_SCHEDULE,
            "without_replacement_scope": ("per_task_per_arm_round" if cardinality == 1 else "canonical_cyclic_K_windows")
                if mode == "per_init_ordinal" else "fixed_diagnostic_video_reuse",
            "schedule_state_origin": origin if restricted else 0,
            "video_ordinal_rule": "init_state_id" if mode == "per_init_ordinal" else "fixed_zero"}
    require_control_selection(selection)
    return selection

def request_init_state_ids(
    *, role: str, init_state_ids: Sequence[int] | None = None, state_count: int | None = None,
    registered_stage1: bool = False, registered_fixed400: bool = False,
) -> tuple[int, ...]:
    """Resolve the existing count API or the registered train diagnostic panel."""
    if role == "test":
        states = tuple(range(50))
        if state_count not in (None, 50) or (init_state_ids is not None and tuple(init_state_ids) != states):
            raise ValueError("sealed Test requires exactly initial states0..49")
        return states
    if init_state_ids is None:
        count = 50 if state_count is None else state_count
        if count not in (10, 50) and not (role == "nonheld_meta" and count == 20 and registered_stage1):
            raise ValueError("count-only Writer requests require 10 or 50 initial states, or registered nonheld20")
        return tuple(range(count))
    states = tuple(init_state_ids)
    if (registered_fixed400 and role == "development_train"
            and states == tuple(range(10, 50)) and state_count == 40):
        return states
    if (role != "development_train" or states not in (TRAIN_DIAGNOSTIC_INIT_STATE_IDS, tuple(range(4)))
            or state_count not in (None, len(states))):
        raise ValueError("explicit Writer init states require development_train states32..35 and count4, "
                         "or registered seen states0..3 and count4")
    return states

def paired_video_sets(selection: Mapping[str, Any], task: int, ordinal: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Use the shared task permutation, independent of worker/checkpoint/cursor."""
    k = int(selection["K"])
    if not 0 <= task < 111:
        raise ValueError("Writer video schedule requires a target40 or Source71 task")
    suite, local_task = ("libero_90", task - 40) if task >= 40 else (SUITE_ORDER[task // 10], task % 10)
    seed, pool = int(selection["seed"]), selection["video_pool"]
    if selection["schedule"] == "conditional_compilation_reserved_seen_v1":
        if ordinal not in range(4):
            raise ValueError("reserved seen-panel video ordinal is outside states0..3")
        return (46 + ordinal,), (46 + (ordinal + 1) % 4,)
    if selection["mode"] == "per_init_ordinal" and pool == list(range(50)):
        correct, other = paired_condition_demo_indices(seed, suite, local_task, ordinal,
            "same_task_other", 50, "without_replacement", k)
        return tuple(sorted(correct)), tuple(sorted(other))
    order = [demo for demo in reference_demo_indices(seed, suite, local_task, 0,
        demo_count=50, sampling_mode="without_replacement", video_count=50) if demo in pool]
    fixed = selection["fixed_videos"].get(str(task))
    if selection["mode"] == "fixed_per_task":
        correct = tuple(sorted(fixed if fixed is not None else order[:k]))
        other = tuple(sorted([demo for demo in order if demo not in correct][:k]))
    else:
        position = ordinal - selection["schedule_state_origin"]
        if k != 1 or not 0 <= position < len(order):
            raise ValueError("finite-pool diagnostic ordinal would repeat or omit a video")
        correct = (order[position],)
        offset = SAME_TASK_OTHER_OFFSET % len(order) or 1
        other = (order[(position + offset) % len(order)],) if len(order) > 1 else ()
    if selection["arm"] == "same_task_other" and len(other) != k:
        raise ValueError("same-task-other requires at least K additional disjoint videos")
    return correct, other

def condition_id(task: int, demos: Sequence[int], *, arm: str = "correct", video_task: int | None = None) -> str:
    prefix = f"task_{task:02d}"
    if arm == "no_video":
        return prefix + "_control_no_video"
    if arm in CONTROL_ARMS:
        prefix += f"_control_{arm}_video_{video_task:02d}"
    return prefix + "_demos_" + "_".join(f"{demo:02d}" for demo in sorted(demos))

def planned_episodes(selection: Mapping[str, Any], task: int) -> list[dict[str, Any]]:
    rows = []
    arm, donor = selection["arm"], video_task_id(selection, task)
    for state in selection["init_state_ids"]:
        ordinal = state if selection["mode"] == "per_init_ordinal" else 0
        correct, other = paired_video_sets(selection, task, ordinal)
        demos = () if arm == "no_video" else other if arm == "same_task_other" else correct
        rows.append({"init_state_id": state, "video_ordinal": ordinal,
                     "condition_id": condition_id(task, demos, arm=arm, video_task=donor), "teacher_demo_indices": list(demos),
                     "paired_correct_demos": list(correct), "paired_other_demos": list(other)})
        if arm in CONTROL_ARMS:
            rows[-1]["video_global_task_id"] = donor
    return rows

def method_metadata(run: Mapping[str, Any], arm: str = "correct") -> dict[str, Any]:
    observer = observer_mode_contract(run["model_config"])
    conditional = run.get("config", {}).get("schema_version") in {
        CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA, LANGUAGE_CONTENT_CONFIG_SCHEMA,
        INITIAL_CONTENT_CONFIG_SCHEMA}
    if conditional:
        config = run["config"]
        parameterization = config["experiment"]["parameterization"]
        deployment_inputs = {
            "direct_lora": [],
            "language_writer": ["exact task language"],
            "video_writer": ["exact task language", "ordered RGB videos", "original frame indices"],
        }[parameterization]
        if arm == "no_video":
            deployment_inputs = []
        method = {
            "schema_version": "ember_conditional_compilation_method_v1",
            "model_config": run["model_config"], "observer": config["observer"],
            "execution_precision": config["execution_precision"],
            "parameterization": parameterization,
            "training_objective": config["optimization"]["loss"],
            "checkpoint_state": ("one shared direct full A/B parameter set" if parameterization == "direct_lora"
                                 else "complete Writer state; only the registered language path is trainable"
                                 if parameterization == "language_writer"
                                 else "complete video Writer including Text/VL/Action Meta and public probe"),
            "deployment_inputs": deployment_inputs,
            "execution_rank": 16, "generated_tensor_count": 76,
            "training_stage": STAGE, "update_version": config["update_version"],
            "macro_cursor": "optimizer_updates", "deployment_frozen_source_vjp": False,
            "source_parameter_training": False, "deployment_grad_context": "no_grad_complete_parameterization",
            "deployment_teacher_labels_loss_optimizer": False,
            "writer_execution": ("none; direct shared A/B parameters" if parameterization == "direct_lora"
                                 else "one pre-rollout text-only call" if parameterization == "language_writer"
                                 else "one pre-rollout video-conditioned call"),
            "teacher_schedule_use": "pairing metadata only" if parameterization != "video_writer"
                                     else "one selected teacher video per condition",
            "teacher_video_values_read": 0 if parameterization != "video_writer" or arm == "no_video" else 1,
        }
        if config["schema_version"] == RELATIONAL_CONFIG_SCHEMA:
            method["study_id"] = config["experiment"]["kind"]
            method["training_pool"] = config["experiment"]["pool"]
        if config["schema_version"] == LANGUAGE_CONTENT_CONFIG_SCHEMA:
            method["study_id"] = config["experiment"]["kind"]
            method["language_content_path"] = config["experiment"]["language_content_path"]
        if config["schema_version"] == INITIAL_CONTENT_CONFIG_SCHEMA:
            method["study_id"] = config["experiment"]["kind"]
            method["initial_content_only"] = True
        if arm in CONTROL_ARMS:
            method["diagnostic_control"] = arm
            method["control_transform"] = ("identity_zero_delta_without_RGB_reads" if arm == "no_video" else
                                            "real_agentview_camera_RGB_before_complete_Writer_forward")
        return method
    cameras = teacher_camera_names(observer["camera_view"])
    patches = len(cameras) * 256
    metadata = {
        "model_config": run["model_config"], "observer": run["config"]["observer"],
        "execution_precision": run["config"]["execution_precision"],
        "checkpoint_state": "strict entire Writer including Text/VL/Action Meta and public probe",
        "frame_stride": 5, "include_last_frame": True, "camera": "_and_".join(cameras) + "_rotated_180",
        "native_image_tokens": patches,
        "execution_rank": 16, "generated_tensor_count": 76, "native_response_shape": [50, 1024],
        "native_response_source": "final_normalized_action_suffix_hidden",
        "visual_token_source": f"actual_final_{patches}_image_patches_and_exact_task_span_tokens",
        "visual_token_gradient": "joint_Text_VL_Action_Meta_complete_Writer_replay",
        "native_read": observer["horizon_read"], "video_order": observer["video_order"],
        "video_representation": "language_queried_video_Core_and_ordered_recurrent_Procedure",
        "process_aggregation": "repeated_full_H50_and_adjacent_E_reads_then_causal_RoPE_with_centered_slot_read",
        "parameter_decoder": "A_matched_Core_conditioned_Procedure_AdaLN_postfusion_and_final_RMSNorm",
        "native_parameter_generation": "complete_A_B_from_eight_shared_family_heads",
        "training_stage": STAGE, "training_objective": "main_fm_plus_video_teaching",
        "deployment_frozen_source_vjp": False, "source_parameter_training": False,
        "deployment_grad_context": "no_grad_complete_Writer_forward",
        "deployment_teacher_labels_loss_optimizer": False, "writer_execution": "one_pre_rollout_call",
        "update_version": run["config"]["update_version"], "macro_cursor": "optimizer_updates",
    }
    if arm in CONTROL_ARMS:
        metadata["diagnostic_control"] = arm
        metadata["control_transform"] = "identity_zero_delta_without_RGB_reads" if arm == "no_video" else (
            f"real_{observer['camera_view']}_camera_RGB_before_complete_Writer_forward")
    if arm == "no_video":
        metadata.update(writer_execution="bypassed_no_video_identity",
                        native_parameter_generation="complete_zero_A_and_B_identity",
                        deployment_grad_context="no_autograd_or_model_forward")
    return metadata

def adapter_metadata(condition: str, checkpoint: Mapping[str, Any]) -> dict[str, str]:
    return {"schema_version": ADAPTER_SCHEMA, "condition_id": condition,
            "writer_checkpoint": str(checkpoint["path"]), "macro": str(checkpoint["macro"])}
