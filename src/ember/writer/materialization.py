"""Compile sealed supervised Writer checkpoints into per-episode complete LoRAs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET, paired_condition_demo_indices, reference_demo_indices,
)
from ember.lora import expected_lora_state_shapes, validate_lora_state
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.data import RawTeacherVideoStore, teacher_camera_names
from ember.writer.learning_data import EVENT_SCHEMA
from ember.writer.materialization_workers import MaterializationWorkers, execution_devices
from ember.writer.training import (CONDITIONAL_CONFIG_SCHEMA, CONDITIONAL_EXPERIMENT, CONFIG_SCHEMA,
                                   CONDITIONAL_UPDATE_VERSION, RUN_SCHEMA, STAGE, TRAINING_SCHEMA, UPDATE_VERSION,
                                   _conditional_config, observer_mode_contract)
from ember.writer.relational_contract import (CONFIG_SCHEMA as RELATIONAL_CONFIG_SCHEMA,
    UPDATE_VERSION as RELATIONAL_UPDATE_VERSION, validate_config as _relational_config,
    registered_stage1_bank_panel)
from ember.writer.video_controls import (CONTROL_ARMS, control_provenance, controlled_frames,
    inspect_diagnostic_contract, require_control_selection, video_task_id)


BANK_SCHEMA = "ember_video_writer_lora_bank_v1"
# This existing execution-protocol kind is also consumed by generic pi05 evaluators.
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
    conditional = config.get("schema_version") in {CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA}
    if conditional:
        (_relational_config if config["schema_version"] == RELATIONAL_CONFIG_SCHEMA else _conditional_config)(config)
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
            or trainer.get("next_macro") != macro or not data_version
            or training != {"schema_version": TRAINING_SCHEMA, "updates": macro,
                            "update_version": config["update_version"], "data_version": data_version}):
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
    registered_stage1: bool = False,
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
    conditional = run.get("config", {}).get("schema_version") in {CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA}
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


def _compile_condition(runtime, store, task, demos, output, checkpoint, *, control=None, video_task=None):
    donor = video_task if video_task is not None else task
    if control is not None and (donor.authority.task_id != control["video_global_task_id"]
                                or task.authority.task_id != control["language_global_task_id"]):
        raise ValueError("actual donor or target language identity differs from the registered video control")
    records = []
    parameterization = getattr(runtime, "parameterization", "video_writer")
    if parameterization == "direct_lora":
        if control is not None:
            raise ValueError("direct shared LoRA has no video control input")
        condition = None
        writer_invocations = 0
    elif parameterization == "language_writer":
        if control is not None:
            raise ValueError("language-only Writer has no video control input")
        condition = runtime.prepare_language(task.authority.language)
        writer_invocations = 1
    elif parameterization == "video_writer":
        if store is None:
            raise ValueError("video Writer materialization requires its registered teacher-video store")
        videos = tuple(store.load(donor.authority.task_id, demo) for demo in demos)
        if any(video.raw_frame_count != donor.episode_lengths[demo] for demo, video in zip(demos, videos, strict=True)):
            raise ValueError("actual teacher frame count differs from its data authority")
        frames, indices = [], []
        for demo, video in zip(demos, videos, strict=True):
            frame, index = torch.from_numpy(video.frames), torch.from_numpy(video.frame_indices)
            record = {"demo_index": demo, "raw_frame_count": video.raw_frame_count,
                      "sampled_frame_count": len(index), "frame_indices": index.tolist()}
            if control is not None:
                content, index, evidence = controlled_frames(index, control=control, demo=demo)
                frame = frame[content]
                record.update(evidence)
            frames.append(frame)
            indices.append(index)
            records.append(record)
        condition = runtime.prepare(tuple(frames), tuple(indices), task.authority.language)
        writer_invocations = 1
    else:
        raise ValueError("unknown conditional Writer parameterization")
    with torch.no_grad():
        generated = runtime.compile(condition)
    return _save_condition(generated, runtime.lora, task, demos, records, output, checkpoint,
                           control=control, parameterization=parameterization,
                           writer_invocations=writer_invocations)


def _save_condition(generated, lora, task, demos, videos, output, checkpoint, *, control=None,
                    parameterization="video_writer", writer_invocations=1):
    state = {name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
             for name, value in generated.items()}
    validate_lora_state(state, lora)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Writer generated nonfinite LoRA parameters")
    identifier = condition_id(task.authority.task_id, demos, arm=control["arm"] if control else "correct",
                              video_task=control["video_global_task_id"] if control else None)
    path = output / f"{identifier}.safetensors"
    save_file(state, str(path), metadata=adapter_metadata(identifier, checkpoint))
    record = {"condition_id": identifier, "global_task_id": task.authority.task_id,
            "suite": task.suite, "task_id": task.suite_task_id, "language": task.authority.language,
            "teacher_demo_indices": list(demos), "teacher_videos": videos,
            "teacher_video_values_read": len(videos), "parameterization": parameterization,
            "adapter": file_record(path),
            "writer_invocations": 0 if control and control["arm"] == "no_video" else writer_invocations,
            "single_complete_rank16": True}
    if control is not None:
        record["video_control"] = control
    return record


def _reusable_conditions(path, *, asset_root, run, checkpoint, selection):
    """Reuse individually valid adapters without accepting their old episode schedule."""
    if path is None:
        return {}
    if selection["arm"] in CONTROL_ARMS:
        raise ValueError("post-hoc controls cannot reuse an untransformed correct/other LoRA")
    from ember.pi05_lora import load_pi05_lora_contract
    from ember.writer.evaluation import _inspect_conditions, validate_information_wall, validate_task_scope

    path = Path(path).resolve()
    manifest = read_json(path)
    lora_path = asset_root / read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]["lora_contract"]
    if (manifest.get("schema_version") != BANK_SCHEMA or manifest.get("kind") != BANK_KIND
            or manifest.get("status") != "sealed" or manifest.get("single_complete_rank16") is not True
            or manifest.get("writer_checkpoint") != checkpoint or manifest.get("source") != run["source"]
            or manifest.get("task_protocol") != run["config"]["data"].get("protocol")
            or manifest.get("method") != method_metadata(run)
            or manifest.get("lora_contract") != file_record(lora_path)
            or Path(manifest.get("asset_root", "")).resolve() != asset_root.resolve()
            or manifest.get("evaluation_role") != selection["evaluation_role"]
            or manifest.get("selection", {}).get("K") != selection["K"]
            or manifest.get("arm") not in {"correct", "same_task_other"}
            or not frozen_authority(manifest.get("materialization_git", {}))):
        raise ValueError("reused LoRAs require identical checkpoint/source/preprocessing/generation contracts")
    validate_task_scope(manifest["tasks"], manifest["evaluation_role"], asset_root,
                        run["config"]["data"].get("protocol"))
    validate_information_wall(manifest)
    _inspect_conditions(manifest, path.parent, load_pi05_lora_contract(lora_path))
    return {row["condition_id"]: row for row in manifest["conditions"]}


def _compile_bank_conditions(*, planned, tasks, output, checkpoint, run, checkpoint_record,
                             workers, reusable, lora_path, no_video, native_transfer=None):
    conditions = {}
    reused = []
    for key in (key for key in planned if key in (reusable or {})):
        record = reusable[key]
        path = output / f"{key}.safetensors"
        os.link(record["adapter"]["path"], path)
        conditions[key] = {**record, "adapter": file_record(path)}
        reused.append(key)
    if no_video:
        from ember.pi05_lora import load_pi05_lora_contract

        lora = load_pi05_lora_contract(lora_path)
        if lora.rank != lora.alpha or lora.rank != 16 or len(lora.targets) != 38 or lora.dropout != 0:
            raise ValueError("no-video identity requires the complete native rank16 LoRA contract")
        zeros = {name: torch.zeros(shape) for name, shape in expected_lora_state_shapes(lora).items()}
        for key, job in planned.items():
            conditions[key] = _save_condition(zeros, lora, tasks[job["task"]], (), [], output,
                                               checkpoint_record, control=job["control"])
    jobs = sorted((job for key, job in planned.items() if key not in conditions),
                  key=lambda job: -sum(tasks[job.get("control", {}).get("video_global_task_id", job["task"])].episode_lengths[demo]
                                       for demo in job["demos"]))
    request = (checkpoint, run, checkpoint_record, tasks, output, native_transfer)
    for job, record in workers.compile(request, jobs) if jobs else ():
        key = job["condition_id"]
        path = output / f"{key}.safetensors"
        if (key not in planned or key in conditions or record.get("condition_id") != key
                or record.get("global_task_id") != job["task"] or record.get("teacher_demo_indices") != job["demos"]
                or ("control" in job and record.get("video_control") != job["control"])
                or record.get("adapter") != file_record(path)):
            raise ValueError("materialization worker returned a duplicate, mismatched, or missing condition file")
        conditions[key] = record
        print(json.dumps({"condition": key, "conditions_ready": len(conditions),
            "newly_compiled": len(conditions) - len(reused), "reused": len(reused),
            "frames": sum(video["sampled_frame_count"] for video in record["teacher_videos"])}), flush=True)
    if set(conditions) != set(planned):
        raise ValueError("materialization cannot seal an incomplete condition bank")
    return conditions, reused


def _materialize(
    *, asset_root: Path, checkpoint: Path, output: Path,
    selection: Mapping[str, Any], workers: MaterializationWorkers,
    run: Mapping[str, Any], checkpoint_record: Mapping[str, Any],
    repository: Mapping[str, Any], reuse_manifest: Path | None = None,
    reusable: Mapping[str, Any] | None = None, diagnostic_contract: Mapping[str, Any] | None = None,
    registered_stage1_panel: Mapping[str, Any] | None = None,
    native_transfer: Mapping[str, Any] | None = None,
) -> Path:
    from ember.writer.learning_data import load_learning_tasks
    from ember.writer.evaluation import validate_task_scope

    role = "train" if selection["evaluation_role"] in {"development_train", "nonheld_meta"} else selection["evaluation_role"]
    tasks = load_learning_tasks(asset_root, selection["task_ids"], role=role,
                                protocol_path=run["config"]["data"].get("protocol"))
    rows = [{"global_task_id": task, "suite": value.suite, "task_id": value.suite_task_id,
             "language": value.authority.language, "split_role": role,
             "teacher_source": file_record(value.authority.path), "episodes": planned_episodes(selection, task)}
            for task, value in tasks.items()]
    validate_task_scope(rows, selection["evaluation_role"], asset_root, run["config"]["data"].get("protocol"))
    task_rows = {row["global_task_id"]: row for row in rows}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    planned = {episode["condition_id"]: {"condition_id": episode["condition_id"],
               "task": row["global_task_id"], "demos": episode["teacher_demo_indices"]}
               for row in rows for episode in row["episodes"]}
    if selection["arm"] in CONTROL_ARMS:
        for job in planned.values():
            job["control"] = control_provenance(selection, job["task"], task_rows)
    lora_path = asset_root / read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]["lora_contract"]
    conditions, reused = _compile_bank_conditions(planned=planned, tasks=tasks, output=output,
        checkpoint=checkpoint, run=run, checkpoint_record=checkpoint_record, workers=workers,
        reusable=reusable, lora_path=lora_path, no_video=selection["arm"] == "no_video",
        native_transfer=native_transfer)
    no_video = selection["arm"] == "no_video"
    conditional = run.get("config", {}).get("schema_version") in {CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA}
    parameterization = run["config"].get("experiment", {}).get("parameterization", "video_writer")
    uses_video = parameterization == "video_writer" and not no_video
    compiler_invocations = sum(int(row.get("writer_invocations", 0)) for row in conditions.values())
    if conditional:
        deployment_inputs = ([] if no_video or parameterization == "direct_lora" else
                             ["exact task language"] if parameterization == "language_writer" else
                             ["exact task language", "ordered RGB videos", "original frame indices"])
        information_wall = {
            "parameterization": parameterization,
            "deployment_inputs": deployment_inputs,
            "teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
            "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
            "materialization_rgb_video_reads": len(conditions) if uses_video else 0,
            "teacher_video_values_read": sum(int(row.get("teacher_video_values_read", 0))
                                               for row in conditions.values()),
            "parameterization_invocations": compiler_invocations,
            "deployment_frozen_source_vjp": False, "deployment_loss_or_optimizer": False,
            "writer_execution_per_unique_condition": ("none" if parameterization == "direct_lora" else
                                                      "cached_native_text_compile" if parameterization == "language_writer" else
                                                      "one_complete_video_compile_per_condition"),
            "outcome_dependent_video_selection": False,
            "shuffled_reversed_wrong_no_video": selection["arm"] in CONTROL_ARMS,
        }
    else:
        information_wall = {
            "deployment_inputs": [] if no_video else ["exact language", "RGB videos", "displayed frame indices"],
            "teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
            "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
            "writer_invocations_per_unique_condition": 0 if no_video else 1,
            "total_writer_invocations": 0 if no_video else len(conditions),
            "deployment_frozen_source_vjp": False, "deployment_loss_or_optimizer": False,
            "outcome_dependent_video_selection": False,
            "shuffled_reversed_wrong_no_video": selection["arm"] in CONTROL_ARMS,
        }
    manifest = {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "status": "sealed",
                "arm": selection["arm"], "evaluation_role": selection["evaluation_role"], "selection": dict(selection),
                "task_protocol": run["config"]["data"].get("protocol"),
                "asset_root": str(asset_root.resolve()), "source": run["source"],
                "writer_checkpoint": checkpoint_record, "materialization_git": repository,
                "lora_contract": file_record(lora_path), "method": method_metadata(run, selection["arm"]),
                "materialization_execution": {"native_frame_chunk": workers.config["observer"].get("frame_chunk") if uses_video else None,
                    "devices": list(map(str, workers.devices)) if not no_video else [],
                    "workers": len(workers.devices) if not no_video else 0,
                    "dispatch": "source_identity_cpu" if no_video else
                        "longest_video_first_dynamic_conditions" if uses_video else "task_parameterization_dynamic_conditions"},
                "tasks": rows, "conditions": [conditions[key] for key in planned], "single_complete_rank16": True,
                "compilation": {"new_conditions": len(conditions) - len(reused), "reused_conditions": len(reused),
                    "reuse_manifest": file_record(reuse_manifest) if reuse_manifest is not None else None,
                    "reused_condition_ids": reused},
                "information_wall": information_wall}
    if diagnostic_contract is not None:
        manifest["diagnostic_contract"] = dict(diagnostic_contract)
        if not conditional:
            manifest["information_wall"]["materialization_rgb_video_reads"] = 0 if no_video else len(conditions)
    elif not conditional:
        manifest["information_wall"]["deployment_inputs"] = ["exact language", "ordered RGB videos", "original frame indices"]
    _attach_stage1_panel_identity(manifest, registered_stage1_panel)
    from ember.writer.native_reader_transfer import attach_manifest

    attach_manifest(manifest, native_transfer)
    path = output / "manifest.json"
    write_json_atomic(path, manifest)
    return path


def _attach_stage1_panel_identity(manifest, panel) -> None:
    if panel is not None:
        manifest["registered_stage1_panel_id"] = panel["id"]


def _validate_conditional_selection(selection: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Bind this study's banks to its two registered training-role panels."""
    if config["schema_version"] == RELATIONAL_CONFIG_SCHEMA:
        registered_stage1_bank_panel(config, selection)
        return
    spec = read_json(REPO_ROOT / config["study_spec"])
    evaluation = spec["evaluation"]
    held, seen = evaluation["diagnostic_held"], evaluation["seen"]
    tasks = selection["task_ids"]
    if tasks == held["task_ids"]:
        states, pool, schedule = held["state_ids"], held["teacher_demos"], VIDEO_SCHEDULE
        permitted_arms = {"correct", "same_task_other", "cross_suite_wrong"}
        role = "development_train"
    elif tasks == seen["task_ids"]:
        states, pool, schedule = seen["state_ids"], list(range(46, 50)), "conditional_compilation_reserved_seen_v1"
        permitted_arms = {"correct"}
        role = "development_train"
    else:
        raise ValueError("conditional bank tasks are outside registered held400 and seen64 panels")
    if (selection["evaluation_role"] != role or selection["K"] != 1
            or selection["mode"] != "per_init_ordinal"
            or selection["seed"] != evaluation["video_schedule_seed"]
            or selection["init_state_ids"] != states or selection["video_pool"] != pool
            or selection["schedule"] != schedule or selection["arm"] not in permitted_arms):
        raise ValueError("conditional bank pairing differs from its registered panel")


def _registered_request_panel(request, run, record):
    schema = run.get("config", {}).get("schema_version")
    if schema not in {CONDITIONAL_CONFIG_SCHEMA, RELATIONAL_CONFIG_SCHEMA}:
        return None
    _validate_conditional_selection(request["selection"], run["config"])
    panel = None
    if schema == RELATIONAL_CONFIG_SCHEMA:
        if run["git"]["commit"] != "7dc95edbba00cf61439700d77fb321eb8df95c07":
            raise ValueError("stage1 bank must use the frozen 7dc training checkpoint")
        panel = registered_stage1_bank_panel(run["config"], request["selection"],
            checkpoint=Path(record["path"]), output=Path(request["output"]))
    arm_id = run["config"]["experiment"]["arm_id"]
    selected_arm = request["selection"]["arm"]
    if arm_id in {"A_direct16", "B_language", "B_S00", "B_S11"} and selected_arm != "correct":
        raise ValueError("direct and language arms reuse their correct result; no video control reruns are registered")
    if arm_id in {"C_video_fm", "D_video_aux", "C_S00", "C_S01", "C_S10", "C_S11"} and selected_arm not in {
            "correct", "same_task_other", "cross_suite_wrong"}:
        raise ValueError("conditional study permits only correct, selected same-task-other, and cross-suite-wrong banks")
    if arm_id in {"C_video_fm", "D_video_aux", "C_S00", "C_S01", "C_S10", "C_S11"} and selected_arm in {
            "same_task_other", "cross_suite_wrong"} and request.get("diagnostic_contract") is None:
        raise ValueError("video controls require the sealed selected-checkpoint diagnostic declaration")
    return panel


def _require_stage1_goal_other_reuse(request, panel, diagnostic, reused) -> None:
    if panel is None or panel["kind"] != "target_other":
        return
    expected = {episode["condition_id"] for episode in planned_episodes(request["selection"], 21)}
    if (request.get("reuse_manifest") is None or diagnostic is None
            or Path(request["reuse_manifest"]).resolve()
            != Path(diagnostic["paired_correct_manifest"]["path"]).resolve()
            or len(expected) != 50 or not expected <= set(reused)):
        raise ValueError("stage1 Goal21 other requires all 50 paired correct LoRAs before GPU launch")


def _materialize_batch(*, asset_root: Path, requests: Sequence[Mapping[str, Any]], device: torch.device | None = None,
                       devices: Sequence[torch.device] | None = None, cpu_threads: int = 4,
                       native_frame_chunk: int | None = None) -> list[Path]:
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("materialization CPU threads must be positive")
    repository = git_state(REPO_ROOT)
    if not frozen_authority(repository):
        raise ValueError("materialization requires a clean pushed detached checkout")
    if not requests:
        raise ValueError("materialization batch must contain at least one request")
    if any(request["selection"]["K"] != 1 for request in requests):
        raise ValueError("canonical unified native Writer materialization requires the trained K=1 condition")
    if native_frame_chunk is not None and (type(native_frame_chunk) is not int or native_frame_chunk <= 0):
        raise ValueError("native frame chunk must be a positive physical batch size")
    outputs = [Path(request["output"]).resolve() for request in requests]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("materialization outputs must be distinct new directories")
    inspected = [inspect_writer_checkpoint(Path(request["checkpoint"])) for request in requests]
    from ember.writer.native_reader_transfer import registered_panels, registered_transfers

    transfers = registered_transfers(requests, inspected)
    panels = registered_panels(requests, inspected, transfers)
    first = inspected[0][0]
    expected = (first["source"], first["model_config"], first["config"]["observer"])
    for run, _ in inspected:
        if (run["source"], run["model_config"], run["config"]["observer"]) != expected:
            raise ValueError("resident batch requires identical source, model, and observer contracts")
    diagnostics = [inspect_diagnostic_contract(request.get("diagnostic_contract"), selection=request["selection"],
                   checkpoint=record, run=run, asset_root=asset_root)
                   for request, (run, record) in zip(requests, inspected, strict=True)]
    reusable = [_reusable_conditions(request.get("reuse_manifest"), asset_root=asset_root,
        run=run, checkpoint=record, selection=request["selection"])
        for request, (run, record) in zip(requests, inspected, strict=True)]
    for request, panel, diagnostic, reused in zip(requests, panels, diagnostics, reusable, strict=True):
        _require_stage1_goal_other_reuse(request, panel, diagnostic, reused)
    selected_devices = execution_devices(device, devices)
    # One asset root fixes LoRA/tokenizer/normalization authorities. Workers may
    # reuse weights for the same checkpoint, never adapted Z/KV/H or generated LoRAs.
    runtime_config = {**first["config"], "model": first["model_config"],
                      "observer": dict(first["config"]["observer"])}
    if native_frame_chunk is not None:
        runtime_config["observer"]["frame_chunk"] = native_frame_chunk
    results, workers = [], None
    with ExitStack() as stack:
        for request, (run, record), reused, diagnostic, panel, transfer in zip(
                requests, inspected, reusable, diagnostics, panels, transfers, strict=True):
            if request["selection"]["arm"] != "no_video" and workers is None:
                workers = stack.enter_context(MaterializationWorkers(asset_root=asset_root, config=runtime_config,
                                               devices=selected_devices, cpu_threads=cpu_threads))
            normalized = {**request, "diagnostic_contract": diagnostic,
                          "registered_stage1_panel": panel, "native_transfer": transfer}
            normalized.pop("native_transfer_cell", None)
            results.append(_materialize(asset_root=asset_root, workers=workers, run=run, reusable=reused,
                           checkpoint_record=record, repository=repository, **normalized))
    return results


def materialize(*, asset_root: Path, checkpoint: Path, output: Path,
                selection: Mapping[str, Any], device: torch.device | None = None, reuse_manifest: Path | None = None,
                devices: Sequence[torch.device] | None = None, cpu_threads: int = 4,
                native_frame_chunk: int | None = None, diagnostic_contract: Mapping[str, Any] | None = None) -> Path:
    return _materialize_batch(asset_root=asset_root, device=device, devices=devices, cpu_threads=cpu_threads,
        native_frame_chunk=native_frame_chunk,
        requests=[{"checkpoint": checkpoint, "output": output, "selection": selection, "reuse_manifest": reuse_manifest,
                   "diagnostic_contract": diagnostic_contract}])[0]


def materialize_requests(*, asset_root: Path, requests: Sequence[Mapping[str, Any]], device: torch.device | None = None,
                         devices: Sequence[torch.device] | None = None, cpu_threads: int = 4,
                         native_frame_chunk: int | None = None) -> list[Path]:
    """Compile complete JSON-request banks with one compatible runtime per GPU."""
    if not isinstance(requests, (list, tuple)):
        raise ValueError("batch requests must be a JSON list")
    fields = {"checkpoint", "output", "role", "task_ids", "k", "arm", "selection_mode",
              "video_pool", "state_count", "init_state_ids", "seed", "fixed_videos", "reuse_manifest", "diagnostic_contract",
              "native_transfer_cell"}
    normalized = []
    for request in requests:
        if not isinstance(request, Mapping) or set(request) - fields:
            raise ValueError("unknown request fields; asset root and device belong to the whole batch")
        stage1_20 = request.get("role") == "nonheld_meta" and request.get("state_count") == 20
        if stage1_20:
            spec = read_json(REPO_ROOT / "configs/relational_support_causality_v1/experiment_spec.json")
            root = Path(spec["outputs"]["planned_run_root"]).resolve() / "materialization"
            support = {row["id"] for row in spec["evaluation"]["stage1"]["panels"]
                       if row["kind"] == "support_correct"}
            output = Path(request["output"]).resolve()
            if (spec["evaluation"].get("active_stage") != "mechanism_core_v1"
                    or output.parent != root or output.name not in support):
                raise ValueError("nonheld20 Writer request is outside stage1 support panels")
        selection = selection_contract(role=request["role"], task_ids=request["task_ids"], cardinality=request["k"],
            arm=request.get("arm", "correct"), mode=request.get("selection_mode", "per_init_ordinal"),
            seed=request.get("seed", DEFAULT_SELECTION_SEED), init_state_ids=request_init_state_ids(
                role=request["role"], init_state_ids=request.get("init_state_ids"),
                state_count=request.get("state_count"), registered_stage1=stage1_20),
            video_pool=request.get("video_pool", tuple(range(50))), fixed_videos=request.get("fixed_videos"))
        normalized.append({"checkpoint": Path(request["checkpoint"]).resolve(),
                           "output": Path(request["output"]).resolve(), "selection": selection,
                           "diagnostic_contract": request.get("diagnostic_contract"),
                           "native_transfer_cell": request.get("native_transfer_cell"),
                           "reuse_manifest": Path(request["reuse_manifest"]).resolve() if request.get("reuse_manifest") else None})
    return _materialize_batch(asset_root=asset_root.resolve(), requests=normalized, device=device,
                              devices=devices, cpu_threads=cpu_threads,
                              native_frame_chunk=native_frame_chunk)


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--requests-json", type=Path, help="Batch request list; shares asset root and devices.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-manifest", type=Path, help="Reuse compatible condition LoRAs and compile only missing videos.")
    parser.add_argument("--diagnostic-contract-json", type=Path, help="Explicit frozen selected-checkpoint video-control or Test400 declaration.")
    parser.add_argument("--role", choices=("development_train", "validation", "test"))
    parser.add_argument("--task-ids", type=_integers)
    parser.add_argument("--k", type=int, choices=(1,))
    parser.add_argument("--arm", choices=("correct", "same_task_other", *CONTROL_ARMS))
    parser.add_argument("--selection-mode", choices=("fixed_per_task", "per_init_ordinal"))
    parser.add_argument("--video-pool", type=_integers)
    parser.add_argument("--fixed-videos-json", type=Path)
    parser.add_argument("--state-count", type=int, choices=(4, 10, 50))
    parser.add_argument("--init-state-ids", type=_integers,
                        help="Explicit train diagnostic panel: 32,33,34,35 (four held videos, once each).")
    parser.add_argument("--seed", type=int)
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--device", default="cuda:0")
    placement.add_argument("--devices", help="Distinct same-node visible devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--native-frame-chunk", type=int,
                        help="Physical native frame batch; preserves every frame and the declared cameras.")
    args = parser.parse_args()
    required = ("checkpoint", "output", "role", "task_ids", "k")
    if args.requests_json is None and any(getattr(args, key) is None for key in required):
        parser.error("single request requires --checkpoint, --output, --role, --task-ids and --k")
    selection_flags = (*required, "arm", "selection_mode", "video_pool", "fixed_videos_json", "state_count", "init_state_ids", "seed", "reuse_manifest", "diagnostic_contract_json")
    if args.requests_json is not None and any(getattr(args, key) != parser.get_default(key) for key in selection_flags):
        parser.error("--requests-json cannot be combined with single-request selection flags")
    defaults = {"arm": "correct", "selection_mode": "per_init_ordinal", "video_pool": tuple(range(50)),
                "seed": DEFAULT_SELECTION_SEED}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    placement = {"devices": tuple(torch.device(value) for value in args.devices.split(","))} if args.devices else {
        "device": torch.device(args.device)}
    if args.requests_json is not None:
        for path in materialize_requests(asset_root=args.asset_root.resolve(), **placement, cpu_threads=args.cpu_threads,
                                         native_frame_chunk=args.native_frame_chunk,
                                         requests=json.loads(args.requests_json.read_text())):
            print(path, flush=True)
        return
    selection = selection_contract(role=args.role, task_ids=args.task_ids, cardinality=args.k,
        arm=args.arm, mode=args.selection_mode, seed=args.seed,
        init_state_ids=request_init_state_ids(role=args.role, init_state_ids=args.init_state_ids, state_count=args.state_count),
        video_pool=args.video_pool, fixed_videos=read_json(args.fixed_videos_json) if args.fixed_videos_json else None)
    print(materialize(asset_root=args.asset_root.resolve(), checkpoint=args.checkpoint.resolve(),
                      output=args.output, selection=selection, **placement, cpu_threads=args.cpu_threads,
                      diagnostic_contract=read_json(args.diagnostic_contract_json) if args.diagnostic_contract_json else None,
                      reuse_manifest=args.reuse_manifest, native_frame_chunk=args.native_frame_chunk), flush=True)
