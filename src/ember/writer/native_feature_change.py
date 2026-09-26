"""Registered C0 feature intervention using the canonical Writer and bank owner."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast


SPEC_PATH = "configs/native_feature_change_causality_v1/experiment_spec.json"
STUDY = "native_feature_change_causality_20260926"
CELLS = ("E0_H0", "E1_H0", "E0_H1", "E1_H1")
REPO_ROOT = Path(__file__).resolve().parents[3]


def spec() -> dict[str, Any]:
    value = read_json(REPO_ROOT / SPEC_PATH)
    evaluation = value["evaluation"]
    if (value.get("schema_version") != "ember_native_feature_change_causality_v1"
            or value.get("study_id") != STUDY
            or value["intervention"]["arms"] != [
                {"id": cell, "E_changes": bool(int(cell[1])), "H_changes": bool(int(cell[-1]))}
                for cell in CELLS]
            or evaluation["task_ids"] != [0, 1, 14, 15, 20, 21, 36, 38]
            or evaluation["state_ids"] != list(range(10))
            or evaluation["native_conditions"] != 80
            or evaluation["banks"] != 320 or evaluation["new_rollouts"] != 320
            or evaluation["pilot_cases_per_arm"] != [[0, 0], [36, 0]]
            or evaluation["capture"]["full_task_ids"] != [0, 14, 21, 38]
            or evaluation["capture"]["full_state_ids"] != [0]
            or evaluation["capture"]["full_cases"] != 16
            or evaluation["video_schedule_seed"] != 20260911
            or value["frozen_input"]["training_commit"] !=
               "dca1b5500ac0f912d56cc1c76382004457e807a4"):
        raise ValueError("native-feature scientific registration changed")
    return value


def registered_selection(study: Mapping[str, Any]) -> dict[str, Any]:
    from ember.writer.materialization import selection_contract

    evaluation = study["evaluation"]
    return selection_contract(
        role="development_train", task_ids=evaluation["task_ids"], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=evaluation["video_schedule_seed"],
        init_state_ids=evaluation["state_ids"], video_pool=range(50),
    )


def panel(cell: str, stage: str) -> dict[str, Any]:
    study = spec()
    if cell not in CELLS or stage not in ("pilot", "remaining"):
        raise ValueError("native-feature panel must be one of the eight frozen stages")
    return {"id": cell, "model": "C0", "kind": "held_correct", "condition": "correct",
            "task_ids": study["evaluation"]["task_ids"],
            "state_ids": study["evaluation"]["state_ids"],
            "rows": 2 if stage == "pilot" else 78, "stage": stage, "study_id": STUDY}


def intervene_packed(
    evidence: torch.Tensor, interactions: torch.Tensor, horizon: torch.Tensor,
    valid_frames: torch.Tensor, projection: torch.nn.Module, *, e: int, h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast each video's actual first feature; retain 50 native horizon positions."""
    if (e not in (0, 1) or h not in (0, 1) or evidence.ndim != 4
            or interactions.shape != evidence.shape[:2] + evidence.shape[-1:]
            or horizon.ndim != 4 or horizon.shape[:2] != evidence.shape[:2]
            or horizon.shape[-2:] != (50, 1024)
            or valid_frames.shape != evidence.shape[:2]
            or valid_frames.dtype != torch.bool
            or not bool(valid_frames[:, 0].all())):
        raise ValueError("native-feature intervention received invalid packed features")
    mask = valid_frames[..., None]
    changed_e = (evidence if e else
                 torch.where(mask[..., None], evidence[:, :1].expand_as(evidence), 0))
    if h:
        return changed_e, interactions, horizon
    changed_h = torch.where(mask[..., None], horizon[:, :1].expand_as(horizon), 0)
    changed_i = projection(changed_h.float().mean(dim=-2).to(changed_h.dtype))
    changed_i = torch.where(mask, changed_i, 0)
    return changed_e, changed_i, changed_h


def compile_grid(writer: torch.nn.Module, policy: torch.nn.Module, condition: tuple):
    """Read one native condition, then use the same Core, Procedure, Compiler and heads."""
    encoded, native = writer.encode_task(policy, *condition, return_trace=True)
    q = native["text_queries"]
    offsets = writer._validated_offsets(condition[2], condition[0].shape[0])
    evidence, interactions, horizon, positions, valid_frames = writer._pack_video_program(
        native["frame_evidence"], native["interactions"], native["horizon"],
        condition[1], offsets,
    )
    valid_tokens = encoded[1]
    if not torch.equal(positions, encoded[3]) or not torch.equal(valid_frames, encoded[4]):
        raise ValueError("native-feature original clock or frame mask changed")
    output = {}
    for cell in CELLS:
        e, h = int(cell[1]), int(cell[-1])
        cell_e, cell_i, cell_h = intervene_packed(
            evidence, interactions, horizon, valid_frames,
            writer.semantic_encoder.interaction_projection, e=e, h=h,
        )
        if cell == "E1_H1":
            core, procedure, attention = encoded[0], encoded[2], encoded[5]
        else:
            core, attention = writer.semantic_core(q, cell_e, valid_frames, valid_tokens)
            procedure = writer.procedure(cell_i, cell_h, cell_e, positions, valid_frames,
                                         valid_tokens)
        generated, compiler_trace = writer.compile_encoded_task(
            core, valid_tokens, procedure, positions, valid_frames, return_trace=True,
        )
        trace = {"q": q, "E": cell_e, "H": cell_h, "I": cell_i,
                 "positions": positions, "valid_frames": valid_frames,
                 "valid_task_tokens": valid_tokens, "frame_attention": attention,
                 "Core": core, "P": procedure, "core_slots": compiler_trace["core_slots"],
                 "centered_P": compiler_trace["procedure_centered"],
                 "procedure_slots": compiler_trace["procedure_slots"],
                 "gamma": compiler_trace["adaln_gamma"],
                 "beta": compiler_trace["adaln_beta"],
                 "fused_slots": compiler_trace["fused_slots"],
                 "output_slots": compiler_trace["output_slots"]}
        output[cell] = generated, trace
    return output


def _cpu_trace(trace: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().to("cpu").contiguous() for key, value in trace.items()}


def materialize_registered_grid(
    *, asset_root: Path, device: torch.device, cpu_threads: int,
    native_frame_chunk: int | None = None,
) -> list[Path]:
    from ember.writer.learning_data import load_learning_tasks
    from ember.writer.materialization import (
        _materialize, _prepare_video_condition, _save_condition, file_record,
        frozen_authority, inspect_writer_checkpoint, planned_episodes,
    )
    from ember.writer.materialization_workers import MaterializationWorkers, ResidentCompiler

    study = spec()
    repo = git_state(REPO_ROOT)
    if not frozen_authority(repo) or device.type != "cuda" or cpu_threads < 1:
        raise ValueError("native-feature materialization requires clean pushed detached CUDA execution")
    checkpoint = Path(study["frozen_input"]["checkpoint"]).resolve()
    run, checkpoint_record = inspect_writer_checkpoint(checkpoint)
    from ember.writer.language_content_contract import validate_config

    validate_config(run["config"])
    if (checkpoint_record["macro"] != 630
            or run["git"]["commit"] != study["frozen_input"]["training_commit"]
            or run["config"]["experiment"]["arm_id"] != "C0"
            or Path(run["source"]["checkpoint"]).resolve() !=
               Path(study["frozen_input"]["source_checkpoint"]).resolve()):
        raise ValueError("native-feature source or frozen C0 checkpoint changed")
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    if any((root / "materialization" / cell).exists() for cell in CELLS) or (root / "features").exists():
        raise ValueError("native-feature conditions were already generated")
    selection = registered_selection(study)
    tasks = load_learning_tasks(asset_root, selection["task_ids"], role="train",
                                protocol_path=run["config"]["data"]["protocol"])
    jobs = [(tasks[task], episode) for task in selection["task_ids"]
            for episode in planned_episodes(selection, task)]
    if len(jobs) != 80 or len({episode["condition_id"] for _, episode in jobs}) != 80:
        raise ValueError("native-feature video schedule is not exactly 80 unique conditions")
    config = dict(run["config"])
    if native_frame_chunk is not None:
        if native_frame_chunk < 1:
            raise ValueError("physical native frame chunk must be positive")
        config["observer"] = {**config["observer"], "frame_chunk": native_frame_chunk}
        config["model"] = {**config["model"], "max_frames_per_encoder_call": native_frame_chunk}
    stage = root / "launch" / "native_feature_staging"
    feature_root = root / "features"
    stage.mkdir(parents=True, exist_ok=False)
    feature_root.mkdir(parents=True, exist_ok=False)
    for cell in CELLS:
        (stage / cell).mkdir()
        (feature_root / cell).mkdir()
    prepared = {cell: {} for cell in CELLS}
    feature_rows = []
    compiler = ResidentCompiler(asset_root, config, device, cpu_threads)
    try:
        compiler.prepare((checkpoint, run, checkpoint_record, tasks, stage, None))
        runtime = compiler.runtime
        for task, episode in jobs:
            identifier, demos = episode["condition_id"], episode["teacher_demo_indices"]
            condition, videos = _prepare_video_condition(runtime, compiler.store, task, demos)
            with torch.inference_mode(), autocast(device):
                grid = compile_grid(runtime.state.writer, runtime.policy, condition)
                if not feature_rows:
                    reference = runtime.compile(condition)
                    actual = grid["E1_H1"][0]
                    if set(reference) != set(actual):
                        raise ValueError("native-feature 11 lost complete canonical LoRA targets")
                    maximum = max(float((actual[key] - reference[key]).abs().max()) for key in actual)
                    squared = sum(float((actual[key] - reference[key]).float().square().sum())
                                  for key in actual)
                    baseline = sum(float(reference[key].float().square().sum()) for key in actual)
                    relative = (squared / max(baseline, 1e-30)) ** .5
                    if maximum > .002 and relative > .02:
                        raise ValueError("native-feature 11 differs materially from canonical Writer")
                    write_json_atomic(root / "launch" / "native_interface_check.json", {
                        "condition_id": identifier, "additional_native_writer_calls": 1,
                        "complete_lora_tensors": len(actual), "max_abs": maximum,
                        "relative_l2": relative, "normal_dtype_tolerance": True,
                    })
                    del reference
            feature = {"condition_id": identifier, "global_task_id": task.authority.task_id,
                       "init_state_id": episode["init_state_id"], "teacher_demo_indices": demos,
                       "teacher_videos": videos, "cells": {}}
            for cell in CELLS:
                generated, trace = grid[cell]
                record = _save_condition(generated, runtime.lora, task, demos, videos,
                                         stage / cell, checkpoint_record)
                prepared[cell][identifier] = record
                path = feature_root / cell / f"{identifier}.pt"
                torch.save(_cpu_trace(trace), path)
                feature["cells"][cell] = file_record(path)
            feature_rows.append(feature)
            print(json.dumps({"native_condition": identifier, "ready": len(feature_rows)}), flush=True)
            del grid, condition
        # Reuse the sole canonical bank manifest builder and its complete source/checkpoint wall.
        workers = MaterializationWorkers(asset_root=asset_root, config=config,
                                         devices=(device,), cpu_threads=cpu_threads)
        output = []
        for cell in CELLS:
            path = _materialize(asset_root=asset_root, checkpoint=checkpoint,
                output=root / "materialization" / cell, selection=selection, workers=workers,
                run=run, checkpoint_record=checkpoint_record, repository=repo,
                precompiled=prepared[cell], extra_manifest={"native_feature_change": {
                    "study_id": STUDY, "cell": cell, "spec_path": SPEC_PATH,
                    "native_read_conditions": 80, "feature_index": str((feature_root / "index.json").resolve()),
                    "source_training_commit": study["frozen_input"]["training_commit"],
                }})
            output.append(path)
        write_json_atomic(feature_root / "index.json", {
            "schema_version": "ember_native_feature_change_feature_index_v1", "study_id": STUDY,
            "materialization_commit": repo["commit"], "checkpoint": checkpoint_record,
            "selection": selection, "conditions": feature_rows,
        })
        # Final banks are hardlinks. Only our known temporary staging is removed.
        shutil.rmtree(stage)
        return output
    finally:
        compiler.close()
