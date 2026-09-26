"""S0 same-forward feature capture through the canonical bank builder."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.learned_initial_content_contract import (
    SPEC_PATH, STUDY, ROOT, bank_panel, panel, selection_for, spec,
)
from ember.writer.runtime import autocast


def _compile_panel(*, compiler, name: str, selection: dict, jobs: list, tasks: dict,
                   checkpoint: Path, run: dict, record: dict, stage: Path,
                   root: Path, repository: dict, asset_root: Path,
                   config: dict, cpu_threads: int) -> Path:
    from ember.writer.materialization import (
        _materialize, _prepare_video_condition, _save_condition,
    )
    from ember.writer.materialization_workers import MaterializationWorkers

    staging = stage / name
    feature_root = root / "features" / name
    staging.mkdir()
    feature_root.mkdir(parents=True, exist_ok=False)
    compiler.prepare((checkpoint, run, record, tasks, staging, None))
    runtime = compiler.runtime
    prepared, features = {}, []
    for task, ep in jobs:
        identifier, demos = ep["condition_id"], ep["teacher_demo_indices"]
        condition, videos = _prepare_video_condition(runtime, compiler.store, task, demos)
        with torch.inference_mode(), autocast(compiler.device):
            encoded, native = runtime.state.writer.encode_task(
                runtime.policy, *condition, return_trace=True)
            generated, slots = runtime.state.writer.compile_encoded_task(
                *encoded[:5], return_trace=True)
        trace = {
            "q": native["text_queries"], "E0": native["packed_evidence"][:, 0],
            "H0": native["packed_horizon"][:, 0],
            "I0": native["packed_interactions"][:, 0],
            "positions": native["positions"], "valid_frames": native["valid_frames"],
            "Core": encoded[0], "P": encoded[2],
            "frame_attention": encoded[5], "valid_task_tokens": encoded[1],
            **{key: value for key, value in slots.items()
               if key not in {"expert", "action_in", "action_out"}},
        }
        if not all(torch.isfinite(value).all() for value in trace.values()
                   if value.is_floating_point()):
            raise ValueError("S0 same-forward feature trace is nonfinite")
        feature_path = feature_root / f"{identifier}.pt"
        torch.save({key: value.detach().cpu().contiguous() for key, value in trace.items()},
                   feature_path)
        prepared[identifier] = _save_condition(
            generated, runtime.lora, task, demos, videos, staging, record)
        features.append({"condition_id": identifier, "global_task_id": task.authority.task_id,
                         "init_state_id": ep["init_state_id"],
                         "teacher_demo_indices": demos, "trace": str(feature_path)})
        print(json.dumps({"panel": name, "condition": identifier,
                          "ready": len(features)}), flush=True)
    index = feature_root / "index.json"
    write_json_atomic(index, {"schema_version": "ember_initial_content_feature_index_v1",
                              "study_id": STUDY, "panel": name,
                              "materialization_commit": repository["commit"],
                              "checkpoint": record, "selection": selection,
                              "conditions": features})
    workers = MaterializationWorkers(asset_root=asset_root, config=config,
                                     devices=(compiler.device,), cpu_threads=cpu_threads)
    return _materialize(
        asset_root=asset_root, checkpoint=checkpoint,
        output=root / "materialization" / name, selection=selection,
        workers=workers, run=run, checkpoint_record=record, repository=repository,
        precompiled=prepared, extra_manifest={"learned_initial_content": {
            "study_id": STUDY, "panel": name, "spec_path": SPEC_PATH,
            "feature_index": str(index),
        }})


def materialize_registered(*, asset_root: Path, device: torch.device, cpu_threads: int,
                           native_frame_chunk: int | None = None) -> list[Path]:
    from ember.writer.learning_data import load_learning_tasks
    from ember.writer.materialization import frozen_authority, inspect_writer_checkpoint, planned_episodes
    from ember.writer.materialization_workers import ResidentCompiler

    study = spec()
    repository = git_state(ROOT)
    if not frozen_authority(repository) or device.type != "cuda" or cpu_threads < 1:
        raise ValueError("S0 bank requires one clean pushed detached CUDA implementation")
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    checkpoint = root / "training/S0/checkpoints/macro_00000630"
    run, record = inspect_writer_checkpoint(checkpoint)
    if run["git"]["commit"] != repository["commit"] or record["macro"] != 630:
        raise ValueError("S0 bank requires the E training checkpoint")
    names = [row["id"] for row in study["evaluation"]["panels"] if row["model"] == "S0"]
    if len(names) != 3 or any((root / "materialization" / name).exists() or
                              (root / "features" / name).exists() for name in names):
        raise ValueError("S0 panel scope or existing materialization changed")
    config = dict(run["config"])
    if native_frame_chunk is not None:
        if native_frame_chunk < 1:
            raise ValueError("physical native frame chunk must be positive")
        config["observer"] = {**config["observer"], "frame_chunk": native_frame_chunk}
        config["model"] = {**config["model"], "max_frames_per_encoder_call": native_frame_chunk}
    stage = root / "launch" / "initial_content_staging"
    stage.mkdir(parents=True, exist_ok=False)
    compiler = ResidentCompiler(asset_root, config, device, cpu_threads)
    try:
        output = []
        for name in names:
            selection = selection_for(name)
            bank_panel(run["config"], selection, checkpoint=checkpoint,
                       output=root / "materialization" / name)
            tasks = load_learning_tasks(asset_root, selection["task_ids"], role="train",
                                        protocol_path=run["config"]["data"]["protocol"])
            jobs = [(tasks[task], ep) for task in selection["task_ids"]
                    for ep in planned_episodes(selection, task)]
            if len(jobs) != panel(name)["rows"] or len({ep["condition_id"] for _, ep in jobs}) != len(jobs):
                raise ValueError("S0 bank planned conditions are incomplete or duplicated")
            output.append(_compile_panel(
                compiler=compiler, name=name, selection=selection, jobs=jobs, tasks=tasks,
                checkpoint=checkpoint, run=run, record=record, stage=stage,
                root=root, repository=repository, asset_root=asset_root,
                config=config, cpu_threads=cpu_threads))
        shutil.rmtree(stage)
        return output
    finally:
        compiler.close()
