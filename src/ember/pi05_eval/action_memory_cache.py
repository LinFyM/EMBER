"""Finite privileged train-only memory preparation; no own rollout or learning."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval.action_memory_geometry import geometry_episode

STUDY = "privileged_action_memory_control_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
SCENE = Path("/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes")
SCHEMA = "ember_privileged_action_memory_manifest_v1"
KIND = "privileged_action_memory_controller"


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def registration(asset_root, root):
    """Use the real existing seen144 state/video schedule and source identity."""
    from ember.operator_writer.scope import task_rows
    from ember.pi05_eval_contract import (load_evaluation_authorities,
                                         inspect_source_checkpoint, inspect_tokenizer)
    from ember.writer.learning_data import load_learning_tasks

    if root.resolve() != ROOT:
        raise ValueError("Only the registered new memory root is authorized")
    spec = read_json(asset_root / "configs/operator_read_write_v1/learning_spec.json")
    tasks, _ = task_rows(asset_root, spec)
    authorities = load_evaluation_authorities(asset_root / spec["source"]["evaluation_config"], asset_root)
    checkpoint = (asset_root / spec["source"]["checkpoint"]).resolve()
    source = inspect_source_checkpoint(authorities, checkpoint.parents[1], checkpoint, evaluation_mode="formal")
    tokenizer = inspect_tokenizer(authorities, (asset_root / spec["source"]["tokenizer"]).resolve())
    learning = load_learning_tasks(asset_root, [row["global_task_id"] for row in tasks],
                                   protocol_path=spec["source"]["data_protocol"])
    conditions = []
    for task in tasks:
        tid = task["global_task_id"]
        for ep in task["episodes"]:
            demo = ep["teacher_demo_indices"][0]
            conditions.append({"condition_id": ep["condition_id"], "global_task_id": tid,
                               "suite": task["suite"], "task_id": task["task_id"],
                               "language": task["language"], "init_state_id": ep["init_state_id"],
                               "teacher_demo": demo, "hdf5": str(learning[tid].authority.path.resolve()),
                               "length": learning[tid].episode_lengths[demo],
                               "geometry_path": str(root / "memory/geometry" / f"task{tid:03d}_demo{demo:02d}.npz"),
                               "source_path": str(root / "memory/source" / f"task{tid:03d}_demo{demo:02d}.npz")})
    from ember.pi05_eval.scene import inspect_registered_scenes
    scene = inspect_registered_scenes(SCENE, tasks, states=(32, 33, 34, 35),
                                      schema="ember_operator_seen_task_scenes_v1")
    return {"schema_version": SCHEMA, "kind": KIND, "study_id": STUDY, "root": str(root),
            "asset_root": str(asset_root), "source": source, "tokenizer": tokenizer,
            "normalization": str((asset_root / spec["source"]["normalization"]).resolve()),
            "policy": authorities.config["policy"], "scene_root": str(SCENE),
            "scene_manifest": {"path": str(SCENE / "manifest.json"), "bytes": (SCENE / "manifest.json").stat().st_size},
            "conditions": conditions, "geometry_definition_git": "9f90a14d",
            "geometry_function": "geometry_episode", "nearest_function": "nearest",
            "geometry_timing": "states[i+1]_back_one_actual_model_timestep_then_forward",
            "action_offset": 1, "position_scale_m": .10, "gripper_scale_m": .04,
            "video_schedule_seed": 20260928, "source_noise_seed": 1729,
            "source_noise_shape": [50, 32], "source_inference_steps": 10,
            "policy_noise_used": False, "privileged_training_diagnostic": True,
            "deployment_candidate": False, "held_or_test_reads": 0,
            "gradient_updates": 0, "tasks": tasks, "scene_count": len(scene["scenes"])}


def prepare_geometry(asset_root, root):
    """Restore only the144 registered train teachers with the original function."""
    from bddl.parsing import scan_tokens
    from ember.task_protocol import load_task_authorities

    started = time.time()
    value = registration(asset_root, root)
    _, manifest = load_task_authorities(asset_root, "configs/libero_24_8_8_coverage_v1/protocol.json")
    rows = {row["global_task_id"]: row for row in manifest["tasks"]}
    cfg = read_json(asset_root / "configs/pi05_writer_data_v1.json")
    assets = (asset_root / cfg["authorities"]["libero_assets"]).resolve()
    libero = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    references = {}
    for row in value["conditions"]:
        task = rows[row["global_task_id"]]
        bddl = libero / "bddl_files" / task["problem_folder"] / task["bddl"]["filename"]
        objects = next(group[1:] for group in scan_tokens(filename=str(bddl))
                       if isinstance(group, list) and group[0] == ":obj_of_interest")
        with h5py.File(row["hdf5"], "r") as handle:
            demo = handle[f"data/demo_{row['teacher_demo']}"]
            geometry = geometry_episode(demo, objects, assets, robosuite)
            full, lengths = [], []
            for i in geometry["frames"]:
                future = np.asarray(demo["actions"][i + 1:i + 51], dtype=np.float32)
                valid = len(future)
                if valid < 5:
                    raise ValueError("The common five executed demo actions must all be real")
                padded = np.repeat(future[-1:], 50, axis=0)
                padded[:valid] = future
                full.append(padded)
                lengths.append(valid)
        signature = geometry["signature"]
        tid = row["global_task_id"]
        if tid in references and references[tid] != signature:
            raise ValueError("Within-task registered teacher point signature differs")
        references[tid] = signature
        row.update(frames=geometry["frames"].tolist(), signature=signature, objects=objects,
                   max_coordinate_error_m=geometry["max_coordinate_error_m"],
                   kinematic_backstep_seconds=geometry["kinematic_backstep_seconds"])
        atomic_npz(Path(row["geometry_path"]), frames=geometry["frames"],
                   absolute=geometry["absolute"], demo_actions50=np.stack(full),
                   demo_valid_lengths=np.asarray(lengths, dtype=np.int32))
    value["geometry_elapsed_seconds"] = time.time() - started
    value["memory_frame_count"] = sum(len(row["frames"]) for row in value["conditions"])
    write_json_atomic(root / "memory/registration.json", value)
    write_json_atomic(root / "memory/demo_action_manifest.json", {**value, "arm": "demo_action"})
    return value


def native_inputs(demo, indices, language, processor):
    """Real canonical dual RGB/state prompt, without actions in the native input."""
    import torch
    from ember.writer.data import _camera_batch
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    obs = demo["obs"]
    states = torch.from_numpy(np.concatenate([obs["ee_states"][indices],
                                               obs["gripper_states"][indices]], axis=1).astype(np.float32)).to("cuda:0")
    if states.shape != (len(indices), 8):
        raise ValueError("Source teacher prompt must use the recorded8state")
    tokens, masks = processor._tokenize_prompts(states, [language] * len(indices))
    return {"observation.images.base_0_rgb": torch.from_numpy(_camera_batch(obs["agentview_rgb"][indices])).to("cuda:0").float().div_(255),
            "observation.images.left_wrist_0_rgb": torch.from_numpy(_camera_batch(obs["eye_in_hand_rgb"][indices])).to("cuda:0").float().div_(255),
            OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: masks}


def predict_frames(policy, processor, demo, row, frames, noise, batch_size):
    import torch

    outputs = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            indices = frames[start:start + batch_size]
            batch = native_inputs(demo, indices, row["language"], processor)
            predicted = policy.predict_action_chunk(batch, noise=noise.expand(len(indices), -1, -1).contiguous(), num_steps=10)
            raw = processor.unnormalize_action(predicted).float().cpu().numpy()
            if raw.shape != (len(indices), 50, 7) or not np.isfinite(raw).all():
                raise ValueError("Source native prediction invalid")
            outputs.append(raw)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), {"batch_size": batch_size, "frames": len(frames),
                                    "seconds": elapsed, "frames_per_second": len(frames) / elapsed,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def build_source(root, shard=0, shards=1, batch_size=32, profile=False):
    """Read-only source Value materialization; no LoRA/native Writer or gradients."""
    import torch
    from ember.pi05_eval.worker_setup import load_policy
    from ember.writer.topology import bind_current_process_to_cuda_numa

    started = time.time()
    value = read_json(root / "memory/registration.json")
    bind_current_process_to_cuda_numa(0)
    stats = read_json(Path(value["normalization"]))["stats"]
    policy, processor, _ = load_policy(Path(value["source"]["model_path"]), stats,
                                       Path(value["tokenizer"]["path"]), value["policy"])
    policy.requires_grad_(False)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    generator = torch.Generator(device="cpu").manual_seed(1729)
    noise = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32).to("cuda:0")
    rows = value["conditions"][shard::shards]
    profile_rows, records = [], []
    for ordinal, row in enumerate(rows):
        path = Path(row["source_path"])
        if path.exists():
            with np.load(path) as cache:
                if cache["frames"].tolist() != row["frames"] or cache["source_actions50"].shape != (len(row["frames"]), 50, 7):
                    raise ValueError("Existing effective source memory differs")
            continue
        frames = np.asarray(row["frames"])
        with h5py.File(row["hdf5"], "r") as handle:
            demo = handle[f"data/demo_{row['teacher_demo']}"]
            outputs, stats_rows = [], []
            if profile and ordinal == 0:
                first = frames[:32]
                kept, a = predict_frames(policy, processor, demo, row, first, noise, 16)
                atomic_npz(root / "memory/first_effective_source_chunk.npz", frames=first, source_actions50=kept)
                _, b = predict_frames(policy, processor, demo, row, first, noise, 32)
                batch_size = 32 if b["frames_per_second"] >= a["frames_per_second"] else 16
                profile_rows.extend([a, b])
                outputs.append(kept)
                frames = frames[len(first):]
                write_json_atomic(root / "memory/source_profile.json", {"condition_id": row["condition_id"], "attempts": profile_rows,
                                  "selected_batch": batch_size, "kept_prediction": "first_effective_source_chunk.npz",
                                  "stop_reason": "Two registered attempts exhausted; no extra scientific frames"})
            if len(frames):
                predicted, facts = predict_frames(policy, processor, demo, row, frames, noise, batch_size)
                outputs.append(predicted)
                stats_rows.append(facts)
        atomic_npz(path, frames=np.asarray(row["frames"]), source_actions50=np.concatenate(outputs))
        records.append({"condition_id": row["condition_id"], "frames": len(row["frames"]), "path": str(path),
                        "batch": batch_size, "chunks": stats_rows})
        write_json_atomic(root / f"memory/source_shard{shard}_progress.json", {"records": records})
    write_json_atomic(root / f"memory/source_shard{shard}_complete.json", {"elapsed_seconds": time.time() - started,
                       "status": "complete", "shard": shard, "shards": shards, "conditions": len(rows),
                       "records": records, "source_frozen": True, "gradients": False,
                       "input_fields": ["dual_RGB", "exact_language", "recorded_8state"],
                       "noise_seed": 1729, "native_ODE_steps": 10, "LoRA_parameters": 0})


def seal_source(root):
    value = read_json(root / "memory/registration.json")
    for row in value["conditions"]:
        with np.load(row["source_path"]) as cache:
            if cache["frames"].tolist() != row["frames"] or cache["source_actions50"].shape != (len(row["frames"]), 50, 7):
                raise ValueError("Cannot register an incomplete source Value memory")
    write_json_atomic(root / "memory/source_action_manifest.json", {**value, "arm": "source_action"})
