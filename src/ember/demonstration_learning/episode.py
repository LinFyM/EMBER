"""One registered longest-video VJP and two train-only canonical interfaces."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_contract import inspect_installed_target_tasks, load_evaluation_authorities
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

from .data import TransferData, _flow_seed, _rng
from .run import REPO, RUN_SCHEMA, SPEC_PATH, STAGE, _one_job


def _profile_event(spec: dict, data: TransferData) -> dict:
    profile = spec["profile"]
    task, teacher = int(profile["task_id"]), int(profile["teacher_demo"])
    others = [demo for demo in range(46) if demo != teacher]
    rng = _rng(spec["data"]["event_seeds"]["query"], task, 288, 0)
    demos = [int(v) for v in rng.choice(others, size=28, replace=False)]
    frames = [int(rng.integers(data.tasks[task].episode_lengths[demo] - 1)) for demo in demos]
    return {"task": task, "visit": 288, "update": None, "kind": "old",
            "teacher_demo": teacher, "flow_seed": _flow_seed(spec, task, 288),
            "queries": [{"demo": d, "frame": f} for d, f in zip(demos, frames, strict=True)]}


def _load_p4(runtime, spec: dict, frozen_git: dict) -> Path:
    root = Path(spec["runtime"]["run_root"]) / "P" / "fresh"
    checkpoint = root / "checkpoints" / "macro_00000004"
    contract = read_json(root / "run_contract.json")
    completion = read_json(root / "completion.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    if (contract.get("schema_version") != RUN_SCHEMA or contract.get("stage") != STAGE
            or contract.get("git") != frozen_git or contract.get("arm") != "P"
            or contract.get("source") != runtime.source or contract.get("lora") != runtime.lora.to_dict()
            or completion.get("updates") != 4 or completion.get("status") != "engineering_complete"
            or manifest.get("next_macro") != 4 or manifest.get("stage") != STAGE
            or manifest.get("run_contract_schema") != RUN_SCHEMA):
        raise ValueError("P4 engineering checkpoint or frozen source changed")
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(runtime.device)), strict=True)
    return checkpoint


def _episode_contract(spec: dict, output: Path, asset_root: Path) -> tuple[dict, dict]:
    from ember.pi05_assets import configure_libero_runtime_assets

    authorities = load_evaluation_authorities(asset_root / spec["source"]["evaluation_config"], asset_root)
    tasks, paths = inspect_installed_target_tasks(
        authorities, role="development_train", state_count=1,
        libero_config_dir=output / "libero_config")
    configure_libero_runtime_assets(Path(paths["assets"]))
    rows = {(task.suite, task.task_id): asdict(task) for task in tasks}
    manifest = read_json(asset_root / spec["data"]["manifest"])
    wanted = {}
    for global_id in (2, 38):
        row = next(row for row in manifest["tasks"] if row["global_task_id"] == global_id)
        key = row["suite"], int(row["task_id"])
        if key not in rows or rows[key]["language"] != row["language"]:
            raise ValueError("interface case is not an official train task")
        wanted[global_id] = rows[key]
    contract = dict(authorities.config)
    contract["libero_paths"] = paths
    contract["parallel"] = {**contract["parallel"], "envs_per_replica": 1}
    contract["diagnostic_stage_predicates"] = {"full_conditions_only": False}
    contract["diagnostic_occupancy_capture"] = {
        "mode": "full", "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"trace_root": str(output / "passive_traces")}}
    return contract, wanted


def profile_and_cases(spec: dict, args, frozen_git: dict, build_runtime) -> None:
    if args.arm is not None or args.resume is not None or args.stop_after is not None:
        raise ValueError("profile/cases only consume the fixed P4 checkpoint")
    if torch.cuda.device_count() != 1:
        raise ValueError("profile/cases require one visible evaluation GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    output = Path(spec["runtime"]["run_root"]) / "profile_cases"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    runtime = build_runtime(args.asset_root, spec, device, evaluation=True)
    checkpoint = _load_p4(runtime, spec, frozen_git)
    data = TransferData(args.asset_root, spec)
    try:
        event = _profile_event(spec, data)
        raw, sampled = data.videos.frame_counts(38, 36)
        if (raw, sampled) != (517, 105):
            raise ValueError("registered longest teaching video identity changed")
        runtime.state.train()
        runtime.state.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = _one_job(runtime, data, event, args.microbatch)
        torch.cuda.synchronize(device)
        profile = {"schema_version": RUN_SCHEMA, "checkpoint": str(checkpoint),
                   "updates": 0, "arm": "P", "result": result,
                   "seconds": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                   "gradient_norms": {
                       "common": float(torch.stack([p.grad.detach().float().norm() for p in runtime.state.common.parameters()
                                                     if p.grad is not None]).norm()),
                       "writer": float(torch.stack([p.grad.detach().float().norm() for p in runtime.state.writer.parameters()
                                                     if p.grad is not None]).norm())},
                   "scientific_qualification": False}
        write_json_atomic(output / "profile.json", profile)
        runtime.state.zero_grad(set_to_none=True)
        runtime.state.eval()
        _run_cases(runtime, data, spec, output, checkpoint)
    finally:
        data.close()


def _run_cases(runtime, data: TransferData, spec: dict, output: Path, checkpoint: Path) -> None:
    contract, tasks = _episode_contract(spec, output, REPO)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not visible.isdigit():
        raise ValueError("canonical EGL case needs a physical visible GPU index")
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(visible))
    rows = []
    try:
        for request in spec["canonical_interface_cases"]:
            task_id = int(request["task_id"])
            demo = int(request["teacher_demo"])
            init_state = int(request["init_state"])
            if (request["arm"] != "P" or request["checkpoint_macro"] != 4
                    or task_id not in tasks or demo != 46 or init_state != 0
                    or request["capture"] != "full_dual_camera"):
                raise ValueError("canonical interface case contract changed")
            condition, raw, sampled = runtime.condition(data, task_id, demo)
            with torch.no_grad():
                generated = runtime.compile(condition)
            validate_lora_state(generated, runtime.lora)
            if any(not torch.isfinite(value).all() for value in generated.values()):
                raise ValueError("compiled interface LoRA is nonfinite")
            copy_task_lora_state_(runtime.policy, generated, runtime.lora)
            task = tasks[task_id]
            envs, states = pool.switch(task)
            started = time.perf_counter()
            result = rollout_shard(
                envs=envs, init_states=states, task=task, state_ids=(0,),
                contract=contract, policy=runtime.policy, preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action)
            if len(result) != 1:
                raise ValueError("canonical interface did not return exactly one episode")
            row = {**result[0], "global_task_id": task_id, "teacher_demo": demo,
                   "teacher_raw_frames": raw, "teacher_sampled_frames": sampled,
                   "compiled_lora": {"rank": 144, "alpha": 144, "targets": 38,
                                     "factors": 76, "checkpoint": str(checkpoint),
                                     "teacher_inputs": "exact language+agentview RGB/positions"},
                   "interface_seconds": time.perf_counter() - started}
            path = output / f"case_global_{task_id:02d}_state_00_demo_46.json"
            write_json_atomic(path, row)
            rows.append({"path": str(path), "bytes": path.stat().st_size,
                         "success": row["success"], "steps": row["steps"],
                         "passive_trace": row.get("continuous_control_trace"),
                         "full_capture": row.get("occupancy_trajectory")})
    finally:
        pool.close()
    write_json_atomic(output / "completion.json", {
        "schema_version": RUN_SCHEMA, "status": "profile_and_two_interfaces_complete",
        "checkpoint": str(checkpoint), "cases": rows,
        "profile": file_record(output / "profile.json"),
        "scientific_qualification": False})
