"""One P6 sequential-compile source-isolation check and canonical train case."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_contract import inspect_installed_target_tasks, load_evaluation_authorities
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

from .data import TransferData
from .run import REPO, RUN_SCHEMA, STAGE


def _load_p6(runtime, spec: dict, frozen_git: dict) -> Path:
    root = Path(spec["runtime"]["run_root"]) / "P" / "fresh"
    checkpoint = root / "checkpoints" / "macro_00000006"
    contract = read_json(root / "run_contract.json")
    completion = read_json(root / "completion.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    if (contract.get("schema_version") != RUN_SCHEMA or contract.get("stage") != STAGE
            or contract.get("git") != frozen_git or contract.get("arm") != "P"
            or contract.get("source") != runtime.source or contract.get("lora") != runtime.lora.to_dict()
            or contract.get("source_identity_before_every_compile") is not True
            or completion.get("updates") != 6 or completion.get("status") != "engineering_complete"
            or manifest.get("next_macro") != 6 or manifest.get("stage") != STAGE
            or manifest.get("run_contract_schema") != RUN_SCHEMA):
        raise ValueError("P6 closure checkpoint or source-isolation contract changed")
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
    row = next(row for row in manifest["tasks"] if row["global_task_id"] == 38)
    key = row["suite"], int(row["task_id"])
    if key not in rows or rows[key]["language"] != row["language"]:
        raise ValueError("global38 is not the official train task")
    contract = dict(authorities.config)
    contract["libero_paths"] = paths
    contract["parallel"] = {**contract["parallel"], "envs_per_replica": 1}
    contract["diagnostic_stage_predicates"] = {"full_conditions_only": False}
    contract["diagnostic_occupancy_capture"] = {
        "mode": "full", "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"trace_root": str(output / "passive_traces")}}
    return contract, rows[key]


def _compile_condition(runtime, data: TransferData, task: int, demo: int) -> tuple[dict, int, int, float]:
    condition, raw, sampled = runtime.condition(data, task, demo)
    started = time.perf_counter()
    with torch.no_grad():
        generated = runtime.compile(condition)
    validate_lora_state(generated, runtime.lora)
    if any(not torch.isfinite(value).all() for value in generated.values()):
        raise ValueError("compiled interface LoRA is nonfinite")
    return generated, raw, sampled, time.perf_counter() - started


def sequential_case(spec: dict, args, frozen_git: dict, build_runtime) -> None:
    if args.arm is not None or args.resume is not None or args.stop_after is not None:
        raise ValueError("sequential case only consumes the fixed fresh P6 checkpoint")
    if torch.cuda.device_count() != 1:
        raise ValueError("sequential case needs one visible evaluation GPU")
    request, = spec["canonical_interface_cases"]
    if (request["arm"] != "P" or request["checkpoint_macro"] != 6
            or request["preceding_install_without_rollout"] != {"task_id": 2, "teacher_demo": 46}
            or request["task_id"] != 38 or request["init_state"] != 0
            or request["teacher_demo"] != 46 or request["capture"] != "full_dual_camera"):
        raise ValueError("single affected interface case contract changed")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not visible.isdigit():
        raise ValueError("canonical EGL case needs a physical visible GPU index")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    output = Path(spec["runtime"]["run_root"]) / "sequential_case"
    output.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(args.asset_root, spec, torch.device("cuda:0"), evaluation=True)
    checkpoint = _load_p6(runtime, spec, frozen_git)
    data = TransferData(args.asset_root, spec)
    try:
        runtime.state.eval()
        first, first_raw, first_sampled, first_seconds = _compile_condition(runtime, data, 2, 46)
        copy_task_lora_state_(runtime.policy, first, runtime.lora)
        installed_delta_norm = runtime.source_delta_norm()
        second, raw, sampled, second_seconds = _compile_condition(runtime, data, 38, 46)
        restored_delta_norm = runtime.source_delta_norm()
        if installed_delta_norm <= 0 or restored_delta_norm != 0 or runtime.source_identity_restores != 2:
            raise ValueError("sequential compile did not restore the physical source identity")
        isolation = {"schema_version": RUN_SCHEMA, "checkpoint": str(checkpoint),
                     "first": {"task": 2, "teacher_demo": 46, "raw_frames": first_raw,
                               "sampled_frames": first_sampled, "compile_seconds": first_seconds,
                               "installed_without_rollout": True},
                     "second": {"task": 38, "teacher_demo": 46, "raw_frames": raw,
                                "sampled_frames": sampled, "compile_seconds": second_seconds},
                     "installed_delta_norm_before_second_compile": installed_delta_norm,
                     "physical_source_delta_norm_after_identity_reset_and_second_compile": restored_delta_norm,
                     "identity_resets_before_compile": runtime.source_identity_restores,
                     "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad)}
        write_json_atomic(output / "source_isolation.json", isolation)
        copy_task_lora_state_(runtime.policy, second, runtime.lora)
        contract, task = _episode_contract(spec, output, REPO)
        pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(visible))
        try:
            envs, states = pool.switch(task)
            started = time.perf_counter()
            result = rollout_shard(
                envs=envs, init_states=states, task=task, state_ids=(0,),
                contract=contract, policy=runtime.policy, preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action)
            if len(result) != 1:
                raise ValueError("canonical interface did not return exactly one episode")
        finally:
            pool.close()
        row = {**result[0], "global_task_id": 38, "teacher_demo": 46,
               "teacher_raw_frames": raw, "teacher_sampled_frames": sampled,
               "compiled_lora": {"rank": 144, "alpha": 144, "targets": 38,
                                 "factors": 76, "checkpoint": str(checkpoint),
                                 "teacher_inputs": "exact language+agentview RGB/positions"},
               "source_isolation": file_record(output / "source_isolation.json"),
               "interface_seconds": time.perf_counter() - started}
        path = output / "case_global_38_state_00_demo_46.json"
        write_json_atomic(path, row)
        write_json_atomic(output / "completion.json", {
            "schema_version": RUN_SCHEMA, "status": "one_sequential_interface_complete",
            "checkpoint": str(checkpoint), "case": file_record(path),
            "source_isolation": file_record(output / "source_isolation.json"),
            "success": row["success"], "steps": row["steps"],
            "passive_trace": row.get("continuous_control_trace"),
            "full_capture": row.get("occupancy_trajectory"),
            "scientific_qualification": False})
    finally:
        data.close()
