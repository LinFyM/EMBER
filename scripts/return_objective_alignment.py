#!/usr/bin/env python3
"""Registered 128-row frozen objective-alignment diagnostic."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import validate_lora_state
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.exploration import alignment_metadata, validate_episode_exploration
from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
    inspect_installed_target_tasks, load_evaluation_authorities, policy_noise_seed)
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import _compile_condition, file_record
from ember.writer.runtime import build_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT.parent / "EMBER"
SPEC_PATH = REPO_ROOT / "configs/return_objective_alignment_v1/experiment_spec.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CELLS = ("P_J0", "P_JS", "RB_J0", "RB_JS")
ORIGINAL_COMMIT = "29634cc9f75143cdd6d70863e6d24ef1a8d3770d"


@lru_cache(maxsize=1)
def _verified_stage_exception():
    spec = read_json(SPEC_PATH)
    exception = spec["execution_stage_exception"]
    pilot = [[cell, 2, 34, 0] for cell in CELLS]
    old_path = Path(exception["bank_spec_path"])
    old = read_json(old_path)
    restored = copy.deepcopy(spec)
    restored.pop("execution_stage_exception")
    restored["materialization"].pop("commit_contract")
    restored["materialization"]["all_new_banks_and_rollouts_one_clean_pushed_detached_commit"] = True
    if (exception["decision"] != "B_repair_canonical_entrypoint_preserve_original_bank_and_pilot"
            or exception["bank_commit"] != ORIGINAL_COMMIT
            or exception["pilot_commit"] != ORIGINAL_COMMIT
            or exception["banks_preserved"] != 16
            or exception["pilot_keys"] != pilot
            or exception["remaining_episodes"] != 124
            or exception["rematerialize_banks"] is not False
            or exception["rerun_pilot"] is not False
            or old_path != Path("/data1/user/ymdai/projects/EMBER-return-objective-alignment-formal/configs/return_objective_alignment_v1/experiment_spec.json")
            or git_state(old_path.parents[2])["commit"] != ORIGINAL_COMMIT
            or restored != old):
        raise ValueError("objective-alignment E/E2 stage exception or original scientific spec changed")
    return exception


def stage_exception(spec):
    exception = _verified_stage_exception()
    if spec.get("execution_stage_exception") != exception:
        raise ValueError("objective-alignment E/E2 stage scope changed")
    return exception


def episode_source_commit(spec, key, current_commit):
    pilot = {tuple(row) for row in stage_exception(spec)["pilot_keys"]}
    return ORIGINAL_COMMIT if key in pilot else current_commit


def authority(*, formal: bool):
    spec = read_json(SPEC_PATH)
    panel = spec["evaluation"]
    conditions = panel["conditions"]
    if (spec["schema_version"] != "ember_return_objective_alignment_v1"
            or tuple(row["name"] for row in panel["cells"]) != CELLS
            or tuple((row["task"], row["teacher_demo"]) for row in conditions) !=
               ((2, 34), (5, 13), (12, 38), (17, 8), (22, 5), (25, 36), (34, 2), (37, 45))
            or any(row["init_state_ids"] != [0, 1, 2, 3] for row in conditions)
            or panel["total_episodes"] != 128 or spec["materialization"]["banks_total"] != 16
            or panel["environment_seed"] != 7 or panel["policy_seed_root"] != 7
            or panel["num_inference_steps"] != 10 or panel["replan_steps"] != 5
            or spec["exploration"]["replica"] != 4
            or spec["exploration"]["seed_root"] != 20260925
            or spec["exploration"]["seed_substream"] != 0x524C
            or spec["exploration"]["temporal_rho"] != .8
            or spec["exploration"]["action_std"] != [0.05] * 6 + [0.1]
            or spec["capture"]["full_cases"] != {"tasks": [2, 12, 22, 34],
                "state": 0, "cells": list(CELLS), "count": 16}):
        raise ValueError("objective-alignment registered matrix changed")
    stage_exception(spec)
    root = Path(spec["resources"]["study_root"])
    config = read_json(ASSET_ROOT / spec["models"]["config"])
    parent = Path(spec["models"]["P"]["parent_checkpoint"])
    run = read_json(parent.parent.parent / "run_contract.json")
    candidate = read_json(Path(spec["models"]["RB"]["completion"]))
    if (run["git"]["commit"] != spec["models"]["P"]["training_commit"]
            or run["config"] != config or parent.name != "macro_00001155"
            or candidate["implementation_commit"] != spec["models"]["RB"]["candidate_commit"]
            or candidate["arm"] != "RB" or candidate["parent_checkpoint"] != str(parent)
            or candidate["weights"] != file_record(Path(spec["models"]["RB"]["weights"]))
            or candidate["alpha"] != spec["models"]["RB"]["alpha"]
            or candidate["parent_adam_inherited"] is not False):
        raise ValueError("frozen parent or saved RB provenance changed")
    git = git_state(REPO_ROOT)
    if formal and (git["branch"] or git["dirty_paths"]
                   or not git_state_is_clean_pushed_or_frozen_authority(git)):
        raise ValueError("formal objective alignment requires one clean pushed detached implementation")
    return spec, root, config, git


def bank_keys(spec):
    keys = tuple((model, row["task"], row["teacher_demo"])
                 for model in ("P", "RB") for row in spec["evaluation"]["conditions"])
    if len(keys) != 16 or len(set(keys)) != 16:
        raise ValueError("objective-alignment bank scope changed")
    return keys


def episode_keys(spec):
    keys = tuple((cell, row["task"], row["teacher_demo"], state)
                 for row in spec["evaluation"]["conditions"]
                 for state in row["init_state_ids"] for cell in CELLS)
    if len(keys) != 128 or len(set(keys)) != 128:
        raise ValueError("objective-alignment episode scope changed")
    return keys


def model_source(spec, model):
    if model not in ("P", "RB"):
        raise ValueError("unknown frozen model")
    row = spec["models"][model]
    return {"model": model, "weights": file_record(Path(row["weights"])),
            "source_commit": row["training_commit"] if model == "P" else row["candidate_commit"],
            "parent_checkpoint": spec["models"]["P"]["parent_checkpoint"],
            "rb_completion": (None if model == "P" else file_record(Path(row["completion"]))) }


def _metadata(spec, root, config):
    evaluation = load_evaluation_authorities(ASSET_ROOT / config["source"]["evaluation_config"], ASSET_ROOT)
    targets, paths = inspect_installed_target_tasks(
        evaluation, role="development_train", state_count=50,
        libero_config_dir=root / "libero_config")
    lookup = {(task.suite, task.task_id): task for task in targets}
    tasks = {}
    for row in spec["evaluation"]["conditions"]:
        task_id = row["task"]
        task = lookup.get((SUITES[task_id // 10], task_id % 10))
        if task is None or task.split_role != "train":
            raise ValueError("objective-alignment task crossed train information wall")
        tasks[task_id] = task
    return evaluation, tasks, paths


def _runtime(config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = build_runtime(ASSET_ROOT, config, torch.device("cuda:0"))
    runtime.policy.requires_grad_(False).eval()
    runtime.state.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise ValueError("Source policy became trainable")
    return runtime


def _bank_path(root, model, task, teacher):
    return root / "banks" / model / f"task_{task:03d}_teacher_{teacher:02d}"


def _episode_path(root, key):
    cell, task, teacher, state = key
    return root / "evaluation" / cell / f"task_{task:03d}_teacher_{teacher:02d}" / f"state_{state:02d}"


def bank_record(spec, root, commit, key):
    model, task, teacher = key
    record = read_json(_bank_path(root, *key) / "bank_record.json")
    adapter = record["condition"]["adapter"]
    exception = stage_exception(spec)
    if (record["schema_version"] != "ember_return_objective_alignment_bank_v1"
            or record["implementation_commit"] != exception["bank_commit"]
            or record["study_spec"] != exception["bank_spec_path"]
            or (record["model"], record["task"], record["teacher"]) != key
            or record["writer"] != model_source(spec, model)
            or record["condition"]["global_task_id"] != task
            or record["condition"]["teacher_demo_indices"] != [teacher]
            or record["condition"]["single_complete_rank16"] is not True
            or record["condition"]["writer_invocations"] != 1
            or adapter != file_record(Path(adapter["path"]))):
        raise ValueError(f"objective-alignment bank provenance changed: {key}")
    return record


def _materialize(spec, root, commit, runtime, store, learning, key):
    model, task, teacher = key
    output = _bank_path(root, *key)
    if output.exists():
        raise ValueError(f"objective-alignment bank already attempted: {key}")
    source = model_source(spec, model)
    runtime.state.load_state_dict(load_file(source["weights"]["path"], device="cpu"), strict=True)
    runtime.state.requires_grad_(False).eval()
    output.mkdir(parents=True)
    condition = _compile_condition(runtime, store, learning[task], (teacher,), output,
                                   {"path": source["weights"]["path"], "macro": 1155})
    validate_lora_state(load_file(condition["adapter"]["path"], device="cpu"), runtime.lora)
    write_json_atomic(output / "bank_record.json", {
        "schema_version": "ember_return_objective_alignment_bank_v1", "implementation_commit": commit,
        "study_spec": str(SPEC_PATH), "model": model, "task": task, "teacher": teacher,
        "writer": source, "condition": condition, "new_writer_forward": 1,
        "training_teacher_actions_read": 0})


def episode_contract(spec, evaluation, paths, task, output, key):
    cell, global_task, teacher, state = key
    enabled = cell.endswith("JS")
    full = global_task in spec["capture"]["full_cases"]["tasks"] and state == 0
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "mode": "compact", "full_conditions": ([{"suite": task.suite,
               "task_id": task.task_id, "init_state_id": state}] if full else []),
               "trajectory_root": str(output / "trajectories"),
               "passive_trace": {"schema_version": "ember_return_objective_alignment_passive_trace_v1",
                                 "spec_path": str(SPEC_PATH),
                                 "trace_root": str(output / "continuous_traces")},
               "training_gradient_use": False, "checkpoint_selection_use": False,
               "validation_use": False, "test_use": False}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
             "full_conditions_only": False, "training_gradient_use": False,
             "checkpoint_selection_use": False, "validation_action_reads": 0,
             "validation_reward_reads": 0, "held_data_use": False}
    return {"role": "development_train", "mode": "formal",
            "environment": evaluation.config["environment"],
            "policy": evaluation.config["policy"], "rng": evaluation.config["rng"],
            "parallel": {"envs_per_replica": 1}, "libero_paths": dict(paths),
            "diagnostic_occupancy_capture": capture,
            "diagnostic_stage_predicates": stage, "adapter": None,
            "diagnostic_exploration": alignment_metadata(enabled=enabled),
            "frozen_objective_alignment": {"spec_path": str(SPEC_PATH), "cell": cell,
                "model": cell.split("_")[0], "global_task": global_task,
                "teacher_demo": teacher, "init_state_id": state,
                "replica": 4, "output": str(output)}}


def _episode(spec, root, commit, runtime, infer, pool, evaluation, tasks, paths, key):
    cell, task_id, teacher, state = key
    output = _episode_path(root, key)
    if output.exists():
        raise ValueError(f"objective-alignment episode already attempted: {key}")
    model = cell.split("_")[0]
    bank = bank_record(spec, root, commit, (model, task_id, teacher))
    lora = load_file(bank["condition"]["adapter"]["path"], device="cpu")
    task = tasks[task_id]
    envs, init_states = pool.switch(asdict(task))
    contract = episode_contract(spec, evaluation, paths, task, output, key)
    output.mkdir(parents=True)
    with infer.activate([lora]):
        rows = rollout_shard(envs=envs[:1], init_states=init_states,
            task=asdict(task), state_ids=(state,), contract=contract,
            policy=runtime.policy, preprocess=runtime.processor,
            postprocess=runtime.processor.unnormalize_action)
    if len(rows) != 1 or rows[0]["init_state_id"] != state:
        raise ValueError("objective-alignment canonical rollout returned wrong row")
    row = rows[0]
    validate_passive_trace_row(row, contract, asdict(task))
    seeds = [policy_noise_seed(7, task.suite, task.task_id, state, index)
             for index in range(len(row["policy_noise_seeds"]))]
    if (row["env_seed"] != 7 or row["policy_seed_root"] != 7
            or row["policy_noise_seeds"] != seeds
            or not validate_episode_exploration(contract, row, replans=len(seeds))
            or row["occupancy_trajectory"]["capture_level"] !=
               ("full" if task_id in spec["capture"]["full_cases"]["tasks"] and state == 0 else "compact")):
        raise ValueError("objective-alignment RNG, exploration or capture changed")
    row.update(cell=cell, model=model, global_task_id=task_id, teacher_demo=teacher,
               bank_record=file_record(_bank_path(root, model, task_id, teacher) / "bank_record.json"))
    write_json_atomic(output / "completion.json", {
        "schema_version": "ember_return_objective_alignment_episode_v1",
        "implementation_commit": commit, "key": list(key), "row": row})


def _worker(stage, spec, root, config, commit, physical_gpu, queue, stop):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        runtime = _runtime(config)
        if stage == "materialize":
            task_ids = [row["task"] for row in spec["evaluation"]["conditions"]]
            learning = load_learning_tasks(ASSET_ROOT, task_ids,
                                           protocol_path=config["data"]["protocol"])
            store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                         frame_stride=5, camera_view="agentview")
            try:
                while not stop.is_set():
                    key = queue.get()
                    if key is None:
                        break
                    _materialize(spec, root, commit, runtime, store, learning, key)
            finally:
                store.close()
        else:
            evaluation, tasks, paths = _metadata(spec, root, config)
            base = episode_contract(spec, evaluation, paths, tasks[2],
                                    root / "evaluation" / "P_J0" / "task_002_teacher_34" / "state_00",
                                    ("P_J0", 2, 34, 0))
            pool = PersistentTaskEnvironmentPool(base, physical_gpu_id=physical_gpu)
            infer = BatchedLoRAInference(runtime.policy, runtime.lora)
            try:
                while not stop.is_set():
                    key = queue.get()
                    if key is None:
                        break
                    _episode(spec, root, commit, runtime, infer, pool, evaluation, tasks, paths, key)
            finally:
                infer.close()
                pool.close()
    except BaseException:
        stop.set()
        raise


def same_common_seed_prefix(left, right):
    common = min(len(left), len(right))
    return common > 0 and left[:common] == right[:common]


def check_pilot(spec, root, commit):
    keys = tuple((cell, 2, 34, 0) for cell in CELLS)
    rows = []
    for key in keys:
        payload = read_json(_episode_path(root, key) / "completion.json")
        if payload["implementation_commit"] != stage_exception(spec)["pilot_commit"] or tuple(payload["key"]) != key:
            raise ValueError("objective-alignment pilot identity changed")
        rows.append(payload["row"])
    if (any(not same_common_seed_prefix(row["policy_noise_seeds"], rows[0]["policy_noise_seeds"])
            for row in rows)
            or rows[0]["bank_record"] != rows[1]["bank_record"]
            or rows[2]["bank_record"] != rows[3]["bank_record"]
            or any(not same_common_seed_prefix(
                row["diagnostic_exploration"]["noise_seeds"],
                rows[0]["diagnostic_exploration"]["noise_seeds"])
                for row in rows)):
        raise ValueError("objective-alignment pilot pairing or bank reuse changed")
    starts = []
    for row in rows:
        with np.load(row["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as trace:
            starts.append((trace["body_positions"][0], trace["eef_pos"][0], trace["gripper_qpos"][0]))
    if any(max(float(np.max(np.abs(left - right))) for left, right in zip(starts[0], start)) > 1e-4
           for start in starts[1:]):
        raise ValueError("objective-alignment pilot initial simulator states diverged")
    from scripts.return_objective_alignment_analysis import check_action_injection

    for key, row in zip(keys, rows, strict=True):
        check_action_injection(spec, row, _episode_path(root, key))
    write_json_atomic(root / "launch" / "pilot_acceptance.json", {
        "schema_version": "ember_return_objective_alignment_pilot_v1",
        "validation_commit": commit, "pilot_commit": ORIGINAL_COMMIT,
        "bank_commit": ORIGINAL_COMMIT, "keys": [list(key) for key in keys],
        "bank_rng_initial_action_trace_valid": True, "score_used_for_acceptance": False})


def _scheduled_jobs(stage, spec, root, commit):
    pilot = tuple((cell, 2, 34, 0) for cell in CELLS)
    if stage == "materialize":
        raise ValueError("objective-alignment original E banks are sealed; no E2 materialization")
    elif stage == "pilot":
        raise ValueError("objective-alignment original E pilot is sealed; no E2 pilot rerun")
    elif stage == "evaluate":
        check_pilot(spec, root, commit)
        jobs = tuple(key for key in episode_keys(spec) if key not in pilot)
        if any(_episode_path(root, key).exists() for key in jobs):
            raise ValueError("objective-alignment remainder attempt already exists")
    else:
        raise ValueError("unknown objective-alignment stage")
    if stage != "materialize":
        for key in bank_keys(spec):
            bank_record(spec, root, commit, key)
    return jobs


def dispatch(stage, spec, root, config, commit, gpus):
    if not gpus or len(gpus) > 2 or len(set(gpus)) != len(gpus):
        raise ValueError("objective alignment needs one or two distinct local physical GPUs")
    receipts = tuple((root / "launch").glob("*_exit_*.json"))
    used = sum(read_json(path)["gpu_hours_conservative"] for path in receipts)
    remaining = spec["resources"]["gpu_hours_hard_max"] - used
    if remaining <= 0:
        raise ValueError("objective-alignment GPU-hour hard budget exhausted")
    jobs = _scheduled_jobs(stage, spec, root, commit)
    horizons = read_json(ASSET_ROOT / config["source"]["evaluation_config"])["environment"]["horizons"]
    jobs = tuple(sorted(jobs, key=lambda key: (-horizons[SUITES[key[1] // 10]], key)))
    context = multiprocessing.get_context("spawn")
    queue, stop = context.Queue(), context.Event()
    for key in jobs:
        queue.put(key)
    for _ in gpus:
        queue.put(None)
    start = time.monotonic()
    deadline = start + remaining * 3600 / len(gpus)
    workers = [context.Process(target=_worker, args=(stage, spec, root, config, commit, gpu, queue, stop),
                               name=f"objective-alignment-{stage}-gpu{gpu}") for gpu in gpus]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=max(0., deadline - time.monotonic()))
    budget_exhausted = any(worker.is_alive() for worker in workers)
    if budget_exhausted:
        stop.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()
    elapsed = time.monotonic() - start
    records = [{"name": worker.name, "pid": worker.pid, "gpu": gpu, "exit_code": worker.exitcode}
               for worker, gpu in zip(workers, gpus, strict=True)]
    write_json_atomic(root / "launch" / f"{stage}_exit_{time.time_ns()}.json", {
        "schema_version": "ember_return_objective_alignment_stage_exit_v1", "stage": stage,
        "implementation_commit": commit, "jobs": len(jobs), "workers": records,
        "wall_seconds": elapsed, "gpu_hours_conservative": elapsed * len(gpus) / 3600,
        "budget_exhausted": budget_exhausted})
    if budget_exhausted or any(row["exit_code"] != 0 for row in records):
        raise RuntimeError(f"objective-alignment {stage} worker failed; partial evidence preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "materialize", "pilot", "evaluate"))
    parser.add_argument("--gpus", type=int, nargs="*")
    args = parser.parse_args()
    spec, root, config, git = authority(formal=args.stage != "inspect")
    if args.stage == "inspect":
        print(json.dumps({"commit": git["commit"], "banks": len(bank_keys(spec)),
            "episodes": len(episode_keys(spec)), "pilot": 4,
            "full": spec["capture"]["full_cases"]["count"],
            "models": {model: model_source(spec, model) for model in ("P", "RB")}}))
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "launch").mkdir(exist_ok=True)
    dispatch(args.stage, spec, root, config, git["commit"], tuple(args.gpus or ()))


if __name__ == "__main__":
    main()
