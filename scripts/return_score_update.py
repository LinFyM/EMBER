#!/usr/bin/env python3
"""Fixed 96-row score-update diagnostic; canonical Writer and rollout own computation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import validate_lora_state
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
    inspect_installed_target_tasks, load_evaluation_authorities, policy_noise_seed)
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import _compile_condition, file_record
from ember.writer.return_credit import candidate_step, norm
from ember.writer.runtime import build_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT.parent / "EMBER"
SPEC_PATH = REPO_ROOT / "configs/return_score_update_v1/experiment_spec.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARMS = ("P", "RAW", "RB")


def authority(*, formal: bool) -> tuple[dict, Path, dict, dict]:
    spec = read_json(SPEC_PATH)
    panel = spec["evaluation"]
    if (spec["schema_version"] != "ember_return_score_update_v1"
            or spec["phase"] != "stage1_96_closed_loop"
            or tuple(panel["task_ids"]) != (2, 5, 12, 17, 22, 25, 34, 37)
            or tuple(panel["init_state_ids"]) != (32, 33)
            or tuple(panel["teacher_demo_ids"]) != (46, 47)
            or tuple(panel["arms"]) != ARMS or panel["total_episodes"] != 96
            or spec["materialization"]["banks_total"] != 48
            or spec["evaluation"]["environment_seed"] != 7):
        raise ValueError("score-update registered stage1 scope changed")
    root = Path(spec["resources"]["study_root"])
    config = read_json(ASSET_ROOT / spec["parent"]["config"])
    checkpoint = Path(spec["parent"]["checkpoint"])
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    if (run["git"]["commit"] != spec["parent"]["training_commit"]
            or run["config"] != config or checkpoint.name != "macro_00001155"):
        raise ValueError("score-update parent training identity changed")
    git = git_state(REPO_ROOT)
    if formal and (git["branch"] or git["dirty_paths"]
                   or not git_state_is_clean_pushed_or_frozen_authority(git)):
        raise ValueError("formal score-update requires clean pushed detached implementation")
    return spec, root, config, git


def episode_keys(spec: dict) -> tuple[tuple[str, int, int, int], ...]:
    panel = spec["evaluation"]
    keys = tuple((arm, task, teacher, state)
                 for task in panel["task_ids"] for state in panel["init_state_ids"]
                 for teacher in panel["teacher_demo_ids"] for arm in ARMS)
    if len(keys) != 96 or len(set(keys)) != 96:
        raise ValueError("score-update episode scope is not exactly 96 unique rows")
    return keys


def bank_keys(spec: dict) -> tuple[tuple[str, int, int], ...]:
    panel = spec["evaluation"]
    keys = tuple((arm, task, teacher) for task in panel["task_ids"]
                 for teacher in panel["teacher_demo_ids"] for arm in ARMS)
    if len(keys) != 48 or len(set(keys)) != 48:
        raise ValueError("score-update bank scope is not exactly 48")
    return keys


def _candidate_source(spec: dict, root: Path, arm: str, commit: str) -> dict:
    if arm == "P":
        path = Path(spec["parent"]["checkpoint"]) / "ecp.safetensors"
        return {"arm": arm, "weights": file_record(path), "parent_macro": 1155,
                "diagnostic_sgd_step": 0, "training_commit": spec["parent"]["training_commit"]}
    path = root / "candidates" / arm / "weights.safetensors"
    completion = read_json(root / "candidates" / arm / "completion.json")
    if (completion["implementation_commit"] != commit or completion["arm"] != arm
            or completion["weights"] != file_record(path)
            or completion["parent_checkpoint"] != spec["parent"]["checkpoint"]
            or completion["alpha"] <= 0 or completion["parent_adam_inherited"] is not False):
        raise ValueError("score-update candidate provenance changed")
    return {"arm": arm, "weights": completion["weights"], "parent_macro": 1155,
            "diagnostic_sgd_step": 1, "gradient": completion["gradient"],
            "alpha": completion["alpha"], "implementation_commit": commit}


def audit_inputs(spec: dict) -> tuple[dict, dict, tuple[str, ...], dict[str, float], float]:
    old_root = Path(spec["gradient_source"]["study_root"])
    old_completion = read_json(old_root / "gradient" / "completion.json")
    if (old_completion["implementation_commit"] != spec["gradient_source"]["implementation_commit"]
            or old_completion["unique_inputs"] != 96
            or old_completion["ten_flow_forward_calls"] != 192):
        raise ValueError("sealed RAW/RB gradient source changed")
    parent_path = Path(spec["parent"]["checkpoint"]) / "ecp.safetensors"
    parent = load_file(str(parent_path), device="cpu")
    gradients = {arm: load_file(str(old_root / spec["gradient_source"][arm]), device="cpu")
                 for arm in ("RAW", "RB")}
    names = tuple(sorted(gradients["RAW"]))
    if (len(parent) != 622 or len(names) != 545 or set(gradients["RB"]) != set(names)
            or not set(names) <= set(parent)
            or any(gradient[name].shape != parent[name].shape or not torch.isfinite(gradient[name]).all()
                   for gradient in gradients.values() for name in names)):
        raise ValueError("score-update gradient/parent parameter map changed")
    lengths = {arm: norm(tuple(gradients[arm][name] for name in names)) for arm in gradients}
    if (abs(lengths["RAW"] - spec["gradient_source"]["raw_norm_reference"]) > 1e-4
            or abs(lengths["RB"] - spec["gradient_source"]["rb_norm_reference"]) > 1e-4):
        raise ValueError("sealed gradient norms changed")
    alpha = spec["intervention"]["base_radius"] / lengths["RAW"]
    if abs(alpha - spec["intervention"]["alpha_reference"]) > 1e-9:
        raise ValueError("shared SGD alpha changed")
    return parent, gradients, names, lengths, alpha


def create_candidates(spec: dict, root: Path, commit: str) -> None:
    if (root / "candidates").exists():
        raise ValueError("score-update candidate attempt already exists")
    parent, gradients, names, lengths, alpha = audit_inputs(spec)
    old_root = Path(spec["gradient_source"]["study_root"])
    _metadata(spec, root, read_json(ASSET_ROOT / spec["parent"]["config"]))
    (root / "candidates").mkdir(parents=True)
    for arm in ("RAW", "RB"):
        output = root / "candidates" / arm
        output.mkdir()
        parameters = tuple((name, torch.nn.Parameter(parent[name].clone())) for name in names)
        info, optimizer = candidate_step(parameters, {name: parent[name] for name in names},
                                         gradients[arm], sign=1, radius=alpha * lengths[arm])
        state = dict(parent)
        state.update({name: parameter.detach().clone() for name, parameter in parameters})
        weights = output / "weights.safetensors"
        save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(weights))
        optimizer_path = output / "fresh_sgd_optimizer.pt"
        torch.save({"parameter_names": names, "state_dict": optimizer}, optimizer_path)
        if any(not torch.equal(state[name], parent[name]) for name in set(parent) - set(names)):
            raise ValueError("score-update fixed Writer buffers changed")
        write_json_atomic(output / "completion.json", {
            "schema_version": "ember_return_score_update_candidate_v1", "arm": arm,
            "implementation_commit": commit, "parent_checkpoint": str(Path(spec["parent"]["checkpoint"])),
            "training_commit": spec["parent"]["training_commit"],
            "gradient_source_commit": spec["gradient_source"]["implementation_commit"],
            "gradient": file_record(old_root / spec["gradient_source"][arm]),
            "weights": file_record(weights), "optimizer_state": file_record(optimizer_path),
            "alpha": alpha, "raw_gradient_norm": lengths["RAW"], "direction_norm": lengths[arm],
            "parent_adam_inherited": False, "sampler_continuation": False,
            "parameter_names": list(names), "fixed_buffer_names": sorted(set(parent) - set(names)), **info})


def _metadata(spec: dict, root: Path, config: dict):
    evaluation = load_evaluation_authorities(ASSET_ROOT / config["source"]["evaluation_config"], ASSET_ROOT)
    targets, paths = inspect_installed_target_tasks(
        evaluation, role="development_train", state_count=50,
        libero_config_dir=root / "libero_config")
    lookup = {(task.suite, task.task_id): task for task in targets}
    tasks = {}
    for task_id in spec["evaluation"]["task_ids"]:
        task = lookup.get((SUITES[task_id // 10], task_id % 10))
        if task is None or task.split_role != "train":
            raise ValueError("score-update task crossed train information wall")
        tasks[task_id] = task
    return evaluation, tasks, paths


def _runtime(config: dict):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = build_runtime(ASSET_ROOT, config, torch.device("cuda:0"))
    runtime.policy.requires_grad_(False).eval()
    runtime.state.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise ValueError("Source policy became trainable")
    return runtime


def _load_writer(runtime, source: dict) -> None:
    state = load_file(source["weights"]["path"], device="cpu")
    runtime.state.load_state_dict(state, strict=True)
    runtime.state.requires_grad_(False).eval()


def _bank_path(root: Path, arm: str, task: int, teacher: int) -> Path:
    return root / "banks" / arm / f"task_{task:03d}_teacher_{teacher:02d}"


def _bank_record(spec: dict, root: Path, commit: str, key: tuple[str, int, int]) -> dict:
    arm, task, teacher = key
    path = _bank_path(root, arm, task, teacher) / "bank_record.json"
    record = read_json(path)
    adapter = record["condition"]["adapter"]
    if (record["schema_version"] != "ember_return_score_update_bank_v1"
            or record["implementation_commit"] != commit
            or record["study_spec"] != str(SPEC_PATH)
            or (record["arm"], record["task"], record["teacher"]) != key
            or record["writer"] != _candidate_source(spec, root, arm, commit)
            or record["condition"]["global_task_id"] != task
            or record["condition"]["teacher_demo_indices"] != [teacher]
            or record["condition"]["single_complete_rank16"] is not True
            or record["condition"]["writer_invocations"] != 1
            or adapter != file_record(Path(adapter["path"]))):
        raise ValueError("score-update bank provenance or complete LoRA changed")
    return record


def _materialize_job(spec, root, commit, runtime, store, learning, key):
    arm, task, teacher = key
    output = _bank_path(root, arm, task, teacher)
    if output.exists():
        _bank_record(spec, root, commit, key)
        return
    source = _candidate_source(spec, root, arm, commit)
    _load_writer(runtime, source)
    output.mkdir(parents=True)
    condition = _compile_condition(runtime, store, learning[task], (teacher,), output,
                                   {"path": source["weights"]["path"], "macro": 1155})
    validate_lora_state(load_file(condition["adapter"]["path"], device="cpu"), runtime.lora)
    write_json_atomic(output / "bank_record.json", {
        "schema_version": "ember_return_score_update_bank_v1", "implementation_commit": commit,
        "study_spec": str(SPEC_PATH), "arm": arm, "task": task, "teacher": teacher,
        "writer": source, "condition": condition, "new_writer_forward": 1,
        "training_teacher_actions_read": 0})


def _contract(evaluation, paths, task, output: Path, *, full: bool) -> dict:
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "mode": "compact", "full_conditions": ([{"suite": task.suite,
               "task_id": task.task_id, "init_state_id": 32}] if full else []),
               "trajectory_root": str(output / "trajectories"),
               "passive_trace": {"schema_version": "ember_return_score_update_passive_trace_v1",
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
            "diagnostic_stage_predicates": stage, "adapter": None}


def _episode_path(root: Path, key: tuple[str, int, int, int]) -> Path:
    arm, task, teacher, state = key
    return root / "evaluation" / arm / f"task_{task:03d}_teacher_{teacher:02d}" / f"state_{state:02d}"


def _episode_job(spec, root, config, commit, runtime, infer, pool, evaluation, tasks, paths, key):
    arm, task_id, teacher, state = key
    output = _episode_path(root, key)
    if output.exists():
        raise ValueError("score-update episode already started; no automatic retry or overwrite")
    if getattr(runtime, "_score_update_loaded_arm", None) != arm:
        _load_writer(runtime, _candidate_source(spec, root, arm, commit))
        runtime._score_update_loaded_arm = arm
    bank = _bank_record(spec, root, commit, (arm, task_id, teacher))
    lora = load_file(bank["condition"]["adapter"]["path"], device="cpu")
    task = tasks[task_id]
    envs, init_states = pool.switch(asdict(task))
    full = (task_id in spec["capture"]["full_cases"]["tasks"]
            and state == 32 and teacher == 46)
    contract = _contract(evaluation, paths, task, output, full=full)
    output.mkdir(parents=True)
    with infer.activate([lora]):
        rows = rollout_shard(envs=envs[:1], init_states=init_states,
            task=asdict(task), state_ids=(state,), contract=contract,
            policy=runtime.policy, preprocess=runtime.processor,
            postprocess=runtime.processor.unnormalize_action)
    if len(rows) != 1 or rows[0]["init_state_id"] != state:
        raise ValueError("score-update canonical rollout returned wrong row")
    row = rows[0]
    validate_passive_trace_row(row, contract, asdict(task))
    seeds = [policy_noise_seed(7, task.suite, task.task_id, state, index)
             for index in range(len(row["policy_noise_seeds"]))]
    if (row["env_seed"] != 7 or row["policy_seed_root"] != 7
            or row["policy_noise_seeds"] != seeds
            or row.get("occupancy_trajectory", {}).get("capture_level") != ("full" if full else "compact")
            or bool(row.get("exploration", {}).get("enabled", False))):
        raise ValueError("score-update paired RNG, exploration or capture changed")
    row.update(global_task_id=task_id, teacher_demo=teacher, arm=arm,
               bank_record=file_record(_bank_path(root, arm, task_id, teacher) / "bank_record.json"))
    write_json_atomic(output / "completion.json", {
        "schema_version": "ember_return_score_update_episode_v1",
        "implementation_commit": commit, "key": list(key), "row": row})


def _worker(stage, spec, root, config, commit, physical_gpu, queue, stop):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        runtime = _runtime(config)
        if stage == "materialize":
            learning = load_learning_tasks(ASSET_ROOT, spec["evaluation"]["task_ids"],
                                           protocol_path=config["data"]["protocol"])
            store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                         frame_stride=5, camera_view="agentview")
            try:
                while not stop.is_set():
                    key = queue.get()
                    if key is None:
                        break
                    _materialize_job(spec, root, commit, runtime, store, learning, key)
            finally:
                store.close()
        else:
            evaluation, tasks, paths = _metadata(spec, root, config)
            base = _contract(evaluation, paths, tasks[2], root / "evaluation", full=False)
            pool = PersistentTaskEnvironmentPool(base, physical_gpu_id=physical_gpu)
            infer = BatchedLoRAInference(runtime.policy, runtime.lora)
            try:
                while not stop.is_set():
                    key = queue.get()
                    if key is None:
                        break
                    _episode_job(spec, root, config, commit, runtime, infer, pool,
                                 evaluation, tasks, paths, key)
            finally:
                infer.close()
                pool.close()
    except BaseException:
        stop.set()
        raise


def _scheduled_jobs(stage: str, spec: dict, root: Path, commit: str,
                    gpus: tuple[int, ...]) -> tuple:
    pilot = tuple((arm, 2, 46, 32) for arm in ARMS)
    if stage == "materialize":
        if not (root / "candidates" / "RAW" / "completion.json").is_file() or not (
                root / "candidates" / "RB" / "completion.json").is_file():
            raise ValueError("score-update candidates must be sealed before materialization")
        jobs = bank_keys(spec)
    elif stage == "pilot":
        if len(gpus) != 1 or any(_episode_path(root, key).exists() for key in pilot):
            raise ValueError("formal pilot must start exactly three untouched rows on one GPU")
        jobs = pilot
    elif stage == "evaluate":
        check_pilot(spec, root, commit)
        jobs = tuple(key for key in episode_keys(spec) if key not in pilot)
        if any(_episode_path(root, key).exists() for key in jobs):
            raise ValueError("score-update remainder has an already-started episode")
    else:
        raise ValueError("unknown score-update stage")
    if stage != "materialize":
        for key in bank_keys(spec):
            _bank_record(spec, root, commit, key)
    return jobs


def dispatch(stage: str, spec: dict, root: Path, config: dict, commit: str,
             gpus: tuple[int, ...]) -> None:
    if not gpus or len(gpus) > 2 or len(set(gpus)) != len(gpus):
        raise ValueError("score-update requires one or two distinct local physical GPUs")
    used = sum(read_json(path)["gpu_hours_conservative"]
               for path in (root / "launch").glob("*_exit_*.json"))
    remaining = spec["resources"]["gpu_hours_hard_max"] - used
    if remaining <= 0:
        raise ValueError("score-update GPU-hour hard budget is exhausted")
    jobs = _scheduled_jobs(stage, spec, root, commit, gpus)
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
    workers = [context.Process(target=_worker,
        args=(stage, spec, root, config, commit, gpu, queue, stop),
        name=f"score-update-{stage}-gpu{gpu}") for gpu in gpus]
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
    records = [{"name": worker.name, "pid": worker.pid, "gpu": gpu,
                "exit_code": worker.exitcode} for worker, gpu in zip(workers, gpus, strict=True)]
    write_json_atomic(root / "launch" / f"{stage}_exit_{time.time_ns()}.json", {
        "schema_version": "ember_return_score_update_stage_exit_v1", "stage": stage,
        "implementation_commit": commit, "jobs": len(jobs), "workers": records,
        "wall_seconds": elapsed, "gpu_hours_conservative": elapsed * len(gpus) / 3600,
        "budget_exhausted": budget_exhausted})
    if budget_exhausted or any(row["exit_code"] != 0 for row in records):
        raise RuntimeError(f"score-update {stage} worker failed; partial evidence preserved")


def check_pilot(spec: dict, root: Path, commit: str) -> None:
    keys = tuple((arm, 2, 46, 32) for arm in ARMS)
    rows = []
    for key in keys:
        record = read_json(_episode_path(root, key) / "completion.json")
        if record["implementation_commit"] != commit or tuple(record["key"]) != key:
            raise ValueError("score-update pilot identity changed")
        row = record["row"]
        if row["bank_record"] != file_record(_bank_path(root, key[0], 2, 46) / "bank_record.json"):
            raise ValueError("score-update pilot bank changed")
        rows.append(row)
    if (any(not row["policy_noise_seeds"] or
            row["policy_noise_seeds"][0] != rows[0]["policy_noise_seeds"][0] for row in rows)
            or any(row["env_seed"] != 7 or row["policy_seed_root"] != 7 for row in rows)):
        raise ValueError("score-update pilot RNG pairing changed")
    starts = []
    for row in rows:
        with np.load(row["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as trace:
            starts.append((trace["body_positions"][0], trace["eef_pos"][0],
                           trace["gripper_qpos"][0]))
    if any(max(float(np.max(np.abs(left - right))) for left, right in zip(starts[0], start)) > 1e-4
           for start in starts[1:]):
        raise ValueError("score-update pilot initial simulator states diverged")
    write_json_atomic(root / "launch" / "pilot_acceptance.json", {
        "schema_version": "ember_return_score_update_pilot_acceptance_v1",
        "implementation_commit": commit, "keys": [list(key) for key in keys],
        "bank_rng_initial_trace_valid": True, "score_used_for_acceptance": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "candidates", "materialize", "pilot", "evaluate"))
    parser.add_argument("--gpus", type=int, nargs="*")
    args = parser.parse_args()
    spec, root, config, git = authority(formal=args.stage != "inspect")
    if args.stage == "inspect":
        parent, gradients, names, lengths, alpha = audit_inputs(spec)
        print(json.dumps({"commit": git["commit"], "banks": len(bank_keys(spec)),
                          "episodes": len(episode_keys(spec)), "pilot": 3,
                          "full": spec["capture"]["full_cases"]["count"],
                          "parent_tensors": len(parent), "gradient_tensors": len(names),
                          "gradient_norms": lengths, "shared_alpha": alpha}))
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "launch").mkdir(exist_ok=True)
    if args.stage == "candidates":
        create_candidates(spec, root, git["commit"])
    else:
        dispatch(args.stage, spec, root, config, git["commit"], tuple(args.gpus or ()))


if __name__ == "__main__":
    main()
