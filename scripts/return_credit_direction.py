#!/usr/bin/env python3
"""Bounded return-credit collection, bank generation, and canonical rollouts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.pi05_eval.crossed_video_predict import scaled_osc_actions
from ember.pi05_assets import write_json_atomic
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.episode import stage_predicate_snapshot
from ember.pi05_eval.return_credit import authority
from ember.pi05_eval.trajectory_capture import _passive_body_registry, validate_passive_trace_row
from ember.writer.flow import flow_mean_lora_gradient
from ember.writer.return_credit import score_cotangent
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
    inspect_installed_target_tasks, load_evaluation_authorities)
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import _compile_condition, file_record
from ember.writer.runtime import build_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT.parent / "EMBER"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _identity(spec, *, formal: bool) -> tuple[Path, dict, dict]:
    root = Path(spec["resources"]["study_root"])
    config = read_json(REPO_ROOT / spec["parent"]["config"])
    run = read_json(Path(spec["parent"]["checkpoint"]).parent.parent / "run_contract.json")
    state = git_state(REPO_ROOT)
    if (run["git"]["commit"] != spec["parent"]["training_commit"]
            or config != run["config"]
            or Path(spec["parent"]["checkpoint"]).name != "macro_00001155"):
        raise ValueError("return-credit parent checkpoint, config or training identity changed")
    if formal and (state["branch"] or state["dirty_paths"]
                   or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("return-credit formal work requires one clean pushed detached implementation")
    return root, config, state


def _task_authority(root: Path, config, spec):
    evaluation = load_evaluation_authorities(ASSET_ROOT / config["source"]["evaluation_config"], ASSET_ROOT)
    targets, paths = inspect_installed_target_tasks(
        evaluation, role="development_train", state_count=50,
        libero_config_dir=root / "libero_config")
    by_key = {(row.suite, row.task_id): row for row in targets}
    tasks = {}
    for task_id in spec["data"]["gradient_tasks"]:
        key = (SUITES[task_id // 10], task_id % 10)
        task = by_key.get(key)
        if task is None or task.split_role != "train":
            raise ValueError("return-credit panel crossed the fixed train task wall")
        tasks[task_id] = task
    return evaluation, tasks, paths


def _base_contract(evaluation, paths, task, output: Path, *, full: bool,
                   collection: dict | None = None):
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "mode": "compact", "full_conditions": ([{
            "suite": task.suite, "task_id": task.task_id, "init_state_id": 32}] if full else []),
        "trajectory_root": str(output / "trajectories"),
        "passive_trace": {"schema_version": "ember_return_credit_passive_trace_v1",
                          "spec_path": str(REPO_ROOT / "configs/return_credit_direction_v1/experiment_spec.json"),
                          "trace_root": str(output / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_action_reads": 0,
        "validation_reward_reads": 0, "held_data_use": False,
    }
    return {"role": "development_train", "mode": "formal",
            "environment": evaluation.config["environment"],
            "policy": evaluation.config["policy"], "rng": evaluation.config["rng"],
            "parallel": {"envs_per_replica": 1}, "libero_paths": dict(paths),
            "diagnostic_occupancy_capture": capture,
            "diagnostic_stage_predicates": stage,
            "return_credit_collection": collection,
            "adapter": None}


def _runtime(config):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    runtime = build_runtime(ASSET_ROOT, config, device)
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise ValueError("return-credit Source became trainable")
    return runtime


def _load_state(runtime, spec, root: Path, arm: str) -> dict:
    if arm == "P":
        path = Path(spec["parent"]["checkpoint"]) / "ecp.safetensors"
    else:
        path = root / "candidates" / arm / "weights.safetensors"
        if not (root / "candidates" / arm / "completion.json").is_file():
            raise ValueError("return-credit candidate has no sealed one-step completion")
    weights = load_file(str(path), device="cpu")
    runtime.state.load_state_dict(weights, strict=True)
    runtime.state.requires_grad_(False).eval()
    runtime.policy.requires_grad_(False).eval()
    return {"path": str(path), "bytes": path.stat().st_size,
            "parent_macro": 1155, "diagnostic_sgd_step": 0 if arm == "P" else 1,
            "arm": arm}


def _bank(runtime, store, learning, root: Path, source: dict, arm: str,
          task_id: int, teacher: int, *, phase: str, state: dict):
    output = root / "banks" / phase / arm / f"task_{task_id:03d}_teacher_{teacher:02d}"
    index = output / "bank_record.json"
    if index.exists():
        row = read_json(index)
        if (row["implementation_commit"] != state["commit"] or row["writer"] != source
                or row["task"] != task_id or row["teacher"] != teacher
                or row["condition"]["adapter"] != file_record(
                    output / f"{row['condition']['condition_id']}.safetensors")):
            # The file check below also validates the exact frozen adapter path.
            raise ValueError("return-credit bank provenance changed")
        return row
    if output.exists():
        raise ValueError("partial return-credit bank exists without completion")
    output.mkdir(parents=True)
    checkpoint = {"path": source["path"], "macro": 1155}
    condition = _compile_condition(runtime, store, learning[task_id], (teacher,),
                                   output, checkpoint)
    record = {"schema_version": "ember_return_credit_bank_v1",
              "implementation_commit": state["commit"], "study_spec": str(REPO_ROOT / "configs/return_credit_direction_v1/experiment_spec.json"),
              "arm": arm, "phase": phase, "task": task_id, "teacher": teacher,
              "writer": source, "condition": condition,
              "training_teacher_actions_read": 0, "new_writer_forward": 1}
    write_json_atomic(index, record)
    return record


def _bank_state(bank):
    path = Path(bank["condition"]["adapter"]["path"])
    if bank["condition"]["adapter"] != file_record(path):
        raise ValueError("return-credit sealed LoRA changed")
    return load_file(str(path), device="cpu")


def _registry_audit(spec, root, config, state, gpu_index):
    evaluation, tasks, paths = _task_authority(root, config, spec)
    contract = _base_contract(evaluation, paths, next(iter(tasks.values())),
                              root / "registry", full=False)
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=gpu_index)
    records = []
    try:
        for task_id, task in tasks.items():
            envs, initial = pool.switch(asdict(task))
            env = envs[0]
            env.seed(7)
            env.reset()
            env.set_init_state(initial[0])
            predicates, values = stage_predicate_snapshot(env)
            bodies, operands = _passive_body_registry(env, {"stage_predicate_states": predicates})
            records.append({"task": task_id, "suite": task.suite, "local_task": task.task_id,
                            "body_registry": bodies, "goal_operands": operands,
                            "initial_predicates": list(values), "env_steps": 0,
                            "policy_forwards": 0})
    finally:
        pool.close()
    write_json_atomic(root / "audit" / "arena_registry.json", {
        "schema_version": "ember_return_credit_registry_audit_v1",
        "implementation_commit": state["commit"], "probe_count": len(records),
        "records": records})


def _collection_groups(spec, pilot):
    rows = [(row, state) for row in spec["collection"]["conditions"]
            for state in row["init_state_ids"]]
    return rows[:1] if pilot else rows


def _engineering_smoke(spec, root, config, state, gpu_index):
    output = root / "development_smoke" / "task_002_state_00"
    if output.exists():
        raise ValueError("return-credit engineering smoke already exists; do not repeat")
    output.mkdir(parents=True)
    evaluation, tasks, paths = _task_authority(root, config, spec)
    runtime = _runtime(config)
    source = _load_state(runtime, spec, root, "P")
    learning = load_learning_tasks(ASSET_ROOT, spec["data"]["gradient_tasks"],
                                   protocol_path=config["data"]["protocol"])
    store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                 frame_stride=5, camera_view="agentview")
    pool = PersistentTaskEnvironmentPool(
        _base_contract(evaluation, paths, tasks[2], output, full=False),
        physical_gpu_id=gpu_index)
    infer = BatchedLoRAInference(runtime.policy, runtime.lora)
    try:
        bank_output = output / "bank"
        bank_output.mkdir()
        condition = _compile_condition(runtime, store, learning[2], (34,), bank_output,
                                       {"path": source["path"], "macro": 1155})
        lora = load_file(condition["adapter"]["path"], device="cpu")
        envs, init_states = pool.switch(asdict(tasks[2]))
        scope = {"global_task": 2, "init_state_id": 0, "replica": 0,
                 "teacher_demo": 34, "output": str(output), "engineering_smoke": True}
        contract = _base_contract(evaluation, paths, tasks[2], output / "replica_0",
                                  full=False, collection=scope)
        with infer.activate([lora]):
            row = rollout_shard(envs=envs[:1], init_states=init_states,
                task=asdict(tasks[2]), state_ids=(0,), contract=contract,
                policy=runtime.policy, preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action)[0]
        validate_passive_trace_row(row, contract, asdict(tasks[2]))
        infer.close()
        payload = torch.load(row["return_credit_collection"]["path"],
                             map_location="cpu", weights_only=True)
        item = payload["decisions"][0]
        runtime.state.requires_grad_(True).train()
        lora_gpu = {name: value.to(runtime.device) for name, value in lora.items()}
        old = item["mu_old_full"][:5, :7].reshape(35)
        cotangent = score_cotangent(item["latent_u"], old, 1., Q=payload["Q"], M=payload["M"])
        gradient, predicted = flow_mean_lora_gradient(runtime.policy, lora_gpu, runtime.lora,
            {key: value.to(runtime.device) for key, value in item["processed"].items()},
            item["flow_noise"].unsqueeze(0).to(runtime.device),
            cotangent.unsqueeze(0).to(runtime.device))
        runtime.state.zero_grad(set_to_none=True)
        video = runtime.prepare(*store_video(store, learning[2], 34), learning[2].authority.language)
        generated = runtime.compile(video)
        torch.autograd.backward(tuple(generated.values()),
            tuple(gradient[name].to(value) for name, value in generated.items()))
        rms = float((predicted[0].float().cpu().reshape(35) - old).square().mean().sqrt())
        active = sum(int(p.grad is not None and bool(torch.count_nonzero(p.grad)))
                     for p in runtime.state.parameters() if p.requires_grad)
        if rms > .01 or active == 0 or any(p.requires_grad for p in runtime.policy.parameters()):
            raise ValueError("return-credit engineering rollout/flow/Writer VJP failed")
        write_json_atomic(output / "completion.json", {
            "schema_version": "ember_return_credit_engineering_smoke_v1",
            "implementation_commit": state["commit"], "formal": False,
            "episode_count": 1, "task": 2, "state": 0, "teacher": 34,
            "replay_rms": rms, "active_writer_parameters": active,
            "source_frozen": True, "Q": payload["Q"], "M": payload["M"],
            "trajectory": row["occupancy_trajectory"],
            "trace": row["continuous_control_trace"]["trace"]})
    finally:
        infer.close()
        store.close()
        pool.close()


def store_video(store, learning_task, demo):
    video = store.load(learning_task.authority.task_id, demo)
    return ((torch.from_numpy(video.frames),), (torch.from_numpy(video.frame_indices),))


def _collect(spec, root, config, state, gpu_index, *, pilot):
    evaluation, tasks, paths = _task_authority(root, config, spec)
    learning = load_learning_tasks(ASSET_ROOT, spec["data"]["gradient_tasks"],
                                   protocol_path=config["data"]["protocol"])
    runtime = _runtime(config)
    source = _load_state(runtime, spec, root, "P")
    store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                 frame_stride=5, camera_view="agentview")
    base = _base_contract(evaluation, paths, tasks[2], root / "collection", full=False)
    pool = PersistentTaskEnvironmentPool(base, physical_gpu_id=gpu_index)
    infer = BatchedLoRAInference(runtime.policy, runtime.lora)
    try:
        for registration, state_id in _collection_groups(spec, pilot):
            task_id = registration["task"]
            output = root / "collection" / "groups" / f"task_{task_id:03d}_state_{state_id:02d}"
            completion = output / "completion.json"
            if completion.exists():
                if pilot:
                    raise ValueError("formal pilot already completed; do not repeat")
                continue
            if output.exists():
                raise ValueError("return-credit collection group has partial preserved output")
            output.mkdir(parents=True)
            task = tasks[task_id]
            bank = _bank(runtime, store, learning, root, source, "P", task_id,
                         registration["teacher_demo"], phase="collection", state=state)
            lora = _bank_state(bank)
            envs, init_states = pool.switch(asdict(task))
            rows = []
            for replica in registration["replicas"]:
                episode_output = output / f"replica_{replica}"
                scope = {"global_task": task_id, "init_state_id": state_id,
                         "replica": replica, "teacher_demo": registration["teacher_demo"],
                         "output": str(output)}
                contract = _base_contract(evaluation, paths, task, episode_output,
                                          full=False, collection=scope)
                with infer.activate([lora]):
                    episode = rollout_shard(envs=envs[:1], init_states=init_states,
                        task=asdict(task), state_ids=(state_id,), contract=contract,
                        policy=runtime.policy, preprocess=runtime.processor,
                        postprocess=runtime.processor.unnormalize_action)[0]
                validate_passive_trace_row(episode, contract, asdict(task))
                episode.update(global_task_id=task_id, replica=replica,
                               teacher_demo=registration["teacher_demo"],
                               bank_record=file_record(root / "banks" / "collection" / "P" /
                                   f"task_{task_id:03d}_teacher_{registration['teacher_demo']:02d}" / "bank_record.json"))
                rows.append(episode)
            write_json_atomic(completion, {"schema_version": "ember_return_credit_collection_group_v1",
                "implementation_commit": state["commit"], "task": task_id, "state": state_id,
                "teacher_demo": registration["teacher_demo"], "rows": rows,
                "successes": sum(row["success"] for row in rows)})
    finally:
        infer.close()
        store.close()
        pool.close()


def _evaluate(spec, root, config, state, gpu_index, arms):
    evaluation, tasks, paths = _task_authority(root, config, spec)
    learning = load_learning_tasks(ASSET_ROOT, spec["data"]["gradient_tasks"],
                                   protocol_path=config["data"]["protocol"])
    runtime = _runtime(config)
    store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                 frame_stride=5, camera_view="agentview")
    base = _base_contract(evaluation, paths, tasks[2], root / "evaluation", full=False)
    pool = PersistentTaskEnvironmentPool(base, physical_gpu_id=gpu_index)
    infer = BatchedLoRAInference(runtime.policy, runtime.lora)
    try:
        for arm in arms:
            source = _load_state(runtime, spec, root, arm)
            for task_id in spec["evaluation"]["task_ids"]:
                task = tasks[task_id]
                envs, init_states = pool.switch(asdict(task))
                for teacher in spec["evaluation"]["teacher_demos"]:
                    output = root / "evaluation" / arm / f"task_{task_id:03d}_teacher_{teacher:02d}"
                    completion = output / "completion.json"
                    if completion.exists():
                        continue
                    if output.exists():
                        raise ValueError("return-credit panel has preserved partial output")
                    output.mkdir(parents=True)
                    bank = _bank(runtime, store, learning, root, source, arm, task_id,
                                 teacher, phase="evaluation", state=state)
                    lora = _bank_state(bank)
                    full = task_id in spec["capture"]["full_cases"]["evaluation_tasks"] and teacher == 46
                    contract = _base_contract(evaluation, paths, task, output, full=full)
                    with infer.activate([lora]):
                        rows = rollout_shard(envs=envs[:1], init_states=init_states,
                            task=asdict(task), state_ids=tuple(spec["evaluation"]["init_state_ids"]),
                            contract=contract, policy=runtime.policy,
                            preprocess=runtime.processor,
                            postprocess=runtime.processor.unnormalize_action)
                    for row in rows:
                        validate_passive_trace_row(row, contract, asdict(task))
                        row.update(global_task_id=task_id, teacher_demo=teacher, arm=arm,
                                   bank_record=file_record(root / "banks" / "evaluation" / arm /
                                       f"task_{task_id:03d}_teacher_{teacher:02d}" / "bank_record.json"))
                    write_json_atomic(completion, {"schema_version": "ember_return_credit_panel_v1",
                        "implementation_commit": state["commit"], "arm": arm,
                        "task": task_id, "teacher_demo": teacher, "rows": rows,
                        "successes": sum(row["success"] for row in rows)})
    finally:
        infer.close()
        store.close()
        pool.close()


def _replay(spec, root, config, state, arms):
    if not (root / "gradient" / "completion.json").is_file():
        raise ValueError("return-credit candidate replay requires sealed gradients")
    runtime = _runtime(config)
    learning = load_learning_tasks(ASSET_ROOT, spec["data"]["gradient_tasks"],
                                   protocol_path=config["data"]["protocol"])
    store = RawTeacherVideoStore(tuple(task.authority for task in learning.values()),
                                 frame_stride=5, camera_view="agentview")
    audit_path = Path("/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924/query_index.jsonl")
    with audit_path.open(encoding="utf-8") as handle:
        osc_audit = json.loads(handle.readline())["osc_channel_audit"]
    infer = BatchedLoRAInference(runtime.policy, runtime.lora)
    try:
        for arm in arms:
            if arm not in ("R", "NEG", "FM"):
                raise ValueError("only three registered candidates need saved-decision replay")
            source = _load_state(runtime, spec, root, arm)
            for condition in spec["collection"]["conditions"]:
                task_id, teacher = condition["task"], condition["teacher_demo"]
                output = root / "replay" / arm / f"task_{task_id:03d}"
                completion = output / "completion.json"
                if completion.exists():
                    continue
                if output.exists():
                    raise ValueError("return-credit candidate replay has preserved partial output")
                output.mkdir(parents=True)
                bank = _bank(runtime, store, learning, root, source, arm, task_id,
                             teacher, phase="replay", state=state)
                lora = _bank_state(bank)
                normalized, environment, osc, old_means, identities = [], [], [], [], []
                for state_id in condition["init_state_ids"]:
                    group = read_json(root / "collection" / "groups" /
                        f"task_{task_id:03d}_state_{state_id:02d}" / "completion.json")
                    for row in group["rows"]:
                        record = row["return_credit_collection"]
                        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
                        if payload["M"] != len(payload["decisions"]):
                            raise ValueError("return-credit candidate replay decision reservoir changed")
                        for decision in payload["decisions"]:
                            batch = {key: value.to(runtime.device)
                                     for key, value in decision["processed"].items()}
                            noise = decision["flow_noise"].unsqueeze(0).to(runtime.device)
                            with torch.inference_mode(), infer.activate([lora]):
                                predicted = runtime.policy.predict_action_chunk(
                                    batch, noise=noise, num_steps=10)
                                env = runtime.processor.unnormalize_action(predicted)
                            if (predicted.shape != (1, 50, 7) or env.shape != (1, 50, 7)
                                    or not torch.isfinite(predicted).all()
                                    or not torch.isfinite(env).all()):
                                raise ValueError("candidate real 10-flow replay changed shape or became nonfinite")
                            p = predicted[0].float().cpu().numpy()
                            e = env[0].float().cpu().numpy()
                            normalized.append(p)
                            environment.append(e)
                            osc.append(scaled_osc_actions(e[None], osc_audit)[0])
                            old_means.append(decision["mu_old_full"].float().numpy())
                            identities.append({"task": task_id, "state": state_id,
                                "replica": row["replica"], "replan": decision["replan_index"],
                                "policy_noise_seed": decision["policy_noise_seed"],
                                "exploration_seed": decision["exploration_seed"],
                                "source_decisions": record["path"]})
                path = output / "predictions.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, normalized=np.stack(normalized),
                        environment=np.stack(environment), osc_first5=np.stack(osc),
                        collection_mu_old=np.stack(old_means))
                write_json_atomic(completion, {"schema_version": "ember_return_credit_replay_v1",
                    "implementation_commit": state["commit"], "arm": arm, "task": task_id,
                    "teacher_demo": teacher, "predictions": len(identities),
                    "identities": identities, "bank_record": file_record(root / "banks" / "replay" /
                        arm / f"task_{task_id:03d}_teacher_{teacher:02d}" / "bank_record.json"),
                    "arrays": file_record(path)})
    finally:
        infer.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "registry-audit", "engineering-smoke", "pilot", "collect", "replay", "evaluate"))
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--arms", nargs="*", choices=("P", "R", "NEG", "FM"))
    args = parser.parse_args()
    spec = authority()
    root, config, state = _identity(spec, formal=args.stage not in (
        "inspect", "registry-audit", "engineering-smoke"))
    if args.stage == "inspect":
        from ember.writer.return_credit import inspect_parent_events
        events = inspect_parent_events(spec)
        print(json.dumps({"tasks": [row["task"] for row in events],
            "teachers": [row["teacher_demo"] for row in events],
            "collection_groups": spec["collection"]["groups"],
            "evaluation_rows": spec["evaluation"]["total_episodes"]}))
        return
    if args.gpu_index is None:
        parser.error("GPU stages require --gpu-index for the physical EGL lock")
    if args.stage == "registry-audit":
        _registry_audit(spec, root, config, state, args.gpu_index)
    elif args.stage == "engineering-smoke":
        _engineering_smoke(spec, root, config, state, args.gpu_index)
    elif args.stage in ("pilot", "collect"):
        _collect(spec, root, config, state, args.gpu_index, pilot=args.stage == "pilot")
    elif args.stage == "replay":
        if not args.arms or len(set(args.arms)) != len(args.arms):
            parser.error("candidate replay needs distinct --arms")
        _replay(spec, root, config, state, args.arms)
    else:
        if not args.arms or len(set(args.arms)) != len(args.arms):
            parser.error("evaluation needs distinct --arms")
        _evaluate(spec, root, config, state, args.gpu_index, args.arms)


if __name__ == "__main__":
    main()
