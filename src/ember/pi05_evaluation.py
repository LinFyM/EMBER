"""Persistent policy workers and strict aggregation for canonical PI05 evaluation."""

from __future__ import annotations

import json
import math
import os
import string
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ember.eval_adapters import (
    WRITER_ADAPTER_KINDS,
    episode_adapter_fields,
    load_evaluation_adapter as _load_evaluation_adapter,
    validate_episode_adapter_fields,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import load_run_contract, policy_noise_seed
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_queue import (
    EvaluationClaim,
    EvaluationShard,
    claim_next,
    complete_job,
    fail_job,
    publish_json_exclusive,
    queue_summary,
    read_json_with_size,
)
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.pi05_processing import libero_policy_input
from ember.writer.topology import bind_current_process_to_cuda_numa, cuda_numa_node


SHARD_RESULT_SCHEMA = "ember_pi05_eval_shard_v1"
_load_policy = load_policy
_validate_worker_assets = validate_worker_assets


def make_policy_noise(
    slots: Sequence[Mapping[str, Any]],
    *,
    root_seed: int,
    suite: str,
    task_id: int,
    chunk_size: int,
    max_action_dim: int,
    device: Any,
) -> tuple[Any, tuple[int, ...]]:
    """Generate order-independent PI05 flow noise for each planning rollout."""

    import torch

    tensors = []
    seeds = []
    for slot in slots:
        seed = policy_noise_seed(
            root_seed,
            suite,
            task_id,
            int(slot["init_state_id"]),
            int(slot["replan_index"]),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        tensors.append(
            torch.randn(
                (chunk_size, max_action_dim),
                dtype=torch.float32,
                generator=generator,
                device="cpu",
            )
        )
        seeds.append(seed)
    return torch.stack(tensors).to(device=device), tuple(seeds)


def _append_worker_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def task_lookup(contract: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    tasks = {
        (row["suite"], int(row["task_id"])): dict(row) for row in contract["tasks"]
    }
    if len(tasks) != len(contract["tasks"]):
        raise Pi05EvaluationError("run contract contains duplicate target tasks")
    return tasks


def _stage_predicate_snapshot(
    env: Any,
    states: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
    owner = getattr(env, "env", env)
    raw_states = (
        tuple(tuple(str(value) for value in state) for state in states)
        if states is not None
        else tuple(
            tuple(str(value) for value in state)
            for state in owner.parsed_problem["goal_state"]
        )
    )
    if not raw_states or not callable(getattr(owner, "_eval_predicate", None)):
        raise Pi05EvaluationError("stage diagnosis requires LIBERO BDDL predicates")
    return raw_states, tuple(bool(owner._eval_predicate(state)) for state in raw_states)


def _update_stage_predicates(env: Any, slot: dict[str, Any]) -> None:
    states, values = _stage_predicate_snapshot(
        env,
        slot["stage_predicate_states"],
    )
    slot["stage_predicate_ever"] = tuple(
        before or current
        for before, current in zip(
            slot["stage_predicate_ever"], values, strict=True
        )
    )
    slot["stage_predicate_peak"] = max(
        int(slot["stage_predicate_peak"]),
        sum(values),
    )
    if values != slot["stage_predicate_last"]:
        slot["stage_predicate_transitions"].append(
            {"step": int(slot["steps"]), "satisfied": list(values)}
        )
        slot["stage_predicate_last"] = values
    slot["stage_predicate_states"] = states


def _start_fixed_episode(
    *,
    env: Any,
    init_state_id: int,
    init_states: Any,
    task: Mapping[str, Any],
    contract: Mapping[str, Any],
    root_seed: int,
    dummy: np.ndarray,
    task_adapter: Any | None,
    capture_occupancy: bool,
) -> dict[str, Any]:
    env.seed(root_seed)
    env.reset()
    observation = env.set_init_state(init_states[init_state_id])
    for _ in range(int(contract["environment"]["dummy_settling_steps"])):
        observation, _, _, _ = env.step(dummy)
    prepared = None
    if task_adapter is not None:
        prepared = task_adapter.prepare_episode(
            suite=str(task["suite"]),
            task_id=int(task["task_id"]),
            init_state_id=init_state_id,
        )
    slot = {
        "init_state_id": init_state_id,
        "obs": observation,
        "steps": 0,
        "replan_index": 0,
        "policy_noise_seeds": [],
        "action_plan": deque(),
        "started": time.monotonic(),
    }
    if prepared is not None:
        slot["writer_lora"] = prepared
    if capture_occupancy:
        slot["replay_observations"] = []
        slot["replay_action_chunks"] = []
    if contract.get("diagnostic_stage_predicates") is not None:
        states, values = _stage_predicate_snapshot(env)
        slot.update(
            {
                "stage_predicate_states": states,
                "stage_predicate_last": values,
                "stage_predicate_ever": values,
                "stage_predicate_peak": sum(values),
                "stage_predicate_transitions": [
                    {"step": 0, "satisfied": list(values)}
                ],
            }
        )
    return slot


def _plan_action_chunks(
    slots: Sequence[dict[str, Any] | None],
    *,
    task: Mapping[str, Any],
    contract: Mapping[str, Any],
    policy: Any,
    preprocess: Any,
    postprocess: Any,
    task_adapter: Any | None,
    root_seed: int,
    replan_steps: int,
) -> None:
    import torch

    planning = [slot for slot in slots if slot is not None and not slot["action_plan"]]
    if not planning:
        return
    batched_adapter = task_adapter is not None and callable(
        getattr(task_adapter, "predict_action_chunk", None)
    )
    groups = (
        [planning]
        if task_adapter is None or batched_adapter
        else [[slot] for slot in planning]
    )
    for group in groups:
        if task_adapter is not None and not batched_adapter:
            task_adapter.install(group[0]["writer_lora"])
        processed = [
            preprocess(libero_policy_input(slot["obs"], str(task["language"])))
            for slot in group
        ]
        batch = {
            key: torch.cat([item[key] for item in processed], dim=0)
            for key in processed[0]
            if isinstance(processed[0][key], torch.Tensor)
        }
        noise, seeds = make_policy_noise(
            group,
            root_seed=root_seed,
            suite=str(task["suite"]),
            task_id=int(task["task_id"]),
            chunk_size=int(policy.config.chunk_size),
            max_action_dim=int(policy.config.max_action_dim),
            device=batch[next(iter(batch))].device,
        )
        with torch.inference_mode():
            predict = (
                policy.predict_action_chunk
                if task_adapter is None or not batched_adapter
                else task_adapter.predict_action_chunk
            )
            arguments = (
                (batch,)
                if task_adapter is None or not batched_adapter
                else ([slot["writer_lora"] for slot in group], batch)
            )
            chunks = predict(
                *arguments,
                noise=noise,
                num_steps=int(contract["policy"]["num_inference_steps"]),
            )
            actions = postprocess(chunks).detach().cpu().numpy()
        for row, (slot, plan, seed) in enumerate(
            zip(group, actions, seeds, strict=True)
        ):
            if "replay_observations" in slot:
                slot["replay_observations"].append(
                    {
                        key: value.detach().to(device="cpu").contiguous()
                        for key, value in processed[row].items()
                        if isinstance(value, torch.Tensor)
                    }
                )
                slot["replay_action_chunks"].append(
                    chunks[row : row + 1].detach().to(device="cpu").contiguous()
                )
            slot["action_plan"].extend(plan[:replan_steps])
            slot["policy_noise_seeds"].append(seed)
            slot["replan_index"] += 1


def rollout_shard(
    *,
    envs: Sequence[Any],
    init_states: Any,
    task: Mapping[str, Any],
    state_ids: Sequence[int],
    contract: Mapping[str, Any],
    policy: Any,
    preprocess: Any,
    postprocess: Any,
    task_adapter: Any | None = None,
) -> list[dict[str, Any]]:
    if not state_ids or len(set(state_ids)) != len(state_ids):
        raise Pi05EvaluationError("evaluation shard state IDs are empty or duplicated")
    dummy = np.asarray(contract["environment"]["dummy_action"], dtype=np.float32)
    max_steps = int(task["horizon"])
    replan_steps = int(contract["policy"]["replan_steps"])
    root_seed = int(contract["rng"]["inference_seed"])
    worker_started = time.monotonic()
    rows: list[dict[str, Any]] = []
    capture_occupancy = contract.get("diagnostic_occupancy_capture") is not None

    active_count = min(len(envs), len(state_ids))
    active_envs = envs[:active_count]
    next_state = active_count
    slots: list[dict[str, Any] | None] = [
        _start_fixed_episode(
            env=env,
            init_state_id=int(state_id),
            init_states=init_states,
            task=task,
            contract=contract,
            root_seed=root_seed,
            dummy=dummy,
            task_adapter=task_adapter,
            capture_occupancy=capture_occupancy,
        )
        for env, state_id in zip(active_envs, state_ids[:active_count], strict=True)
    ]
    policy.reset()
    while any(slot is not None for slot in slots):
        _plan_action_chunks(
            slots,
            task=task,
            contract=contract,
            policy=policy,
            preprocess=preprocess,
            postprocess=postprocess,
            task_adapter=task_adapter,
            root_seed=root_seed,
            replan_steps=replan_steps,
        )

        for slot_index, (env, slot) in enumerate(zip(active_envs, slots, strict=True)):
            if slot is None:
                continue
            obs, _, done, _ = env.step(slot["action_plan"].popleft())
            slot["obs"] = obs
            slot["steps"] += 1
            if "stage_predicate_states" in slot:
                _update_stage_predicates(env, slot)
            if not bool(done) and slot["steps"] < max_steps:
                continue
            finished = time.monotonic()
            row = {
                "suite": task["suite"],
                "task_id": int(task["task_id"]),
                "split_role": task["split_role"],
                "language": task["language"],
                "init_state_id": int(slot["init_state_id"]),
                "env_seed": root_seed,
                "policy_seed_root": root_seed,
                "policy_noise_seeds": list(slot["policy_noise_seeds"]),
                "success": bool(done),
                "steps": int(slot["steps"]),
                "wall_seconds": finished - float(slot["started"]),
                "finished_at": finished - worker_started,
            }
            if "stage_predicate_states" in slot:
                row["stage_predicates"] = {
                    "schema_version": "ember_pi05_stage_predicate_episode_v1",
                    "predicates": [
                        list(state) for state in slot["stage_predicate_states"]
                    ],
                    "transitions": slot["stage_predicate_transitions"],
                    "ever_satisfied": list(slot["stage_predicate_ever"]),
                    "final_satisfied": list(slot["stage_predicate_last"]),
                    "peak_satisfied_count": int(slot["stage_predicate_peak"]),
                }
            if capture_occupancy:
                import torch

                capture = contract["diagnostic_occupancy_capture"]
                root = Path(str(capture["trajectory_root"]))
                root.mkdir(parents=True, exist_ok=True)
                path = root / (
                    f"{task['suite']}_task_{int(task['task_id']):02d}_"
                    f"state_{int(slot['init_state_id']):03d}.pt"
                )
                torch.save(
                    {
                        "schema_version": "ember_writer_occupancy_trajectory_v1",
                        "suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "init_state_id": int(slot["init_state_id"]),
                        "success": bool(done),
                        "steps": int(slot["steps"]),
                        "policy_noise_seeds": tuple(slot["policy_noise_seeds"]),
                        "observations": tuple(slot["replay_observations"]),
                        "action_chunks": tuple(slot["replay_action_chunks"]),
                    },
                    path,
                )
                row["occupancy_trajectory"] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "replans": len(slot["replay_observations"]),
                }
            row.update(
                episode_adapter_fields(contract, task_adapter, slot.get("writer_lora"))
            )
            rows.append(row)
            if next_state < len(state_ids):
                slots[slot_index] = _start_fixed_episode(
                    env=env,
                    init_state_id=int(state_ids[next_state]),
                    init_states=init_states,
                    task=task,
                    contract=contract,
                    root_seed=root_seed,
                    dummy=dummy,
                    task_adapter=task_adapter,
                    capture_occupancy=capture_occupancy,
                )
                next_state += 1
            else:
                slots[slot_index] = None
    return sorted(rows, key=lambda row: int(row["init_state_id"]))


def validate_shard_result(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    shard: EvaluationShard,
) -> list[dict[str, Any]]:
    _validate_shard_header(payload, contract=contract, shard=shard)
    task = task_lookup(contract).get(shard.task_key)
    if task is None:
        raise Pi05EvaluationError(
            f"raw evaluation shard task is outside contract: {shard.job_id}"
        )
    rows = [dict(row) for row in payload.get("rows", [])]
    expected_ids = set(int(value) for value in shard.init_state_ids)
    actual_ids = [int(row.get("init_state_id", -1)) for row in rows]
    if len(rows) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise Pi05EvaluationError(
            f"raw evaluation shard state coverage changed: {shard.job_id}"
        )
    for row in rows:
        _validate_episode_row(row, contract=contract, shard=shard, task=task)
    return sorted(rows, key=lambda row: int(row["init_state_id"]))


def _validate_shard_header(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    shard: EvaluationShard,
) -> None:
    observed_shard = dict(payload.get("shard", {}))
    if "init_state_ids" in observed_shard:
        observed_shard["init_state_ids"] = tuple(observed_shard["init_state_ids"])
    if (
        payload.get("schema_version") != SHARD_RESULT_SCHEMA
        or payload.get("contract_reference") != contract["contract_reference"]
        or payload.get("job_id") != shard.job_id
        or observed_shard != asdict(shard)
    ):
        raise Pi05EvaluationError(
            f"raw evaluation shard contract changed: {shard.job_id}"
        )
    producer = payload.get("producer", {})
    started_unix = float(payload.get("started_unix", float("nan")))
    finished_unix = float(payload.get("finished_unix", float("nan")))
    if (
        not producer.get("worker_id")
        or len(str(producer.get("claim_token", ""))) != 32
        or int(producer.get("attempt", 0)) <= 0
        or not math.isfinite(started_unix)
        or not math.isfinite(finished_unix)
        or finished_unix < started_unix
    ):
        raise Pi05EvaluationError(
            f"raw evaluation shard producer is invalid: {shard.job_id}"
        )


def _validate_episode_row(
    row: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    shard: EvaluationShard,
    task: Mapping[str, Any],
) -> None:
    root_seed = int(contract["rng"]["inference_seed"])
    replan_steps = int(contract["policy"]["replan_steps"])
    state_id = int(row["init_state_id"])
    steps = int(row.get("steps", 0))
    seeds = [int(value) for value in row.get("policy_noise_seeds", [])]
    expected_seeds = [
        policy_noise_seed(root_seed, shard.suite, shard.task_id, state_id, index)
        for index in range(math.ceil(steps / replan_steps))
    ]
    adapter = contract.get("adapter")
    adapter_valid = validate_episode_adapter_fields(
        adapter,
        row,
        suite=shard.suite,
        task_id=shard.task_id,
        init_state_id=state_id,
    )
    stage = row.get("stage_predicates")
    if contract.get("diagnostic_stage_predicates") is None:
        stage_valid = stage is None
    else:
        final_predicates = (
            stage.get("final_satisfied", ()) if isinstance(stage, Mapping) else ()
        )
        stage_valid = (
            isinstance(stage, Mapping)
            and stage.get("schema_version")
            == "ember_pi05_stage_predicate_episode_v1"
            and bool(final_predicates)
            and bool(all(final_predicates)) == bool(row.get("success"))
        )
    valid = (
        row.get("suite") == shard.suite
        and int(row.get("task_id", -1)) == shard.task_id
        and row.get("language") == task["language"]
        and row.get("split_role") == task["split_role"]
        and type(row.get("success")) is bool
        and 1 <= steps <= shard.horizon
        and int(row.get("env_seed", -1)) == root_seed
        and int(row.get("policy_seed_root", -1)) == root_seed
        and seeds == expected_seeds
        and adapter_valid
        and stage_valid
    )
    if not valid:
        raise Pi05EvaluationError(
            f"raw evaluation row contract changed: {shard.job_id}"
        )


def _complete_published_shard(
    *,
    output_dir: Path,
    queue_path: Path,
    claim: EvaluationClaim,
    worker_id: str,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    relative = Path("shards") / f"{claim.shard.job_id}.json"
    path = output_dir / relative
    if not path.exists():
        return None
    payload, artifact_bytes = read_json_with_size(path)
    rows = validate_shard_result(payload, contract=contract, shard=claim.shard)
    complete_job(
        queue_path,
        job_id=claim.shard.job_id,
        worker_id=worker_id,
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=len(rows),
        successes=sum(bool(row["success"]) for row in rows),
    )
    return rows


@dataclass
class WorkerRuntime:
    output_dir: Path
    queue_path: Path
    worker_id: str
    gpu_index: int
    gpu_slot: int
    replica: int
    numa_node: int
    gpu_uuid: str
    gpu_name: str
    cpu_affinity: tuple[int, ...]
    contract: dict[str, Any]
    tasks: dict[tuple[str, int], dict[str, Any]]
    policy: Any
    task_adapter: Any | None
    preprocess: Any
    postprocess: Any
    pool: PersistentTaskEnvironmentPool
    event_path: Path


def _parse_worker_assignment(
    worker_id: str, contract: Mapping[str, Any]
) -> tuple[int, int, int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        gpu_text, replica_text = worker_id.split("-r", 1)
        gpu_index, replica = int(gpu_text), int(replica_text)
    except ValueError as error:
        raise Pi05EvaluationError(
            f"invalid PI05 evaluator worker ID: {worker_id}"
        ) from error
    physical_gpu_ids = tuple(
        int(value)
        for value in contract["parallel"].get(
            "physical_gpu_ids",
            range(int(contract["parallel"].get("physical_gpu_count", 8))),
        )
    )
    valid = (
        visible == str(gpu_index)
        and gpu_index in physical_gpu_ids
        and 0 <= replica < int(contract["parallel"]["replicas_per_gpu"])
    )
    if not valid:
        raise Pi05EvaluationError(
            "worker ID does not match its one physical visible GPU"
        )
    return gpu_index, physical_gpu_ids.index(gpu_index), replica


def _initialize_worker(
    output_dir: Path,
    worker_id: str,
    *,
    writer_generation: bool = False,
) -> WorkerRuntime:
    import torch

    output_dir = output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    queue_path = output_dir / "queue.sqlite3"
    gpu_index, gpu_slot, replica = _parse_worker_assignment(worker_id, contract)
    if writer_generation:
        adapter = contract.get("adapter")
        generators = int(contract["parallel"].get("writer_generators_per_gpu", 0))
        if (
            not isinstance(adapter, Mapping)
            or adapter.get("kind") not in WRITER_ADAPTER_KINDS
            or not 0 <= replica < generators
        ):
            raise Pi05EvaluationError("invalid Writer generator worker assignment")
    os.environ.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(gpu_index),
        LIBERO_CONFIG_PATH=str((output_dir / "libero_config").resolve()),
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Pi05EvaluationError(
            "each PI05 evaluator worker must see exactly one CUDA GPU"
        )
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    numa_node = cuda_numa_node(0)
    if affinity is None or numa_node is None:
        raise Pi05EvaluationError("PI05 evaluator requires GPU-local NUMA affinity")
    model_path, normalization, tokenizer_path = validate_worker_assets(contract)
    torch.manual_seed(int(contract["rng"]["inference_seed"]))
    torch.cuda.manual_seed(int(contract["rng"]["inference_seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    policy, preprocess, postprocess = load_policy(
        model_path,
        normalization["stats"],
        tokenizer_path,
        contract["policy"],
    )
    task_adapter = _load_evaluation_adapter(
        policy,
        contract,
        device=torch.device("cuda:0"),
        writer_generation=writer_generation,
    )
    return WorkerRuntime(
        output_dir=output_dir,
        queue_path=queue_path,
        worker_id=worker_id,
        gpu_index=gpu_index,
        gpu_slot=gpu_slot,
        replica=replica,
        numa_node=numa_node,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        gpu_name=str(torch.cuda.get_device_name(0)),
        cpu_affinity=affinity,
        contract=contract,
        tasks=task_lookup(contract),
        policy=policy,
        task_adapter=task_adapter,
        preprocess=preprocess,
        postprocess=postprocess,
        pool=PersistentTaskEnvironmentPool(
            contract,
            physical_gpu_id=gpu_index,
        ),
        event_path=output_dir / "workers" / f"{worker_id}.jsonl",
    )


def _publish_claim_result(
    runtime: WorkerRuntime,
    claim: EvaluationClaim,
    rows: list[dict[str, Any]],
    started_unix: float,
) -> bool:
    payload = {
        "schema_version": SHARD_RESULT_SCHEMA,
        "contract_reference": runtime.contract["contract_reference"],
        "job_id": claim.shard.job_id,
        "shard": asdict(claim.shard),
        "producer": {
            "worker_id": runtime.worker_id,
            "claim_token": claim.claim_token,
            "attempt": claim.attempt,
        },
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "rows": rows,
    }
    validate_shard_result(payload, contract=runtime.contract, shard=claim.shard)
    relative = Path("shards") / f"{claim.shard.job_id}.json"
    try:
        artifact_bytes = publish_json_exclusive(runtime.output_dir / relative, payload)
    except Pi05EvaluationError:
        adopted = _complete_published_shard(
            output_dir=runtime.output_dir,
            queue_path=runtime.queue_path,
            claim=claim,
            worker_id=runtime.worker_id,
            contract=runtime.contract,
        )
        if adopted is None:
            raise
        return True
    complete_job(
        runtime.queue_path,
        job_id=claim.shard.job_id,
        worker_id=runtime.worker_id,
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=len(rows),
        successes=sum(bool(row["success"]) for row in rows),
    )
    return False


def _execute_claim(runtime: WorkerRuntime, claim: EvaluationClaim) -> bool:
    published = _complete_published_shard(
        output_dir=runtime.output_dir,
        queue_path=runtime.queue_path,
        claim=claim,
        worker_id=runtime.worker_id,
        contract=runtime.contract,
    )
    if published is not None:
        return True
    task = runtime.tasks[claim.task_key]
    envs, init_states = runtime.pool.switch(task)
    started_unix = time.time()
    rows = rollout_shard(
        envs=envs,
        init_states=init_states,
        task=task,
        state_ids=claim.shard.init_state_ids,
        contract=runtime.contract,
        policy=runtime.policy,
        preprocess=runtime.preprocess,
        postprocess=runtime.postprocess,
        task_adapter=runtime.task_adapter,
    )
    return _publish_claim_result(runtime, claim, rows, started_unix)


def _run_writer_bootstrap(runtime: WorkerRuntime, invocation_id: str) -> None:
    from ember.writer.evaluation_runtime import run_writer_generation_phase

    run_writer_generation_phase(
        runtime,
        invocation_id=invocation_id,
        append_event=_append_worker_event,
    )


def _drain_claim_queue(
    runtime: WorkerRuntime,
    *,
    worker_id: str,
) -> tuple[int, int]:
    completed = adopted = 0
    preferred_task: tuple[str, int] | None = None
    while (
        claim := claim_next(
            runtime.queue_path,
            worker_id=worker_id,
            preferred_task=preferred_task,
            physical_gpu=runtime.gpu_slot,
        )
    ) is not None:
        preferred_task = claim.task_key
        try:
            adopted += int(_execute_claim(runtime, claim))
            completed += 1
        except Exception as error:
            fail_job(
                runtime.queue_path,
                job_id=claim.shard.job_id,
                worker_id=worker_id,
                claim_token=claim.claim_token,
                error=repr(error),
            )
            raise
    return completed, adopted


def run_worker(
    *,
    output_dir: Path,
    worker_id: str,
    writer_generator: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    invocation_id = os.environ.get("EMBER_PI05_EVAL_INVOCATION_ID", "")
    if len(invocation_id) != 32 or any(
        character not in string.hexdigits for character in invocation_id
    ):
        raise Pi05EvaluationError("evaluator worker lacks a launcher invocation lease")
    contract = load_run_contract(output_dir / "run_contract.json")
    event_path = output_dir / "workers" / f"{worker_id}.jsonl"
    process_started_unix = time.time()
    _append_worker_event(
        event_path,
        {
            "event": "process_started",
            "unix": process_started_unix,
            "worker_id": worker_id,
            "pid": os.getpid(),
            "invocation_id": invocation_id,
            "contract_reference": contract["contract_reference"],
        },
    )
    runtime: WorkerRuntime | None = None
    completed = 0
    adopted = 0
    try:
        runtime = _initialize_worker(
            output_dir,
            worker_id,
            writer_generation=writer_generator,
        )
        ready_unix = time.time()
        _append_worker_event(
            event_path,
            {
                "event": "ready",
                "unix": ready_unix,
                "worker_id": worker_id,
                "pid": os.getpid(),
                "invocation_id": invocation_id,
                "physical_gpu": runtime.gpu_index,
                "gpu_uuid": runtime.gpu_uuid,
                "gpu_name": runtime.gpu_name,
                "replica": runtime.replica,
                "numa_node": runtime.numa_node,
                "cpu_affinity": list(runtime.cpu_affinity),
                "model_load_seconds": ready_unix - process_started_unix,
                "contract_reference": runtime.contract["contract_reference"],
                "writer_generator": writer_generator,
            },
        )
        if writer_generator:
            _run_writer_bootstrap(runtime, invocation_id)
        completed, adopted = _drain_claim_queue(runtime, worker_id=worker_id)
    except Exception as error:
        _append_worker_event(
            event_path,
            {
                "event": "failed",
                "unix": time.time(),
                "worker_id": worker_id,
                "pid": os.getpid(),
                "invocation_id": invocation_id,
                "contract_reference": contract["contract_reference"],
                "error": repr(error),
            },
        )
        raise
    finally:
        if runtime is not None:
            runtime.pool.close()
    summary = {
        "event": "finished",
        "unix": time.time(),
        "worker_id": worker_id,
        "pid": os.getpid(),
        "invocation_id": invocation_id,
        "contract_reference": contract["contract_reference"],
        "completed_shards": completed,
        "adopted_shards": adopted,
        "wall_seconds": time.time() - process_started_unix,
        "queue": queue_summary(runtime.queue_path),
    }
    _append_worker_event(event_path, summary)
    return summary
