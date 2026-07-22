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
    episode_adapter_fields,
    load_evaluation_adapter as _load_evaluation_adapter,
    validate_episode_adapter_fields,
)
from ember.libero_evaluation import sha256_file
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import load_run_contract, policy_noise_seed
from ember.pi05_eval_queue import (
    EvaluationClaim,
    EvaluationShard,
    claim_next,
    complete_job,
    fail_job,
    publish_json_exclusive,
    queue_summary,
    read_json_with_sha256,
)
from ember.pi05_processing import Pi05LiberoProcessor, libero_policy_input
from ember.writer.topology import bind_current_process_to_cuda_numa, cuda_numa_node


SHARD_RESULT_SCHEMA = "ember_pi05_eval_shard_v1"


def _load_policy(
    model_path: Path,
    stats: Mapping[str, Any],
    tokenizer_path: Path,
    policy_contract: Mapping[str, Any],
) -> tuple[Any, Pi05LiberoProcessor, Any]:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION, OBS_STATE

    config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(config, PI05Config):
        raise Pi05EvaluationError("evaluation checkpoint did not resolve to PI05Config")
    config.device = "cuda:0"
    config.dtype = str(policy_contract["precision"])
    config.chunk_size = int(policy_contract["chunk_size"])
    config.n_action_steps = int(policy_contract["n_action_steps"])
    config.num_inference_steps = int(policy_contract["num_inference_steps"])
    config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(int(policy_contract["state_dim"]),)
    )
    config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION, shape=(int(policy_contract["action_dim"]),)
    )
    policy = PI05Policy.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=True,
    ).to("cuda:0").eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    processor = Pi05LiberoProcessor(
        stats,
        tokenizer_path,
        config.tokenizer_max_length,
        "cuda:0",
    )
    return policy, processor, processor.unnormalize_action


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


class PersistentTaskEnvironmentPool:
    """Keep one task's vector of raw LIBERO environments alive between shards."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        from libero.libero import benchmark

        self.contract = contract
        self.suites = {
            name: benchmark.get_benchmark_dict()[name]()
            for name in contract["environment"]["horizons"]
        }
        self.current_key: tuple[str, int] | None = None
        self.envs: list[Any] = []
        self.init_states: Any = None

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.envs = []
        self.init_states = None
        self.current_key = None

    def switch(self, task: Mapping[str, Any]) -> tuple[list[Any], Any]:
        key = task["suite"], int(task["task_id"])
        if self.current_key == key:
            return self.envs, self.init_states
        self.close()
        from libero.libero.envs import OffScreenRenderEnv

        bddl = (
            Path(self.contract["libero_paths"]["bddl_files"])
            / task["problem_folder"]
            / task["bddl_file"]
        )
        init_path = (
            Path(self.contract["libero_paths"]["init_states"])
            / task["suite"]
            / task["init_states_file"]
        )
        if (
            bddl.stat().st_size != int(task["bddl_bytes"])
            or sha256_file(bddl) != task["bddl_sha256"]
            or init_path.stat().st_size != int(task["init_states_bytes"])
            or sha256_file(init_path) != task["init_states_sha256"]
        ):
            raise Pi05EvaluationError(f"installed task assets changed: {key}")
        suite = self.suites[task["suite"]]
        installed = suite.get_task(int(task["task_id"]))
        if installed.language != task["language"]:
            raise Pi05EvaluationError(f"installed task language changed: {key}")
        self.init_states = suite.get_task_init_states(int(task["task_id"]))
        env_count = min(
            int(self.contract["parallel"]["envs_per_replica"]),
            len(task["init_state_ids"]),
        )
        self.envs = [
            OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=int(self.contract["environment"]["render_resolution"]),
                camera_widths=int(self.contract["environment"]["render_resolution"]),
            )
            for _ in range(env_count)
        ]
        self.current_key = key
        return self.envs, self.init_states


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
    return slot


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
    import torch

    if not state_ids or len(set(state_ids)) != len(state_ids):
        raise Pi05EvaluationError("evaluation shard state IDs are empty or duplicated")
    dummy = np.asarray(contract["environment"]["dummy_action"], dtype=np.float32)
    max_steps = int(task["horizon"])
    replan_steps = int(contract["policy"]["replan_steps"])
    root_seed = int(contract["rng"]["inference_seed"])
    worker_started = time.monotonic()
    rows: list[dict[str, Any]] = []

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
        )
        for env, state_id in zip(active_envs, state_ids[:active_count], strict=True)
    ]
    policy.reset()
    while any(slot is not None for slot in slots):
        planning = [slot for slot in slots if slot is not None and not slot["action_plan"]]
        if planning:
            planning_groups = [planning] if task_adapter is None else [[slot] for slot in planning]
            for group in planning_groups:
                if task_adapter is not None:
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
                    suite=str(task["suite"]), task_id=int(task["task_id"]),
                    chunk_size=int(policy.config.chunk_size),
                    max_action_dim=int(policy.config.max_action_dim),
                    device=batch[next(iter(batch))].device,
                )
                with torch.inference_mode():
                    chunks = policy.predict_action_chunk(
                        batch, noise=noise, num_steps=int(contract["policy"]["num_inference_steps"])
                    )
                    actions = postprocess(chunks).detach().cpu().numpy()
                for slot, plan, seed in zip(group, actions, seeds, strict=True):
                    slot["action_plan"].extend(plan[:replan_steps])
                    slot["policy_noise_seeds"].append(seed)
                    slot["replan_index"] += 1

        for slot_index, (env, slot) in enumerate(zip(active_envs, slots, strict=True)):
            if slot is None:
                continue
            obs, _, done, _ = env.step(slot["action_plan"].popleft())
            slot["obs"] = obs
            slot["steps"] += 1
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
        raise Pi05EvaluationError(f"raw evaluation shard task is outside contract: {shard.job_id}")
    rows = [dict(row) for row in payload.get("rows", [])]
    expected_ids = set(int(value) for value in shard.init_state_ids)
    actual_ids = [int(row.get("init_state_id", -1)) for row in rows]
    if len(rows) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise Pi05EvaluationError(f"raw evaluation shard state coverage changed: {shard.job_id}")
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
        or payload.get("contract_sha256") != contract["contract_sha256"]
        or payload.get("job_id") != shard.job_id
        or observed_shard != asdict(shard)
    ):
        raise Pi05EvaluationError(f"raw evaluation shard contract changed: {shard.job_id}")
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
        raise Pi05EvaluationError(f"raw evaluation shard producer is invalid: {shard.job_id}")


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
    )
    if not valid:
        raise Pi05EvaluationError(f"raw evaluation row contract changed: {shard.job_id}")


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
    payload, digest = read_json_with_sha256(path)
    rows = validate_shard_result(payload, contract=contract, shard=claim.shard)
    complete_job(
        queue_path,
        job_id=claim.shard.job_id,
        worker_id=worker_id,
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_sha256=digest,
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
    replica: int
    numa_node: int
    gpu_uuid: str
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
) -> tuple[int, int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        gpu_text, replica_text = worker_id.split("-r", 1)
        gpu_index, replica = int(gpu_text), int(replica_text)
    except ValueError as error:
        raise Pi05EvaluationError(f"invalid PI05 evaluator worker ID: {worker_id}") from error
    valid = (
        visible == str(gpu_index)
        and 0 <= gpu_index < 8
        and 0 <= replica < int(contract["parallel"]["replicas_per_gpu"])
    )
    if not valid:
        raise Pi05EvaluationError("worker ID does not match its one physical visible GPU")
    return gpu_index, replica


def _validate_worker_assets(contract: Mapping[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    normalization_path = Path(contract["normalization"]["path"])
    if sha256_file(normalization_path) != contract["normalization"]["sha256"]:
        raise Pi05EvaluationError("source-only normalization changed after queue creation")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    model_path = Path(contract["model"]["model_path"])
    for record in contract["model"]["model_files"]:
        relative = Path(record["path"]).relative_to("ema_policy")
        path = model_path / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise Pi05EvaluationError(f"PI05 model file changed after queue creation: {path}")
    tokenizer_path = Path(contract["tokenizer"]["path"])
    if (
        not tokenizer_path.is_file()
        or tokenizer_path.stat().st_size != int(contract["tokenizer"]["bytes"])
        or sha256_file(tokenizer_path) != contract["tokenizer"]["sha256"]
    ):
        raise Pi05EvaluationError("OpenPI tokenizer changed after queue creation")
    return model_path, normalization, tokenizer_path


def _initialize_worker(output_dir: Path, worker_id: str) -> WorkerRuntime:
    import torch

    output_dir = output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    queue_path = output_dir / "queue.sqlite3"
    gpu_index, replica = _parse_worker_assignment(worker_id, contract)
    os.environ.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(gpu_index),
        LIBERO_CONFIG_PATH=str((output_dir / "libero_config").resolve()),
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Pi05EvaluationError("each PI05 evaluator worker must see exactly one CUDA GPU")
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    numa_node = cuda_numa_node(0)
    if affinity is None or numa_node is None:
        raise Pi05EvaluationError("PI05 evaluator requires GPU-local NUMA affinity")
    model_path, normalization, tokenizer_path = _validate_worker_assets(contract)
    torch.manual_seed(int(contract["rng"]["inference_seed"]))
    torch.cuda.manual_seed(int(contract["rng"]["inference_seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    policy, preprocess, postprocess = _load_policy(
        model_path,
        normalization["stats"],
        tokenizer_path,
        contract["policy"],
    )
    task_adapter = _load_evaluation_adapter(
        policy,
        contract,
        device=torch.device("cuda:0"),
    )
    return WorkerRuntime(
        output_dir=output_dir,
        queue_path=queue_path,
        worker_id=worker_id,
        gpu_index=gpu_index,
        replica=replica,
        numa_node=numa_node,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        cpu_affinity=affinity,
        contract=contract,
        tasks=task_lookup(contract),
        policy=policy,
        task_adapter=task_adapter,
        preprocess=preprocess,
        postprocess=postprocess,
        pool=PersistentTaskEnvironmentPool(contract),
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
        "contract_sha256": runtime.contract["contract_sha256"],
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
        digest = publish_json_exclusive(runtime.output_dir / relative, payload)
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
        rows_sha256=digest,
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


def run_worker(*, output_dir: Path, worker_id: str) -> dict[str, Any]:
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
            "contract_sha256": contract["contract_sha256"],
        },
    )
    runtime: WorkerRuntime | None = None
    completed = 0
    adopted = 0
    preferred_task: tuple[str, int] | None = None
    try:
        runtime = _initialize_worker(output_dir, worker_id)
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
                "replica": runtime.replica,
                "numa_node": runtime.numa_node,
                "cpu_affinity": list(runtime.cpu_affinity),
                "model_load_seconds": ready_unix - process_started_unix,
                "contract_sha256": runtime.contract["contract_sha256"],
            },
        )
        while (claim := claim_next(
            runtime.queue_path,
            worker_id=worker_id,
            preferred_task=preferred_task,
        )) is not None:
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
    except Exception as error:
        _append_worker_event(
            event_path,
            {
                "event": "failed",
                "unix": time.time(),
                "worker_id": worker_id,
                "pid": os.getpid(),
                "invocation_id": invocation_id,
                "contract_sha256": contract["contract_sha256"],
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
        "contract_sha256": contract["contract_sha256"],
        "completed_shards": completed,
        "adopted_shards": adopted,
        "wall_seconds": time.time() - process_started_unix,
        "queue": queue_summary(runtime.queue_path),
    }
    _append_worker_event(event_path, summary)
    return summary
