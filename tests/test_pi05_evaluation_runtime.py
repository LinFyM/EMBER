from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, policy_noise_seed
from ember.pi05_eval_queue import (
    EvaluationShard,
    claim_next,
    complete_job,
    initialize_queue,
    publish_json_exclusive,
)
from ember.pi05_evaluation import (
    SHARD_RESULT_SCHEMA,
    _complete_published_shard,
    _load_evaluation_adapter,
    _validate_worker_assets,
    make_policy_noise,
    rollout_shard,
    validate_shard_result,
)
from ember.pi05_eval_results import aggregate_run


def _contract(output_dir: Path) -> dict:
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "mode": "smoke",
        "arm": "frozen_pi05_source_base",
        "role": "test",
        "output_dir": str(output_dir),
        "content_hash_policy": "disabled_by_owner",
        "model": {"optimizer_step": 1},
        "normalization": {"bytes": 1},
        "tokenizer": {"bytes": 1},
        "rng": {"inference_seed": 7},
        "policy": {"replan_steps": 5},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "split_role": "train",
                "language": "task zero",
                "init_state_ids": [0, 1],
            }
        ],
    }
    contract["contract_reference"] = f"{RUN_CONTRACT_SCHEMA}:test-contract"
    return contract


def _rows() -> list[dict]:
    return [
        {
            "suite": "libero_spatial",
            "task_id": 0,
            "split_role": "train",
            "language": "task zero",
            "init_state_id": state_id,
            "env_seed": 7,
            "policy_seed_root": 7,
            "policy_noise_seeds": [
                policy_noise_seed(7, "libero_spatial", 0, state_id, 0)
            ],
            "success": state_id == 0,
            "steps": 1,
            "wall_seconds": 0.1,
            "finished_at": 0.1,
        }
        for state_id in (0, 1)
    ]


def _payload(contract: dict, shard: EvaluationShard) -> dict:
    return {
        "schema_version": SHARD_RESULT_SCHEMA,
        "contract_reference": contract["contract_reference"],
        "job_id": shard.job_id,
        "shard": asdict(shard),
        "producer": {"worker_id": "0-r0", "claim_token": "a" * 32, "attempt": 1},
        "started_unix": 10.0,
        "finished_unix": 12.0,
        "rows": _rows(),
    }


def test_static_source_sft_rows_remain_batched_without_task_expert_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    contract["adapter"] = {
        "kind": "shared_source_sft_lora",
        "arm": "source_sft",
        "lora_state_sha256": "8" * 64,
    }
    contract["arm"] = "source_sft"
    shard = EvaluationShard(
        job_id="source-sft-job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    payload = _payload(contract, shard)
    for row in payload["rows"]:
        row["policy_adapter_sha256"] = "8" * 64
    assert all("task_expert" not in row for row in payload["rows"])
    assert len(validate_shard_result(payload, contract=contract, shard=shard)) == 2
    payload["rows"][0]["policy_adapter_sha256"] = "9" * 64
    with pytest.raises(Pi05EvaluationError, match="row contract changed"):
        validate_shard_result(payload, contract=contract, shard=shard)


def test_static_source_sft_adapter_is_installed_once_not_returned_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.source_sft.inference as inference

    calls = []

    class FakeStaticAdapter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(inference, "FrozenSourceSFTAdapter", FakeStaticAdapter)
    contract = {
        "mode": "smoke",
        "model": {"source": "base"},
        "adapter": {"kind": "shared_source_sft_lora"},
        "tasks": [
            {"suite": "libero_spatial", "task_id": 1},
            {"suite": "libero_object", "task_id": 3},
        ],
    }
    policy = object()
    result = _load_evaluation_adapter(policy, contract, device=torch.device("cpu"))
    assert result is None
    assert len(calls) == 1
    assert calls[0]["policy"] is policy
    assert calls[0]["task_keys"] == (
        ("libero_spatial", 1),
        ("libero_object", 3),
    )


def test_worker_asset_validation_uses_paths_and_sizes(tmp_path: Path) -> None:
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"stats": {}}) + "\n", encoding="utf-8")
    model_path = tmp_path / "policy"
    model_path.mkdir()
    model = model_path / "model.safetensors"
    model.write_bytes(b"model-a")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"token-a")
    contract = {
        "normalization": {
            "path": str(normalization),
            "bytes": normalization.stat().st_size,
        },
        "model": {
            "model_path": str(model_path),
            "frozen_policy_subdir": "policy",
            "model_files": [
                {
                    "path": "policy/model.safetensors",
                    "bytes": model.stat().st_size,
                }
            ],
        },
        "tokenizer": {
            "path": str(tokenizer),
            "bytes": tokenizer.stat().st_size,
        },
    }
    assert _validate_worker_assets(contract)[0] == model_path
    model.write_bytes(b"model-longer")
    with pytest.raises(Pi05EvaluationError, match="model file changed"):
        _validate_worker_assets(contract)
    model.write_bytes(b"model-a")
    tokenizer.write_bytes(b"token-longer")
    with pytest.raises(Pi05EvaluationError, match="tokenizer changed"):
        _validate_worker_assets(contract)


def test_policy_noise_is_invariant_to_batch_order() -> None:
    slots = [
        {"init_state_id": state_id, "replan_index": replan}
        for state_id, replan in ((3, 0), (1, 4), (8, 2))
    ]
    forward, forward_seeds = make_policy_noise(
        slots,
        root_seed=7,
        suite="libero_goal",
        task_id=2,
        chunk_size=5,
        max_action_dim=7,
        device=torch.device("cpu"),
    )
    reverse, reverse_seeds = make_policy_noise(
        list(reversed(slots)),
        root_seed=7,
        suite="libero_goal",
        task_id=2,
        chunk_size=5,
        max_action_dim=7,
        device=torch.device("cpu"),
    )
    by_seed = {
        seed: tensor for seed, tensor in zip(forward_seeds, forward, strict=True)
    }
    for seed, tensor in zip(reverse_seeds, reverse, strict=True):
        assert torch.equal(tensor, by_seed[seed])


def _observation(value: int) -> dict:
    image = torch.full((4, 4, 3), value, dtype=torch.uint8).numpy()
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
        "robot0_gripper_qpos": [0.0, 0.0],
    }


class _FakeEnv:
    def __init__(self, success_after: int = 2) -> None:
        self.action_steps = 0
        self.success_after = success_after

    def seed(self, seed: int) -> None:
        assert seed == 7

    def reset(self) -> dict:
        return {}

    def set_init_state(self, state: int) -> dict:
        self.action_steps = 0
        return _observation(state)

    def step(self, action):
        if float(action[-1]) != -1.0:
            self.action_steps += 1
        return (
            _observation(self.action_steps),
            0.0,
            self.action_steps >= self.success_after,
            {},
        )


class _FakePolicy:
    class Config:
        chunk_size = 50
        max_action_dim = 32

    config = Config()

    def __init__(self) -> None:
        self.batch_sizes = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch, *, noise, num_steps):
        assert noise.shape[1:] == (50, 32)
        assert num_steps == 10
        self.batch_sizes.append(int(noise.shape[0]))
        return torch.zeros((noise.shape[0], 50, 7), dtype=torch.float32)


def _preprocess(value):
    assert "observation.images.right_wrist_0_rgb" not in value
    return {
        key: tensor.unsqueeze(0)
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }


def test_rollout_executes_arbitrary_state_shard_with_per_row_noise() -> None:

    contract = {
        "environment": {
            "dummy_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "dummy_settling_steps": 10,
        },
        "policy": {"replan_steps": 5, "num_inference_steps": 10},
        "rng": {"inference_seed": 7},
        "adapter": {
            "kind": "shared_source_sft_lora",
            "lora_state_sha256": "8" * 64,
        },
    }
    task = {
        "suite": "libero_goal",
        "task_id": 4,
        "split_role": "test",
        "language": "put the bowl on top of the cabinet",
        "horizon": 300,
    }
    policy = _FakePolicy()
    rows = rollout_shard(
        envs=(_FakeEnv(), _FakeEnv()),
        init_states=tuple(range(10)),
        task=task,
        state_ids=(7, 2, 9),
        contract=contract,
        policy=policy,
        preprocess=_preprocess,
        postprocess=lambda value: value,
    )
    assert [row["init_state_id"] for row in rows] == [2, 7, 9]
    assert all(row["steps"] == 2 and row["success"] for row in rows)
    assert max(policy.batch_sizes) == 2
    for row in rows:
        assert row["policy_adapter_sha256"] == "8" * 64
        assert row["policy_noise_seeds"] == [
            policy_noise_seed(7, "libero_goal", 4, row["init_state_id"], 0)
        ]


def test_diagnostic_rollout_captures_requeryable_occupancy(tmp_path: Path) -> None:
    contract = {
        "environment": {
            "dummy_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "dummy_settling_steps": 10,
        },
        "policy": {"replan_steps": 5, "num_inference_steps": 10},
        "rng": {"inference_seed": 7},
        "diagnostic_occupancy_capture": {
            "trajectory_root": str(tmp_path / "trajectories")
        },
    }
    task = {
        "suite": "libero_goal",
        "task_id": 4,
        "split_role": "validation",
        "language": "put the bowl on top of the cabinet",
        "horizon": 300,
    }
    rows = rollout_shard(
        envs=(_FakeEnv(success_after=7),),
        init_states=tuple(range(10)),
        task=task,
        state_ids=(3,),
        contract=contract,
        policy=_FakePolicy(),
        preprocess=_preprocess,
        postprocess=lambda value: value,
    )
    record = rows[0]["occupancy_trajectory"]
    trajectory = torch.load(record["path"], map_location="cpu", weights_only=False)
    assert record["bytes"] > 0
    assert record["replans"] == 2
    assert trajectory["schema_version"] == "ember_pi05_occupancy_trajectory_v1"
    assert len(trajectory["observations"]) == len(trajectory["action_chunks"]) == 2
    assert len(trajectory["policy_noise_seeds"]) == 2


def test_sealed_stage_diagnosis_records_bddl_predicate_transitions() -> None:
    class StageEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__(success_after=3)
            self.env = self
            self.parsed_problem = {
                "goal_state": [
                    ["stage_one", "object"],
                    ["stage_two", "object"],
                ]
            }

        def _eval_predicate(self, state) -> bool:
            threshold = 1 if state[0] == "stage_one" else 3
            return self.action_steps >= threshold

    contract = {
        "environment": {
            "dummy_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "dummy_settling_steps": 10,
        },
        "policy": {"replan_steps": 5, "num_inference_steps": 10},
        "rng": {"inference_seed": 7},
        "diagnostic_stage_predicates": {
            "schema_version": "ember_pi05_stage_predicate_capture_v1"
        },
    }
    rows = rollout_shard(
        envs=(StageEnv(),),
        init_states=tuple(range(10)),
        task={
            "suite": "libero_10",
            "task_id": 1,
            "split_role": "validation",
            "language": "complete both stages",
            "horizon": 520,
        },
        state_ids=(0,),
        contract=contract,
        policy=_FakePolicy(),
        preprocess=_preprocess,
        postprocess=lambda value: value,
    )

    stage = rows[0]["stage_predicates"]
    assert rows[0]["success"] is True
    assert stage["predicates"] == [
        ["stage_one", "object"],
        ["stage_two", "object"],
    ]
    assert stage["transitions"] == [
        {"step": 0, "satisfied": [False, False]},
        {"step": 1, "satisfied": [True, False]},
        {"step": 3, "satisfied": [True, True]},
    ]
    assert stage["ever_satisfied"] == [True, True]
    assert stage["final_satisfied"] == [True, True]
    assert stage["peak_satisfied_count"] == 2


def test_task_expert_is_prepared_once_and_reinstalled_for_each_replan() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.prepared = []
            self.installed = []

        def prepare_episode(self, **identity):
            value = SimpleNamespace(evidence=dict(identity))
            self.prepared.append(value)
            return value

        def install(self, prepared) -> None:
            self.installed.append(prepared)

    adapter = FakeAdapter()
    contract = {
        "environment": {
            "dummy_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "dummy_settling_steps": 10,
        },
        "policy": {"replan_steps": 5, "num_inference_steps": 10},
        "rng": {"inference_seed": 7},
    }
    task = {
        "suite": "libero_goal",
        "task_id": 4,
        "split_role": "test",
        "language": "put the bowl on top of the cabinet",
        "horizon": 300,
    }
    adapted = rollout_shard(
        envs=(_FakeEnv(success_after=7),),
        init_states=tuple(range(10)),
        task=task,
        state_ids=(3,),
        contract=contract,
        policy=_FakePolicy(),
        preprocess=_preprocess,
        postprocess=lambda value: value,
        task_adapter=adapter,
    )
    assert adapted[0]["steps"] == 7
    assert len(adapter.prepared) == 1
    assert adapter.installed == [adapter.prepared[0], adapter.prepared[0]]
    assert adapted[0]["task_expert"] == {
        "suite": "libero_goal",
        "task_id": 4,
        "init_state_id": 3,
    }


def test_shard_validation_rejects_wrong_policy_schedule(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    payload = json.loads(json.dumps(_payload(contract, shard)))
    assert len(validate_shard_result(payload, contract=contract, shard=shard)) == 2
    payload["rows"][0]["policy_noise_seeds"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="row contract changed"):
        validate_shard_result(payload, contract=contract, shard=shard)


def test_aggregate_requires_queue_size_and_exact_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    queue = tmp_path / "queue.sqlite3"
    initialize_queue(queue, (shard,), contract_reference=contract["contract_reference"])
    claim = claim_next(queue, worker_id="0-r0")
    assert claim is not None
    relative = Path("shards/job.json")
    artifact_bytes = publish_json_exclusive(
        tmp_path / relative, _payload(contract, shard)
    )
    complete_job(
        queue,
        job_id=shard.job_id,
        worker_id="0-r0",
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=2,
        successes=1,
    )
    result = aggregate_run(tmp_path)
    assert result["overall"]["successes"] == 1
    assert result["overall"]["episodes"] == 2


def test_aggregate_rejects_queue_counts_that_differ_from_raw_rows(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    queue = tmp_path / "queue.sqlite3"
    initialize_queue(queue, (shard,), contract_reference=contract["contract_reference"])
    claim = claim_next(queue, worker_id="0-r0")
    assert claim is not None
    relative = Path("shards/job.json")
    artifact_bytes = publish_json_exclusive(
        tmp_path / relative, _payload(contract, shard)
    )
    complete_job(
        queue,
        job_id=shard.job_id,
        worker_id="0-r0",
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=2,
        successes=0,
    )
    with pytest.raises(Pi05EvaluationError, match="queue summary differs"):
        aggregate_run(tmp_path)


def test_aggregate_uses_full_launcher_window_and_validates_worker_topology(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    contract["parallel"] = {"replicas_per_gpu": 1, "worker_count": 8}
    (tmp_path / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    queue = tmp_path / "queue.sqlite3"
    initialize_queue(queue, (shard,), contract_reference=contract["contract_reference"])
    claim = claim_next(queue, worker_id="0-r0")
    assert claim is not None
    payload = _payload(contract, shard)
    payload["producer"] = {
        "worker_id": "0-r0",
        "claim_token": claim.claim_token,
        "attempt": claim.attempt,
    }
    relative = Path("shards/job.json")
    artifact_bytes = publish_json_exclusive(tmp_path / relative, payload)
    complete_job(
        queue,
        job_id=shard.job_id,
        worker_id="0-r0",
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=2,
        successes=1,
    )
    invocation_id = "a" * 32
    worker_ids = [f"{gpu}-r0" for gpu in range(8)]
    completion = {
        "schema_version": "ember_pi05_eval_launcher_completion_v1",
        "contract_reference": contract["contract_reference"],
        "invocation_id": invocation_id,
        "started_unix": 1.0,
        "finished_unix": 21.0,
        "wall_seconds": 20.0,
        "worker_ids": worker_ids,
        "return_codes": {worker_id: 0 for worker_id in worker_ids},
    }
    publish_json_exclusive(tmp_path / "launcher_completion.json", completion)
    for gpu, worker_id in enumerate(worker_ids):
        path = tmp_path / "workers" / f"{worker_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        events = (
            {
                "event": "process_started",
                "unix": 2.0,
                "worker_id": worker_id,
                "pid": 1000 + gpu,
                "invocation_id": invocation_id,
                "contract_reference": contract["contract_reference"],
            },
            {
                "event": "ready",
                "unix": 5.0,
                "worker_id": worker_id,
                "pid": 1000 + gpu,
                "invocation_id": invocation_id,
                "physical_gpu": gpu,
                "gpu_uuid": f"GPU-{gpu}",
                "gpu_name": "NVIDIA A40",
                "replica": 0,
                # Physical indices do not imply an even NUMA split; gpu01 is
                # a seven-device host whose GPU3 is attached to node 1.
                "numa_node": 0 if gpu < 3 else 1,
                "cpu_affinity": [gpu],
                "model_load_seconds": 3.0,
                "contract_reference": contract["contract_reference"],
            },
            {
                "event": "finished",
                "unix": 20.0,
                "worker_id": worker_id,
                "pid": 1000 + gpu,
                "invocation_id": invocation_id,
                "completed_shards": int(gpu == 0),
                "adopted_shards": 0,
                "contract_reference": contract["contract_reference"],
            },
        )
        path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
    result = aggregate_run(tmp_path)
    assert result["overall"]["evaluation_wall_seconds"] == 20.0
    assert result["overall"]["shard_execution_window_seconds"] == 2.0
    assert len(result["workers"]) == 8


def test_recovery_adopts_durable_orphan_from_previous_claim(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    queue = tmp_path / "queue.sqlite3"
    initialize_queue(queue, (shard,), contract_reference=contract["contract_reference"])
    old = claim_next(queue, worker_id="0-r0")
    assert old is not None
    payload = _payload(contract, shard)
    payload["producer"] = {
        "worker_id": "0-r0",
        "claim_token": old.claim_token,
        "attempt": old.attempt,
    }
    publish_json_exclusive(tmp_path / "shards/job.json", payload)

    initialize_queue(
        queue,
        (shard,),
        contract_reference=contract["contract_reference"],
        recover_claims=True,
    )
    current = claim_next(queue, worker_id="0-r0")
    assert current is not None and current.claim_token != old.claim_token
    adopted = _complete_published_shard(
        output_dir=tmp_path,
        queue_path=queue,
        claim=current,
        worker_id="0-r0",
        contract=contract,
    )
    assert adopted is not None and len(adopted) == 2


def test_aggregate_rejects_raw_file_size_change(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    queue = tmp_path / "queue.sqlite3"
    initialize_queue(queue, (shard,), contract_reference=contract["contract_reference"])
    claim = claim_next(queue, worker_id="0-r0")
    assert claim is not None
    relative = Path("shards/job.json")
    artifact_bytes = publish_json_exclusive(
        tmp_path / relative, _payload(contract, shard)
    )
    complete_job(
        queue,
        job_id=shard.job_id,
        worker_id="0-r0",
        claim_token=claim.claim_token,
        rows_path=relative.as_posix(),
        rows_bytes=artifact_bytes,
        row_count=2,
        successes=1,
    )
    (tmp_path / relative).write_text("{}\n", encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="size changed"):
        aggregate_run(tmp_path)
