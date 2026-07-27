from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import sqlite3

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_queue import (
    EvaluationTask,
    build_cost_balanced_shards,
    claim_next,
    complete_job,
    initialize_queue,
    publish_json_exclusive,
    queue_summary,
    read_json_with_sha256,
    validate_worker_layout,
)


def _tasks() -> tuple[EvaluationTask, ...]:
    return tuple(
        EvaluationTask(suite, task_id, horizon, tuple(range(50)))
        for task_id, (suite, horizon) in enumerate(
            (
                ("libero_spatial", 220),
                ("libero_object", 280),
                ("libero_goal", 300),
                ("libero_10", 520),
            )
        )
    )


def _drain_queue_process(path: str, worker: int, output) -> None:
    claimed = []
    queue_path = Path(path)
    while (claim := claim_next(queue_path, worker_id=f"{worker}-r0")) is not None:
        claimed.append(claim.shard.job_id)
    output.put(claimed)


def test_cost_balanced_shards_cover_every_state_once() -> None:
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    assert len(shards) == 26
    assert len({shard.job_id for shard in shards}) == len(shards)
    covered = {
        (shard.suite, shard.task_id, state_id)
        for shard in shards
        for state_id in shard.init_state_ids
    }
    assert len(covered) == 200
    long_shards = [shard for shard in shards if shard.suite == "libero_10"]
    assert {len(shard.init_state_ids) for shard in long_shards} == {6, 7}
    assert {shard.preferred_gpu for shard in long_shards} == set(range(8))
    assert all(
        shard.preferred_gpu is None for shard in shards if shard.suite != "libero_10"
    )
    ordinary_shards = [shard for shard in shards if shard.suite != "libero_10"]
    assert len(ordinary_shards) == 18
    assert max(len(shard.init_state_ids) for shard in ordinary_shards) == 9
    assert {
        shard.estimated_cost
        for shard in ordinary_shards
        if len(shard.init_state_ids) == 9
    } == {1980, 2520, 2700}


def test_max_horizon_states_follow_actual_gpu_count_before_dynamic_work(
    tmp_path: Path,
) -> None:
    tasks = (
        EvaluationTask("libero_goal", 0, 300, tuple(range(8))),
        EvaluationTask("libero_10", 1, 520, tuple(range(8))),
        EvaluationTask("libero_10", 2, 520, tuple(range(8))),
    )
    shards = build_cost_balanced_shards(
        tasks, env_batch_size=8, physical_gpu_count=4
    )
    long_shards = [shard for shard in shards if shard.horizon == 520]
    assert len(long_shards) == 8
    assert {shard.preferred_gpu for shard in long_shards} == set(range(4))
    assert all(len(shard.init_state_ids) == 2 for shard in long_shards)

    path = tmp_path / "queue.sqlite3"
    initialize_queue(path, shards, contract_sha256="8" * 64)
    for gpu in range(4):
        first = claim_next(path, worker_id=f"{gpu}-r0", physical_gpu=gpu)
        second = claim_next(path, worker_id=f"{gpu}-r1", physical_gpu=gpu)
        assert first is not None and first.shard.preferred_gpu == gpu
        assert second is not None and second.shard.preferred_gpu == gpu
    ordinary = claim_next(path, worker_id="0-r2", physical_gpu=0)
    assert ordinary is not None and ordinary.shard.preferred_gpu is None


def test_idle_gpu_steals_other_gpu_long_work_before_ordinary(
    tmp_path: Path,
) -> None:
    tasks = (
        EvaluationTask("libero_goal", 0, 300, tuple(range(8))),
        EvaluationTask("libero_10", 1, 520, tuple(range(8))),
        EvaluationTask("libero_10", 2, 520, tuple(range(8))),
    )
    shards = build_cost_balanced_shards(
        tasks, env_batch_size=8, physical_gpu_count=2
    )
    gpu_zero_long = sum(shard.preferred_gpu == 0 for shard in shards)
    assert gpu_zero_long > 0

    path = tmp_path / "queue.sqlite3"
    initialize_queue(path, shards, contract_sha256="9" * 64)
    for replica in range(gpu_zero_long):
        local = claim_next(
            path,
            worker_id=f"0-r{replica}",
            physical_gpu=0,
        )
        assert local is not None and local.shard.preferred_gpu == 0

    stolen = claim_next(
        path,
        worker_id=f"0-r{gpu_zero_long}",
        preferred_task=("libero_goal", 0),
        physical_gpu=0,
    )
    assert stolen is not None
    assert stolen.shard.horizon == 520
    assert stolen.shard.preferred_gpu == 1


def test_max_horizon_states_fill_replica_slots_before_dynamic_work(
    tmp_path: Path,
) -> None:
    tasks = (
        EvaluationTask("libero_goal", 0, 300, tuple(range(8))),
        EvaluationTask("libero_10", 1, 520, tuple(range(50))),
        EvaluationTask("libero_10", 2, 520, tuple(range(50))),
    )
    shards = build_cost_balanced_shards(
        tasks,
        env_batch_size=8,
        physical_gpu_count=4,
        replicas_per_gpu=6,
    )
    long_shards = [shard for shard in shards if shard.horizon == 520]
    assert len(long_shards) == 48
    assert {len(shard.init_state_ids) for shard in long_shards} == {2, 3}
    assert {
        gpu: sum(shard.preferred_gpu == gpu for shard in long_shards)
        for gpu in range(4)
    } == {gpu: 12 for gpu in range(4)}

    path = tmp_path / "queue.sqlite3"
    initialize_queue(path, shards, contract_sha256="7" * 64)
    for gpu in range(4):
        for replica in range(6):
            claim = claim_next(
                path,
                worker_id=f"{gpu}-r{replica}",
                physical_gpu=gpu,
            )
            assert claim is not None and claim.shard.horizon == 520
            assert claim.shard.preferred_gpu == gpu
    for gpu in range(4):
        for replica in range(6):
            claim = claim_next(
                path,
                worker_id=f"{gpu}-r{replica}",
                physical_gpu=gpu,
            )
            assert claim is not None and claim.shard.horizon == 520
            assert claim.shard.preferred_gpu == gpu
    ordinary = claim_next(path, worker_id="0-r0", physical_gpu=0)
    assert ordinary is not None and ordinary.shard.preferred_gpu is None


def test_four_gpu_six_replica_panel_keeps_two_ordinary_waves_after_long_work(
    tmp_path: Path,
) -> None:
    tasks = tuple(
        EvaluationTask(suite, task_id, horizon, tuple(range(50)))
        for task_id, (suite, horizon) in enumerate(
            (
                ("libero_spatial", 220),
                ("libero_spatial", 220),
                ("libero_object", 280),
                ("libero_object", 280),
                ("libero_goal", 300),
                ("libero_goal", 300),
                ("libero_10", 520),
                ("libero_10", 520),
            )
        )
    )
    shards = build_cost_balanced_shards(
        tasks,
        env_batch_size=8,
        physical_gpu_count=4,
        replicas_per_gpu=6,
    )
    long_shards = [shard for shard in shards if shard.horizon == 520]
    ordinary_shards = [shard for shard in shards if shard.horizon != 520]
    assert len(long_shards) == 48
    assert len(ordinary_shards) == 48
    assert max(len(shard.init_state_ids) for shard in ordinary_shards) == 7
    assert sum(len(shard.init_state_ids) for shard in ordinary_shards) == 300

    covered = {
        (shard.suite, shard.task_id, state_id)
        for shard in shards
        for state_id in shard.init_state_ids
    }
    assert len(covered) == 400

    path = tmp_path / "queue.sqlite3"
    initialize_queue(path, shards, contract_sha256="6" * 64)
    for gpu in range(4):
        for replica in range(6):
            for wave in range(2):
                claim = claim_next(
                    path,
                    worker_id=f"{gpu}-r{replica}-w{wave}",
                    physical_gpu=gpu,
                )
                assert claim is not None and claim.shard.horizon == 520
    for gpu in range(4):
        for replica in range(6):
            claim = claim_next(
                path,
                worker_id=f"{gpu}-r{replica}-ordinary",
                physical_gpu=gpu,
            )
            assert claim is not None and claim.shard.horizon != 520
    assert queue_summary(path)["status_counts"] == {"claimed": 72, "pending": 24}


def test_queue_claim_completion_and_contract_resume(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    initialize_queue(path, shards, contract_sha256="a" * 64)

    first = claim_next(path, worker_id="0-r0", physical_gpu=0)
    assert first is not None and first.shard.preferred_gpu == 0
    preferred = claim_next(
        path,
        worker_id="1-r0",
        preferred_task=("libero_10", 3),
        physical_gpu=1,
    )
    assert preferred is not None and preferred.shard.preferred_gpu == 1
    complete_job(
        path,
        job_id=first.shard.job_id,
        worker_id="0-r0",
        claim_token=first.claim_token,
        rows_path=f"shards/{first.shard.job_id}.json",
        rows_sha256="b" * 64,
        row_count=len(first.shard.init_state_ids),
        successes=2,
    )
    assert queue_summary(path)["completed_rows"] == len(first.shard.init_state_ids)
    initialize_queue(path, shards, contract_sha256="a" * 64, recover_claims=True)
    summary = queue_summary(path)
    assert summary["status_counts"] == {"complete": 1, "pending": len(shards) - 1}
    with pytest.raises(Pi05EvaluationError, match="another contract"):
        initialize_queue(path, shards, contract_sha256="c" * 64)


def test_concurrent_claims_are_unique(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    initialize_queue(path, shards, contract_sha256="d" * 64)

    def drain(worker: int) -> list[str]:
        claimed = []
        while (claim := claim_next(path, worker_id=f"{worker}-r0")) is not None:
            claimed.append(claim.shard.job_id)
        return claimed

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = [job for rows in executor.map(drain, range(8)) for job in rows]
    assert len(claims) == len(shards)
    assert len(set(claims)) == len(shards)
    assert queue_summary(path)["status_counts"] == {"claimed": len(shards)}


def test_multiprocess_claims_are_unique(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    initialize_queue(path, shards, contract_sha256="9" * 64)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(target=_drain_queue_process, args=(str(path), worker, output))
        for worker in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    claims = [job for _ in processes for job in output.get(timeout=5)]
    assert len(claims) == len(shards)
    assert len(set(claims)) == len(shards)


def test_recovered_claim_rejects_late_old_worker_and_wrong_row_count(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    initialize_queue(path, shards, contract_sha256="e" * 64)
    old = claim_next(path, worker_id="0-r0")
    assert old is not None
    initialize_queue(path, shards, contract_sha256="e" * 64, recover_claims=True)
    current = claim_next(path, worker_id="0-r0")
    assert current is not None and current.shard.job_id == old.shard.job_id
    assert current.attempt == old.attempt + 1
    assert current.claim_token != old.claim_token

    with pytest.raises(Pi05EvaluationError, match="does not own"):
        complete_job(
            path,
            job_id=old.shard.job_id,
            worker_id="0-r0",
            claim_token=old.claim_token,
            rows_path=f"shards/{old.shard.job_id}.json",
            rows_sha256="f" * 64,
            row_count=len(old.shard.init_state_ids),
            successes=0,
        )
    with pytest.raises(Pi05EvaluationError, match="row count differs"):
        complete_job(
            path,
            job_id=current.shard.job_id,
            worker_id="0-r0",
            claim_token=current.claim_token,
            rows_path=f"shards/{current.shard.job_id}.json",
            rows_sha256="f" * 64,
            row_count=len(current.shard.init_state_ids) - 1,
            successes=0,
        )


def test_resume_rejects_tampered_scheduler_columns(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    shards = build_cost_balanced_shards(_tasks(), env_batch_size=8)
    initialize_queue(path, shards, contract_sha256="1" * 64)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE jobs SET estimated_cost=estimated_cost+1 WHERE job_id=?",
            (shards[0].job_id,),
        )
    with pytest.raises(Pi05EvaluationError, match="identities changed"):
        initialize_queue(path, shards, contract_sha256="1" * 64)


def test_worker_layout_requires_symmetric_supported_replicas() -> None:
    for replicas in (1, 2, 3, 4, 5, 6):
        validate_worker_layout(
            (f"{gpu}-r{replica}" for gpu in range(8) for replica in range(replicas)),
            replicas,
        )
    with pytest.raises(Pi05EvaluationError, match="symmetric"):
        validate_worker_layout(("0-r0", "1-r0"), 1)
    validate_worker_layout(
        (f"{gpu}-r{replica}" for gpu in range(4) for replica in range(2)),
        2,
        physical_gpu_count=4,
    )
    validate_worker_layout(
        (f"{gpu}-r{replica}" for gpu in (2, 3) for replica in range(5)),
        5,
        physical_gpu_ids=(2, 3),
    )
    with pytest.raises(Pi05EvaluationError, match="profile"):
        validate_worker_layout(("0-r0",), 7)


def test_shard_json_publish_is_durable_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "shards" / "one.json"
    digest = publish_json_exclusive(path, {"rows": [1], "job_id": "one"})
    value, observed = read_json_with_sha256(path)
    assert observed == digest
    assert value == {"job_id": "one", "rows": [1]}
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        publish_json_exclusive(path, {"rows": [2], "job_id": "one"})
    assert read_json_with_sha256(path)[0]["rows"] == [1]
