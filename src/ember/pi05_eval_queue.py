"""Cost-balanced, coordinator-free work queue for canonical PI05 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import RUNTIME_REPLICA_PROFILES


QUEUE_SCHEMA = "ember_pi05_eval_queue_v3"


@dataclass(frozen=True)
class EvaluationTask:
    suite: str
    task_id: int
    horizon: int
    init_state_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationShard:
    job_id: str
    ordinal: int
    suite: str
    task_id: int
    horizon: int
    init_state_ids: tuple[int, ...]
    estimated_cost: int
    preferred_gpu: int | None = None

    @property
    def task_key(self) -> tuple[str, int]:
        return self.suite, self.task_id


@dataclass(frozen=True)
class EvaluationClaim:
    shard: EvaluationShard
    attempt: int
    claim_token: str

    @property
    def task_key(self) -> tuple[str, int]:
        return self.shard.task_key


def _job_id(suite: str, task_id: int, state_ids: Sequence[int]) -> str:
    encoded = json.dumps(
        [suite, task_id, list(state_ids)], separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _task_chunks(
    task: EvaluationTask, *, env_batch_size: int, target_cost: int
) -> list[tuple[int, ...]]:
    state_ids = tuple(int(value) for value in task.init_state_ids)
    invalid = (
        task.horizon <= 0
        or not state_ids
        or len(set(state_ids)) != len(state_ids)
        or min(state_ids) < 0
    )
    if invalid:
        raise Pi05EvaluationError(f"invalid evaluation task: {task.suite}/{task.task_id}")
    waves = max(1, round(target_cost / (task.horizon * env_batch_size)))
    shard_size = waves * env_batch_size
    return [
        state_ids[start : start + shard_size]
        for start in range(0, len(state_ids), shard_size)
    ]


def _validate_shard_coverage(
    tasks: Sequence[EvaluationTask], shards: Sequence[EvaluationShard]
) -> None:
    covered = {
        (shard.suite, shard.task_id, state_id)
        for shard in shards
        for state_id in shard.init_state_ids
    }
    expected = {
        (task.suite, task.task_id, state_id)
        for task in tasks
        for state_id in task.init_state_ids
    }
    if covered != expected or sum(len(value.init_state_ids) for value in shards) != len(expected):
        raise Pi05EvaluationError("evaluation shards do not cover fixed states exactly once")


def build_cost_balanced_shards(
    tasks: Sequence[EvaluationTask],
    *,
    env_batch_size: int,
    target_cost: int | None = None,
    physical_gpu_count: int = 8,
    replicas_per_gpu: int = 1,
) -> tuple[EvaluationShard, ...]:
    """Pre-balance max-horizon states across worker slots, then shard other work."""

    if (
        not tasks
        or env_batch_size <= 0
        or physical_gpu_count <= 0
        or replicas_per_gpu <= 0
    ):
        raise Pi05EvaluationError("evaluation sharding needs tasks and a positive env batch")
    task_keys = [(task.suite, task.task_id) for task in tasks]
    if len(set(task_keys)) != len(task_keys):
        raise Pi05EvaluationError("evaluation sharding received duplicate tasks")
    if target_cost is None:
        target_cost = max(task.horizon for task in tasks) * env_batch_size
    if target_cost <= 0:
        raise Pi05EvaluationError("evaluation shard target cost must be positive")

    shards: list[EvaluationShard] = []
    ordinal = 0

    def append_shard(
        task: EvaluationTask,
        state_ids: tuple[int, ...],
        *,
        preferred_gpu: int | None,
    ) -> None:
        nonlocal ordinal
        shards.append(
            EvaluationShard(
                job_id=_job_id(task.suite, task.task_id, state_ids),
                ordinal=ordinal,
                suite=task.suite,
                task_id=task.task_id,
                horizon=task.horizon,
                init_state_ids=state_ids,
                estimated_cost=task.horizon * len(state_ids),
                preferred_gpu=preferred_gpu,
            )
        )
        ordinal += 1

    priority_horizon = max(task.horizon for task in tasks)
    priority_tasks = tuple(task for task in tasks if task.horizon == priority_horizon)
    ordinary_tasks = tuple(task for task in tasks if task.horizon != priority_horizon)
    worker_slot_count = physical_gpu_count * replicas_per_gpu
    for gpu in range(physical_gpu_count):
        for replica in range(replicas_per_gpu):
            slot = gpu * replicas_per_gpu + replica
            for task in priority_tasks:
                state_ids = tuple(task.init_state_ids[slot::worker_slot_count])
                if state_ids:
                    append_shard(task, state_ids, preferred_gpu=gpu)

    by_task = [
        _task_chunks(task, env_batch_size=env_batch_size, target_cost=target_cost)
        for task in ordinary_tasks
    ]
    for shard_index in range(max((len(value) for value in by_task), default=0)):
        for task, task_shards in zip(ordinary_tasks, by_task, strict=True):
            if shard_index >= len(task_shards):
                continue
            state_ids = task_shards[shard_index]
            append_shard(task, state_ids, preferred_gpu=None)
    _validate_shard_coverage(tasks, shards)
    return tuple(shards)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _shard_payload(shard: EvaluationShard) -> str:
    return json.dumps(asdict(shard), sort_keys=True, separators=(",", ":"))


def publish_json_exclusive(path: Path, value: Any) -> str:
    """Durably publish immutable JSON without replacing an earlier attempt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Pi05EvaluationError(
                f"immutable evaluation shard already exists: {path}"
            ) from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def read_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(f"invalid evaluation shard JSON: {path}") from error
    if not isinstance(value, dict):
        raise Pi05EvaluationError(f"evaluation shard is not a JSON object: {path}")
    return value, hashlib.sha256(encoded).hexdigest()


def initialize_queue(
    path: Path,
    shards: Sequence[EvaluationShard],
    *,
    contract_sha256: str,
    recover_claims: bool = False,
    retry_failed: bool = False,
) -> None:
    """Create or validate a queue; optionally recover claims after all workers exit."""

    if len(contract_sha256) != 64 or not shards:
        raise Pi05EvaluationError("invalid evaluation queue contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        shard.job_id: (
            shard.ordinal,
            shard.suite,
            shard.task_id,
            shard.estimated_cost,
            shard.preferred_gpu,
            _shard_payload(shard),
        )
        for shard in shards
    }
    if len(expected) != len(shards):
        raise Pi05EvaluationError("evaluation shards contain duplicate job IDs")
    with closing(_connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE,
                    suite TEXT NOT NULL, task_id INTEGER NOT NULL,
                    estimated_cost INTEGER NOT NULL, preferred_gpu INTEGER,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL, worker_id TEXT, claimed_unix REAL,
                    claim_token TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                    finished_unix REAL, rows_path TEXT, rows_sha256 TEXT, row_count INTEGER,
                    successes INTEGER, error TEXT
                )"""
            )
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if not metadata:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (("schema_version", QUEUE_SCHEMA), ("contract_sha256", contract_sha256)),
                )
                connection.executemany(
                        """INSERT INTO jobs(
                        job_id, ordinal, suite, task_id, estimated_cost,
                        preferred_gpu, payload, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (
                        (
                            shard.job_id,
                            shard.ordinal,
                            shard.suite,
                            shard.task_id,
                            shard.estimated_cost,
                            shard.preferred_gpu,
                            expected[shard.job_id][-1],
                        )
                        for shard in shards
                    ),
                )
            else:
                if metadata != {
                    "schema_version": QUEUE_SCHEMA,
                    "contract_sha256": contract_sha256,
                }:
                    raise Pi05EvaluationError("evaluation queue belongs to another contract")
                observed = {
                    row["job_id"]: (
                        int(row["ordinal"]),
                        row["suite"],
                        int(row["task_id"]),
                        int(row["estimated_cost"]),
                        int(row["preferred_gpu"])
                        if row["preferred_gpu"] is not None
                        else None,
                        row["payload"],
                    )
                    for row in connection.execute(
                        """SELECT job_id, ordinal, suite, task_id, estimated_cost,
                        preferred_gpu, payload FROM jobs"""
                    )
                }
                if observed != expected:
                    raise Pi05EvaluationError("evaluation queue shard identities changed")
                if recover_claims:
                    connection.execute(
                        """UPDATE jobs SET status='pending', worker_id=NULL, claimed_unix=NULL,
                        claim_token=NULL
                        WHERE status='claimed'"""
                    )
                if retry_failed:
                    connection.execute(
                        """UPDATE jobs SET status='pending', worker_id=NULL, claimed_unix=NULL,
                        claim_token=NULL, finished_unix=NULL, rows_path=NULL,
                        rows_sha256=NULL, row_count=NULL, successes=NULL, error=NULL
                        WHERE status='failed'"""
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _decode_shard(payload: str) -> EvaluationShard:
    value = json.loads(payload)
    value["init_state_ids"] = tuple(value["init_state_ids"])
    return EvaluationShard(**value)


def claim_next(
    path: Path,
    *,
    worker_id: str,
    preferred_task: tuple[str, int] | None = None,
    physical_gpu: int | None = None,
) -> EvaluationClaim | None:
    """Claim GPU-affine max-horizon work before ordinary dynamic work."""

    if not worker_id:
        raise Pi05EvaluationError("evaluation worker ID is empty")
    if physical_gpu is not None and physical_gpu < 0:
        raise Pi05EvaluationError("evaluation worker physical GPU is invalid")
    with closing(_connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if physical_gpu is None and preferred_task is None:
                row = connection.execute(
                    """SELECT job_id, payload FROM jobs WHERE status='pending'
                    ORDER BY estimated_cost DESC, ordinal ASC LIMIT 1"""
                ).fetchone()
            elif physical_gpu is None:
                row = connection.execute(
                    """SELECT job_id, payload FROM jobs WHERE status='pending'
                    ORDER BY estimated_cost DESC,
                    CASE WHEN suite=? AND task_id=? THEN 0 ELSE 1 END,
                    ordinal ASC LIMIT 1""",
                    preferred_task,
                ).fetchone()
            elif preferred_task is None:
                row = connection.execute(
                    """SELECT job_id, payload FROM jobs WHERE status='pending'
                    ORDER BY CASE
                        WHEN preferred_gpu=? THEN 0
                        WHEN preferred_gpu IS NULL THEN 1
                        ELSE 2 END,
                    estimated_cost DESC, ordinal ASC LIMIT 1""",
                    (physical_gpu,),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT job_id, payload FROM jobs WHERE status='pending'
                    ORDER BY CASE
                        WHEN preferred_gpu=? THEN 0
                        WHEN preferred_gpu IS NULL THEN 1
                        ELSE 2 END,
                    CASE WHEN suite=? AND task_id=? THEN 0 ELSE 1 END,
                    estimated_cost DESC, ordinal ASC LIMIT 1""",
                    (physical_gpu, *preferred_task),
                ).fetchone()
            if row is None:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            changed = connection.execute(
                """UPDATE jobs SET status='claimed', worker_id=?, claimed_unix=?,
                claim_token=?, attempt=attempt+1
                WHERE job_id=? AND status='pending'""",
                (worker_id, time.time(), claim_token, row["job_id"]),
            ).rowcount
            if changed != 1:
                raise Pi05EvaluationError("evaluation queue claim lost atomic ownership")
            attempt = int(
                connection.execute(
                    "SELECT attempt FROM jobs WHERE job_id=?", (row["job_id"],)
                ).fetchone()["attempt"]
            )
            connection.commit()
            return EvaluationClaim(
                shard=_decode_shard(row["payload"]),
                attempt=attempt,
                claim_token=claim_token,
            )
        except Exception:
            connection.rollback()
            raise


def complete_job(
    path: Path,
    *,
    job_id: str,
    worker_id: str,
    claim_token: str,
    rows_path: str,
    rows_sha256: str,
    row_count: int,
    successes: int,
) -> None:
    """Commit a shard only after its immutable raw-row file is published."""

    if (
        len(claim_token) != 32
        or not rows_path.startswith("shards/")
        or Path(rows_path).is_absolute()
        or ".." in Path(rows_path).parts
        or len(rows_sha256) != 64
        or row_count <= 0
        or not 0 <= successes <= row_count
    ):
        raise Pi05EvaluationError("invalid completed evaluation shard summary")
    with closing(_connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        owner = connection.execute(
            """SELECT payload FROM jobs WHERE job_id=? AND status='claimed'
            AND worker_id=? AND claim_token=?""",
            (job_id, worker_id, claim_token),
        ).fetchone()
        if owner is None:
            connection.rollback()
            raise Pi05EvaluationError("evaluation worker does not own completed shard")
        expected_rows = len(_decode_shard(owner["payload"]).init_state_ids)
        if row_count != expected_rows:
            connection.rollback()
            raise Pi05EvaluationError(
                f"completed shard row count differs: expected={expected_rows} actual={row_count}"
            )
        changed = connection.execute(
            """UPDATE jobs SET status='complete', finished_unix=?, rows_path=?, rows_sha256=?,
            row_count=?, successes=? WHERE job_id=? AND status='claimed' AND worker_id=?
            AND claim_token=?""",
            (
                time.time(),
                rows_path,
                rows_sha256,
                row_count,
                successes,
                job_id,
                worker_id,
                claim_token,
            ),
        ).rowcount
        if changed != 1:
            connection.rollback()
            raise Pi05EvaluationError("evaluation worker does not own completed shard")
        connection.commit()


def fail_job(
    path: Path, *, job_id: str, worker_id: str, claim_token: str, error: str
) -> None:
    with closing(_connect(path)) as connection:
        changed = connection.execute(
            """UPDATE jobs SET status='failed', finished_unix=?, error=?
            WHERE job_id=? AND status='claimed' AND worker_id=? AND claim_token=?""",
            (time.time(), error, job_id, worker_id, claim_token),
        ).rowcount
        if changed != 1:
            raise Pi05EvaluationError("evaluation worker does not own failed shard")


def queue_summary(path: Path) -> dict[str, Any]:
    with closing(_connect(path)) as connection:
        counts = {
            row["status"]: int(row["count"])
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            )
        }
        totals = connection.execute(
            """SELECT COALESCE(SUM(row_count), 0) AS rows,
            COALESCE(SUM(successes), 0) AS successes FROM jobs WHERE status='complete'"""
        ).fetchone()
        return {
            "status_counts": counts,
            "completed_rows": int(totals["rows"]),
            "successes": int(totals["successes"]),
        }


def completed_jobs(path: Path) -> tuple[dict[str, Any], ...]:
    with closing(_connect(path)) as connection:
        return tuple(
            dict(row)
            for row in connection.execute(
                """SELECT job_id, ordinal, suite, task_id, payload, rows_sha256,
                rows_path, row_count, successes, worker_id, attempt
                FROM jobs WHERE status='complete'
                ORDER BY ordinal"""
            )
        )


def failed_jobs(path: Path) -> tuple[dict[str, Any], ...]:
    with closing(_connect(path)) as connection:
        return tuple(
            dict(row)
            for row in connection.execute(
                """SELECT job_id, suite, task_id, worker_id, attempt, error, finished_unix
                FROM jobs WHERE status='failed' ORDER BY ordinal"""
            )
        )


def validate_worker_layout(
    worker_ids: Iterable[str],
    replicas_per_gpu: int,
    physical_gpu_count: int = 8,
    *,
    physical_gpu_ids: Iterable[int] | None = None,
) -> None:
    """Require the same non-zero replica count on every physical GPU."""

    if physical_gpu_ids is None:
        gpu_ids = tuple(range(physical_gpu_count))
    else:
        gpu_ids = tuple(int(value) for value in physical_gpu_ids)
    if (
        replicas_per_gpu not in RUNTIME_REPLICA_PROFILES
        or not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise Pi05EvaluationError("PI05 evaluator worker profile is invalid")
    parsed = [tuple(value.rsplit("-r", 1)) for value in worker_ids]
    expected = {
        (str(gpu), str(replica))
        for gpu in gpu_ids
        for replica in range(replicas_per_gpu)
    }
    if set(parsed) != expected or len(parsed) != len(expected):
        raise Pi05EvaluationError("evaluation workers are not symmetric across physical GPUs")
