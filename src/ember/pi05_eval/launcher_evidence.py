"""Validate cumulative launcher evidence for resumable PI05 evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_queue import read_json_with_sha256


def _read_invocation_events(path: Path) -> list[dict[str, Any]]:
    try:
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError("launcher invocation lineage is invalid") from error
    if any(not isinstance(row, dict) for row in events):
        raise Pi05EvaluationError("launcher invocation lineage is invalid")
    return events


def _validate_start(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> None:
    if (
        row.get("contract_sha256") != contract["contract_sha256"]
        or tuple(row.get("worker_ids", ()))
        != tuple(launcher.get("worker_ids", ()))
    ):
        raise Pi05EvaluationError("launcher invocation start evidence changed")


def _close_attempt(
    opened: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    event = terminal.get("event")
    if (
        event not in {"failed", "completed"}
        or terminal.get("invocation_id") != opened.get("invocation_id")
    ):
        raise Pi05EvaluationError("launcher invocation IDs are not sequential")
    wall_seconds = float(terminal.get("wall_seconds", 0))
    if wall_seconds <= 0 or not math.isclose(
        float(terminal.get("unix", 0)) - float(opened.get("unix", 0)),
        wall_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise Pi05EvaluationError("launcher invocation timing changed")
    return {
        "event": event,
        "invocation_id": str(terminal["invocation_id"]),
        "started_unix": float(opened["unix"]),
        "finished_unix": float(terminal["unix"]),
        "wall_seconds": wall_seconds,
        "return_codes": terminal.get("return_codes"),
    }


def _parse_attempts(
    events: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    opened: Mapping[str, Any] | None = None
    for row in events:
        if row.get("event") in {"started", "resume_started"}:
            if opened is not None:
                raise Pi05EvaluationError("launcher invocation start evidence changed")
            _validate_start(row, contract, launcher)
            opened = row
        else:
            if opened is None:
                raise Pi05EvaluationError(
                    "launcher invocation terminal evidence changed"
                )
            attempts.append(_close_attempt(opened, row))
            opened = None
    if opened is not None:
        raise Pi05EvaluationError("launcher invocation completion lineage changed")
    return attempts


def _validate_final_attempt(
    attempts: Sequence[Mapping[str, Any]],
    launcher: Mapping[str, Any],
) -> None:
    if (
        not attempts
        or attempts[-1]["event"] != "completed"
        or attempts[-1]["invocation_id"] != launcher.get("invocation_id")
        or any(row["event"] == "completed" for row in attempts[:-1])
        or not math.isclose(
            float(attempts[-1]["started_unix"]),
            float(launcher.get("started_unix", 0)),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(attempts[-1]["finished_unix"]),
            float(launcher.get("finished_unix", 0)),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise Pi05EvaluationError("launcher invocation completion lineage changed")


def _load_failures(
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], Path]]:
    failures: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted((output_dir / "failures").glob("launcher_*.json")):
        failure, _ = read_json_with_sha256(path)
        invocation_id = str(failure.get("invocation_id", ""))
        if (
            failure.get("schema_version") != "ember_pi05_eval_launcher_failure_v1"
            or not invocation_id
            or invocation_id in failures
        ):
            raise Pi05EvaluationError("launcher failure lineage changed")
        failures[invocation_id] = failure, path
    expected = {row["invocation_id"] for row in attempts if row["event"] == "failed"}
    if set(failures) != expected:
        raise Pi05EvaluationError("launcher failure artifacts are incomplete")
    return failures


def _cumulative_attempt_evidence(
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    failures: Mapping[str, tuple[Mapping[str, Any], Path]],
    *,
    total_shards: int,
) -> tuple[list[dict[str, Any]], int]:
    completed_before_final = 0
    evidence = []
    for row in attempts:
        record = dict(row)
        if row["event"] == "failed":
            failure, failure_path = failures[row["invocation_id"]]
            completed = int(
                failure.get("queue", {}).get("status_counts", {}).get("complete", 0)
            )
            if (
                failure.get("return_codes") != row["return_codes"]
                or completed < completed_before_final
                or completed >= total_shards
            ):
                raise Pi05EvaluationError("launcher failure queue evidence changed")
            completed_before_final = completed
            record["completed_shards"] = completed
            record["failure_artifact"] = str(failure_path.relative_to(output_dir))
        evidence.append(record)
    return evidence, completed_before_final


def _smoke_fallback(
    launcher: Mapping[str, Any],
    *,
    total_shards: int,
) -> dict[str, Any]:
    wall_seconds = float(launcher["wall_seconds"])
    return {
        "active_wall_seconds": wall_seconds,
        "completed_before_final_attempt": 0,
        "attempts": [
            {
                "event": "completed",
                "invocation_id": launcher["invocation_id"],
                "wall_seconds": wall_seconds,
                "completed_shards": total_shards,
            }
        ],
    }


def launcher_attempt_summary(
    output_dir: Path,
    contract: Mapping[str, Any],
    launcher: Mapping[str, Any],
    workers: Sequence[Mapping[str, Any]],
    *,
    total_shards: int,
) -> dict[str, Any]:
    """Return exact active time and cumulative shard counts across resumes."""

    path = output_dir / "invocations.jsonl"
    if not path.is_file():
        if contract.get("mode") != "smoke" or (output_dir / "failures").exists():
            raise Pi05EvaluationError("launcher invocation lineage is missing")
        return _smoke_fallback(launcher, total_shards=total_shards)
    attempts = _parse_attempts(_read_invocation_events(path), contract, launcher)
    _validate_final_attempt(attempts, launcher)
    failures = _load_failures(output_dir, attempts)
    evidence, completed_before_final = _cumulative_attempt_evidence(
        output_dir,
        attempts,
        failures,
        total_shards=total_shards,
    )
    completed_in_final = sum(int(row["completed_shards"]) for row in workers)
    if completed_before_final + completed_in_final != total_shards:
        raise Pi05EvaluationError("launcher cumulative completed-shard evidence changed")
    evidence[-1]["completed_shards"] = completed_in_final
    return {
        "active_wall_seconds": sum(float(row["wall_seconds"]) for row in attempts),
        "completed_before_final_attempt": completed_before_final,
        "attempts": evidence,
    }
