"""Audit source-base competence/headroom for result-blind confirmation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy, _make_condition_env
from ember.gate_zero_base_competence import (
    ArmParallelContext,
    _broadcast,
    _close_parallel,
    _gather,
)
from ember.gate_zero_checkpoint import (
    CHECKPOINT_MANIFEST,
    sha256_file,
    validate_source_base_checkpoint,
)
from ember.gate_zero_evidence import (
    GateZeroEvidenceError,
    deterministic_state_partition,
    load_gate_zero_evidence_spec,
    select_confirmation_tasks,
    validate_bound_authority,
)
from ember.gate_zero_task_local_rl.runtime import scoped_policy_execution_horizon
from ember.specification_probe import ResetAuditEnv, _run_upstream_eval


class GateZeroBaseDifficultyAuditError(RuntimeError):
    """Raised when the result-blind source difficulty audit changes authority."""


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroBaseDifficultyAuditError(f"difficulty audit changed: {label}")


def _safe_relative(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        raise GateZeroBaseDifficultyAuditError(f"invalid relative authority: {label}")
    return value


def load_difficulty_audit_spec(
    path: Path, evidence_path: Path, split_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise GateZeroBaseDifficultyAuditError("difficulty audit authority unreadable") from error
    evidence = load_gate_zero_evidence_spec(evidence_path, split_path)
    _require(spec.get("schema_version"), 1, "schema")
    _require(
        spec.get("status"),
        "predeclared_before_source_base_difficulty_outcomes",
        "status",
    )
    _require(spec.get("surface"), "libero90_source_only_base_difficulty_audit", "surface")
    authority = spec.get("authority", {})
    _require(authority.get("evidence_contract_sha256"), sha256_file(evidence_path), "evidence SHA")
    _require(authority.get("split_reseal_sha256"), sha256_file(split_path), "split SHA")
    _require(
        spec.get("task_ids"),
        evidence["confirmation_selection"]["candidate_task_ids"],
        "candidate tasks",
    )
    if not set(spec["task_ids"]) <= set(split["active_split"]["source"]):
        raise GateZeroBaseDifficultyAuditError("difficulty audit escaped source tasks")
    _require(spec.get("batch_size"), 8, "batch size")
    _require(spec.get("episodes_per_task"), 32, "episodes per task")
    if len(spec.get("policy_rng_seeds", [])) != 4 or len(
        set(spec["policy_rng_seeds"])
    ) != 4:
        raise GateZeroBaseDifficultyAuditError("policy RNG batches changed")
    if len(spec.get("evaluator_seed_starts", [])) != 4:
        raise GateZeroBaseDifficultyAuditError("evaluator seed batches changed")
    _require(spec.get("execution_horizon"), 16, "execution horizon")
    _require(spec.get("model_action_chunk_size"), 50, "model chunk")
    for surface in ("validation", "held", "locked"):
        _require(spec.get(f"{surface}_numeric_access"), False, f"{surface} access")
    partition = spec.get("partition", {})
    _require(partition.get("seed"), evidence["state_partition"]["seed"], "partition seed")
    _require(partition.get("role"), "train", "partition role")
    _require(partition.get("selected_state_count"), 32, "partition count")
    selection = spec.get("selection", {})
    for key in (
        "minimum_base_successes",
        "minimum_base_failures",
        "minimum_selected_tasks",
        "maximum_selected_tasks",
    ):
        _require(selection.get(key), evidence["confirmation_selection"][key], key)
    parallel = spec.get("parallel", {})
    _require(parallel.get("allowed_world_sizes"), [1, 2, 4], "world sizes")
    _require(parallel.get("assignment"), "task_index_mod_world_size", "assignment")
    _require(spec.get("resources", {}).get("maximum_concurrent_gpus"), 4, "GPU ceiling")
    for key in (
        "source_base_checkpoint_relative_path",
        "source_competence_result_relative_path",
    ):
        _safe_relative(authority.get(key), key)
    return spec, evidence, split


def assigned_tasks(spec: Mapping[str, Any], *, rank: int, world_size: int) -> list[int]:
    if world_size not in spec["parallel"]["allowed_world_sizes"] or not 0 <= rank < world_size:
        raise GateZeroBaseDifficultyAuditError("invalid difficulty-audit topology")
    return [
        task_id
        for index, task_id in enumerate(spec["task_ids"])
        if index % world_size == rank
    ]


def _initialize_audit_parallel(spec: Mapping[str, Any]) -> ArmParallelContext:
    import torch

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise GateZeroBaseDifficultyAuditError("invalid torchrun rank environment") from error
    assigned_tasks(spec, rank=rank, world_size=world_size)
    if not torch.cuda.is_available():
        raise GateZeroBaseDifficultyAuditError("difficulty audit requires CUDA")
    torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        initialized = True
    return ArmParallelContext(rank, local_rank, world_size, initialized)


def set_physical_init_state_ids(env: Any, indices: Sequence[int]) -> list[int]:
    """Set the actual pre-reset LIBERO state counters, then verify worker state."""

    expected = [int(value) for value in indices]
    if len(expected) != len(set(expected)) or any(not 0 <= value < 50 for value in expected):
        raise GateZeroBaseDifficultyAuditError("invalid physical init-state batch")
    env.set_attr("init_state_id", expected)
    observed = list(env.call("init_state_id"))
    if observed != expected:
        raise GateZeroBaseDifficultyAuditError("physical init-state assignment failed")
    return observed


def _validate_upstream_authority(
    spec: Mapping[str, Any], *, checkpoint: Path, competence_result: Path
) -> dict[str, Any]:
    authority = spec["authority"]
    _require(
        sha256_file(checkpoint / CHECKPOINT_MANIFEST),
        authority["source_base_checkpoint_manifest_sha256"],
        "checkpoint manifest SHA",
    )
    manifest = validate_source_base_checkpoint(checkpoint)
    _require(manifest["step"], authority["expected_checkpoint_step"], "checkpoint step")
    _require(manifest["checkpoint_role"], authority["expected_checkpoint_role"], "checkpoint role")
    _require(
        sha256_file(competence_result),
        authority["source_competence_result_sha256"],
        "competence result SHA",
    )
    try:
        competence = json.loads(competence_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroBaseDifficultyAuditError("competence result unreadable") from error
    _require(competence.get("status"), "source_competence_passed", "competence status")
    return {
        "checkpoint_manifest_sha256": authority["source_base_checkpoint_manifest_sha256"],
        "checkpoint_step": manifest["step"],
        "checkpoint_role": manifest["checkpoint_role"],
        "source_competence_result_sha256": authority["source_competence_result_sha256"],
    }


def _task_authority(task_id: int, indices: Sequence[int]) -> tuple[str, dict[str, Any]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite("libero_90")
    task = suite.get_task(task_id)
    states = np.asarray(get_task_init_states(suite, task_id))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return task.language, {
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "bddl_filename": task.bddl_file,
        "bddl_sha256": sha256_file(bddl),
        "init_state_filename": task.init_states_file,
        "init_state_file_sha256": sha256_file(init_file),
        "physical_init_state_indices": list(indices),
        "physical_init_state_sha256": [
            hashlib.sha256(np.ascontiguousarray(states[index]).tobytes()).hexdigest()
            for index in indices
        ],
    }


def _relative_videos(output_dir: Path, paths: Sequence[str]) -> list[str]:
    result = []
    for raw in paths:
        try:
            result.append(Path(raw).resolve().relative_to(output_dir.resolve()).as_posix())
        except ValueError as error:
            raise GateZeroBaseDifficultyAuditError("video escaped output root") from error
    return result


def _evaluate_task(
    *,
    spec: Mapping[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    task_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    partition = deterministic_state_partition(task_id=task_id, seed=spec["partition"]["seed"])
    indices = partition["train"]
    language, authority = _task_authority(task_id, indices)
    rows: list[dict[str, Any]] = []
    videos: list[str] = []
    elapsed = 0.0
    for batch_index in range(4):
        state_ids = indices[batch_index * 8 : (batch_index + 1) * 8]
        env = ResetAuditEnv(
            _make_condition_env(
                {"task_suite": spec["task_suite"], "task_id": task_id},
                {"name": f"task{task_id}_batch{batch_index}", "batch_size": 8, "mode": "async"},
            )
        )
        try:
            set_physical_init_state_ids(env, state_ids)
            batch_spec = {
                "episodes_per_task": 8,
                "max_videos_per_arm": spec["max_videos_per_task"] if batch_index == 0 else 0,
                "policy_rng_seed": spec["policy_rng_seeds"][batch_index],
                "seed_start": spec["evaluator_seed_starts"][batch_index],
            }
            with scoped_policy_execution_horizon(
                runtime[0],
                execution_horizon=spec["execution_horizon"],
                expected_model_chunk_size=spec["model_action_chunk_size"],
            ):
                metrics, seconds = _run_upstream_eval(
                    spec=batch_spec,
                    runtime=runtime,
                    env=env,
                    videos_dir=output_dir / "videos" / f"task_{task_id}" / f"batch_{batch_index}",
                )
            elapsed += seconds
        finally:
            env.close()
        reset = env.reset_events
        if len(reset) != 1 or reset[0]["before"] != state_ids:
            raise GateZeroBaseDifficultyAuditError("physical reset identity changed")
        if reset[0]["after"] != [value + 8 for value in state_ids]:
            raise GateZeroBaseDifficultyAuditError("physical reset stride changed")
        expected_seeds = list(range(batch_spec["seed_start"], batch_spec["seed_start"] + 8))
        if reset[0]["seeds"] != expected_seeds or len(metrics["per_episode"]) != 8:
            raise GateZeroBaseDifficultyAuditError("evaluator seed/episode identity changed")
        videos.extend(_relative_videos(output_dir, metrics.get("video_paths", [])))
        for slot, episode in enumerate(metrics["per_episode"]):
            rows.append(
                {
                    "task_id": task_id,
                    "batch_index": batch_index,
                    "batch_slot": slot,
                    "physical_init_state_index": state_ids[slot],
                    "physical_init_state_sha256": authority["physical_init_state_sha256"][
                        batch_index * 8 + slot
                    ],
                    "policy_rng_seed": batch_spec["policy_rng_seed"],
                    "evaluator_seed": expected_seeds[slot],
                    "execution_horizon": spec["execution_horizon"],
                    "success": bool(episode["success"]),
                    "sum_reward": float(episode["sum_reward"]),
                    "max_reward": float(episode["max_reward"]),
                }
            )
    if len(rows) != spec["episodes_per_task"] or len(
        {row["physical_init_state_index"] for row in rows}
    ) != spec["episodes_per_task"]:
        raise GateZeroBaseDifficultyAuditError("difficulty audit rows repeat or omit state")
    return {
        "task_id": task_id,
        "language": language,
        "partition": partition,
        "task_authority": authority,
        "successes": sum(int(row["success"]) for row in rows),
        "episodes": len(rows),
        "episode_rows": rows,
        "video_paths": videos,
        "eval_seconds": elapsed,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_checksums(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and not path.name.startswith("gpu_telemetry_")
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files
    )
    (output_dir / "checksums.sha256").write_text(text, encoding="utf-8")


def _prepare_tracker(
    spec: Mapping[str, Any], context: ArmParallelContext, output_dir: Path
) -> Any:
    error, tracker = None, None
    if context.is_primary:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            unexpected = [
                path.name
                for path in output_dir.iterdir()
                if not (path.name.startswith("gpu_telemetry_") and path.suffix == ".csv")
            ]
            if unexpected:
                raise GateZeroBaseDifficultyAuditError(
                    f"refusing non-fresh audit output: {unexpected}"
                )
            import trackio

            trackio.init(
                project=spec["tracking"]["project"],
                name=output_dir.name,
                group=spec["tracking"]["group"],
                config={"world_size": context.world_size, "task_ids": spec["task_ids"]},
                auto_log_gpu=True,
                gpu_log_interval=1.0,
            )
            tracker = trackio
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
    error = _broadcast(context, error)
    if error:
        raise GateZeroBaseDifficultyAuditError(error)
    return tracker


def _evaluate_local_tasks(
    spec: Mapping[str, Any],
    context: ArmParallelContext,
    *,
    checkpoint: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    task_ids = assigned_tasks(spec, rank=context.rank, world_size=context.world_size)
    runtime = _load_policy(
        checkpoint / "pretrained_model",
        {"task_suite": spec["task_suite"], "task_id": task_ids[0]},
    )
    records = []
    for task_id in task_ids:
        record = _evaluate_task(
            spec=spec, runtime=runtime, task_id=task_id, output_dir=output_dir
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "source_base_difficulty_task",
                    "rank": context.rank,
                    "task_id": task_id,
                    "successes": record["successes"],
                    "episodes": record["episodes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return records


def _publish_result(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    split: Mapping[str, Any],
    upstream: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    world_size: int,
    wall_seconds: float,
    tracker: Any,
) -> dict[str, Any]:
    counts = {row["task_id"]: row["successes"] for row in tasks}
    try:
        selection = select_confirmation_tasks(evidence, split, counts)
        selection_status = "confirmation_tasks_selected_before_lora_outcomes"
    except GateZeroEvidenceError as error:
        candidate = evidence["confirmation_selection"]
        episodes = candidate["audit_rollouts_per_task"]
        eligible = sorted(
            task_id
            for task_id, successes in counts.items()
            if successes >= candidate["minimum_base_successes"]
            and episodes - successes >= candidate["minimum_base_failures"]
        )
        selection = {
            "outcome_authority": "frozen_base_only",
            "base_success_counts": {str(key): value for key, value in sorted(counts.items())},
            "eligible_task_ids": eligible,
            "selected_task_ids": [],
            "bounded_recovery_required": True,
            "diagnosis": str(error),
        }
        selection_status = "insufficient_base_headroom_candidates"
    selection_manifest = {
        "schema_version": 1,
        "status": selection_status,
        **selection,
        "physical_state_partitions": {
            str(row["task_id"]): row["partition"]
            for row in tasks
            if row["task_id"] in selection["selected_task_ids"]
        },
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "locked_numeric_access": False,
    }
    selection_text = json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    result = {
        "schema_version": 1,
        "status": (
            "source_base_difficulty_audit_completed"
            if selection["selected_task_ids"]
            else "source_base_difficulty_audit_requires_bounded_recovery"
        ),
        "surface": spec["surface"],
        "config_sha256": sha256_file(args.config),
        "evidence_contract_sha256": sha256_file(args.evidence_contract),
        "split_reseal_sha256": sha256_file(args.split_reseal),
        "upstream": upstream,
        "tasks": tasks,
        "selection_manifest_sha256": hashlib.sha256(selection_text.encode()).hexdigest(),
        "parallel": {"world_size": world_size, "physical_gpus": args.physical_gpus},
        "total_environment_episodes": sum(row["episodes"] for row in tasks),
        "wall_seconds": wall_seconds,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }
    _atomic_json(args.output_dir / "confirmation_selection_manifest.json", selection_manifest)
    _atomic_json(args.output_dir / "difficulty_result.json", result)
    _atomic_json(
        args.output_dir / "eval_info.json",
        {
            "overall": {
                "surface": spec["surface"],
                "status": result["status"],
                "episodes": result["total_environment_episodes"],
            },
            "per_task": [
                {
                    "task_group": "libero_90:source_base_difficulty",
                    "task_id": row["task_id"],
                    "metrics": {
                        "successes": [value["success"] for value in row["episode_rows"]],
                        "video_paths": row["video_paths"],
                    },
                }
                for row in tasks
            ],
        },
    )
    build_eval_gallery(args.output_dir)
    _write_checksums(args.output_dir)
    if args.latest_link:
        update_latest_link(args.output_dir, args.latest_link)
    tracker.log(
        {
            **{f"difficulty/task_{row['task_id']}_successes": row["successes"] for row in tasks},
            "difficulty/selected_tasks": len(selection["selected_task_ids"]),
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec, evidence, split = load_difficulty_audit_spec(
        args.config, args.evidence_contract, args.split_reseal
    )
    validate_bound_authority(evidence, repo_root=args.repo_root, output_root=args.output_root)
    upstream = _validate_upstream_authority(
        spec, checkpoint=args.checkpoint, competence_result=args.competence_result
    )
    context = _initialize_audit_parallel(spec)
    tracker = None
    try:
        tracker = _prepare_tracker(spec, context, args.output_dir)
        started = time.perf_counter()
        local = _evaluate_local_tasks(
            spec,
            context,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
        )
        gathered = _gather(context, local)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        tasks = sorted(
            (row for rows in gathered or [] for row in rows),
            key=lambda row: row["task_id"],
        )
        result = _publish_result(
            args=args,
            spec=spec,
            evidence=evidence,
            split=split,
            upstream=upstream,
            tasks=tasks,
            world_size=context.world_size,
            wall_seconds=time.perf_counter() - started,
            tracker=tracker,
        )
        tracker.finish()
        tracker = None
        return result
    finally:
        if tracker is not None:
            tracker.finish()
        _close_parallel(context)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config evidence-contract split-reseal repo-root output-root checkpoint "
        "competence-result output-dir latest-link"
    ).split():
        parser.add_argument(f"--{name}", type=Path, required=name != "latest-link")
    parser.add_argument("--physical-gpus", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            args.output_dir / f"failure_packet_rank_{os.environ.get('RANK', '0')}.json",
            {
                "schema_version": 1,
                "status": "source_base_difficulty_audit_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "gate_zero_authorized": False,
                "writer_authorized": False,
            },
        )
        raise
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
