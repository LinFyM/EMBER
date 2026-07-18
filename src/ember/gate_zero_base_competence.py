"""Evaluate the frozen Gate 0 source-base competence prerequisite in arm parallel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy, _make_condition_env
from ember.gate_zero_checkpoint import (
    CHECKPOINT_MANIFEST,
    sha256_file,
    validate_source_base_checkpoint,
)
from ember.gate_zero_contract import load_gate_zero_contract
from ember.specification_probe import (
    ResetAuditEnv,
    _run_upstream_eval,
    apply_prompt_override,
    paired_gap_summary,
)


class GateZeroBaseCompetenceError(RuntimeError):
    """Raised when source competence evidence is outside the frozen contract."""


@dataclass(frozen=True)
class ArmParallelContext:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroBaseCompetenceError(f"{label} changed: {actual!r} != {expected!r}")


def _pair_map(spec: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pair in spec.get("hard_negative_pairs", []):
        left, right = pair.get("left"), pair.get("right")
        if not isinstance(left, int) or not isinstance(right, int) or left == right:
            raise GateZeroBaseCompetenceError("invalid hard-negative pair")
        if left in mapping or right in mapping:
            raise GateZeroBaseCompetenceError("hard-negative pair members overlap")
        mapping[left], mapping[right] = right, left
    return mapping


def validate_source_competence_spec(
    spec: dict[str, Any], gate_zero_path: Path, phase0_path: Path
) -> None:
    if spec.get("schema_version") != 1:
        raise GateZeroBaseCompetenceError("unsupported source competence schema")
    if spec.get("status") != "predeclared_before_source_policy_rollout_outcomes":
        raise GateZeroBaseCompetenceError("source competence was not predeclared")
    gate_zero = load_gate_zero_contract(gate_zero_path, phase0_path)
    phase0 = tomllib.loads(phase0_path.read_text(encoding="utf-8"))
    authority = spec.get("authority", {})
    _require_equal(
        authority.get("gate_zero_contract_sha256"),
        sha256_file(gate_zero_path),
        "Gate 0 contract SHA256",
    )
    _require_equal(
        authority.get("phase0_contract_sha256"),
        sha256_file(phase0_path),
        "Phase 0 contract SHA256",
    )
    _require_equal(spec.get("surface"), "resealed_libero90_source_tasks_3_4_only", "surface")
    _require_equal(spec.get("task_suite"), "libero_90", "task suite")
    _require_equal(spec.get("task_ids"), gate_zero["base_competence"]["task_ids"], "source tasks")
    if not set(spec["task_ids"]) <= set(phase0["splits"]["source"]):
        raise GateZeroBaseCompetenceError("competence task IDs must remain source tasks")
    _require_equal(spec.get("conditions"), ["correct", "swapped"], "diagnostic conditions")
    _require_equal(spec.get("batch_size"), gate_zero["base_competence"]["batch_size"], "batch size")
    _require_equal(spec.get("episodes_per_task"), spec["batch_size"], "episode count")
    _require_equal(spec.get("use_async_envs"), True, "async evaluator")
    _require_equal(
        spec.get("official_init_state_indices"),
        gate_zero["base_competence"]["official_init_state_indices"],
        "official init-state authority",
    )
    if spec["official_init_state_indices"] != list(range(spec["batch_size"], 2 * spec["batch_size"])):
        raise GateZeroBaseCompetenceError("official init-state IDs must match one upstream reset")
    mapping = _pair_map(spec)
    if set(mapping) != set(spec["task_ids"]) or any(mapping.get(value) != key for key, value in mapping.items()):
        raise GateZeroBaseCompetenceError("hard-negative map must cover the source tasks")
    for seed_name in ("seed_start", "policy_rng_seed", "bootstrap_seed"):
        if not isinstance(spec.get(seed_name), int) or spec[seed_name] < 0:
            raise GateZeroBaseCompetenceError(f"invalid {seed_name}")
    if spec.get("bootstrap_replicates", 0) < 1000:
        raise GateZeroBaseCompetenceError("bootstrap replicate count is too small")
    decision = spec.get("decision", {})
    _require_equal(
        decision.get("correct_prompt_minimum_successes_per_task"),
        gate_zero["base_competence"]["correct_prompt_minimum_successes_per_task"],
        "source competence threshold",
    )
    _require_equal(decision.get("bounded_recovery"), gate_zero["base_competence"]["bounded_recovery"], "bounded recovery")
    _require_equal(decision.get("recovery_max_steps"), gate_zero["base_competence"]["recovery_max_steps"], "recovery maximum")
    for forbidden in ("gate_minus_one_decision_authorized", "gate_zero_decision_authorized", "writer_authorized"):
        _require_equal(decision.get(forbidden), False, forbidden)
    evaluation = spec.get("evaluation_contract", {})
    for required in (
        "same_batch_across_arms",
        "same_mode_across_arms",
        "same_seed_init_mapping_across_arms",
        "same_policy_rng_seed_across_arms",
        "upstream_rollout_unchanged",
        "prompt_override_preserves_environment_goal",
    ):
        _require_equal(evaluation.get(required), True, required)
    _require_equal(evaluation.get("cross_batch_pooling"), False, "cross-batch pooling")
    parallel = spec.get("parallel", {})
    _require_equal(parallel.get("allowed_world_sizes"), [1, 2, 4], "parallel world sizes")
    _require_equal(parallel.get("preferred_world_size"), 4, "preferred arm-parallel world")
    _require_equal(parallel.get("assignment"), "canonical_arm_index_mod_world_size", "arm assignment")
    for required in ("one_policy_process_per_rank", "rank_zero_result_writer"):
        _require_equal(parallel.get(required), True, required)
    resources = spec.get("resources", {})
    _require_equal(resources.get("maximum_concurrent_gpus"), 4, "competence GPU ceiling")
    if resources.get("minimum_free_memory_mib", 0) < 10240:
        raise GateZeroBaseCompetenceError("competence memory headroom weakened")
    _require_equal(authority.get("expected_checkpoint_step"), gate_zero["base_fit"]["scientific_checkpoint_step"], "checkpoint step")
    _require_equal(authority.get("expected_checkpoint_role"), gate_zero["base_fit"]["checkpoint"]["scientific_policy_role"], "checkpoint role")
    relative = authority.get("source_base_output_relative_path")
    if not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise GateZeroBaseCompetenceError("source-base output authority must be relative")


def load_source_competence_spec(
    path: Path, gate_zero_path: Path, phase0_path: Path
) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroBaseCompetenceError("invalid source competence TOML") from error
    validate_source_competence_spec(spec, gate_zero_path, phase0_path)
    return spec


def _canonical_arms(spec: dict[str, Any]) -> list[tuple[int, str]]:
    return [(task_id, condition) for task_id in spec["task_ids"] for condition in spec["conditions"]]


def assigned_competence_arms(
    spec: dict[str, Any], *, rank: int, world_size: int
) -> list[tuple[int, str]]:
    if world_size not in spec["parallel"]["allowed_world_sizes"] or not 0 <= rank < world_size:
        raise GateZeroBaseCompetenceError("invalid arm-parallel rank topology")
    return [arm for index, arm in enumerate(_canonical_arms(spec)) if index % world_size == rank]


def resolve_competence_prompt(
    spec: dict[str, Any], task_id: int, condition: str, languages: dict[int, str]
) -> str:
    if task_id not in languages:
        raise GateZeroBaseCompetenceError(f"missing language for task {task_id}")
    if condition == "correct":
        return languages[task_id]
    if condition == "swapped":
        return languages[_pair_map(spec)[task_id]]
    raise GateZeroBaseCompetenceError(f"unknown competence condition: {condition}")


def _validate_arm_set(spec: dict[str, Any], arms: list[dict[str, Any]]) -> None:
    expected = set(_canonical_arms(spec))
    actual = {(arm.get("task_id"), arm.get("condition")) for arm in arms}
    if actual != expected or len(arms) != len(expected):
        raise GateZeroBaseCompetenceError("competence arm set is incomplete or duplicated")
    for arm in arms:
        successes = arm.get("successes")
        if not isinstance(successes, list) or len(successes) != spec["episodes_per_task"]:
            raise GateZeroBaseCompetenceError("competence arm episode count changed")
        if not all(isinstance(value, (bool, np.bool_)) for value in successes):
            raise GateZeroBaseCompetenceError("competence success values must be boolean")


def decide_source_competence(
    spec: dict[str, Any], arms: list[dict[str, Any]], *, mechanics_valid: bool
) -> dict[str, Any]:
    common = {
        "gate_minus_one_authorized": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "task_local_oracle_fit_authorized": False,
    }
    if not mechanics_valid:
        return {**common, "status": "stopped", "reason": "mechanics_identity_failure", "failure_class": "implementation"}
    _validate_arm_set(spec, arms)
    correct = {
        task_id: sum(
            next(
                arm["successes"]
                for arm in arms
                if arm["task_id"] == task_id and arm["condition"] == "correct"
            )
        )
        for task_id in spec["task_ids"]
    }
    swapped = {
        task_id: sum(
            next(
                arm["successes"]
                for arm in arms
                if arm["task_id"] == task_id and arm["condition"] == "swapped"
            )
        )
        for task_id in spec["task_ids"]
    }
    minimum = spec["decision"]["correct_prompt_minimum_successes_per_task"]
    if any(value < minimum for value in correct.values()):
        return {
            **common,
            "status": "source_competence_failed",
            "reason": "correct_prompt_below_frozen_task_minimum",
            "failure_class": spec["decision"]["failure_class"],
            "correct_successes": correct,
            "swapped_successes_diagnostic": swapped,
            "bounded_recovery": spec["decision"]["bounded_recovery"],
            "bounded_recovery_max_steps": spec["decision"]["recovery_max_steps"],
        }
    return {
        **common,
        "status": "source_competence_passed",
        "reason": "every_source_task_met_frozen_correct_prompt_minimum",
        "correct_successes": correct,
        "swapped_successes_diagnostic": swapped,
        "task_local_oracle_fit_authorized": True,
    }


def _initialize_parallel(spec: dict[str, Any]) -> ArmParallelContext:
    import torch

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise GateZeroBaseCompetenceError("invalid torchrun rank environment") from error
    assigned_competence_arms(spec, rank=rank, world_size=world_size)
    if not torch.cuda.is_available():
        raise GateZeroBaseCompetenceError("source competence requires CUDA")
    torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        initialized = True
    return ArmParallelContext(rank, local_rank, world_size, initialized)


def _close_parallel(context: ArmParallelContext) -> None:
    if context.initialized:
        import torch

        torch.distributed.destroy_process_group()


def _broadcast(context: ArmParallelContext, value: Any) -> Any:
    if context.world_size == 1:
        return value
    import torch

    payload = [value if context.is_primary else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _gather(context: ArmParallelContext, value: Any) -> list[Any] | None:
    if context.world_size == 1:
        return [value]
    import torch

    result = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(value, result, dst=0)
    return result


def _validate_source_base_result(
    spec: dict[str, Any], source_base_output: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    if source_base_output.name != PurePosixPath(spec["authority"]["source_base_output_relative_path"]).name:
        raise GateZeroBaseCompetenceError("source-base output directory name changed")
    result_path = source_base_output / "training_result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checksum = (source_base_output / "checksums.sha256").read_text(encoding="utf-8").split()
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroBaseCompetenceError("source-base completion result is missing") from error
    if checksum != [sha256_file(result_path), "training_result.json"]:
        raise GateZeroBaseCompetenceError("source-base result checksum changed")
    if result.get("status") != "source_base_fit_completed_pending_competence":
        raise GateZeroBaseCompetenceError("source-base fit has not completed")
    manifest = validate_source_base_checkpoint(checkpoint_dir)
    authority = spec["authority"]
    _require_equal(manifest.get("step"), authority["expected_checkpoint_step"], "checkpoint step")
    _require_equal(manifest.get("checkpoint_role"), authority["expected_checkpoint_role"], "checkpoint role")
    _require_equal(manifest.get("authorities", {}).get("gate_zero_contract_sha256"), authority["gate_zero_contract_sha256"], "checkpoint Gate 0 authority")
    if result.get("final_checkpoint", {}).get("manifest_sha256") != sha256_file(checkpoint_dir / CHECKPOINT_MANIFEST):
        raise GateZeroBaseCompetenceError("source-base result does not bind final checkpoint")
    if result.get("gate_zero_authorized") is not False or result.get("writer_authorized") is not False:
        raise GateZeroBaseCompetenceError("source-base result contains premature authorization")
    return {
        "result_sha256": sha256_file(result_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_dir / CHECKPOINT_MANIFEST),
        "checkpoint_schema_version": manifest["schema_version"],
        "checkpoint_step": manifest["step"],
        "checkpoint_role": manifest["checkpoint_role"],
    }


def _task_authorities(spec: dict[str, Any]) -> tuple[dict[int, str], list[dict[str, Any]]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite(spec["task_suite"])
    languages: dict[int, str] = {}
    rows = []
    for task_id in spec["task_ids"]:
        task = suite.get_task(task_id)
        languages[task_id] = task.language
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = np.asarray(get_task_init_states(suite, task_id))
        indices = spec["official_init_state_indices"]
        rows.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
                "bddl_filename": task.bddl_file,
                "bddl_sha256": sha256_file(bddl),
                "init_state_filename": task.init_states_file,
                "init_state_file_sha256": sha256_file(init_path),
                "init_state_indices": indices,
                "init_state_sha256": [
                    hashlib.sha256(np.ascontiguousarray(init_states[index]).tobytes()).hexdigest()
                    for index in indices
                ],
            }
        )
    return languages, rows


def _relative_videos(output_dir: Path, paths: Sequence[str]) -> list[str]:
    relative = []
    for raw in paths:
        resolved = Path(raw).resolve()
        try:
            relative.append(resolved.relative_to(output_dir.resolve()).as_posix())
        except ValueError as error:
            raise GateZeroBaseCompetenceError("evaluation video escaped output directory") from error
    return relative


def _evaluate_arm(
    *,
    spec: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    task_id: int,
    condition: str,
    prompt: str,
    output_dir: Path,
) -> dict[str, Any]:
    batch_size = spec["batch_size"]
    env = ResetAuditEnv(
        _make_condition_env(
            {**spec, "task_id": task_id},
            {"name": f"{task_id}_{condition}", "batch_size": batch_size, "mode": "async"},
        )
    )
    try:
        override = apply_prompt_override(env, prompt, batch_size=batch_size)
        metrics, elapsed = _run_upstream_eval(
            spec=spec,
            runtime=runtime,
            env=env,
            videos_dir=output_dir / "videos" / f"task_{task_id}" / condition,
        )
        final_init_ids = list(env.call("init_state_id"))
    finally:
        env.close()
    if len(env.reset_events) != 1:
        raise GateZeroBaseCompetenceError("upstream rollout must perform exactly one reset")
    reset = env.reset_events[0]
    expected_before = list(range(batch_size))
    expected_after = spec["official_init_state_indices"]
    expected_seeds = list(range(spec["seed_start"], spec["seed_start"] + batch_size))
    episodes = metrics["per_episode"]
    if len(episodes) != batch_size:
        raise GateZeroBaseCompetenceError("upstream evaluator returned the wrong episode count")
    return {
        "task_id": task_id,
        "condition": condition,
        "prompt": prompt,
        "seeds": expected_seeds,
        "official_rollout_init_state_indices": expected_after,
        "init_state_ids_before_reset": reset["before"],
        "init_state_ids_after_reset": reset["after"],
        "init_state_ids_after_rollout": final_init_ids,
        "prompt_override": override,
        "mechanics_valid": (
            override["mechanically_valid"]
            and reset["before"] == expected_before
            and reset["after"] == expected_after
            and reset["seeds"] == expected_seeds
        ),
        "successes": [bool(episode["success"]) for episode in episodes],
        "sum_rewards": [float(episode["sum_reward"]) for episode in episodes],
        "max_rewards": [float(episode["max_reward"]) for episode in episodes],
        "video_paths": _relative_videos(output_dir, metrics.get("video_paths", [])),
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
        and not path.name.startswith("failure_packet")
    )
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _eval_info(spec: dict[str, Any], arms: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": spec["surface"],
            "status": decision["status"],
            "episodes": sum(len(arm["successes"]) for arm in arms),
            "successes": sum(sum(arm["successes"]) for arm in arms),
            "gate_zero_authorized": False,
            "writer_authorized": False,
        },
        "per_task": [
            {
                "task_group": f"{spec['task_suite']}:{arm['condition']}",
                "task_id": arm["task_id"],
                "metrics": {
                    "successes": arm["successes"],
                    "sum_rewards": arm["sum_rewards"],
                    "max_rewards": arm["max_rewards"],
                    "video_paths": arm["video_paths"],
                },
            }
            for arm in arms
        ],
    }


def _prepare_competence_run(
    spec: dict[str, Any],
    context: ArmParallelContext,
    *,
    source_base_output: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[int, str], list[dict[str, Any]], Any]:
    primary_error = None
    checkpoint_evidence = None
    languages = None
    task_authorities = None
    tracker = None
    if context.is_primary:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            unexpected = [
                path.name
                for path in output_dir.iterdir()
                if not (
                    path.is_file()
                    and path.name.startswith("gpu_telemetry_")
                    and path.suffix == ".csv"
                )
            ]
            if unexpected:
                raise GateZeroBaseCompetenceError(f"refusing non-fresh output: {unexpected}")
            checkpoint_evidence = _validate_source_base_result(
                spec, source_base_output, checkpoint_dir
            )
            languages, task_authorities = _task_authorities(spec)
            import trackio

            trackio.init(
                project=spec["tracking"]["project"],
                name=output_dir.name,
                group=spec["tracking"]["group"],
                config={"world_size": context.world_size, "surface": spec["surface"]},
                auto_log_gpu=True,
                gpu_log_interval=1.0,
                auto_log_cpu=True,
                cpu_log_interval=1.0,
            )
            tracker = trackio
        except BaseException as error:
            primary_error = f"{type(error).__name__}: {error}"
    primary_error = _broadcast(context, primary_error)
    if primary_error is not None:
        raise GateZeroBaseCompetenceError(primary_error)
    return (
        _broadcast(context, checkpoint_evidence),
        _broadcast(context, languages),
        _broadcast(context, task_authorities),
        tracker,
    )


def _evaluate_local_competence_arms(
    spec: dict[str, Any],
    context: ArmParallelContext,
    checkpoint_dir: Path,
    output_dir: Path,
    languages: dict[int, str],
) -> list[dict[str, Any]]:
    runtime = _load_policy(
        checkpoint_dir / "pretrained_model", {**spec, "task_id": spec["task_ids"][0]}
    )
    local_arms = []
    for task_id, condition in assigned_competence_arms(
        spec, rank=context.rank, world_size=context.world_size
    ):
        arm = _evaluate_arm(
            spec=spec,
            runtime=runtime,
            task_id=task_id,
            condition=condition,
            prompt=resolve_competence_prompt(spec, task_id, condition, languages),
            output_dir=output_dir,
        )
        local_arms.append(arm)
        print(
            json.dumps(
                {
                    "event": "source_competence_arm",
                    "rank": context.rank,
                    "task_id": task_id,
                    "condition": condition,
                    "successes": sum(arm["successes"]),
                    "episodes": len(arm["successes"]),
                    "mechanics_valid": arm["mechanics_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return local_arms


def _ordered_gathered_arms(
    spec: dict[str, Any], gathered: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    arms = [arm for rows in gathered for arm in rows]
    order = {value: index for index, value in enumerate(_canonical_arms(spec))}
    arms.sort(key=lambda arm: order[(arm["task_id"], arm["condition"])])
    return arms


def _build_competence_result(
    spec: dict[str, Any],
    *,
    spec_path: Path,
    gate_zero_path: Path,
    phase0_path: Path,
    checkpoint_evidence: dict[str, Any],
    task_authorities: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    decision: dict[str, Any],
    comparison: dict[str, Any],
    context: ArmParallelContext,
    physical_gpus: str,
    wall_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": decision["status"],
        "surface": spec["surface"],
        "spec_filename": spec_path.name,
        "spec_sha256": sha256_file(spec_path),
        "gate_zero_contract_sha256": sha256_file(gate_zero_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "checkpoint": checkpoint_evidence,
        "task_authorities": task_authorities,
        "arms": arms,
        "correct_vs_swapped": comparison,
        "decision": decision,
        "interpretation": spec["interpretation"],
        "parallel": {
            "world_size": context.world_size,
            "assignment": spec["parallel"]["assignment"],
        },
        "resources": {"cuda_visible_physical_gpus": physical_gpus, **spec["resources"]},
        "wall_seconds": wall_seconds,
        "tracking": {
            "backend": "trackio",
            "project": spec["tracking"]["project"],
            "run": output_dir.name,
            "dashboard_command": "trackio show --project EMBER_gate0",
        },
    }


def _publish_competence_result(
    spec: dict[str, Any],
    result: dict[str, Any],
    arms: list[dict[str, Any]],
    decision: dict[str, Any],
    *,
    output_dir: Path,
    latest_link: Path | None,
    tracker: Any,
) -> None:
    _atomic_json(output_dir / "competence_result.json", result)
    _atomic_json(output_dir / "eval_info.json", _eval_info(spec, arms, decision))
    build_eval_gallery(output_dir)
    _write_checksums(output_dir)
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    tracker.log(
        {
            **{
                f"competence/task_{task_id}_correct_successes": decision.get(
                    "correct_successes", {}
                ).get(task_id, 0)
                for task_id in spec["task_ids"]
            },
            "competence/passed": int(decision["status"] == "source_competence_passed"),
        }
    )
    tracker.finish()


def run_source_competence(
    *,
    spec_path: Path,
    gate_zero_path: Path,
    phase0_path: Path,
    source_base_output: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    latest_link: Path | None,
    physical_gpus: str,
) -> dict[str, Any]:
    spec = load_source_competence_spec(spec_path, gate_zero_path, phase0_path)
    context = _initialize_parallel(spec)
    tracker = None
    try:
        checkpoint_evidence, languages, task_authorities, tracker = _prepare_competence_run(
            spec,
            context,
            source_base_output=source_base_output,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )
        started = time.perf_counter()
        local_arms = _evaluate_local_competence_arms(
            spec, context, checkpoint_dir, output_dir, languages
        )
        gathered = _gather(context, local_arms)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        arms = _ordered_gathered_arms(spec, gathered)
        mechanics_valid = all(arm["mechanics_valid"] for arm in arms)
        decision = decide_source_competence(spec, arms, mechanics_valid=mechanics_valid)
        comparison = paired_gap_summary(
            arms,
            left="correct",
            right="swapped",
            seed=spec["bootstrap_seed"],
            replicates=spec["bootstrap_replicates"],
        )
        comparison["diagnostic_only"] = True
        result = _build_competence_result(
            spec,
            spec_path=spec_path,
            gate_zero_path=gate_zero_path,
            phase0_path=phase0_path,
            checkpoint_evidence=checkpoint_evidence,
            task_authorities=task_authorities,
            arms=arms,
            decision=decision,
            comparison=comparison,
            context=context,
            physical_gpus=physical_gpus,
            wall_seconds=time.perf_counter() - started,
            output_dir=output_dir,
        )
        _publish_competence_result(
            spec,
            result,
            arms,
            decision,
            output_dir=output_dir,
            latest_link=latest_link,
            tracker=tracker,
        )
        tracker = None
        return result
    except BaseException:
        if tracker is not None:
            tracker.finish()
        raise
    finally:
        _close_parallel(context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-zero-contract", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--source-base-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path)
    parser.add_argument("--physical-gpus", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = run_source_competence(
            spec_path=args.config.resolve(),
            gate_zero_path=args.gate_zero_contract.resolve(),
            phase0_path=args.phase0_contract.resolve(),
            source_base_output=args.source_base_output.resolve(),
            checkpoint_dir=args.checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link,
            physical_gpus=args.physical_gpus,
        )
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        _atomic_json(
            args.output_dir / f"failure_packet_rank_{rank}.json",
            {
                "schema_version": 1,
                "status": "error",
                "rank": int(rank),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"status": result["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
