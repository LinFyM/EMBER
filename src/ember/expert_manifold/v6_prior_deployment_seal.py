"""Artifact-backed deployment seal for the frozen-v6 Program residual."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ember.eval_adapters import validate_writer_episode
from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority
from ember.writer.errors import WriterModelError
from ember.writer.evaluation_cache import (
    validate_writer_cache_manifest,
    writer_cache_manifest_path,
    writer_cache_requests,
)


V6_PRIOR_DEPLOYMENT_SEAL_SCHEMA = (
    "ember_pi05_v6_condition_program_residual_deployment_seal_v1"
)
_RUN_SCHEMA = "ember_pi05_target_eval_launch_v2"
_RESULT_SCHEMA = "ember_pi05_target_eval_results_v2"
_PROFILE_SCHEMA = "ember_pi05_writer_generation_profile_v1"
_ADAPTER_SCHEMA = "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
_SEALED_DEPLOYMENT_CONFIG_SCHEMA = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v2"
)
_ARM = "expert_manifold_v6_condition_residual_correct"
_PANEL = "same_fixed_longest_first_request_panel_all_candidates"
_SELECTION = (
    "highest_measured_fixed_panel_loras_per_second_with_stable_" "longest_video_batch"
)
_ZERO_READS = {
    "teacher_action_reads": 0,
    "teacher_state_reads": 0,
    "reward_reads": 0,
    "terminal_reads": 0,
}
_RUNTIME_ARTIFACT_ROOT = Path("runs/outputs")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"artifact is not an object: {path}")
    return value


def _relative_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError:
        storage_root = (repo_root / _RUNTIME_ARTIFACT_ROOT).resolve()
        relative = _RUNTIME_ARTIFACT_ROOT / resolved.relative_to(storage_root)
    return {"path": relative.as_posix(), "bytes": resolved.stat().st_size}


def _resolve_evidence_path(repo_root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("deployment evidence path is not repo-relative")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        if relative.parts[:2] != _RUNTIME_ARTIFACT_ROOT.parts:
            raise
        resolved.relative_to((repo_root / _RUNTIME_ARTIFACT_ROOT).resolve())
    return resolved


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("deployment seal contract changed")


def _validate_measurement(
    row: Mapping[str, Any],
    *,
    batch_size: int,
    panel_size: int,
    measured_runs: int,
) -> tuple[bool, list[str], list[int]]:
    repeats = [float(value) for value in row["repeat_wall_seconds"]]
    sampled = [int(value) for value in row["sampled_frame_counts"]]
    entry_ids = [str(value) for value in row["entry_ids"]]
    forward_batches = [int(value) for value in row["forward_batch_sizes_per_repeat"]]
    expected_batches = [
        min(batch_size, panel_size - offset)
        for offset in range(0, panel_size, batch_size)
    ]
    wall = float(row["wall_seconds"])
    throughput = float(row["loras_per_second"])
    allocated = int(row["peak_allocated_bytes"])
    reserved = int(row["peak_reserved_bytes"])
    total = int(row["device_total_bytes"])
    headroom = int(row["memory_headroom_bytes"])
    required_headroom = int(row["required_memory_headroom_bytes"])
    observed = {
        "repeat_count": len(repeats),
        "generated_entries": int(row["generated_entries"]),
        "max_batch": int(row["max_observed_forward_batch_size"]),
        "forward_batches": forward_batches,
        "shared_panel": row.get("comparison_panel_shared_across_candidates"),
        "longest": row.get("longest_video_included"),
        "panel_entries": int(row["panel_entry_count"]),
        "entry_count": len(entry_ids),
        "unique_entries": len(set(entry_ids)),
        "sample_count": len(sampled),
        "sample_total": int(row["panel_total_sampled_frames"]),
        "sample_max": int(row["max_sampled_video_frames"]),
        "headroom": headroom,
    }
    expected = {
        "repeat_count": measured_runs,
        "generated_entries": panel_size * measured_runs,
        "max_batch": max(expected_batches),
        "forward_batches": expected_batches,
        "shared_panel": True,
        "longest": True,
        "panel_entries": panel_size,
        "entry_count": panel_size,
        "unique_entries": panel_size,
        "sample_count": panel_size,
        "sample_total": sum(sampled),
        "sample_max": max(sampled),
        "headroom": total - reserved,
    }
    _require(observed == expected)
    _require(all(value > 0 and math.isfinite(value) for value in repeats))
    _require(all(value > 0 for value in sampled))
    _require(wall > 0 and math.isfinite(wall))
    _require(math.isclose(wall, sum(repeats), rel_tol=1e-9, abs_tol=1e-6))
    _require(throughput > 0 and math.isfinite(throughput))
    _require(
        math.isclose(
            throughput,
            int(row["generated_entries"]) / wall,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    _require(allocated > 0 and reserved >= allocated)
    _require(total > reserved and required_headroom > 0)
    stable = max(repeats) / min(repeats) <= 1.25 and headroom >= required_headroom
    _require(bool(row.get("stable")) == stable)
    return stable, entry_ids, sampled


def _throughput_profile_matches(
    result: Mapping[str, Any], required_sizes: list[int]
) -> bool:
    try:
        sizes = [int(value) for value in result["profiled_writer_model_batch_sizes"]]
        selected = int(result["selected_writer_model_batch_size"])
        measurements = [
            dict(value) for value in result["writer_generation_measurements"]
        ]
        rows = {int(value["batch_size"]): value for value in measurements}
        measured_runs = int(result["measured_runs_per_batch"])
        _require(sizes == required_sizes == sorted(set(sizes)))
        _require(selected in sizes and set(rows) == set(sizes))
        _require(len(rows) == len(measurements))
        _require(measured_runs >= 2 and int(result["warmup_runs_per_batch"]) >= 1)
        _require(result.get("throughput_comparison_panel") == _PANEL)
        _require(result.get("selection_rule") == _SELECTION)
        stable_rows = []
        reference: tuple[list[str], list[int]] | None = None
        panel_size = max(sizes)
        for batch_size, row in rows.items():
            stable, entry_ids, sampled = _validate_measurement(
                row,
                batch_size=batch_size,
                panel_size=panel_size,
                measured_runs=measured_runs,
            )
            current = entry_ids, sampled
            reference = current if reference is None else reference
            _require(current == reference)
            if stable:
                stable_rows.append(row)
        _require(bool(stable_rows))
        best = max(
            stable_rows,
            key=lambda row: (float(row["loras_per_second"]), int(row["batch_size"])),
        )
        _require(selected == int(best["batch_size"]))
        return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _adapter_matches(config: Mapping[str, Any], adapter: Mapping[str, Any]) -> bool:
    wall = adapter.get("information_wall", {})
    writer = adapter.get("writer_asset", {})
    residual = writer.get("residual_state", {})
    return (
        adapter.get("schema_version") == _ADAPTER_SCHEMA
        and adapter.get("kind") == "expert_manifold_writer"
        and adapter.get("arm") == _ARM
        and adapter.get("video_condition") == "correct"
        and adapter.get("video_schedule", {}).get("sampling_mode")
        == "without_replacement"
        # The v3 change is training-only reconciliation state.  The deployed
        # Balanced-v2 feature, Program memory and frozen decoder are unchanged,
        # so the existing live deployment seal remains the direct graph proof.
        and adapter.get("config", {}).get("schema")
        in {config.get("schema_version"), _SEALED_DEPLOYMENT_CONFIG_SCHEMA}
        and writer.get("architecture") == config.get("writer", {}).get("architecture")
        and int(writer.get("program_residual_value_count", -1))
        == int(config.get("program_residual", {}).get("value_count", -2))
        and int(writer.get("deployment_trainable_parameter_count", -1)) == 0
        and int(writer.get("generated_lora_tensor_count", -1)) == 76
        and int(writer.get("method_macro", -1)) == 0
        and residual.get("kind") == "fresh_elementwise_zero"
        and int(residual.get("bytes", -1)) == 0
        and int(residual.get("tensor_count", -1)) == 0
        and wall.get("video_is_only_dynamic_value") is True
        and wall.get("language_only_lora_path") is False
        and wall.get("deployment_expert_bank_read") is False
        and all(wall.get(name) == value for name, value in _ZERO_READS.items())
    )


def _profile_matches(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    try:
        evaluation = config["evaluation"]
        required_sizes = [
            int(value) for value in evaluation["required_writer_model_batch_sizes"]
        ]
        git = run["git"]
        adapter = run["adapter"]
        parallel = run["parallel"]
        tasks = run["tasks"]
        physical = parallel["physical_gpu_ids"]
        preflight = result["preflight"]
        expected_ids = {request.entry_id for request in writer_cache_requests(run)}
        rows = result["writer_generation_measurements"]
        measured_ids = set(rows[0]["entry_ids"])
        observed_run = {
            "schema": run.get("schema_version"),
            "mode": run.get("mode"),
            "role": run.get("role"),
            "hash": run.get("content_hash_policy"),
            "git": git_state_is_clean_pushed_or_frozen_authority(git),
            "adapter": _adapter_matches(config, adapter),
            "task_count": len(tasks),
            "task_states": all(
                row.get("init_state_ids") == [0, 1, 2, 3] for row in tasks
            ),
            "request_count": len(expected_ids),
            "physical_count": int(parallel.get("physical_gpu_count", -1)),
            "replicas": int(parallel.get("replicas_per_gpu", -1)),
            "generators": int(parallel.get("writer_generators_per_gpu", -1)),
            "batch": int(parallel.get("writer_generation_batch_size", -1)),
            "physical_ids": len(physical),
        }
        expected_run = {
            "schema": _RUN_SCHEMA,
            "mode": "smoke",
            "role": "validation",
            "hash": "disabled_by_owner",
            "git": True,
            "adapter": True,
            "task_count": 8,
            "task_states": True,
            "request_count": max(required_sizes),
            "physical_count": 1,
            "replicas": 1,
            "generators": 1,
            "batch": max(required_sizes),
            "physical_ids": 1,
        }
        _require(observed_run == expected_run)
        observed_result = {
            "git": result.get("git"),
            "schema": result.get("schema_version"),
            "reference": result.get("contract_reference"),
            "root": result.get("root"),
            "device": result.get("device"),
            "hash": result.get("content_hash_policy"),
            "released": result.get("writer_modules_released"),
            "source_reused": result.get("source_policy_reused"),
            "panel_ids": measured_ids,
            "preflight_ids": preflight.get("physical_gpu_ids"),
            "preflight_apps": preflight.get("compute_applications"),
            "preflight_names": preflight.get("device_names"),
            "throughput": adapter.get("evaluation_authority", {}).get(
                "throughput_policy"
            ),
        }
        expected_result = {
            "git": git,
            "schema": _PROFILE_SCHEMA,
            "reference": run.get("contract_reference"),
            "root": run.get("output_dir"),
            "device": "NVIDIA A40",
            "hash": "disabled_by_owner",
            "released": True,
            "source_reused": True,
            "panel_ids": expected_ids,
            "preflight_ids": physical,
            "preflight_apps": [],
            "preflight_names": ["NVIDIA A40"],
            "throughput": evaluation.get("throughput_policy"),
        }
        _require(observed_result == expected_result)
        counters = {
            name: int(result.get(name, -1))
            for name in (*_ZERO_READS, "oom_count", "nonfinite_count")
        }
        _require(counters == dict.fromkeys(counters, 0))
        _require(_throughput_profile_matches(result, required_sizes))
        return True
    except (IndexError, KeyError, TypeError, ValueError, WriterModelError):
        return False


def _vertical_contract_matches(
    config: Mapping[str, Any],
    profile_run: Mapping[str, Any],
    run: Mapping[str, Any],
    selected_batch: int,
) -> bool:
    git = run.get("git", {})
    adapter = run.get("adapter", {})
    parallel = run.get("parallel", {})
    tasks = run.get("tasks", [])
    observed = {
        "schema": run.get("schema_version"),
        "mode": run.get("mode"),
        "role": run.get("role"),
        "hash": run.get("content_hash_policy"),
        "git_valid": git_state_is_clean_pushed_or_frozen_authority(git),
        "git_matched": git == profile_run.get("git"),
        "adapter_matched": adapter == profile_run.get("adapter"),
        "adapter_valid": _adapter_matches(config, adapter),
        "task_count": len(tasks),
        "state0": all(row.get("init_state_ids") == [0] for row in tasks),
        "task_keys": [(row.get("suite"), row.get("task_id")) for row in tasks],
        "physical_count": int(parallel.get("physical_gpu_count", -1)),
        "replicas": int(parallel.get("replicas_per_gpu", -1)),
        "generators": int(parallel.get("writer_generators_per_gpu", -1)),
        "batch": int(parallel.get("writer_generation_batch_size", -1)),
        "physical_ids": len(parallel.get("physical_gpu_ids", [])),
    }
    expected = {
        "schema": _RUN_SCHEMA,
        "mode": "smoke",
        "role": "validation",
        "hash": "disabled_by_owner",
        "git_valid": True,
        "git_matched": True,
        "adapter_matched": True,
        "adapter_valid": True,
        "task_count": 8,
        "state0": True,
        "task_keys": [
            (row.get("suite"), row.get("task_id"))
            for row in profile_run.get("tasks", [])
        ],
        "physical_count": 1,
        "replicas": 1,
        "generators": 1,
        "batch": selected_batch,
        "physical_ids": 1,
    }
    return observed == expected


def _vertical_rows_match(
    adapter: Mapping[str, Any],
    rows: object,
    expected_requests: tuple[Any, ...],
) -> bool:
    if not isinstance(rows, list) or len(rows) != len(expected_requests):
        return False
    try:
        expected_keys = {
            (request.suite, request.task_id, request.init_state_id)
            for request in expected_requests
        }
        observed_keys = {
            (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
            for row in rows
        }
        valid = [
            validate_writer_episode(
                adapter,
                row.get("writer"),
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                init_state_id=int(row["init_state_id"]),
            )
            and type(row.get("success")) is bool
            and int(row.get("steps", 0)) >= 1
            for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return observed_keys == expected_keys and all(valid)


def _vertical_lifecycle_matches(
    run: Mapping[str, Any],
    results: Mapping[str, Any],
    selected_batch: int,
) -> bool:
    adapter = run.get("adapter", {})
    parallel = run.get("parallel", {})
    workers = results.get("workers", [])
    attempts = results.get("launcher_attempts", {}).get("attempts", [])
    generation = results.get("writer_generation", {})
    launcher = results.get("launcher", {})
    physical = parallel.get("physical_gpu_ids", [])
    preflight = launcher.get("preflight", {})
    worker = workers[0] if len(workers) == 1 else {}
    observed = {
        "schema": results.get("schema_version"),
        "reference": results.get("contract_reference"),
        "adapter": results.get("adapter"),
        "episodes": int(results.get("overall", {}).get("episodes", -1)),
        "worker_count": len(workers),
        "gpu": worker.get("gpu_name"),
        "reloaded": worker.get("source_policy_reloaded"),
        "attempt_count": len(attempts),
        "attempt_event": attempts[0].get("event") if len(attempts) == 1 else None,
        "return_codes": launcher.get("return_codes"),
        "preflight_ids": preflight.get("physical_gpu_ids"),
        "preflight_apps": preflight.get("compute_applications"),
        "preflight_names": preflight.get("device_names"),
        "generator_workers": int(generation.get("generator_workers", -1)),
        "assigned": int(generation.get("assigned_entries", -1)),
        "generated": int(generation.get("generated_entries", -1)),
        "reused": int(generation.get("reused_entries", -1)),
        "batches": int(generation.get("generated_batches", -1)),
        "batch_sizes": generation.get("generation_batch_size"),
        "observed_batch": int(generation.get("max_observed_forward_batch_size", -1)),
        "redundant": int(generation.get("redundant_writer_forwards", -1)),
        "roundoff": generation.get("batch_shape_bf16_roundoff_accepted"),
        "source_reused": generation.get(
            "all_source_policy_processes_reused_for_rollout"
        ),
        "released": generation.get("all_writer_modules_released"),
        "not_reloaded": generation.get("all_source_policies_not_reloaded"),
        "gpu_names": generation.get("gpu_names"),
    }
    worker_id = str(worker.get("worker_id"))
    expected = {
        "schema": _RESULT_SCHEMA,
        "reference": run.get("contract_reference"),
        "adapter": adapter,
        "episodes": 8,
        "worker_count": 1,
        "gpu": "NVIDIA A40",
        "reloaded": False,
        "attempt_count": 1,
        "attempt_event": "completed",
        "return_codes": {worker_id: 0},
        "preflight_ids": physical,
        "preflight_apps": [],
        "preflight_names": ["NVIDIA A40"],
        "generator_workers": 1,
        "assigned": 8,
        "generated": 8,
        "reused": 0,
        "batches": 1,
        "batch_sizes": [selected_batch],
        "observed_batch": 8,
        "redundant": 0,
        "roundoff": True,
        "source_reused": True,
        "released": True,
        "not_reloaded": True,
        "gpu_names": ["NVIDIA A40"],
    }
    return observed == expected


def _vertical_cache_matches(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_requests: tuple[Any, ...],
) -> bool:
    adapter = run.get("adapter", {})
    descriptor = run.get("writer_lora_cache", {})
    storage = descriptor.get("lora_storage_per_entry", {})
    expected_storage = (
        adapter.get("writer_asset", {})
        .get("writer_state", {})
        .get("template_lora_storage", {})
    )
    observed = {
        "descriptor": manifest.get("descriptor"),
        "entries": manifest.get("entry_ids"),
        "storage": storage,
        "tensor_count": int(storage.get("tensor_count", -1)),
        "tensor_bytes": int(storage.get("tensor_bytes", -1)),
        "dtype_counts": storage.get("dtype_tensor_counts"),
    }
    expected = {
        "descriptor": descriptor,
        "entries": [request.entry_id for request in expected_requests],
        "storage": expected_storage,
        "tensor_count": 76,
        "tensor_bytes": 2_641_920,
        "dtype_counts": {"BF16": 72, "F32": 4},
    }
    return observed == expected


def _vertical_matches(
    config: Mapping[str, Any],
    profile_run: Mapping[str, Any],
    run: Mapping[str, Any],
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected_batch: int,
) -> bool:
    try:
        expected_requests = writer_cache_requests(run)
        checks = (
            _vertical_contract_matches(config, profile_run, run, selected_batch),
            _vertical_rows_match(
                run.get("adapter", {}),
                results.get("rows"),
                expected_requests,
            ),
            _vertical_lifecycle_matches(run, results, selected_batch),
            _vertical_cache_matches(run, manifest, expected_requests),
        )
    except (KeyError, TypeError, ValueError, WriterModelError):
        return False
    return all(checks)


def assemble_v6_prior_evaluation_smoke_evidence(
    *,
    config: Mapping[str, Any],
    profile_root: Path,
    vertical_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Derive one seal from a matched throughput profile and vertical smoke."""

    profile_root = profile_root.resolve()
    vertical_root = vertical_root.resolve()
    profile_path = profile_root / "writer_generation_profile.json"
    vertical_path = vertical_root / "results.json"
    profile_run = _read_json(profile_root / "run_contract.json")
    profile = _read_json(profile_path)
    vertical_run = _read_json(vertical_root / "run_contract.json")
    results = _read_json(vertical_path)
    manifest = validate_writer_cache_manifest(vertical_run, verify_entry_files=False)
    manifest_path = writer_cache_manifest_path(vertical_run)
    manifest_path.resolve().relative_to(vertical_root)
    selected = int(profile.get("selected_writer_model_batch_size", -1))
    if not _profile_matches(config, profile_run, profile) or not _vertical_matches(
        config,
        profile_run,
        vertical_run,
        results,
        manifest,
        selected,
    ):
        raise ExpertManifoldError("residual deployment evidence is incomplete")
    commit = str(profile_run.get("git", {}).get("commit", ""))
    if not commit:
        raise ExpertManifoldError("residual deployment commit is missing")
    return {
        "schema": V6_PRIOR_DEPLOYMENT_SEAL_SCHEMA,
        "run_commit": commit,
        "writer_model_batch_size": selected,
        "profile": _relative_record(profile_path, repo_root),
        "vertical": _relative_record(vertical_path, repo_root),
        "cache_manifest": _relative_record(manifest_path, repo_root),
    }


def evaluation_artifact_matches(
    *,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    repo_root: Path,
    commit_in_active_lineage: Callable[[str], bool],
) -> bool:
    required = {
        "schema",
        "run_commit",
        "writer_model_batch_size",
        "profile",
        "vertical",
        "cache_manifest",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        return False
    try:
        records = [evidence[name] for name in ("profile", "vertical", "cache_manifest")]
        if any(
            not isinstance(record, Mapping) or set(record) != {"path", "bytes"}
            for record in records
        ):
            return False
        paths = [
            _resolve_evidence_path(repo_root, record["path"]) for record in records
        ]
        paths[2].relative_to(paths[1].parent)
        if any(
            path.stat().st_size != int(record["bytes"])
            for path, record in zip(paths, records, strict=True)
        ):
            return False
        if (
            paths[0].name != "writer_generation_profile.json"
            or paths[1].name != "results.json"
            or paths[2].name != "cache_manifest.json"
        ):
            return False
        assembled = assemble_v6_prior_evaluation_smoke_evidence(
            config=config,
            profile_root=paths[0].parent,
            vertical_root=paths[1].parent,
            repo_root=repo_root,
        )
        commit = str(evidence["run_commit"])
        return (
            evidence.get("schema") == V6_PRIOR_DEPLOYMENT_SEAL_SCHEMA
            and assembled == dict(evidence)
            and commit_in_active_lineage(commit)
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ExpertManifoldError,
        WriterModelError,
    ):
        return False
