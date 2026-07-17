"""Build the pinned, leakage-safe LIBERO-90 authority and data-quality report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ember.contracts import load_contract, validate_contract
from ember.libero_data import (
    REQUIRED_DATASETS,
    AuditResult,
    ManifestError,
    audit_demonstration_file,
    compute_normalization,
    load_hub_surface,
    sha256_file,
)
from ember.libero_report import render_report
SCENE_RE = re.compile(r"^([A-Z]+(?:_[A-Z]+)*_SCENE\d+)_")


def _sha256_array_row(row: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(row)
    identity = f"{contiguous.dtype.str}:{contiguous.shape}:".encode("ascii")
    return hashlib.sha256(identity + contiguous.tobytes()).hexdigest()


def _split_lookup(contract: dict[str, Any]) -> dict[int, str]:
    return {
        index: split
        for split in ("source", "validation", "held_out")
        for index in contract["splits"][split]
    }


def _bddl_authority(
    *, task: Any, split: str, bddl_path: Path, parser: Any, task_index: int
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": task.bddl_file,
        "sha256": sha256_file(bddl_path),
        "semantic_access_policy": (
            "identity_only_not_parsed" if split == "held_out" else "source_or_validation_semantics"
        ),
    }
    if split == "held_out":
        return record
    problem = parser(str(bddl_path))
    bddl_language = " ".join(problem["language_instruction"])
    record.update(
        {
            "problem_name": problem["problem_name"],
            "language": bddl_language,
            "language_matches_task_map": bddl_language == task.language,
            "fixture_categories": sorted(problem["fixtures"]),
            "fixture_instances": sorted(
                value for values in problem["fixtures"].values() for value in values
            ),
            "object_categories": sorted(problem["objects"]),
            "object_instances": sorted(
                value for values in problem["objects"].values() for value in values
            ),
            "initial_state": problem["initial_state"],
            "goal_state": problem["goal_state"],
            "objects_of_interest": problem["obj_of_interest"],
        }
    )
    return record


def _init_state_authority(*, suite: Any, index: int, task: Any, path: Path) -> dict[str, Any]:
    init_states = np.asarray(suite.get_task_init_states(index))
    if init_states.ndim != 2 or init_states.shape[0] != 50:
        raise ManifestError(f"official init-state count differs from 50: task {index}")
    return {
        "filename": task.init_states_file,
        "sha256": sha256_file(path),
        "count": int(init_states.shape[0]),
        "state_dimension": int(init_states.shape[1]),
        "dtype": str(init_states.dtype),
        "row_sha256": [_sha256_array_row(row) for row in init_states],
    }


def load_task_authority(
    *, contract: dict[str, Any], libero_config_root: Path
) -> list[dict[str, Any]]:
    """Load the pinned task map, BDDL semantics, and official evaluation init states."""

    config_path = libero_config_root / "config.yaml"
    try:
        runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid LIBERO runtime config: {config_path}") from error
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_root.resolve())
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    suite = get_benchmark_dict()["libero_90"]()
    if suite.n_tasks != contract["datasets"]["libero_90"]["task_count"]:
        raise ManifestError("installed LIBERO task map does not contain 90 tasks")
    split_lookup = _split_lookup(contract)
    records: list[dict[str, Any]] = []
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        scene_match = SCENE_RE.match(task.name)
        if scene_match is None:
            raise ManifestError(f"task name lacks a canonical scene prefix: {index}")
        bddl_path = Path(runtime_config["bddl_files"]) / task.problem_folder / task.bddl_file
        init_path = Path(runtime_config["init_states"]) / task.problem_folder / task.init_states_file
        if not bddl_path.is_file() or not init_path.is_file():
            raise ManifestError(f"missing BDDL or init-state authority: task {index}")
        split = split_lookup[index]
        records.append(
            {
                "task_index": index,
                "task_name": task.name,
                "language": task.language,
                "scene": scene_match.group(1),
                "split": split,
                "bddl": _bddl_authority(
                    task=task,
                    split=split,
                    bddl_path=bddl_path,
                    parser=robosuite_parse_problem,
                    task_index=index,
                ),
                "init_states": _init_state_authority(
                    suite=suite, index=index, task=task, path=init_path
                ),
            }
        )
    return records


def _factor_coverage(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "scenes": lambda task: [task["scene"]],
        "object_categories": lambda task: task["bddl"].get("object_categories", []),
        "fixture_categories": lambda task: task["bddl"].get("fixture_categories", []),
        "goal_predicates": lambda task: [goal[0] for goal in task["bddl"].get("goal_state", [])],
    }
    coverage: dict[str, Any] = {}
    for label, extractor in dimensions.items():
        held_out_evaluable = label == "scenes"
        values_by_split = {
            split: sorted(
                {value for task in tasks if task["split"] == split for value in extractor(task)}
            )
            for split in ("source", "validation", "held_out")
        }
        source_values = set(values_by_split["source"])
        coverage[label] = {
            "values_by_split": values_by_split,
            "validation_absent_from_source": sorted(
                set(values_by_split["validation"]) - source_values
            ),
            "held_out_absent_from_source": (
                sorted(set(values_by_split["held_out"]) - source_values)
                if held_out_evaluable
                else None
            ),
            "held_out_coverage_status": (
                "evaluated_from_task_name_scene"
                if held_out_evaluable
                else "not_evaluated_due_to_access_policy"
            ),
            "held_out_semantics_access": (
                "task-name scene only"
                if held_out_evaluable
                else "not parsed; no held label access"
            ),
        }
    return coverage


def _git_commit(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _aggregate_quality(
    tasks: list[dict[str, Any]], *, access_counts: dict[str, int], contract: dict[str, Any]
) -> dict[str, Any]:
    warning_tasks: dict[str, list[int]] = {}
    for task in tasks:
        for warning in task["quality"]["warnings"]:
            warning_tasks.setdefault(warning["code"], []).append(task["task_index"])
    interpretations = {
        "legacy_env_bddl_basename_mismatch": (
            "legacy producer provenance only; canonical HDF5 BDDL basename and task map match"
        ),
        "legacy_env_bddl_suite": (
            "legacy producer provenance only; canonical filename and pinned BDDL authority match"
        ),
        "bddl_task_map_language_mismatch": (
            "task-map and HDF5 agree; preserve the parsed BDDL wording as a distinct audited field"
        ),
    }
    issues = [
        {
            "severity": "note",
            "code": code,
            "task_count": len(indices),
            "sample_task_indices": indices[:5],
            "interpretation": interpretations.get(code, "documented non-blocking data note"),
        }
        for code, indices in sorted(warning_tasks.items())
    ]
    if contract["datasets"]["libero_90"].get("hub_card_license_conflict"):
        issues.append(
            {
                "severity": "note",
                "code": "hub_card_license_conflict",
                "interpretation": "the original LIBERO dataset release is the recorded license authority",
            }
        )
    return {
        "schema_version": 1,
        "status": "pass_with_documented_notes" if issues else "pass",
        "dimensions": {
            "completeness": "90/90 task files and 50/50 demonstrations per task",
            "uniqueness": "task names, task indices, and HDF5 basenames are one-to-one",
            "validity": "all HDF5 schemas, dtypes, frame dimensions, and aggregate counts validated",
            "integrity": "all local bytes and SHA256 identities match the pinned Hub LFS tree",
            "leakage": "numeric values read only from source/source_base_fit; validation and held-out are metadata-only",
        },
        "access_counts": access_counts,
        "issues": issues,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifacts(
    *,
    output_dir: Path,
    latest_link: Path | None,
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    quality_report: dict[str, Any],
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ManifestError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "normalization_source_only.json", normalization)
        _write_json(temporary / "quality_report.json", quality_report)
        (temporary / "index.html").write_text(
            render_report(manifest, quality_report), encoding="utf-8"
        )
        checksums = []
        for name in ("index.html", "manifest.json", "normalization_source_only.json", "quality_report.json"):
            checksums.append(f"{sha256_file(temporary / name)}  {name}")
        (temporary / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if latest_link is not None:
        latest_link = Path(os.path.abspath(latest_link))
        latest_link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(latest_link):
            if not latest_link.is_symlink():
                raise ManifestError(f"latest link path is not a symlink: {latest_link}")
            latest_link.unlink()
        latest_link.symlink_to(os.path.relpath(output_dir, latest_link.parent))


def _audit_jobs(
    *,
    authorities: list[dict[str, Any]],
    hub: dict[str, Any],
    dataset: dict[str, Any],
    dataset_root: Path,
    source_bounds: list[int],
) -> list[dict[str, Any]]:
    expected = {f"{task['task_name']}_demo.hdf5" for task in authorities}
    if set(hub["files"]) != expected:
        missing = sorted(expected - set(hub["files"]))
        extra = sorted(set(hub["files"]) - expected)
        raise ManifestError(
            f"task map and Hub filenames differ: missing={missing[:3]}, extra={extra[:3]}"
        )
    source_episodes = tuple(range(source_bounds[0], source_bounds[1] + 1))
    jobs = []
    for authority in authorities:
        filename = f"{authority['task_name']}_demo.hdf5"
        hub_entry = hub["files"][filename]
        jobs.append(
            {
                "path": dataset_root / filename,
                "task_index": authority["task_index"],
                "task_name": authority["task_name"],
                "split": authority["split"],
                "language": authority["language"],
                "bddl_basename": authority["bddl"]["filename"],
                "expected_tag": dataset["hdf5_tag"],
                "expected_demos": dataset["demos_per_task"],
                "expected_sha256": hub_entry["sha256"],
                "expected_bytes": hub_entry["bytes"],
                "normalization_episodes": (
                    source_episodes if authority["split"] == "source" else ()
                ),
            }
        )
    return jobs


def _run_audits(jobs: list[dict[str, Any]], workers: int) -> list[AuditResult]:
    results: list[AuditResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_demonstration_file, **job): job["task_index"] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"audited_task={result.record['task_index']} completed={completed}/90", flush=True)
    return sorted(results, key=lambda result: result.record["task_index"])


def _merge_task_records(
    authorities: list[dict[str, Any]], results: list[AuditResult]
) -> list[dict[str, Any]]:
    by_index = {result.record["task_index"]: result for result in results}
    tasks = []
    for authority in authorities:
        record = by_index[authority["task_index"]].record
        record.update({key: value for key, value in authority.items() if key not in record})
        if record["bddl"].get("language_matches_task_map") is False:
            record["quality"]["warnings"].append(
                {
                    "code": "bddl_task_map_language_mismatch",
                    "message": "task-map/HDF5 instruction and parsed BDDL instruction differ",
                }
            )
            record["quality"]["warning_count"] += 1
            record["quality"]["status"] = "pass_with_note"
        tasks.append(record)
    if len(tasks) != 90 or len({task["task_name"] for task in tasks}) != 90:
        raise ManifestError("task records are incomplete or not unique")
    return tasks


def _summary(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tasks": len(tasks),
        "source": sum(task["split"] == "source" for task in tasks),
        "validation": sum(task["split"] == "validation" for task in tasks),
        "held_out": sum(task["split"] == "held_out" for task in tasks),
        "demonstrations": sum(task["demonstrations"]["count"] for task in tasks),
        "frames": sum(task["demonstrations"]["steps"] for task in tasks),
        "hdf5_bytes": sum(task["hdf5"]["bytes"] for task in tasks),
    }


def _manifest_document(
    *,
    workspace: Path,
    contract: dict[str, Any],
    dataset: dict[str, Any],
    hub: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    upstreams = contract["upstreams"]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generation_git_commit": _git_commit(workspace),
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "subdir": dataset["subdir"],
            "license": dataset["license"],
            "hub_lfs_file_count": hub["file_count"],
            "hub_lfs_total_bytes": hub["total_bytes"],
        },
        "upstream_authority": {
            "libero_official_commit": upstreams["libero_official"]["commit"],
            "libero_runtime_commit": upstreams["libero_runtime"]["commit"],
            "task_map_git_blob": upstreams["libero_task_map"]["git_blob_sha"],
            "bddl_tree_sha": upstreams["libero_official"]["bddl_tree_sha"],
            "init_states_tree_sha": upstreams["libero_official"]["init_states_tree_sha"],
        },
        "access_contract": {
            "source_numeric_episode_pool": "source_base_fit",
            "validation_hdf5_access": "metadata_only",
            "held_out_hdf5_access": "metadata_only",
            "held_out_bddl_access": "identity_hash_only_not_semantically_parsed",
            "demo_model_xml_access": "forbidden_and_not_serialized",
        },
        "dataset_schema": {
            path: {
                "dtype": dtype,
                "tail_shape": list(tail) if tail is not None else "task_dependent",
            }
            for path, (dtype, tail) in REQUIRED_DATASETS.items()
        },
        "summary": _summary(tasks),
        "factor_coverage": _factor_coverage(tasks),
        "tasks": tasks,
    }


def build_manifest(
    *,
    workspace: Path,
    contract_path: Path,
    dataset_root: Path,
    hub_tree_path: Path,
    libero_config_root: Path,
    output_dir: Path,
    latest_link: Path | None,
    workers: int,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    validate_contract(contract)
    dataset = contract["datasets"]["libero_90"]
    hub = load_hub_surface(
        hub_tree_path,
        subdir=dataset["subdir"],
        expected_file_count=dataset["file_count"],
        expected_total_bytes=dataset["total_bytes"],
    )
    authorities = load_task_authority(contract=contract, libero_config_root=libero_config_root)
    source_bounds = contract["episode_authority"]["source_base_fit"]
    jobs = _audit_jobs(
        authorities=authorities,
        hub=hub,
        dataset=dataset,
        dataset_root=dataset_root,
        source_bounds=source_bounds,
    )
    results = _run_audits(jobs, workers)
    tasks = _merge_task_records(authorities, results)
    normalization = compute_normalization(
        results,
        source_task_indices=contract["splits"]["source"],
        episode_bounds=source_bounds,
    )
    access_counts = {
        "source_normalization_values": sum(
            task["access_policy"] == "source_normalization_values" for task in tasks
        ),
        "metadata_only": sum(task["access_policy"] == "metadata_only" for task in tasks),
        "validation_or_held_numeric_values": 0,
    }
    manifest = _manifest_document(
        workspace=workspace, contract=contract, dataset=dataset, hub=hub, tasks=tasks
    )
    quality_report = _aggregate_quality(
        tasks, access_counts=access_counts, contract=contract
    )
    write_artifacts(
        output_dir=output_dir,
        latest_link=latest_link,
        manifest=manifest,
        normalization=normalization,
        quality_report=quality_report,
    )
    return {"summary": manifest["summary"], "quality": quality_report["status"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=Path("configs/phase0.toml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hub-tree", type=Path, required=True)
    parser.add_argument("--libero-config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    return args


def main() -> int:
    args = _parse_args()
    result = build_manifest(
        workspace=args.workspace.resolve(),
        contract_path=args.contract.resolve(),
        dataset_root=args.dataset_root.resolve(),
        hub_tree_path=args.hub_tree.resolve(),
        libero_config_root=args.libero_config_root.resolve(),
        output_dir=args.output_dir,
        latest_link=args.latest_link,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
