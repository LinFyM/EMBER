"""Specification-only LIBERO source filtering and post-seal data audit.

The two public stages are intentionally separate.  ``seal_overlap_audit`` only
reads installed LIBERO task specifications.  ``seal_source_data`` accepts the
hash of that first artifact before it opens any demonstration file.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ember.libero_data import (
    AuditResult,
    audit_demonstration_file,
    compute_normalization,
)
from ember.libero_evaluation import canonical_sha256, sha256_file
from ember.pi05_assets import Pi05EvaluationError, prepare_libero_config, write_json_atomic


SCHEMA = "ember_pi05_source_overlap_v1"
DATA_SCHEMA = "ember_pi05_source_manifest_v1"
NORMALIZATION_SCHEMA = "ember_pi05_source_normalization_v1"
SOURCE_SUITE = "libero_90"
TARGET_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_BENCHMARK_SHA256 = "9c07414d4c75f5de50d9c8b3fefbda81ae72b3eae158c1af1d115e77063efd18"
EXPECTED_TASK_MAP_SHA256 = "0c950df0a785aa55de968bb38ccd865d2017f71ddbe6f48cfd05ac0742b6d62d"
# SHA256 over sorted ``suite\0task_id\0per-file-sha256\n`` rows.
EXPECTED_BDDL_AGGREGATE_SHA256 = "9cd625af0e19fac1150b6f3365a6ce40e4a3a6d5d758bab2aa17168cd8e2d4cd"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Equivalence:
    target_suite: str
    target_task_id: int
    source_task_ids: tuple[int, ...]
    reason: str


# This table is the reviewed result of comparing all 90 x 40 pairs.  It is
# coupled to exact installed benchmark/task-map/BDDL hashes below; a changed
# specification fails closed instead of silently reusing the judgment.
EQUIVALENCES = (
    Equivalence("libero_goal", 3, (8,), "ordered open-top-drawer then bowl-in composition"),
    Equivalence("libero_goal", 4, (10, 25, 31), "akita black bowl on cabinet top-side"),
    Equivalence("libero_goal", 7, (20, 44), "turn on flat stove"),
    Equivalence("libero_goal", 8, (9, 30), "akita black bowl on plate"),
    Equivalence("libero_goal", 9, (27,), "wine bottle on wine-rack top region"),
    Equivalence("libero_object", 0, (46, 50), "alphabet soup in basket"),
    Equivalence("libero_object", 1, (47,), "BDDL cream_cheese object in basket"),
    Equivalence("libero_object", 4, (48,), "ketchup in basket"),
    Equivalence("libero_object", 5, (49, 54), "tomato sauce in basket"),
    Equivalence("libero_object", 6, (51,), "butter in basket"),
    Equivalence("libero_object", 7, (52,), "milk in basket"),
    Equivalence("libero_object", 9, (53,), "orange juice in basket"),
    Equivalence("libero_10", 5, (77,), "book in caddy back compartment"),
)


NEAR_MISSES = (
    {
        "source_task_ids": [2, 29],
        "target": {"suite": "libero_goal", "task_id": 3},
        "decision": "retain",
        "reason": "source top drawer is initially open and the open>place composition is absent",
    },
    {
        "source_task_ids": [12, 13, 14],
        "target": {"suite": "libero_spatial", "task_ids": list(range(10))},
        "decision": "retain",
        "reason": "three same-type bowls and back/front/middle source selector differ from target tasks",
    },
    {
        "source_task_ids": [15],
        "target": {"suite": "libero_goal", "task_id": 4},
        "decision": "retain",
        "reason": "source selects the middle of three same-type bowls",
    },
    {
        "source_task_ids": [38],
        "target": {"suite": "libero_10", "task_id": 2},
        "decision": "retain",
        "reason": "source stove is initially on and task selects right-of-two moka pots; target must turn on one pot task",
    },
)


def excluded_source_ids() -> tuple[int, ...]:
    values = sorted(task_id for group in EQUIVALENCES for task_id in group.source_task_ids)
    if len(values) != len(set(values)):
        raise Pi05EvaluationError("one LIBERO-90 source task has multiple exclusion owners")
    return tuple(values)


def active_source_ids() -> tuple[int, ...]:
    excluded = set(excluded_source_ids())
    return tuple(task_id for task_id in range(90) if task_id not in excluded)


def _balanced_section(text: str, name: str) -> str:
    start = text.lower().find(f"(:{name}")
    if start < 0:
        raise Pi05EvaluationError(f"BDDL has no :{name} section")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return " ".join(text[start : index + 1].split())
    raise Pi05EvaluationError(f"unterminated BDDL :{name} section")


def _optional_section(text: str, name: str) -> str | None:
    if f"(:{name}" not in text.lower():
        return None
    return _balanced_section(text, name)


def _task_surface(suite_name: str, suite: Any, task_id: int, bddl_root: Path) -> dict[str, Any]:
    task = suite.get_task(task_id)
    bddl_path = bddl_root / task.problem_folder / task.bddl_file
    text = bddl_path.read_text(encoding="utf-8")
    return {
        "suite": suite_name,
        "task_id": task_id,
        "task_name": str(task.name),
        "language": str(task.language).lower(),
        "problem_folder": str(task.problem_folder),
        "bddl_file": str(task.bddl_file),
        "bddl_bytes": bddl_path.stat().st_size,
        "bddl_sha256": sha256_file(bddl_path),
        "specification": {
            "fixtures": _optional_section(text, "fixtures"),
            "objects": _optional_section(text, "objects"),
            "obj_of_interest": _optional_section(text, "obj_of_interest"),
            "init": _balanced_section(text, "init"),
            "goal": _balanced_section(text, "goal"),
        },
    }


def _bddl_aggregate(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["suite"], item["task_id"])):
        digest.update(
            f"{row['suite']}\0{row['task_id']}\0{row['bddl_sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _installed_specifications() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="ember-source-spec-") as config_dir:
        paths = prepare_libero_config(Path(config_dir))
        from libero.libero import benchmark
        import libero.libero.benchmark as benchmark_module

        benchmark_path = Path(benchmark_module.__file__).resolve()
        task_map_path = benchmark_path.parent / "libero_suite_task_map.py"
        authorities = {
            "distribution": "hf-libero",
            "version": "0.1.4",
            "benchmark_module_sha256": sha256_file(benchmark_path),
            "task_map_module_sha256": sha256_file(task_map_path),
        }
        if authorities["benchmark_module_sha256"] != EXPECTED_BENCHMARK_SHA256:
            raise Pi05EvaluationError("installed LIBERO benchmark authority changed")
        if authorities["task_map_module_sha256"] != EXPECTED_TASK_MAP_SHA256:
            raise Pi05EvaluationError("installed LIBERO task-map authority changed")
        bddl_root = Path(paths["bddl_files"])
        suites = {
            name: benchmark.get_benchmark_dict()[name]()
            for name in (*TARGET_SUITES, SOURCE_SUITE)
        }
        target_rows = [
            _task_surface(name, suites[name], task_id, bddl_root)
            for name in TARGET_SUITES
            for task_id in range(suites[name].n_tasks)
        ]
        source_rows = [
            _task_surface(SOURCE_SUITE, suites[SOURCE_SUITE], task_id, bddl_root)
            for task_id in range(suites[SOURCE_SUITE].n_tasks)
        ]
    return authorities, target_rows, source_rows


def _decorate_source_rows(source_rows: list[dict[str, Any]]) -> set[int]:
    excluded = set(excluded_source_ids())
    match_by_source = {
        source_id: {
            "target_suite": group.target_suite,
            "target_task_id": group.target_task_id,
            "reason": group.reason,
        }
        for group in EQUIVALENCES
        for source_id in group.source_task_ids
    }
    for row in source_rows:
        task_id = int(row["task_id"])
        row["decision"] = "exclude" if task_id in excluded else "active"
        row["exact_match"] = match_by_source.get(task_id)
    return excluded


def _overlap_record(
    *,
    sealed_utc: str,
    authorities: dict[str, Any],
    aggregate: str,
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    excluded: set[int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "sealed_utc": sealed_utc,
        "purpose": "filter LIBERO-90 exact semantic/composition overlap with target LIBERO-40",
        "algorithm": {
            "name": "manual_role_bddl_full_task_equivalence_v1",
            "seed": None,
            "pair_count_reviewed": 90 * 40,
            "rule": "exclude only complete ordered task equivalence after BDDL-confirmed aliases; retain primitive/subtask containment",
            "aliases": [
                "put/place",
                "bowl/black bowl only when BDDL type is akita_black_bowl",
                "rack/wine rack only when BDDL type is wine_rack",
                "cream cheese box/cream cheese only when BDDL type is cream_cheese",
                "white/wooden cabinet only for language-unspecified same-role cabinet regions",
            ],
            "required_equal_surfaces": [
                "ordered operations",
                "task-relevant object and fixture types",
                "destination relation and selector",
                "source selector",
                "object multiplicity",
                "task-relevant initial predicate truth",
                "complete goal/composition",
            ],
            "ignored_surfaces": ["scene identity", "coordinates", "distractors", "BDDL instance numbering"],
        },
        "information_wall": {
            "read": [
                "task identity",
                "task name",
                "language",
                "BDDL filename/content",
                "scene",
                "objects/fixtures/roles",
                "task-relevant init/goal predicates",
            ],
            "forbidden": [
                "action",
                "reward",
                "proprio",
                "terminal",
                "normalization",
                "policy outcome",
            ],
        },
        "authorities": {**authorities, "bddl_130_aggregate_sha256": aggregate},
        "summary": {
            "source_tasks": len(source_rows),
            "target_tasks": len(target_rows),
            "excluded_source_tasks": len(excluded),
            "active_source_tasks": len(source_rows) - len(excluded),
            "excluded_source_task_ids": sorted(excluded),
            "active_source_task_ids": list(active_source_ids()),
        },
        "equivalences": [
            {
                "target": {"suite": item.target_suite, "task_id": item.target_task_id},
                "source_task_ids": list(item.source_task_ids),
                "reason": item.reason,
            }
            for item in EQUIVALENCES
        ],
        "near_misses": list(NEAR_MISSES),
        "target_tasks": target_rows,
        "source_tasks": source_rows,
    }


def seal_overlap_audit(output_path: Path, *, sealed_utc: str) -> dict[str, Any]:
    """Seal the complete specification-only source/target overlap decision."""

    authorities, target_rows, source_rows = _installed_specifications()
    aggregate = _bddl_aggregate([*target_rows, *source_rows])
    if aggregate != EXPECTED_BDDL_AGGREGATE_SHA256:
        raise Pi05EvaluationError("installed 130-task BDDL authority changed")
    excluded = _decorate_source_rows(source_rows)
    value = _overlap_record(
        sealed_utc=sealed_utc,
        authorities=authorities,
        aggregate=aggregate,
        target_rows=target_rows,
        source_rows=source_rows,
        excluded=excluded,
    )
    if value["summary"]["active_source_tasks"] != 71:
        raise Pi05EvaluationError("reviewed source overlap count changed")
    write_json_atomic(output_path, value)
    return value


def _load_overlap(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise Pi05EvaluationError("unsupported source overlap audit")
    if tuple(value.get("summary", {}).get("active_source_task_ids", ())) != active_source_ids():
        raise Pi05EvaluationError("source overlap audit active IDs changed")
    if value.get("authorities", {}).get("bddl_130_aggregate_sha256") != EXPECTED_BDDL_AGGREGATE_SHA256:
        raise Pi05EvaluationError("source overlap audit BDDL authority changed")
    return value


def _hdf5_aggregate(records: Iterable[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "task_id": int(record["task_index"]),
                "filename": record["hdf5"]["filename"],
                "bytes": int(record["hdf5"]["bytes"]),
                "sha256": record["hdf5"]["sha256"],
            }
            for record in sorted(records, key=lambda item: int(item["task_index"]))
        ]
    )


def seal_source_data(
    *,
    overlap_path: Path,
    prior_manifest_path: Path,
    data_root: Path,
    manifest_path: Path,
    normalization_path: Path,
    sealed_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit active HDF5 files and compute stats only after overlap is sealed."""

    overlap = _load_overlap(overlap_path)
    overlap_sha256 = sha256_file(overlap_path)
    prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    if len(prior.get("tasks", [])) != 90:
        raise Pi05EvaluationError("provenance manifest is not the pinned 90-task corpus")
    prior_by_id = {int(row["task_index"]): row for row in prior["tasks"]}
    source_by_id = {int(row["task_id"]): row for row in overlap["source_tasks"]}
    results: list[AuditResult] = []
    for task_id in active_source_ids():
        old = prior_by_id[task_id]
        spec = source_by_id[task_id]
        if old["task_name"] != spec["task_name"] or old["language"] != spec["language"]:
            raise Pi05EvaluationError(f"task {task_id} provenance and specification disagree")
        if old["bddl"]["sha256"] != spec["bddl_sha256"]:
            raise Pi05EvaluationError(f"task {task_id} BDDL provenance changed")
        results.append(
            audit_demonstration_file(
                data_root / old["hdf5"]["filename"],
                task_index=task_id,
                task_name=old["task_name"],
                split="train",
                language=old["language"],
                bddl_basename=old["bddl"]["filename"],
                expected_tag="libero-v1",
                expected_demos=50,
                expected_sha256=old["hdf5"]["sha256"],
                expected_bytes=int(old["hdf5"]["bytes"]),
                normalization_episodes=tuple(range(50)),
            )
        )
    records = [result.record for result in results]
    aggregate = _hdf5_aggregate(records)
    manifest = {
        "schema_version": DATA_SCHEMA,
        "sealed_utc": sealed_utc,
        "overlap_audit_sha256": overlap_sha256,
        "provenance_manifest": {
            "path": str(prior_manifest_path.resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(prior_manifest_path),
            "usage": "immutable file/BDDL identities only; retired split fields ignored",
        },
        "dataset": {
            "repo_id": prior["dataset"]["repo_id"],
            "revision": prior["dataset"]["revision"],
            "subdir": prior["dataset"]["subdir"],
        },
        "summary": {
            "active_tasks": len(records),
            "episodes": sum(int(row["demonstrations"]["count"]) for row in records),
            "frames": sum(int(row["demonstrations"]["steps"]) for row in records),
            "hdf5_bytes": sum(int(row["hdf5"]["bytes"]) for row in records),
            "hdf5_aggregate_sha256": aggregate,
            "active_source_task_ids": list(active_source_ids()),
            "excluded_source_task_ids": list(excluded_source_ids()),
        },
        "access_order": "overlap audit sealed before any active HDF5 numeric read",
        "tasks": records,
    }
    write_json_atomic(manifest_path, manifest)
    raw_normalization = compute_normalization(
        results,
        train_task_indices=list(active_source_ids()),
        episode_bounds=[0, 49],
    )
    normalization = {
        "schema_version": NORMALIZATION_SCHEMA,
        "sealed_utc": sealed_utc,
        "overlap_audit_sha256": overlap_sha256,
        "source_manifest_sha256": sha256_file(manifest_path),
        "hdf5_aggregate_sha256": aggregate,
        "authority": {
            "source_suite": SOURCE_SUITE,
            "active_task_ids": list(active_source_ids()),
            "episodes_per_task": 50,
            "numeric_rows": int(raw_normalization["observation.state"]["count"]),
            "validation_or_test_numeric_reads": 0,
        },
        "feature_definitions": raw_normalization["feature_definitions"],
        "quantile_method": "numpy.quantile_linear_q01_q10_q50_q90_q99",
        "stats": {
            "observation.state": raw_normalization["observation.state"],
            "action": raw_normalization["action"],
        },
    }
    write_json_atomic(normalization_path, normalization)
    return manifest, normalization


def write_checksums(config_dir: Path, filenames: Iterable[str]) -> None:
    rows = []
    for filename in filenames:
        path = config_dir / filename
        if not path.is_file():
            raise Pi05EvaluationError(f"missing source-corpus artifact: {path}")
        rows.append(f"{sha256_file(path)}  {filename}")
    path = config_dir / "checksums.sha256"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
