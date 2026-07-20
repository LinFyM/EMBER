#!/usr/bin/env python3
"""Seal data identities around the specification-only EMBER LIBERO-90 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ember.libero_data import (
    REQUIRED_DATASETS,
    audit_demonstration_file,
    compute_normalization,
    load_hub_surface,
    sha256_file,
)
from ember.libero_split import (
    ProtocolError,
    SCHEMA_VERSION,
    SPLIT_ORDER,
    build_factor_artifact,
    build_split_artifact,
    load_specification_surface,
    seal_split,
)


DATASET_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DATASET_TREE_SHA256 = "b935f12c002c801cafb611ad8072af196278ce788adf016c1e358b8464a61f37"
DATASET_FILE_COUNT = 90
DATASET_TOTAL_BYTES = 66_658_085_995
EXPECTED_DEMOS = 50
EXPECTED_HDF5_TAG = "libero-v1"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _init_state_record(path: Path) -> dict[str, Any]:
    import torch

    values = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(values, "shape") or len(values) != 50:
        raise ProtocolError(f"invalid official init-state file: {path.name}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "count": len(values),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _audit_task(
    *,
    row: dict[str, Any],
    split_name: str,
    hdf5_root: Path,
    bddl_root: Path,
    init_root: Path,
    authority: dict[str, Any],
) -> Any:
    task_index = row["task_index"]
    hdf5_name = f"{row['task_name']}_demo.hdf5"
    result = audit_demonstration_file(
        hdf5_root / hdf5_name,
        task_index=task_index,
        task_name=row["task_name"],
        split=split_name,
        language=row["language"],
        bddl_basename=row["bddl_basename"],
        expected_tag=EXPECTED_HDF5_TAG,
        expected_demos=EXPECTED_DEMOS,
        expected_sha256=authority["sha256"],
        expected_bytes=authority["bytes"],
        normalization_episodes=(tuple(range(EXPECTED_DEMOS)) if split_name == "train" else ()),
    )
    bddl_path = bddl_root / row["bddl_basename"]
    init_path = init_root / row["init_states_basename"]
    result.record.update(
        {
            "scene": row["scene"],
            "scene_family": row["scene_family"],
            "task_family": row["task_family"],
            "bddl": {
                "filename": bddl_path.name,
                "bytes": bddl_path.stat().st_size,
                "sha256": sha256_file(bddl_path),
            },
            "init_states": _init_state_record(init_path),
        }
    )
    return result


def _require_one_authority(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    variants = {json.dumps(record[field], sort_keys=True) for record in records}
    if len(variants) != 1:
        raise ProtocolError(f"{field} authority varies across LIBERO-90 tasks")
    return json.loads(next(iter(variants)))


def _audit_all_tasks(
    rows: list[dict[str, Any]],
    split: dict[str, list[int]],
    surface: dict[str, Any],
    hdf5_root: Path,
    bddl_root: Path,
    init_root: Path,
) -> list[Any]:
    split_by_id = {task: name for name in SPLIT_ORDER for task in split[name]}
    results = []
    for row in rows:
        hdf5_name = f"{row['task_name']}_demo.hdf5"
        authority = surface["files"].get(hdf5_name)
        if authority is None:
            raise ProtocolError(f"missing Hub authority for {hdf5_name}")
        result = _audit_task(
            row=row,
            split_name=split_by_id[row["task_index"]],
            hdf5_root=hdf5_root,
            bddl_root=bddl_root,
            init_root=init_root,
            authority=authority,
        )
        results.append(result)
        print(f"audited task {row['task_index']:02d}/89 ({result.record['split']})", flush=True)
    return results


def _build_manifest(
    records: list[dict[str, Any]],
    surface: dict[str, Any],
    factor_sha256: str,
    split_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "surface_contract": {
            "split_generation": "language_scene_and_role_factors_only",
            "split_forbidden_fields": [
                "action",
                "proprio",
                "reward",
                "terminal",
                "normalization",
                "policy_outcome",
            ],
            "train_numeric_access": "state_and_action_for_train_only_normalization",
            "validation_numeric_access": "metadata_only",
            "test_numeric_access": "metadata_only",
        },
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "subdir": "libero_90",
            "tree_metadata_sha256": DATASET_TREE_SHA256,
            "hdf5_file_count": surface["file_count"],
            "hdf5_total_bytes": surface["total_bytes"],
        },
        "protocol_references": {
            "factor_table_sha256": factor_sha256,
            "split_sha256": split_sha256,
        },
        "dataset_schema": {
            key: {"dtype": value[0], "tail_shape": value[1]}
            for key, value in REQUIRED_DATASETS.items()
        },
        "summary": {
            "tasks": len(records),
            "demonstrations": sum(record["demonstrations"]["count"] for record in records),
            "frames": sum(record["demonstrations"]["steps"] for record in records),
            "split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
            "quality": dict(sorted(Counter(record["quality"]["status"] for record in records).items())),
            "controller": _require_one_authority(records, "controller"),
            "camera": _require_one_authority(records, "camera"),
        },
        "tasks": records,
    }


def _audit_data(
    rows: list[dict[str, Any]],
    split: dict[str, list[int]],
    dataset_root: Path,
    tree_metadata: Path,
    factor_sha256: str,
    split_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from libero.libero import get_libero_path

    if sha256_file(tree_metadata) != DATASET_TREE_SHA256:
        raise ProtocolError("the local Hub tree metadata does not match the pinned revision")
    surface = load_hub_surface(
        tree_metadata,
        subdir="libero_90",
        expected_file_count=DATASET_FILE_COUNT,
        expected_total_bytes=DATASET_TOTAL_BYTES,
    )
    hdf5_root = dataset_root / "libero_90"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_90"
    init_root = Path(get_libero_path("init_states")) / "libero_90"
    results = _audit_all_tasks(rows, split, surface, hdf5_root, bddl_root, init_root)
    normalization = compute_normalization(
        results,
        train_task_indices=split["train"],
        episode_bounds=[0, EXPECTED_DEMOS - 1],
    )
    return _build_manifest(
        [result.record for result in results], surface, factor_sha256, split_sha256
    ), normalization


def _write_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    checksums = "".join(
        f"{_sha256_bytes(content)}  {name}\n" for name, content in sorted(artifacts.items())
    ).encode("utf-8")
    artifacts = {**artifacts, "checksums.sha256": checksums}
    if output_dir.exists():
        actual = {path.name for path in output_dir.iterdir() if path.is_file()}
        if actual != set(artifacts):
            raise ProtocolError("existing protocol directory has a different file set")
        for name, content in artifacts.items():
            if (output_dir / name).read_bytes() != content:
                raise ProtocolError(f"existing sealed artifact differs: {name}")
        print(f"verified existing sealed protocol: {output_dir}")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".libero90-protocol-", dir=output_dir.parent))
    try:
        for name, content in artifacts.items():
            (temporary / name).write_bytes(content)
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"sealed protocol: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tree-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _suite, rows, authority = load_specification_surface()
    split, solver = seal_split(rows)
    factor_bytes = _canonical_json(build_factor_artifact(rows, authority))
    split_bytes = _canonical_json(
        build_split_artifact(
            rows,
            split,
            solver,
            sha256_file(Path(__file__).resolve()),
            _sha256_bytes(factor_bytes),
        )
    )
    manifest, normalization = _audit_data(
        rows,
        split,
        args.dataset_root.resolve(),
        args.tree_metadata.resolve(),
        _sha256_bytes(factor_bytes),
        _sha256_bytes(split_bytes),
    )
    _write_artifacts(
        args.output_dir.resolve(),
        {
            "factor_table.json": factor_bytes,
            "split.json": split_bytes,
            "data_manifest.json": _canonical_json(manifest),
            "normalization_train_only.json": _canonical_json(normalization),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
