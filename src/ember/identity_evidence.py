"""Canonical evidence primitives for evaluation-identity diagnostics."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class IdentityProbeError(RuntimeError):
    """Raised when the evaluation-identity contract is invalid or violated."""


def as_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return np.asarray(value)
    return None


def _tree_leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, Mapping):
        leaves: dict[str, Any] = {}
        for key in sorted(tree, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_tree_leaves(tree[key], path))
        return leaves
    if isinstance(tree, (list, tuple)):
        leaves = {}
        for index, value in enumerate(tree):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            leaves.update(_tree_leaves(value, path))
        return leaves
    return {prefix or "$": tree}


def _leaf_identity(value: Any) -> tuple[dict[str, Any], bytes]:
    array = as_numpy(value)
    if array is not None:
        if array.dtype.hasobject:
            raise IdentityProbeError("Object arrays are not valid identity evidence")
        canonical = np.ascontiguousarray(array)
        raw = canonical.tobytes(order="C")
        metadata: dict[str, Any] = {
            "kind": "array",
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if canonical.size and np.issubdtype(canonical.dtype, np.number):
            finite = canonical[np.isfinite(canonical)]
            if finite.size:
                metadata["min"] = float(finite.min())
                metadata["max"] = float(finite.max())
        return metadata, raw
    if value is None or isinstance(value, (bool, int, float, str)):
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        return {
            "kind": "scalar",
            "type": type(value).__name__,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }, raw
    raise IdentityProbeError(f"Unsupported identity leaf: {type(value).__name__}")


def canonical_tree_summary(tree: Any) -> dict[str, Any]:
    """Hash a nested tree while binding paths, dtypes, shapes, and values."""

    leaves = _tree_leaves(tree)
    summaries: dict[str, dict[str, Any]] = {}
    root = hashlib.sha256()
    for path in sorted(leaves):
        metadata, raw = _leaf_identity(leaves[path])
        summaries[path] = metadata
        root.update(path.encode("utf-8"))
        root.update(b"\0")
        root.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        root.update(b"\0")
        root.update(raw)
        root.update(b"\0")
    return {"sha256": root.hexdigest(), "leaves": summaries}


def _compare_leaf(left: Any, right: Any, atol: float, rtol: float) -> tuple[bool, bool, float]:
    left_array = as_numpy(left)
    right_array = as_numpy(right)
    if left_array is None or right_array is None:
        exact = type(left) is type(right) and left == right
        return exact, exact, 0.0
    same_contract = left_array.shape == right_array.shape and left_array.dtype == right_array.dtype
    if not same_contract:
        return False, False, float("inf")
    exact = bool(np.array_equal(left_array, right_array, equal_nan=True))
    if not np.issubdtype(left_array.dtype, np.number):
        return exact, exact, 0.0
    within = bool(np.allclose(left_array, right_array, atol=atol, rtol=rtol, equal_nan=True))
    if not left_array.size:
        return exact, within, 0.0
    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    max_abs = float(finite.max()) if finite.size else (0.0 if exact else float("inf"))
    return exact, within, max_abs


def _leaf_difference_stats(left: Any, right: Any) -> dict[str, Any] | None:
    left_array = as_numpy(left)
    right_array = as_numpy(right)
    if left_array is None or right_array is None:
        return None
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return {"contract_mismatch": True}
    if not np.issubdtype(left_array.dtype, np.number):
        return None
    left_float = left_array.astype(np.float64)
    right_float = right_array.astype(np.float64)
    equal = left_array == right_array
    if np.issubdtype(left_array.dtype, np.inexact):
        equal = equal | (np.isnan(left_array) & np.isnan(right_array))
    unequal = ~equal
    difference = np.abs(left_float - right_float)
    finite = difference[np.isfinite(difference)]
    total = int(left_array.size)
    return {
        "contract_mismatch": False,
        "total_count": total,
        "unequal_count": int(unequal.sum()),
        "unequal_fraction": float(unequal.sum() / total) if total else 0.0,
        "mean_abs": float(finite.mean()) if finite.size else 0.0,
        "max_abs": float(finite.max()) if finite.size else 0.0,
    }


def _path_domain(path: str) -> str:
    if "pixels" in path:
        return "pixels"
    if any(token in path for token in ("agent_pos", "state", "eef_", "gripper_", "joint")):
        return "state"
    if any(token in path for token in ("reward", "terminated", "truncated", "done")):
        return "outcome"
    if "action" in path:
        return "action"
    return "other"


def compare_trees(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    """Compare two nested trees with exact and declared-tolerance outcomes."""

    left_leaves = _tree_leaves(left)
    right_leaves = _tree_leaves(right)
    all_paths = sorted(set(left_leaves) | set(right_leaves))
    mismatched: list[str] = []
    outside_tolerance: list[str] = []
    leaf_differences: dict[str, dict[str, Any]] = {}
    max_abs = 0.0
    for path in all_paths:
        if path not in left_leaves or path not in right_leaves:
            mismatched.append(path)
            outside_tolerance.append(path)
            max_abs = float("inf")
            continue
        exact, within, leaf_max = _compare_leaf(left_leaves[path], right_leaves[path], atol, rtol)
        if not exact:
            mismatched.append(path)
            difference_stats = _leaf_difference_stats(left_leaves[path], right_leaves[path])
            if difference_stats is not None:
                leaf_differences[path] = difference_stats
        if not within:
            outside_tolerance.append(path)
        max_abs = max(max_abs, leaf_max)
    return {
        "exact": not mismatched,
        "within_tolerance": not outside_tolerance,
        "max_abs": max_abs,
        "mismatched_paths": mismatched,
        "outside_tolerance_paths": outside_tolerance,
        "leaf_differences": leaf_differences,
        "mismatch_domains": sorted({_path_domain(path) for path in mismatched}),
        "left_sha256": canonical_tree_summary(left)["sha256"],
        "right_sha256": canonical_tree_summary(right)["sha256"],
    }


def _validate_probe_core(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise IdentityProbeError("Identity probe schema must be version 1")
    if spec.get("surface") != "official_overlap_mechanics_only":
        raise IdentityProbeError("Identity probe must remain on the official overlap surface")
    if spec.get("task_suite") != "libero_spatial" or spec.get("task_id") != 0:
        raise IdentityProbeError("First identity probe is locked to libero_spatial task 0")
    for key in ("fixed_steps", "policy_steps"):
        if not isinstance(spec.get(key), int) or not 1 <= spec[key] <= 10:
            raise IdentityProbeError(f"{key} must be an integer from 1 through 10")


def _validate_probe_batches(spec: Mapping[str, Any]) -> None:
    batch_sizes = spec.get("policy_batch_sizes")
    valid = isinstance(batch_sizes, list) and bool(batch_sizes)
    valid = valid and batch_sizes == sorted(set(batch_sizes))
    valid = valid and batch_sizes[0] == 1 and batch_sizes[-1] <= 112
    if not valid:
        raise IdentityProbeError("Policy batch sizes must be sorted, unique, and bounded by 112")
    conditions = spec.get("env_conditions", [])
    names = {item.get("name") for item in conditions}
    modes = {item.get("mode") for item in conditions}
    if len(names) != len(conditions) or modes != {"sync", "async"}:
        raise IdentityProbeError("Environment conditions need unique sync and async entries")
    if any(item.get("batch_size") not in {1, 2} for item in conditions):
        raise IdentityProbeError("Environment identity batches are predeclared as 1 or 2")


def _validate_probe_comparisons(spec: Mapping[str, Any]) -> None:
    names = {item.get("name") for item in spec["env_conditions"]}
    for pair in spec.get("comparison_pairs", []):
        if pair.get("left") not in names or pair.get("right") not in names:
            raise IdentityProbeError("Comparison references an unknown condition")
        if not pair.get("shared_indices"):
            raise IdentityProbeError("Comparison must declare shared logical indices")


def _validate_probe_tolerances(spec: Mapping[str, Any]) -> None:
    for key in ("observation_atol", "observation_rtol", "action_atol", "action_rtol"):
        if not isinstance(spec.get(key), (int, float)) or spec[key] < 0:
            raise IdentityProbeError(f"{key} must be non-negative")


def load_probe_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the bounded, overlap-only Gate -1 specification."""

    with Path(path).open("rb") as handle:
        spec = tomllib.load(handle)
    _validate_probe_core(spec)
    _validate_probe_batches(spec)
    _validate_probe_comparisons(spec)
    _validate_probe_tolerances(spec)
    return spec
