"""Sealed protocol, normalization, and model assets for pi0.5 LIBERO evaluation."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ember.libero_evaluation import sha256_file


class Pi05EvaluationError(RuntimeError):
    """Raised when the sealed pi0.5 evaluation contract would be violated."""


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise Pi05EvaluationError("unsupported LIBERO 24/8/8 protocol schema")
    suites = protocol.get("split", {}).get("suites", {})
    if set(suites) != {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        raise Pi05EvaluationError("protocol must contain exactly four official LIBERO suites")
    for suite, roles in suites.items():
        values = [*roles["train"], *roles["validation"], *roles["test"]]
        if len(roles["train"]) != 6 or len(roles["validation"]) != 2 or len(roles["test"]) != 2:
            raise Pi05EvaluationError(f"suite {suite} is not a 6/2/2 split")
        if sorted(values) != list(range(10)):
            raise Pi05EvaluationError(f"suite {suite} does not cover local task IDs 0..9 once")
    return protocol


def prepare_libero_config(config_dir: Path) -> dict[str, str]:
    package = importlib.util.find_spec("libero")
    if package is None or package.origin is None:
        raise Pi05EvaluationError("installed LIBERO package cannot be located")
    benchmark_root = Path(package.origin).resolve().parent / "libero"
    configured_assets = os.environ.get("EMBER_LIBERO_ASSETS_ROOT")
    assets_root = (
        Path(configured_assets).expanduser().resolve()
        if configured_assets
        else benchmark_root / "assets"
    )
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(assets_root),
    }
    for name in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not Path(paths[name]).exists():
            raise Pi05EvaluationError(f"missing LIBERO {name}: {paths[name]}")
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config_dir / "config.yaml", paths)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return paths


def _suite_languages(protocol: dict[str, Any], config_dir: Path) -> dict[str, dict[int, str]]:
    prepare_libero_config(config_dir)
    from libero.libero import benchmark

    result: dict[str, dict[int, str]] = {}
    for suite_name in protocol["split"]["suites"]:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        result[suite_name] = {
            task_id: str(suite.get_task(task_id).language) for task_id in range(suite.n_tasks)
        }
    return result


def _dataset_language_index(dataset_root: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_root / "meta" / "tasks.parquet")
    frame = table.to_pandas()
    if "task_index" not in frame.columns:
        raise Pi05EvaluationError("dataset task table has no task_index column")
    result = {str(language): int(row.task_index) for language, row in frame.iterrows()}
    if len(result) != 40:
        raise Pi05EvaluationError(f"expected 40 dataset tasks, found {len(result)}")
    return result


def _role_global_ids(
    protocol: dict[str, Any], dataset_root: Path, config_dir: Path
) -> tuple[dict[str, list[int]], dict[str, dict[int, int]]]:
    language_to_global = _dataset_language_index(dataset_root)
    languages = _suite_languages(protocol, config_dir)
    roles = {"train": [], "validation": [], "test": []}
    local_to_global: dict[str, dict[int, int]] = {}
    for suite_name, suite_roles in protocol["split"]["suites"].items():
        local_to_global[suite_name] = {}
        for local_id, language in languages[suite_name].items():
            if language not in language_to_global:
                raise Pi05EvaluationError(f"dataset is missing language: {language}")
            local_to_global[suite_name][local_id] = language_to_global[language]
        for role in roles:
            roles[role].extend(local_to_global[suite_name][i] for i in suite_roles[role])
    checks = ((roles["train"], 24), (roles["validation"], 8), (roles["test"], 8))
    if any(len(set(ids)) != expected for ids, expected in checks):
        raise Pi05EvaluationError("global dataset role IDs are not unique")
    return roles, local_to_global


DATASET_REPO = "HuggingFaceVLA/libero"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
_FILESYSTEM_LOCAL = threading.local()


def _read_dataset_columns(
    dataset_root: Path, relative_path: str, columns: list[str]
) -> Any:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    local_path = dataset_root / relative_path
    if local_path.is_file():
        return pq.read_table(local_path, columns=columns)
    if not hasattr(_FILESYSTEM_LOCAL, "client"):
        _FILESYSTEM_LOCAL.client = HfFileSystem()
    remote_path = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/{relative_path}"
    with _FILESYSTEM_LOCAL.client.open(remote_path, "rb") as source:
        return pq.read_table(source, columns=columns)


def compute_train_only_stats(
    protocol_path: Path,
    dataset_root: Path,
    output_path: Path,
    workers: int = 16,
) -> None:
    from huggingface_hub import list_repo_files

    protocol = load_protocol(protocol_path)
    with tempfile.TemporaryDirectory(prefix="ember-libero-stats-") as config_dir:
        roles, local_to_global = _role_global_ids(protocol, dataset_root, Path(config_dir))
    train_ids = np.asarray(sorted(roles["train"]), dtype=np.int64)
    relative_paths = sorted(
        path
        for path in list_repo_files(
            DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION
        )
        if path.startswith("data/") and path.endswith(".parquet")
    )
    if not relative_paths:
        raise Pi05EvaluationError("dataset has no numeric parquet files")

    def task_ids_in_file(path: str) -> tuple[str, frozenset[int]]:
        table = _read_dataset_columns(dataset_root, path, ["task_index"])
        return path, frozenset(int(value) for value in table["task_index"].to_pylist())

    with ThreadPoolExecutor(max_workers=workers) as executor:
        file_tasks = dict(executor.map(task_ids_in_file, relative_paths))
    train_set = frozenset(int(task_id) for task_id in train_ids)
    source_only_paths = [
        path for path in relative_paths if file_tasks[path] and file_tasks[path] <= train_set
    ]
    covered = frozenset().union(*(file_tasks[path] for path in source_only_paths))
    if covered != train_set:
        raise Pi05EvaluationError(
            f"source-only parquet selection misses train task IDs: {sorted(train_set - covered)}"
        )

    def numeric_rows(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = _read_dataset_columns(
            dataset_root,
            path,
            ["task_index", "observation.state", "action"],
        )
        task_ids = np.asarray(table["task_index"].to_numpy(zero_copy_only=False))
        if not frozenset(int(value) for value in np.unique(task_ids)) <= train_set:
            raise Pi05EvaluationError("held task appeared in a source-only parquet file")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        return task_ids, states, actions

    with ThreadPoolExecutor(max_workers=workers) as executor:
        numeric = list(executor.map(numeric_rows, source_only_paths))
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    selected_counts = {int(task_id): 0 for task_id in train_ids}
    for task_ids, state, action in numeric:
        states.append(state)
        actions.append(action)
        unique, counts = np.unique(task_ids, return_counts=True)
        for task_id, count in zip(unique, counts, strict=True):
            selected_counts[int(task_id)] += int(count)
    if not states or any(count == 0 for count in selected_counts.values()):
        raise Pi05EvaluationError("one or more source tasks contributed no numeric rows")
    state = np.concatenate(states)
    action = np.concatenate(actions)
    value = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_total_numeric_files": len(relative_paths),
        "source_only_numeric_files": len(source_only_paths),
        "source_only_file_selection": "all task_index values are in the sealed 24-task development-train role",
        "selected_train_rows": int(state.shape[0]),
        "selected_train_global_task_ids": sorted(int(x) for x in train_ids),
        "validation_global_task_ids_not_read": sorted(roles["validation"]),
        "test_global_task_ids_not_read": sorted(roles["test"]),
        "local_to_global_task_ids": local_to_global,
        "per_train_task_rows": selected_counts,
        "quantile_method": "numpy.quantile_linear_q01_q99",
        "stats": _normalization_stats(state, action),
    }
    write_json_atomic(output_path, value)


def _normalization_stats(state: np.ndarray, action: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in (("observation.state", state), ("action", action)):
        result[name] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0, method="linear").tolist(),
            "q99": np.quantile(values, 0.99, axis=0, method="linear").tolist(),
        }
    return result


def create_model_manifest(model_path: Path, output_path: Path) -> None:
    weights = model_path / "model.safetensors"
    config = model_path / "config.json"
    if not weights.is_file() or not config.is_file():
        raise Pi05EvaluationError(f"incomplete pi0.5 model snapshot: {model_path}")
    metadata = model_path / ".cache/huggingface/download/model.safetensors.metadata"
    if not metadata.is_file():
        raise Pi05EvaluationError("model snapshot has no Hugging Face revision metadata")
    revision = metadata.read_text(encoding="utf-8").splitlines()[0]
    if len(revision) != 40:
        raise Pi05EvaluationError("invalid Hugging Face model revision metadata")
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "model_repo": "lerobot/pi05_base",
            "model_revision": revision,
            "weights_filename": weights.name,
            "weights_bytes": weights.stat().st_size,
            "weights_sha256": sha256_file(weights),
            "config_sha256": sha256_file(config),
        },
    )
