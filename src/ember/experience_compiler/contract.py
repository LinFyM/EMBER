"""Frozen assets, metadata panels and disjoint initial-state streams."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from ember.operator_writer.data import TASKS as TASKS36
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.learning_data import load_learning_tasks


ASSET_ROOT = Path("/data1/user/ymdai/projects/EMBER")
RUN_ROOT = Path("/data1/user/ymdai/ember_runs/functional_revision_learning_20261010")
SCHEMA = "ember_functional_revision_compiler_v1"
STAGE = "functional_revision_learning_20261010"
SEED = 20261010
EVENT_SCHEMA = "ember_actual_compiler_events_v2"
_SOURCE_SPEC = read_json(ASSET_ROOT / "configs/operator_read_write_v1/learning_spec.json")
SOURCE = dict(_SOURCE_SPEC["source"])
MT_PATH = Path(_SOURCE_SPEC["evaluation"]["mt_checkpoint"])
_T_ROOT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T")
T_RESULTS = _T_ROOT / "evaluation/2340/correct400/results.json"
MT_RESULTS = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400/results.json")
_LEARNING_CONTRACT = Path("/data1/user/ymdai/ember_runs/denoising_return_writer_20261006/readouts/parent/seen/SDE/evaluation/run_contract.json")
_FORMAL_CONTRACT = T_RESULTS.with_name("run_contract.json")
_TASK_FIELDS = ("suite", "task_id", "language", "bddl_file", "bddl_bytes", "problem_folder",
                "horizon", "init_states_file", "init_states_bytes", "installed_init_state_count",
                "split_role")


def condition_seed(*coordinates: int, domain: int = 0) -> int:
    values = np.random.SeedSequence([SEED, domain, *map(int, coordinates)]).generate_state(2)
    return ((int(values[0]) << 32) | int(values[1])) & ((1 << 63) - 1)


def training_tasks(asset_root: Path = ASSET_ROOT):
    """Only metadata for the fixed 24+12 allowlist; no HDF5 labels are opened."""
    return load_learning_tasks(Path(asset_root), TASKS36, role="train",
                               protocol_path=SOURCE["data_protocol"])


def _environment(path: Path, role: str, asset_root: Path) -> dict:
    original = read_json(path)
    result = {key: deepcopy(original[key]) for key in ("environment", "libero_paths", "policy", "rng")}
    tasks = []
    for item in original["tasks"]:
        task = {key: item[key] for key in _TASK_FIELDS}
        if task["installed_init_state_count"] != 50:
            raise ValueError("compilation requires the original full50 initial-state pool")
        task["global_task_id"] = (40 + task["task_id"] if task["suite"] == "libero_90"
                                  else 10 * SUITE_ORDER.index(task["suite"]) + task["task_id"])
        task["init_state_ids"] = list(range(50))
        tasks.append(task)
    result.update(tasks=tasks, role=role)
    # The supplied asset root owns tokenizer/data/runtime assets; historical
    # checkpoint roots remain read-only references and are never copied.
    for key, value in result["libero_paths"].items():
        if str(value).startswith(str(ASSET_ROOT) + "/"):
            result["libero_paths"][key] = str(Path(asset_root) / Path(value).relative_to(ASSET_ROOT))
    return result


def learning_environment(*, asset_root: Path = ASSET_ROOT) -> dict:
    result = _environment(_LEARNING_CONTRACT, "train", Path(asset_root))
    if tuple(task["global_task_id"] for task in result["tasks"]) != TASKS36:
        raise ValueError("learning environment differs from the fixed36 metadata order")
    return result


def formal_environment(*, asset_root: Path = ASSET_ROOT) -> dict:
    result = _environment(_FORMAL_CONTRACT, "validation", Path(asset_root))
    # Final400 must use the same registered scene restoration as strong T/MT.
    # Learning practice intentionally does not inherit the old seen32..35 scene.
    result["operator_read_write_scene"] = deepcopy(read_json(_FORMAL_CONTRACT)["operator_read_write_scene"])
    if len(result["tasks"]) != 8 or any(task["split_role"] != "validation" for task in result["tasks"]):
        raise ValueError("formal environment must be the fixed validation8")
    return result


def state_stream(event, excluded=()):
    """Cycle full50 permutations forever; excluded query/final IDs never occur.

    This is an adaptation state source, not a practice/read-count cap. Reuse
    becomes possible after exhausting the allowed pool and must be recorded.
    """
    seed = event["seed"] if isinstance(event, Mapping) else event.seed
    excluded = frozenset(map(int, excluded))
    if excluded - set(range(50)) or len(excluded) == 50:
        raise ValueError("initial-state exclusions must leave a legal full50 subset")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5354]))
    while True:
        for state_id in rng.permutation(50):
            if int(state_id) not in excluded:
                yield int(state_id)
