"""Frozen assets, metadata panels and disjoint initial-state streams."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from ember.expert_manifold.video_schedule import condition_demo_index
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
T_BANK = _T_ROOT / "banks/2340/manifest.json"
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


def query_seed(seed: int, position: int) -> int:
    """Independent query root in LIBERO/NumPy's actual uint32 seed domain."""
    return int(np.random.SeedSequence([seed, position, 0xA5DE]).generate_state(1)[0])


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


def panel_contract(tasks=None, *, asset_root: Path = ASSET_ROOT) -> dict:
    tasks = training_tasks(asset_root) if tasks is None else tasks
    protocol = read_json(Path(asset_root) / SOURCE["data_protocol"])
    manifest = read_json(Path(asset_root) / protocol["data_manifest"])
    ordered = [int(row["global_task_id"]) for row in manifest["tasks"]
               if row["split_role"] == "train" and int(row["global_task_id"]) in tasks]
    selected = []
    # The original manifest owns panel order, even if a caller reorders its map.
    for suite in SUITE_ORDER:
        matches = [task_id for task_id in ordered if task_id < 40 and tasks[task_id].suite == suite]
        if len(matches) < 2:
            raise ValueError("fixed train panel needs two target tasks from each suite")
        selected.extend(matches[:2])
    conditions = []
    for task_id in selected:
        task = tasks[task_id]
        teachers = np.random.default_rng(np.random.SeedSequence([SEED, 0x5041, task_id])).permutation(50)[:2]
        for ordinal, demo in enumerate(teachers):
            demo = int(demo)
            conditions.append({"condition_id": f"train_task{task_id:03d}_demo{demo:02d}",
                               "task_id": task_id, "suite": task.suite,
                               "suite_task_id": task.suite_task_id, "language": task.authority.language,
                               "teacher_demo": demo, "final_state_ids": [32, 33, 34],
                               "seed": condition_seed(task_id, demo, ordinal, domain=0x5041)})
    return {"schema_version": SCHEMA, "task_ids": selected, "final_state_ids": [32, 33, 34],
            "conditions": conditions, "selection": "first_two_target_train_tasks_per_suite_in_manifest_order"}


def formal400_mapping(*, asset_root: Path = ASSET_ROOT) -> tuple[dict, ...]:
    """Canonical correct/other video ordinals, checked against all original400."""
    bank = read_json(T_BANK)
    video_seed = int(read_json(Path(bank["spec"]["path"]))["evaluation"]["video_schedule_seed"])
    environment = formal_environment(asset_root=asset_root)
    metadata = {(row["suite"], row["task_id"]): row for row in environment["tasks"]}
    output = []
    for task in bank["tasks"]:
        suite, local = task["suite"], int(task["task_id"])
        row = metadata[(suite, local)]
        episodes = task["episodes"]
        if len(episodes) != 50 or {int(item["init_state_id"]) for item in episodes} != set(range(50)):
            raise ValueError("reference formal bank does not contain a full50 state panel")
        for episode in episodes:
            final = int(episode["init_state_id"])
            demos = [condition_demo_index(video_seed, suite, local, final, condition=arm,
                     demo_count=50, sampling_mode="without_replacement")
                     for arm in ("correct", "same_task_other")]
            if (episode["teacher_demo_indices"] != [demos[0]]
                    or episode["paired_correct_demos"] != [demos[0]]
                    or episode["paired_other_demos"] != [demos[1]]
                    or episode["video_ordinal"] != final):
                raise ValueError("canonical video schedule differs from the reference400 mapping")
            output.append({"condition_id": episode["condition_id"], "task_id": row["global_task_id"],
                           "suite": suite, "suite_task_id": local, "language": row["language"],
                           "teacher_demo": demos[0], "init_state_id": final, "final_state_ids": [final],
                           "video_ordinal": final, "paired_correct_demos": [demos[0]],
                           "paired_other_demos": [demos[1]], "video_schedule_seed": video_seed,
                           "seed": condition_seed(row["global_task_id"], demos[0], final, domain=0xF04)})
    if len(output) != 400 or len({row["condition_id"] for row in output}) != 400:
        raise ValueError("formal mapping must have400 different task-video conditions")
    return tuple(output)


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
