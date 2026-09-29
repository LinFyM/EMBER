"""Registered 36 seen-task evaluation scope shared by bank, scene and capture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.task_protocol import load_task_authorities
from ember.writer.materialization import planned_episodes, selection_contract

from .data import TASKS


REPO = Path(__file__).resolve().parents[3]
PATH = REPO / "configs/operator_read_write_v1/seen_task_scope.json"
CAPTURE_PATH = REPO / "configs/operator_read_write_v1/seen_task_capture.json"
FROZEN_SCOPE_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-seen-task-formal"
    "/configs/operator_read_write_v1/seen_task_scope.json")
ROLE = "operator_seen_training36"
STATES = (32, 33, 34, 35)
SCHEMA = "ember_operator_seen_task_scope_v1"


def registration() -> dict:
    value = read_json(PATH)
    if (value.get("schema_version") != SCHEMA
            or value.get("study_id") != "operator_seen_task_diagnosis_20260929"
            or value.get("evaluation_role") != ROLE
            or tuple(value.get("global_task_ids", ())) != TASKS
            or tuple(value.get("init_state_ids", ())) != STATES
            or value.get("full_init_state_id") != 32
            or value.get("video_schedule_seed") != 20260928
            or value.get("video_pool") != [0, 49]
            or value.get("run_root") != "/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929"
            or value.get("checkpoint_t") != "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1800/T/train/attempts/continuation/checkpoints/macro_00001800"
            or value.get("checkpoint_mt") != "/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors"
            or value.get("training_git_t") != "fcc23cd15cc475530c385e354670efee6bacfa12"
            or value.get("training_git_mt") != "3ebb979b"):
        raise ValueError("operator seen-task registration changed")
    return value


def task_keys(protocol: Mapping, meta_protocol: Mapping) -> tuple[tuple[str, int], ...]:
    registered = registration()
    source = set(map(int, meta_protocol["active_source_task_ids"]))
    auxiliary = protocol.get("auxiliary_train", {})
    if (auxiliary.get("suite") != "libero_90"
            or auxiliary.get("global_task_id_offset") != 40
            or tuple(auxiliary.get("task_ids", ())) != tuple(task - 40 for task in TASKS[24:])):
        raise ValueError("seen-task support differs from the actual coverage36 train allowlist")
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    keys = []
    for global_id in registered["global_task_ids"]:
        if global_id < 40:
            suite, task = suites[global_id // 10], global_id % 10
            if task not in protocol["split"]["suites"][suite]["train"]:
                raise ValueError("seen-task target crosses development train role")
        else:
            suite, task = "libero_90", global_id - 40
            if task not in source or task not in auxiliary["task_ids"]:
                raise ValueError("seen-task support crosses audited coverage training role")
        keys.append((suite, task))
    if len(keys) != 36 or len(set(keys)) != 36:
        raise ValueError("seen-task keys changed")
    return tuple(keys)


def selection() -> dict:
    value = registration()
    result = selection_contract(
        role="nonheld_meta", task_ids=value["global_task_ids"], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=value["video_schedule_seed"],
        init_state_ids=STATES, video_pool=range(50))
    result["evaluation_role"] = ROLE
    return result


def task_rows(asset_root: Path, spec: Mapping) -> tuple[list[dict], list[dict]]:
    _, manifest = load_task_authorities(asset_root, spec["source"]["data_protocol"])
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    selected, conditions = [], []
    for global_id in registration()["global_task_ids"]:
        original = rows[global_id]
        if original["split_role"] != "train":
            raise ValueError("seen-task teacher is outside actual train allowlist")
        episodes = planned_episodes(selection(), global_id)
        if len(episodes) != 4 or len({row["teacher_demo_indices"][0] for row in episodes}) != 4:
            raise ValueError("seen-task four-state video schedule changed")
        selected.append({"global_task_id": global_id, "suite": original["suite"],
                         "task_id": original["task_id"], "language": original["language"],
                         "split_role": "train",
                         "episodes": episodes})
        conditions.extend({"condition_id": episode["condition_id"],
                           "global_task_id": global_id,
                           "teacher_demo": episode["teacher_demo_indices"][0]}
                          for episode in episodes)
    if len(selected) != 36 or len(conditions) != 144:
        raise ValueError("seen-task registered size changed")
    return selected, conditions


def capture_registration(tasks: list) -> tuple[list[dict], Path, Path]:
    scope = registration()
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": 32}
            for task in tasks]
    expected = [{"suite": suite, "task_id": task, "init_state_id": 32}
                for suite, task in task_keys_from_ids(scope["global_task_ids"])]
    if full != expected:
        raise ValueError("seen-task full capture cases changed")
    return full, CAPTURE_PATH, Path(scope["run_root"])


def capture_expectations(bank: Mapping, bank_path: Path, tasks: list,
                         output_dir: Path | None = None) -> dict:
    """One registered geometry for old400 and the new 36-by-4 panel."""
    from . import bank as owner

    seen = bank.get("evaluation_scope") is not None
    macro = bank_path.parent.name
    eval_root = bank_path.parent.parent.parent / "evaluation"
    if seen:
        full, capture, root = capture_registration(tasks)
        expected_bank = root / bank["mode"] / "banks" / macro / "manifest.json"
        if bank["mode"] not in ("T", "MT"):
            raise ValueError("seen-task capture arm changed")
        canonical = eval_root / "correct144"
        repair = eval_root / "attempts/role_authority_repair/correct144"
        admission_fix = eval_root / "attempts/gpu_admission_fix/correct144"
        if output_dir is not None and output_dir.resolve() not in (
                canonical.resolve(), repair.resolve(), admission_fix.resolve()):
            raise ValueError("seen-task evaluation output is outside the registered attempts")
        if output_dir is not None and output_dir.resolve() == repair.resolve() and bank["mode"] != "T":
            raise ValueError("seen-task role-authority repair belongs only to the failed T queue")
        return dict(full=full, capture=capture, study=registration()["study_id"],
                    output=output_dir if output_dir is not None else canonical,
                    role=ROLE, states=STATES,
                    task_count=36, expected_bank=expected_bank)
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": 0}
            for task in tasks]
    public_beta = bank.get("mode") == owner.PUBLIC_BETA_MODE
    pilot = bank.get("mode") in owner.PILOT_ARMS
    return dict(full=full,
                capture=(owner.PUBLIC_BETA_CAPTURE_PATH if public_beta else
                         owner.PILOT_CAPTURE_PATH if pilot else
                         owner.SPEC_PATH.parent / "official_capture.json"),
                study=(owner.PUBLIC_BETA_STUDY if public_beta else
                       "operator_public_function_pilot_20260929" if pilot else
                       "operator_read_write_learning_20260928"),
                output=(eval_root / "correct400" if macro in ("270", "300") else
                        eval_root / macro / "correct400"),
                role="validation", states=tuple(range(50)), task_count=8,
                expected_bank=(owner.PUBLIC_BETA_ROOT / owner.PUBLIC_BETA_MODE / "banks/1800/manifest.json"
                               if public_beta else owner.PILOT_ROOT / bank["mode"] / "banks/1890/manifest.json"
                               if pilot else None))


def task_keys_from_ids(ids) -> tuple[tuple[str, int], ...]:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    return tuple((suites[task // 10], task % 10) if task < 40 else ("libero_90", task - 40)
                 for task in ids)


def inspect_bank_scope(bank: Mapping, spec: Mapping, path: Path, source: Mapping,
                       task_keys: tuple, evaluation_role: str, require_formal: bool,
                       task_init_state_ids: Mapping | None) -> Path:
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.writer.materialization import file_record

    from . import bank as owner

    scope = registration()
    if read_json(FROZEN_SCOPE_PATH) != scope:
        raise ValueError("seen-task evaluation-only scope differs from frozen bank source")
    mode = bank.get("mode")
    if mode not in ("T", "MT"):
        raise ValueError("seen-task bank arm changed")
    macro = 1800 if mode == "T" else 300
    scene_root = Path(scope["run_root"]) / "scenes"
    tasks, conditions = task_rows(Path(bank["asset_root"]), spec)
    expected = (
        (path, Path(scope["run_root"]) / mode / "banks" / str(macro) / "manifest.json"),
        (bank.get("schema_version"), owner.BANK_SCHEMA), (bank.get("kind"), owner.KIND),
        (bank.get("evaluation_scope"), file_record(FROZEN_SCOPE_PATH)),
        (bank.get("materialization_git", {}).get("branch"), ""),
        (bank.get("materialization_git", {}).get("dirty_paths"), []),
        (bank.get("spec"), file_record(owner.SEALED_SPEC_PATH if mode == "MT" else
                                       owner.CONTINUATION1800_FROZEN_SPEC_PATH)),
        (bank.get("checkpoint"), scope["checkpoint_mt" if mode == "MT" else "checkpoint_t"]),
        (bank.get("source"), source), (bank.get("scene_root"), str(scene_root)),
        (bank.get("tasks"), tasks), (len(bank.get("conditions", ())), 144),
        (evaluation_role, ROLE), (require_formal, True),
        (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}),
        (bank.get("information_wall"), {"teacher_video_values_read": 0 if mode == "MT" else 144,
                                         "teacher_runtime_reads": 0, "deployment_adapters": 1,
                                         "validation_test_gradients": False}),
    )
    if (not owner.source_matches(bank["source"], source)
            or not bank.get("materialization_git", {}).get("commit")
            or bank.get("materialization_git", {}).get("pushed_ref") not in
                ("origin/codex/demonstration-transfer", "origin/main")
            or any(actual != wanted for actual, wanted in expected)
            or any({key: row[key] for key in ("condition_id", "global_task_id", "teacher_demo")}
                   != condition for row, condition in zip(bank["conditions"], conditions, strict=True))
            or task_init_state_ids is not None and any(
                tuple(task_init_state_ids.get((row["suite"], row["task_id"]), ())) != STATES
                for row in tasks)):
        raise ValueError("seen-task bank/source/task/condition scope changed")
    if mode == "T":
        materialization = read_json(path.parent / "materialization_contract.json")
        if any(materialization.get(key) != bank.get(key) for key in
               ("checkpoint", "spec", "training_git", "source", "lora",
                "evaluation_scope", "materialization_git")):
            raise ValueError("seen-task materialization lineage changed")
    inspect_registered_scenes(scene_root, tasks, states=STATES,
                              schema="ember_operator_seen_task_scenes_v1")
    return scene_root


def inspect_official_scope(bank: Mapping, spec: Mapping, path: Path, source: Mapping,
                           task_keys: tuple, evaluation_role: str, require_formal: bool,
                           task_init_state_ids: Mapping | None) -> None:
    """Preserve all original 400-scene reader boundaries during scope extraction."""
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.writer.materialization import file_record

    from . import bank as owner

    mode = bank["mode"]
    macro = 300 if mode == "MT" else int(Path(bank["checkpoint"]).name.split("_")[-1])
    continuation = macro in owner.CONTINUATION_EVALUATION_MACROS
    window1350 = macro in owner.CONTINUATION1350_EVALUATION_MACROS
    window1800 = macro in owner.CONTINUATION1800_EVALUATION_MACROS
    window2340 = macro in owner.CONTINUATION2340_EVALUATION_MACROS
    pilot = macro in owner.PILOT_CHECKPOINTS and mode in owner.PILOT_ARMS
    registered_spec = (owner.CONTINUATION2340_FROZEN_SPEC_PATH if window2340 else
                       owner.PILOT_FROZEN_SPEC_PATH if pilot else
                       owner.CONTINUATION1800_FROZEN_SPEC_PATH if window1800 else
                       owner.CONTINUATION1350_FROZEN_SPEC_PATH if window1350 else
                       owner.CONTINUATION_FROZEN_SPEC_PATH if continuation else owner.SEALED_SPEC_PATH)
    current_spec = owner.specification(owner.CONTINUATION2340_SPEC_PATH if window2340 else
                                       owner.PILOT_SPEC_PATH if pilot else
                                       owner.CONTINUATION1800_SPEC_PATH if window1800 else
                                       owner.CONTINUATION1350_SPEC_PATH if window1350 else
                                       owner.CONTINUATION_SPEC_PATH if continuation else owner.SPEC_PATH)
    expected_path = Path(spec["run_root"]) / mode / "banks" / str(macro) / "manifest.json"
    tasks, conditions = owner.task_rows(spec, Path(bank["asset_root"]))
    expected = (
        (path, expected_path.resolve()), (bank.get("schema_version"), owner.BANK_SCHEMA),
        (bank.get("kind"), owner.KIND), (bank["spec"], file_record(Path(bank["spec"]["path"]))),
        (Path(bank["spec"]["path"]).resolve(), registered_spec.resolve()),
        ({key: value for key, value in spec.items() if key != "budget"},
         {key: value for key, value in current_spec.items() if key != "budget"}),
        (spec.get("schema_version"), "ember_operator_read_write_learning_v1"),
        ("arm" in bank, False),
        (bank["source"], source), (bank["scene_root"], str(owner.SCENE_ROOT)),
        (bank.get("loss_variant") if pilot or window2340 else None,
         owner.PILOT_ARMS[mode] if pilot else "full" if window2340 else None),
        (bank.get("materialization_git", {}).get("branch") if window2340 else None,
         "" if window2340 else None),
        (bank.get("materialization_git", {}).get("dirty_paths") if window2340 else None,
         [] if window2340 else None),
        (bank.get("pilot") if pilot else None, spec["pilot"] if pilot else None),
        (bank.get("parent_checkpoint") if pilot else None,
         str(Path(spec["continuation"]["parent_run_root"]) /
             "T/train/attempts/continuation/checkpoints/macro_00001800") if pilot else None),
        (evaluation_role, "validation"), (require_formal, True), (bank["tasks"], tasks),
        (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}),
        (bank["information_wall"], {"teacher_video_values_read": 0 if mode == "MT" else 400,
                                     "teacher_runtime_reads": 0, "deployment_adapters": 1,
                                     "validation_test_gradients": False}),
    )
    if (mode not in ("T", "U", "MT", *owner.PILOT_ARMS)
            or not owner.source_matches(bank["source"], source)
            or window2340 and (mode != "T" or not bank.get("materialization_git", {}).get("commit")
                               or bank["materialization_git"].get("pushed_ref") not in
                               ("origin/codex/demonstration-transfer", "origin/main"))
            or any(actual != wanted for actual, wanted in expected)
            or len(bank["conditions"]) != 400
            or any({k: row[k] for k in ("condition_id", "global_task_id", "teacher_demo")} != condition
                   for row, condition in zip(bank["conditions"], conditions, strict=True))):
        raise ValueError("operator official bank provenance/scope changed")
    if window2340:
        materialization = read_json(path.parent / "materialization_contract.json")
        if any(materialization.get(key) != bank.get(key) for key in
               ("checkpoint", "spec", "training_git", "source", "lora",
                "materialization_git", "loss_variant")):
            raise ValueError("2340 materialization lineage changed")
    if task_init_state_ids is not None and any(
            tuple(task_init_state_ids.get((row["suite"], row["task_id"]), ())) != tuple(range(50))
            for row in tasks):
        raise ValueError("operator official state scope changed")
    inspect_registered_scenes(owner.SCENE_ROOT, tasks)


def create_scenes(asset_root: Path, gpu_index: int) -> Path:
    """Seal post-dummy complete starts once; both frozen policies restore them."""
    from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
    from ember.pi05_eval.scene import _scene_snapshot, inspect_registered_scenes, scene_path
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_installed_target_tasks

    scope = registration()
    root = Path(scope["run_root"])
    scene_root = root / "scenes"
    scene_root.mkdir(parents=True, exist_ok=True)
    authorities = load_evaluation_authorities(
        asset_root / "configs/libero_24_8_8_coverage_v1/evaluation.json", asset_root)
    tasks, paths = inspect_installed_target_tasks(
        authorities, role=ROLE, state_count=4, libero_config_dir=root / "scene_libero_config")
    if tuple((row.suite, row.task_id) for row in tasks) != task_keys(
            authorities.protocol, authorities.meta_protocol):
        raise ValueError("seen-task installed scene keys changed")
    contract = {"libero_paths": paths, "environment": authorities.config["environment"],
                "parallel": {"envs_per_replica": 1}}
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=gpu_index)
    dummy = np.asarray(authorities.config["environment"]["dummy_action"], dtype=np.float64)
    rows = []
    try:
        for task in tasks:
            envs, init_states = pool.switch(vars(task))
            env = envs[0]
            for state in STATES:
                path = scene_path(scene_root, vars(task), state)
                if not path.exists():
                    env.seed(7)
                    env.reset()
                    observation = env.set_init_state(init_states[state])
                    for _ in range(10):
                        observation, _, _, _ = env.step(dummy)
                    names = sorted(env.env.obj_body_id)
                    goals = [list(row) for row in env.env.parsed_problem["goal_state"]]
                    snapshot = _scene_snapshot(env, observation, names, goals, image=True)
                    temporary = path.with_suffix(".partial.npz")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **snapshot)
                    os.replace(temporary, path)
                rows.append({"suite": task.suite, "task_id": task.task_id, "state": state,
                             "path": str(path), "bytes": path.stat().st_size})
    finally:
        pool.close()
    manifest = scene_root / "manifest.json"
    expected = {"schema_version": "ember_operator_seen_task_scenes_v1", "seed": 7,
                "dummy_steps": 10, "scope": scope["schema_version"], "scenes": rows}
    if manifest.exists():
        if read_json(manifest) != expected:
            raise ValueError("published seen-task scene registry changed")
    else:
        write_json_atomic(manifest, expected)
    inspect_registered_scenes(scene_root, [vars(task) for task in tasks],
                              states=STATES, schema="ember_operator_seen_task_scenes_v1")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", choices=("scenes",))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    args = parser.parse_args()
    print(create_scenes(args.asset_root, args.gpu_index))


if __name__ == "__main__":
    main()
