"""One registered repair of the seen-task scene observation snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

from . import scope


ROOT = Path(scope.registration()["run_root"])
ATTEMPT = ROOT / "attempts/scene_canonical144"
SCENES = ATTEMPT / "scenes"
SCHEMA = "ember_operator_seen_scene_canonicalization_v1"


def source_bank_path(mode: str) -> Path:
    if mode not in ("T", "MT"):
        raise ValueError("canonical scene repair has only T1800 and MT300")
    return ROOT / mode / "banks" / ("1800" if mode == "T" else "300") / "manifest.json"


def bank_path(mode: str) -> Path:
    return ATTEMPT / mode / "banks" / ("1800" if mode == "T" else "300") / "manifest.json"


def _lineage(mode: str) -> dict:
    return {"schema_version": SCHEMA,
            "source_bank": file_record(source_bank_path(mode)),
            "source_scene_manifest": file_record(ROOT / "scenes/manifest.json"),
            "canonical_scene_manifest": file_record(SCENES / "manifest.json"),
            "teacher_factors_recomputed": False,
            "physical_state_source": "same_saved_post_dummy_sim_model_controller"}


def create_scenes(asset_root: Path, gpu_index: int) -> Path:
    """Reobserve each saved physical state, then prove strict restore before publication."""
    from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
    from ember.pi05_eval.scene import (_assert_scene_pair, _restore_scene,
                                       _scene_snapshot, inspect_registered_scenes, scene_path)
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_installed_target_tasks
    from .run import frozen_git

    registered = scope.registration()
    old_manifest = read_json(ROOT / "scenes/manifest.json")
    authorities = load_evaluation_authorities(
        asset_root / "configs/libero_24_8_8_coverage_v1/evaluation.json", asset_root)
    tasks, paths = inspect_installed_target_tasks(
        authorities, role=scope.ROLE, state_count=4,
        libero_config_dir=ROOT / "scene_libero_config")
    if tuple((row.suite, row.task_id) for row in tasks) != scope.task_keys(
            authorities.protocol, authorities.meta_protocol):
        raise ValueError("canonical scene repair task allowlist changed")
    inspect_registered_scenes(ROOT / "scenes", [vars(task) for task in tasks],
                              states=scope.STATES,
                              schema="ember_operator_seen_task_scenes_v1")
    contract = {"libero_paths": paths, "environment": authorities.config["environment"],
                "parallel": {"envs_per_replica": 1}}
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=gpu_index)
    dummy = np.asarray(authorities.config["environment"]["dummy_action"], dtype=np.float64)
    SCENES.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for task in tasks:
            envs, init_states = pool.switch(vars(task))
            env = envs[0]
            names = sorted(env.env.obj_body_id)
            goals = [list(row) for row in env.env.parsed_problem["goal_state"]]
            for state in scope.STATES:
                old_path = scene_path(ROOT / "scenes", vars(task), state)
                with np.load(old_path, allow_pickle=False) as source:
                    old = {name: source[name] for name in source.files}
                env.seed(7)
                env.reset()
                observation = env.set_init_state(init_states[state])
                for _ in range(10):
                    observation, _, _, _ = env.step(dummy)
                observation = _restore_scene(env, old)
                canonical = _scene_snapshot(env, observation, names, goals, image=True)
                for key in ("model_body_names", "model_body_pos", "model_body_quat",
                            "sim_state", "controller_goal_pos", "controller_goal_ori"):
                    if not np.array_equal(canonical[key], old[key]):
                        raise ValueError(f"canonical scene changed saved physical state: {key}")
                _assert_scene_pair(env, _restore_scene(env, canonical), names, goals,
                                   canonical, image=True)
                path = scene_path(SCENES, vars(task), state)
                if path.exists():
                    with np.load(path, allow_pickle=False) as previous:
                        if set(previous.files) != set(canonical) or any(
                                not np.array_equal(previous[key], canonical[key])
                                for key in canonical):
                            raise ValueError("partial canonical scene differs from this saved source")
                else:
                    temporary = path.with_suffix(".partial.npz")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **canonical)
                    temporary.replace(path)
                rows.append({"suite": task.suite, "task_id": task.task_id,
                             "state": state, "path": str(path), "bytes": path.stat().st_size})
    finally:
        pool.close()
    if len(rows) != 144 or old_manifest.get("schema_version") != "ember_operator_seen_task_scenes_v1":
        raise ValueError("canonical scene repair did not cover all old cases")
    manifest = SCENES / "manifest.json"
    expected = {"schema_version": "ember_operator_seen_task_scenes_v1", "seed": 7,
                "dummy_steps": 10, "scope": registered["schema_version"],
                "canonicalization": SCHEMA, "source_manifest": file_record(ROOT / "scenes/manifest.json"),
                "evaluation_git": frozen_git(continuation=True), "scenes": rows}
    if manifest.exists():
        if read_json(manifest) != expected:
            raise ValueError("published canonical scene manifest differs")
    else:
        write_json_atomic(manifest, expected)
    inspect_registered_scenes(SCENES, [vars(task) for task in tasks],
                              states=scope.STATES, schema="ember_operator_seen_task_scenes_v1")
    return manifest


def register_bank(mode: str) -> Path:
    """Bind old verified factor files to the corrected scene without copying them."""
    from .run import frozen_git

    manifest = read_json(SCENES / "manifest.json")
    materialization_git = frozen_git(continuation=True)
    if (manifest.get("canonicalization") != SCHEMA
            or manifest.get("evaluation_git") != materialization_git):
        raise ValueError("canonical scene manifest is not published")
    old = read_json(source_bank_path(mode))
    if old.get("mode") != mode or old.get("evaluation_scope") is None:
        raise ValueError("canonical scene repair source bank changed")
    output = bank_path(mode)
    wanted = {**old, "scene_root": str(SCENES),
              "materialization_git": materialization_git,
              "scene_repair": _lineage(mode)}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if read_json(output) != wanted:
            raise ValueError("published canonical bank differs")
    else:
        write_json_atomic(output, wanted)
    return output


def inspect_bank_scope(bank: Mapping, spec: Mapping, path: Path, source: Mapping,
                       task_keys: tuple, evaluation_role: str, require_formal: bool,
                       task_init_state_ids: Mapping | None) -> Path:
    from ember.pi05_eval.scene import inspect_registered_scenes

    mode = bank.get("mode")
    if path != bank_path(mode).resolve() or bank.get("scene_repair") != _lineage(mode):
        raise ValueError("canonical scene bank path or source lineage changed")
    old_path = source_bank_path(mode)
    old = read_json(old_path)
    scope.inspect_bank_scope(old, spec, old_path, source, task_keys,
                             evaluation_role, require_formal, task_init_state_ids)
    published_git = read_json(SCENES / "manifest.json").get("evaluation_git")
    if (not isinstance(published_git, dict)
            or bank.get("materialization_git") != published_git
            or published_git.get("branch") != ""
            or published_git.get("dirty_paths") != []
            or published_git.get("pushed_ref") not in
            ("origin/codex/demonstration-transfer", "origin/main")):
        raise ValueError("canonical scene/bank frozen Git lineage changed")
    wanted = {**old, "scene_root": str(SCENES),
              "materialization_git": published_git,
              "scene_repair": _lineage(mode)}
    if bank != wanted:
        raise ValueError("canonical scene bank changed old LoRA factors or task map")
    inspect_registered_scenes(SCENES, old["tasks"], states=scope.STATES,
                              schema="ember_operator_seen_task_scenes_v1")
    return SCENES


def capture_expectations(bank: Mapping, bank_path_value: Path, tasks: list,
                         output_dir: Path | None) -> dict:
    full, capture, _ = scope.capture_registration(tasks)
    output = ATTEMPT / bank["mode"] / "evaluation/correct144"
    if output_dir is not None and output_dir.resolve() != output.resolve():
        raise ValueError("canonical scene evaluation output is outside its registered attempt")
    return {"full": full, "capture": capture, "study": scope.registration()["study_id"],
            "output": output, "role": scope.ROLE, "states": scope.STATES,
            "task_count": 36, "expected_bank": bank_path(bank["mode"]).resolve()}
