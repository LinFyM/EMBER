"""Sealed two-task original-rollout queries and crossed materialized conditions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.pi05_assets import Pi05EvaluationError, prepare_libero_config
from ember.pi05_eval.approach_channel_replay import _osc_audit
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.episode import start_fixed_episode
from ember.pi05_eval.frozen_prefix import POLICY_PANEL, TASK_KEYS, _original_panel
from ember.pi05_eval.prefix_replay import _controller_state, _roles, _state_error
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_processing import libero_policy_input


SCHEMA = "ember_crossed_video_field_manifest_v1"
QUERY_SCHEMA = "ember_crossed_video_field_query_v1"
SPEC_RELATIVE = Path("configs/crossed_video_action_field_v1/experiment_spec.json")
TASK_GLOBAL = {("libero_object", 4): 14, ("libero_goal", 1): 21}
STEPS = (0, 10, 20)
SOURCE_PANELS = ("C_correct", "B", "C_wrong")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Pi05EvaluationError(f"crossed-video JSON authority invalid: {path}")
    return value


def _asset(condition: Mapping[str, Any]) -> dict[str, Any]:
    adapter = condition["adapter"]
    path = Path(adapter["path"])
    if not path.is_file() or path.stat().st_size != int(adapter["bytes"]):
        raise Pi05EvaluationError("crossed-video sealed adapter file changed")
    return {
        "condition_id": condition["condition_id"],
        "global_task_id": int(condition["global_task_id"]),
        "teacher_demo_indices": list(condition["teacher_demo_indices"]),
        "adapter": dict(adapter),
    }


def _authorities(repo_root: Path, study_root: Path, formal: bool):
    spec_path = repo_root / SPEC_RELATIVE
    spec = read_json(spec_path)
    expected_tasks = [
        {"global_task_id": 14, "suite": "libero_object", "task_id": 4},
        {"global_task_id": 21, "suite": "libero_goal", "task_id": 1},
    ]
    if (spec.get("schema_version") != "ember_crossed_video_action_field_spec_v1"
            or spec.get("tasks") != expected_tasks
            or spec.get("state_ids") != list(range(50))
            or spec.get("query_control_steps") != list(STEPS)
            or spec.get("correct_predictions") != 15000
            or spec.get("total_predictions") != 15900
            or formal and study_root.resolve() != Path(spec["run_root"]).resolve()):
        raise Pi05EvaluationError("crossed-video frozen specification changed")
    reference = Path(spec["reference_root"])
    c_contract = read_json(reference / "evaluation" / POLICY_PANEL["C_correct"] / "run_contract.json")
    model = c_contract["model"]
    keys = tuple((row["suite"], int(row["task_id"])) for row in c_contract["tasks"])
    if len(keys) != 8 or set(TASK_KEYS) - set(keys):
        raise Pi05EvaluationError("crossed-video original held8 scope changed")
    return spec_path, reference, c_contract, model, keys


def _panels(reference: Path, model: Mapping[str, Any], keys: tuple):
    panels = {}
    for name in SOURCE_PANELS:
        contract, rows, results_path = _original_panel(
            root=reference, name=POLICY_PANEL[name], model=model, keys=keys,
        )
        bank_path = reference / "materialization" / POLICY_PANEL[name] / "manifest.json"
        tasks = tuple(SimpleNamespace(suite=s, task_id=t, init_state_ids=tuple(range(50)))
                      for s, t in keys)
        inspected = inspect_static_task_lora_adapter(
            manifest_path=bank_path, source=model, tasks=tasks,
            evaluation_role="development_train", require_formal=True,
        )
        if inspected != contract["adapter"]:
            raise Pi05EvaluationError(f"crossed-video {name} bank differs from original evaluation")
        panels[name] = {
            "contract": contract, "rows": rows, "results_path": results_path,
            "bank": read_json(bank_path), "bank_path": bank_path,
        }
    return panels


def _bank_conditions(panels: Mapping[str, Any]) -> dict[str, Any]:
    banks = {}
    for name, panel in panels.items():
        conditions = {row["condition_id"]: row for row in panel["bank"]["conditions"]}
        selected = {}
        for key in TASK_KEYS:
            global_id = TASK_GLOBAL[key]
            episodes = next(row["episodes"] for row in panel["bank"]["tasks"]
                            if int(row["global_task_id"]) == global_id)
            if len(episodes) != 50 or {int(e["init_state_id"]) for e in episodes} != set(range(50)):
                raise Pi05EvaluationError(f"crossed-video {name} state mapping changed")
            state_conditions = {int(e["init_state_id"]): _asset(conditions[e["condition_id"]])
                                for e in episodes}
            if name == "C_correct":
                pool = list(state_conditions.values())
                demos = [item["teacher_demo_indices"][0] for item in pool]
                if sorted(demos) != list(range(50)) or len({x["condition_id"] for x in pool}) != 50:
                    raise Pi05EvaluationError("crossed-video correct50 pool changed")
                selected[str(global_id)] = {
                    "pool_by_video_demo": sorted(pool, key=lambda item: item["teacher_demo_indices"][0]),
                    "per_state": {str(state): state_conditions[state] for state in range(50)},
                }
            else:
                selected[str(global_id)] = {
                    "per_state": {str(state): state_conditions[state] for state in range(50)},
                }
        banks[name] = {
            "manifest": {"path": str(panel["bank_path"]),
                         "bytes": panel["bank_path"].stat().st_size},
            "lora_contract": panel["bank"]["lora_contract"],
            "selection": panel["bank"]["selection"],
            "conditions": selected,
        }
    if any(banks[name]["lora_contract"] != banks["C_correct"]["lora_contract"]
           for name in SOURCE_PANELS):
        raise Pi05EvaluationError("crossed-video banks have different LoRA topology")
    return banks


def _state_queries(suite: str, task_id: int, state: int,
                   panels: Mapping[str, Any], banks: Mapping[str, Any]) -> list[dict[str, Any]]:
    global_id = TASK_GLOBAL[(suite, task_id)]
    source_rows = {name: panels[name]["rows"][(suite, task_id, state)]
                   for name in SOURCE_PANELS}
    c_row = source_rows["C_correct"]
    original = c_row["occupancy_trajectory"]
    path = Path(original["path"])
    if not path.is_file() or path.stat().st_size != int(original["bytes"]):
        raise Pi05EvaluationError("crossed-video original C trajectory missing")
    correct = banks["C_correct"]["conditions"][str(global_id)]
    diagonal = correct["per_state"][str(state)]
    if c_row["horizon_writer_lora"]["condition_id"] != diagonal["condition_id"]:
        raise Pi05EvaluationError("crossed-video C diagonal changed")
    other_demo = c_row["horizon_writer_lora"]["paired_other_demos"]
    if len(other_demo) != 1 or other_demo[0] == diagonal["teacher_demo_indices"][0]:
        raise Pi05EvaluationError("crossed-video scheduled other mapping changed")
    other = correct["pool_by_video_demo"][other_demo[0]]
    if other["teacher_demo_indices"] != other_demo:
        raise Pi05EvaluationError("crossed-video other is absent from correct50 pool")
    for name in ("B", "C_wrong"):
        actual = source_rows[name]["horizon_writer_lora"]["condition_id"]
        expected = banks[name]["conditions"][str(global_id)]["per_state"][str(state)]["condition_id"]
        if actual != expected:
            raise Pi05EvaluationError(f"crossed-video {name} original state map changed")
    result = []
    for step in STEPS:
        index = step // 5
        seed = policy_noise_seed(7, suite, task_id, state, index)
        if c_row["policy_noise_seeds"][index] != seed:
            raise Pi05EvaluationError("crossed-video original C RNG changed")
        result.append({
            "query_id": f"{suite}:{task_id}:{state}:{step}",
            "global_task_id": global_id, "suite": suite, "task_id": task_id,
            "init_state_id": state, "control_step": step, "global_replan_index": index,
            "policy_noise_seed": seed, "language": c_row["language"],
            "C_original_trajectory": dict(original),
            "C_original_results_path": str(panels["C_correct"]["results_path"]),
            "C_diagonal_condition_id": diagonal["condition_id"],
            "C_scheduled_other_condition_id": other["condition_id"],
            "B_original_trajectory": dict(source_rows["B"]["occupancy_trajectory"]),
            "B_condition_id": source_rows["B"]["horizon_writer_lora"]["condition_id"],
            "wrong_condition_id": source_rows["C_wrong"]["horizon_writer_lora"]["condition_id"],
        })
    return result


def build_manifest(repo_root: Path, study_root: Path, *, formal: bool) -> dict[str, Any]:
    spec_path, reference, c_contract, model, keys = _authorities(repo_root, study_root, formal)
    panels = _panels(reference, model, keys)
    banks = _bank_conditions(panels)
    query_plan = [query for suite, task_id in TASK_KEYS for state in range(50)
                  for query in _state_queries(suite, task_id, state, panels, banks)]
    if len(query_plan) != 300 or len({row["query_id"] for row in query_plan}) != 300:
        raise Pi05EvaluationError("crossed-video query manifest incomplete")
    return {
        "schema_version": SCHEMA, "spec_path": str(spec_path),
        "spec_bytes": spec_path.stat().st_size,
        "reference_root": str(reference), "model": model,
        "normalization": c_contract["normalization"],
        "tokenizer": c_contract["tokenizer"], "policy": c_contract["policy"],
        "environment": c_contract["environment"], "libero_paths": c_contract["libero_paths"],
        "tasks": [row for row in c_contract["tasks"]
                  if (row["suite"], int(row["task_id"])) in TASK_KEYS],
        "original_C_contract_path": str(reference / "evaluation" /
                                         POLICY_PANEL["C_correct"] / "run_contract.json"),
        "banks": banks, "query_plan": query_plan,
        "information_wall": {
            "original_saved_C_environment_actions_only": True,
            "new_teacher_or_expert_numeric_reads": 0,
            "predicted_action_environment_steps": 0,
            "training_or_checkpoint_selection": False,
        },
    }


def _saved_c_actions(records: Mapping[int, Mapping[str, Any]],
                     suite: str, task_id: int, state_id: int):
    import torch

    source = records[0]["C_original_trajectory"]
    original = torch.load(source["path"], map_location="cpu", weights_only=True)
    seeds = tuple(policy_noise_seed(7, suite, task_id, state_id, i) for i in range(5))
    if (original["schema_version"] != "ember_pi05_diagnostic_trajectory_v2"
            or original["suite"] != suite or int(original["task_id"]) != task_id
            or int(original["init_state_id"]) != state_id
            or len(original["states"]) < 5 or len(original["executed_action_prefixes"]) < 4
            or tuple(original["replan_steps"][:5]) != (0, 5, 10, 15, 20)
            or tuple(original["policy_noise_seeds"][:5]) != seeds):
        raise Pi05EvaluationError("crossed-video original C saved trajectory changed")
    actions = np.concatenate([
        np.asarray(value, dtype=np.float32)
        for value in original["executed_action_prefixes"][:4]
    ], axis=0)
    if actions.shape != (20, 7) or not np.isfinite(actions).all():
        raise Pi05EvaluationError("crossed-video original C environment commands invalid")
    return original, actions


def _capture_query(*, env: Any, obs: Mapping[str, Any], task: Mapping[str, Any],
                   row: Mapping[str, Any], original: Mapping[str, Any],
                   study_root: Path, names: tuple[str, ...],
                   roles: Mapping[str, Any]) -> dict[str, Any]:
    step = int(row["control_step"])
    raw = libero_policy_input(obs, task["language"])
    error = _state_error(raw["observation.state"], original["states"][step // 5], step=step)
    owner = env.env
    positions = np.stack([
        np.asarray(owner.sim.data.body_xpos[owner.obj_body_id[name]],
                   dtype=np.float32).copy() for name in names
    ])
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy()
    stem = (f"{row['suite']}_task_{row['task_id']:02d}_"
            f"state_{row['init_state_id']:03d}_step_{step:02d}")
    path = study_root / "queries" / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, schema_version=np.asarray(QUERY_SCHEMA),
            agentview_image=np.asarray(obs["agentview_image"], dtype=np.uint8),
            wrist_image=np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
            state8=np.asarray(raw["observation.state"], dtype=np.float32),
            sim_state=np.asarray(env.get_sim_state(), dtype=np.float64),
            object_names=np.asarray(names), object_positions=positions,
            eef_pos=eef,
            eef_quat=np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            target_center_direction=positions[names.index(roles["target"])] - eef,
        )
    return {
        **row, "query_path": str(path), "query_bytes": path.stat().st_size,
        "state_replay_error": error, "target_body": roles["target"],
        "object_roles": roles, "controller_state": _controller_state(env),
        "osc_channel_audit": _osc_audit(env),
        "observation_source": "canonical reset+settling10+original C environment actions",
        "teacher_or_expert_numeric_reads": 0,
    }


def acquire_queries(manifest: Mapping[str, Any], study_root: Path, *,
                    smoke: bool = False) -> list[dict[str, Any]]:
    if manifest["schema_version"] != SCHEMA or len(manifest["query_plan"]) != 300:
        raise Pi05EvaluationError("crossed-video query authority changed")
    paths = prepare_libero_config(study_root / "launch" / "libero_config")
    if paths != manifest["libero_paths"]:
        raise Pi05EvaluationError("crossed-video installed LIBERO paths changed")
    contract = {
        "environment": manifest["environment"], "libero_paths": paths,
        "parallel": {"envs_per_replica": 1},
    }
    physical_gpu_id = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=physical_gpu_id)
    tasks = {(row["suite"], int(row["task_id"])): row for row in manifest["tasks"]}
    planned = {(row["suite"], int(row["task_id"]), int(row["init_state_id"])): {}
               for row in manifest["query_plan"]}
    for row in manifest["query_plan"]:
        planned[(row["suite"], int(row["task_id"]), int(row["init_state_id"]))][row["control_step"]] = row
    index = []
    try:
        for suite, task_id in TASK_KEYS:
            if smoke and (suite, task_id) != ("libero_goal", 1):
                continue
            task = tasks[(suite, task_id)]
            envs, init_states = pool.switch(task)
            env = envs[0]
            for state_id in range(50):
                if smoke and state_id != 0:
                    continue
                records = planned[(suite, task_id, state_id)]
                original, actions = _saved_c_actions(records, suite, task_id, state_id)
                dummy = np.asarray(manifest["environment"]["dummy_action"], dtype=np.float32)
                slot = start_fixed_episode(
                    env=env, init_state_id=state_id, init_states=init_states,
                    task=task, contract=contract, root_seed=7, dummy=dummy,
                    task_adapter=None, capture_level=None,
                )
                obs = slot["obs"]
                names, roles = _roles(env, task)
                for step in range(21):
                    if step in STEPS:
                        index.append(_capture_query(
                            env=env, obs=obs, task=task, row=records[step],
                            original=original, study_root=study_root,
                            names=names, roles=roles,
                        ))
                    if step < 20:
                        obs, _, _, _ = env.step(actions[step])
    finally:
        pool.close()
    if len(index) != (3 if smoke else 300):
        raise Pi05EvaluationError("crossed-video query acquisition incomplete")
    return index
