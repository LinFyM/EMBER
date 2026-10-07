"""Train-only privileged CPU controller for the fixed 144-condition NN study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json, source_reference_matches


KIND = "privileged_action_memory_controller"
STUDY = "privileged_action_memory_control_20261007"
MANIFEST_SCHEMA = "ember_privileged_action_memory_manifest_v1"
EVAL_SCHEMA = "ember_privileged_action_memory_evaluation_v1"
EPISODE_SCHEMA = "ember_privileged_action_memory_episode_v1"
PASSIVE_TAG = "ember_privileged_action_memory_passive_capture_v1"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
SCENE_ROOT = Path("/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929"
                  "/attempts/scene_canonical144/scenes")


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size}


def _expected_conditions() -> dict[tuple[str, int, int], dict[str, Any]]:
    from ember.operator_writer.scope import registration, selection, task_keys_from_ids
    from ember.writer.materialization import planned_episodes

    scope = registration()
    result = {}
    for global_id, (suite, task_id) in zip(
        scope["global_task_ids"], task_keys_from_ids(scope["global_task_ids"]), strict=True
    ):
        for episode in planned_episodes(selection(), global_id):
            result[(suite, task_id, int(episode["init_state_id"]))] = {
                "global_task_id": global_id, "condition_id": episode["condition_id"],
                "teacher_demo": episode["teacher_demo_indices"][0],
            }
    return result


def _cache_records(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    result = {}
    for name in ("geometry_path", "source_path") if arm == "source_action" else ("geometry_path",):
        path = Path(row[name]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise Pi05EvaluationError("action-memory cache is missing or outside the study root")
        result[name] = _file_record(path)
    return result


def _validate_study_metadata(value: Mapping[str, Any], path: Path, source: Mapping[str, Any],
                             role: str, formal: bool) -> None:
    from ember.operator_writer.scope import ROLE

    if (not path.is_relative_to(ROOT) or value.get("schema_version") != MANIFEST_SCHEMA
            or value.get("kind") != KIND or value.get("study_id") != STUDY
            or value.get("root") != str(ROOT) or value.get("scene_root") != str(SCENE_ROOT)
            or value.get("arm") not in ("demo_action", "source_action")
            or role != ROLE or not formal
            or source.get("optimizer_step") != 1000
            or not source_reference_matches(value.get("source"), source)):
        raise Pi05EvaluationError("privileged action-memory study/source/role changed")
    fixed = {"geometry_definition_git": "9f90a14d", "action_offset": 1,
             "position_scale_m": .10, "gripper_scale_m": .04, "video_schedule_seed": 20260928,
             "source_noise_seed": 1729, "source_noise_shape": [50, 32], "source_inference_steps": 10,
             "privileged_training_diagnostic": True, "deployment_candidate": False,
             "held_or_test_reads": 0, "gradient_updates": 0, "policy_noise_used": False}
    if any(value.get(key) != wanted for key, wanted in fixed.items()):
        raise Pi05EvaluationError("action-memory fixed geometry/value/information-wall metadata changed")


def _signature(row: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    signature = tuple(tuple(point) for point in row["signature"])
    if (not signature or len(set(signature)) != len(signature)
            or any(len(point) != 3 or point[1] not in ("body", "site")
                   or not all(isinstance(item, str) and item for item in point) for point in signature)):
        raise Pi05EvaluationError("action-memory object body/site signature changed")
    return signature


def _validate_condition(row: Mapping[str, Any], expected: Mapping[str, Any], task: Any) -> None:
    if (any(row.get(name) != wanted for name, wanted in expected.items())
            or row.get("language") != task.language
            or list(row["frames"]) != list(range(0, int(row["length"]) - 5, 5))
            or not row["frames"]):
        raise Pi05EvaluationError("action-memory teacher mapping or legal future frames changed")


def inspect_manifest(
    *, manifest_path: Path, source: Mapping[str, Any], tasks: Sequence[Any],
    evaluation_role: str, require_formal: bool,
) -> dict[str, Any]:
    """Inspect only metadata and registered identities; cache arrays load per episode."""
    from ember.operator_writer.scope import STATES
    from ember.pi05_eval.scene import inspect_registered_scenes

    path = manifest_path.resolve()
    value = read_json(path)
    _validate_study_metadata(value, path, source, evaluation_role, require_formal)
    expected = _expected_conditions()
    task_by_key = {(str(task.suite), int(task.task_id)): task for task in tasks}
    if (len(task_by_key) != 36 or len(tasks) != 36
            or set(task_by_key) != {key[:2] for key in expected}
            or any(tuple(task.init_state_ids) != STATES for task in tasks)):
        raise Pi05EvaluationError("privileged action-memory evaluation requires the original seen144")
    conditions = value.get("conditions", ())
    seen, signatures, records = set(), {}, {}
    for row in conditions:
        key = (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        signature = _signature(row)
        if key in seen or key not in expected:
            raise Pi05EvaluationError("action-memory condition is repeated or outside seen144")
        _validate_condition(row, expected[key], task_by_key[key[:2]])
        if key[:2] in signatures and signatures[key[:2]] != signature:
            raise Pi05EvaluationError("action-memory signature changed within one task")
        seen.add(key)
        signatures[key[:2]] = signature
        records[row["condition_id"]] = _cache_records(row, value["arm"])
    if seen != set(expected) or len(conditions) != 144:
        raise Pi05EvaluationError("privileged action-memory manifest does not cover seen144")
    inspect_registered_scenes(SCENE_ROOT, [dict(suite=k[0], task_id=k[1]) for k in task_by_key],
                              states=STATES, schema="ember_operator_seen_task_scenes_v1")
    return {**value, "schema_version": EVAL_SCHEMA, "evaluation_role": evaluation_role,
            "manifest": _file_record(path), "scene_manifest": _file_record(SCENE_ROOT / "manifest.json"),
            "cache_files": records, "privileged": True, "training_tasks_only": True,
            "ember_score": False, "rollout_model_loaded": False, "policy_noise_consumed": False}


def _capture_contract(adapter: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]],
                      output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "selection_path": adapter["manifest"]["path"],
        "selection_bytes": adapter["manifest"]["bytes"], "study_id": STUDY,
        "mode": "compact", "full_conditions": [
            {"suite": task["suite"], "task_id": int(task["task_id"]), "init_state_id": 32}
            for task in tasks],
        "trajectory_root": str(output_dir / "trajectories"),
        "passive_trace": {"schema_version": PASSIVE_TAG,
                          "trace_root": str(output_dir / "continuous_traces")},
        "privileged": True, "training_tasks_only": True, "ember_score": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return capture, stage


def attach_contract(contract: dict[str, Any]) -> None:
    adapter = contract["adapter"]
    policy = contract["policy"]
    output_dir = Path(contract["output_dir"]).resolve()
    if (not output_dir.is_relative_to(ROOT) or contract["mode"] != "formal"
            or contract["rng"]["inference_seed"] != 7
            or contract.get("diagnostic_exploration") is not None
            or any(contract.get(name) is not None for name in (
                "diagnostic_occupancy_capture", "diagnostic_stage_predicates", "diagnostic_task_subset"))
            or any(policy.get(key) != value for key, value in {
                "replan_steps": 5, "action_dim": 7, "chunk_size": 50, "num_inference_steps": 10}.items())):
        raise Pi05EvaluationError("action-memory execution or fixed capture scope changed")
    contract["privileged_action_memory_scene"] = {
        "root": adapter["scene_root"], "manifest": adapter["scene_manifest"]}
    capture, stage = _capture_contract(adapter, contract["tasks"], output_dir)
    contract["diagnostic_occupancy_capture"] = capture
    contract["diagnostic_stage_predicates"] = stage
    contract["policy"] = {**policy, "actor": KIND, "num_inference_steps": 0,
                          "source_value_num_inference_steps": 10 if adapter["arm"] == "source_action" else None,
                          "model_loaded": False, "policy_noise_consumed": False}
    contract["parallel"] = {**contract["parallel"], "one_policy_per_worker": False,
                            "cpu_controller_per_worker": True, "gpu_use": "EGL_renderer_context_only"}


def validate_contract(contract: Mapping[str, Any]) -> None:
    from ember.operator_writer.scope import ROLE, STATES

    adapter = contract.get("adapter") or {}
    tasks = contract["tasks"]
    expected = _expected_conditions()
    cases = {(str(task["suite"]), int(task["task_id"]), int(state))
             for task in tasks for state in task["init_state_ids"]}
    capture, stage = _capture_contract(adapter, tasks, Path(contract["output_dir"]).resolve())
    if (adapter.get("kind") != KIND or adapter.get("schema_version") != EVAL_SCHEMA
            or contract.get("role") != ROLE or contract.get("mode") != "formal"
            or not Path(contract["output_dir"]).resolve().is_relative_to(ROOT)
            or cases != set(expected) or len(tasks) != 36
            or any(tuple(task["init_state_ids"]) != STATES for task in tasks)
            or contract.get("diagnostic_occupancy_capture") != capture
            or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("diagnostic_exploration") is not None
            or contract["rng"]["inference_seed"] != 7
            or any(contract["policy"].get(key) != wanted for key, wanted in {
                "actor": KIND, "num_inference_steps": 0, "policy_noise_consumed": False,
                "model_loaded": False, "chunk_size": 50, "action_dim": 7, "replan_steps": 5,
                "source_value_num_inference_steps": 10 if adapter["arm"] == "source_action" else None}.items())
            or contract.get("privileged_action_memory_scene") != {
                "root": str(SCENE_ROOT), "manifest": adapter["scene_manifest"]}):
        raise Pi05EvaluationError("privileged action-memory run/capture contract changed")


def _episode_evidence(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": EPISODE_SCHEMA, "study_id": STUDY, "arm": adapter["arm"],
            **{key: row[key] for key in ("condition_id", "global_task_id", "suite", "task_id",
                                        "init_state_id", "teacher_demo", "hdf5")},
            "cache_files": adapter["cache_files"][row["condition_id"]],
            "privileged": True, "training_tasks_only": True, "ember_score": False,
            "controller": "cpu_absolute_geometry_1nn", "rollout_model_loaded": False,
            "rollout_flow_steps": 0, "policy_noise_consumed": False,
            "proposal_kind": "recorded_future_with_nonexecuted_last_action_padding"
            if adapter["arm"] == "demo_action" else "frozen_source_full50_prediction",
            "action_origin": "recorded_teacher_post_action_offset1" if adapter["arm"] == "demo_action"
            else "frozen_source1000_ode10_common_noise_seed1729"}


def validate_episode(adapter: Mapping[str, Any], evidence: Any, *,
                     suite: str, task_id: int, init_state_id: int) -> bool:
    rows = [row for row in adapter["conditions"] if
            (row["suite"], int(row["task_id"]), int(row["init_state_id"])) ==
            (suite, task_id, init_state_id)]
    return len(rows) == 1 and evidence == _episode_evidence(adapter, rows[0])


@dataclass(frozen=True)
class PreparedActionMemory:
    key: str
    evidence: dict[str, Any]
    signature: tuple[tuple[str, str, str], ...]
    frames: np.ndarray
    absolute: np.ndarray
    actions: np.ndarray
    valid_lengths: np.ndarray


class PrivilegedActionMemoryController:
    """Retrieve an unchanged raw five-action prefix; keep all flow/model work offline."""

    kind = KIND

    def __init__(self, contract: Mapping[str, Any]) -> None:
        validate_contract(contract)
        self.adapter = contract["adapter"]
        if not source_reference_matches(self.adapter.get("source"), contract["model"]):
            raise Pi05EvaluationError("action-memory worker source provenance changed")
        path = Path(contract["normalization"]["path"])
        if not path.is_file() or path.stat().st_size != contract["normalization"]["bytes"]:
            raise Pi05EvaluationError("action-memory frozen normalization changed")
        stats = read_json(path)["stats"]["action"]
        self.low = np.asarray(stats["q01"], dtype=np.float32)
        self.high = np.asarray(stats["q99"], dtype=np.float32)
        if (self.low.shape != (7,) or self.high.shape != (7,)
                or not np.isfinite([self.low, self.high]).all() or (self.high < self.low).any()):
            raise Pi05EvaluationError("action-memory source action quantiles are invalid")
        self.conditions = {(row["suite"], int(row["task_id"]), int(row["init_state_id"])): row
                           for row in self.adapter["conditions"]}
        self.cache: dict[str, PreparedActionMemory] = {}
        self.current_task: tuple[str, int] | None = None

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedActionMemory:
        row = self.conditions.get((suite, task_id, init_state_id))
        if row is None:
            raise Pi05EvaluationError("action-memory episode is outside the registered seen144")
        if self.current_task != (suite, task_id):
            self.cache.clear()
            self.current_task = suite, task_id
        key = row["condition_id"]
        if key not in self.cache:
            if _cache_records(row, self.adapter["arm"]) != self.adapter["cache_files"][key]:
                raise Pi05EvaluationError("action-memory episode cache changed")
            self.cache[key] = self._load_memory(row)
        return self.cache[key]

    def _load_memory(self, row: Mapping[str, Any]) -> PreparedActionMemory:
        with np.load(row["geometry_path"], allow_pickle=False) as data:
            frames, absolute = data["frames"], data["absolute"]
            if self.adapter["arm"] == "demo_action":
                actions, valid = data["demo_actions50"], data["demo_valid_lengths"]
            else:
                with np.load(row["source_path"], allow_pickle=False) as source:
                    if not np.array_equal(source["frames"], frames):
                        raise Pi05EvaluationError("source and geometry memory frame alignment changed")
                    actions = source["source_actions50"]
                valid = np.full(len(frames), 50, dtype=np.int64)
        expected_valid = (np.minimum(50, int(row["length"]) - frames - 1)
                          if self.adapter["arm"] == "demo_action" else np.full(len(frames), 50))
        if (not np.issubdtype(frames.dtype, np.integer) or not np.issubdtype(valid.dtype, np.integer)
                or not np.array_equal(frames, row["frames"])
                or absolute.shape != (len(frames), 14 + 12 * len(row["signature"]))
                or actions.shape != (len(frames), 50, 7) or valid.shape != (len(frames),)
                or not np.array_equal(valid, expected_valid) or (valid < 5).any()
                or not np.isfinite(absolute).all() or not np.isfinite(actions).all()):
            raise Pi05EvaluationError("action-memory finite/shape/five-real-action contract changed")
        return PreparedActionMemory(row["condition_id"], _episode_evidence(self.adapter, row),
                                    tuple(tuple(point) for point in row["signature"]),
                                    frames, absolute, actions, valid)

    def plan_slots(self, envs: Sequence[Any], slots: Sequence[dict[str, Any] | None], *,
                   task: Mapping[str, Any], root_seed: int, replan_steps: int) -> None:
        from ember.pi05_eval.action_memory_geometry import live_query, nearest
        from ember.pi05_eval.trajectory_capture import record_replan
        from ember.pi05_eval_contract import policy_noise_seed
        from ember.pi05_processing import libero_policy_input
        import torch

        if envs is None or replan_steps != 5:
            raise Pi05EvaluationError("action-memory control must execute the unchanged first five actions")
        for env, slot in zip(envs, slots, strict=True):
            if slot is None or slot["action_plan"] or slot.get("prefix_terminal", False):
                continue
            memory = slot["episode_adapter"]
            query, cache_delta = live_query(env, slot["obs"], memory.signature)
            indices, distances = nearest(query[None, :], memory.absolute)
            index = int(indices[0])
            plan = np.asarray(memory.actions[index], dtype=np.float32).copy()
            normalized = 2.0 * (plan - self.low) / (self.high - self.low + 1e-6) - 1.0
            valid_mask = np.arange(50) < int(memory.valid_lengths[index])
            raw_input = libero_policy_input(slot["obs"], str(task["language"]))
            processed = {name: value.unsqueeze(0) for name, value in raw_input.items()
                         if isinstance(value, torch.Tensor)}
            evidence = {"condition_id": memory.key, "teacher_position": index,
                        "teacher_frame": int(memory.frames[index]), "absolute_squared_distance": float(distances[0]),
                        "query_absolute": torch.from_numpy(query.copy()),
                        "action_origin": memory.evidence["action_origin"],
                        "proposal_kind": memory.evidence["proposal_kind"],
                        "valid_length": int(memory.valid_lengths[index]), "live_eef_cache_delta": cache_delta,
                        "policy_noise_consumed": False, "rollout_flow_steps": 0}
            record_replan(slot, raw_input, processed, torch.from_numpy(normalized[None]), plan[:5],
                          command_kind=KIND, raw_chunk=plan, valid_action_mask=valid_mask,
                          replan_metadata=evidence)
            slot["action_plan"].extend(plan[:5])
            slot["policy_noise_seeds"].append(policy_noise_seed(
                root_seed, str(task["suite"]), int(task["task_id"]),
                int(slot["init_state_id"]), int(slot["replan_index"])))
            slot["replan_index"] += 1

    def close(self) -> None:
        self.cache.clear()
