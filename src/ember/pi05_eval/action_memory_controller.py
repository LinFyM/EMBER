"""Finite T50 handoff; reuse the sealed demo NN and canonical queue/physics."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json, source_reference_matches

KIND = "privileged_action_memory_controller"
STUDY = "teacher_state_handoff_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
PRIOR_ROOT = ROOT.parent / "privileged_action_memory_control_20261007"
SEALED = PRIOR_ROOT / "frozen_controller/src/ember/pi05_eval"
MANIFEST_SCHEMA = "ember_teacher_state_handoff_manifest_v1"
EVAL_SCHEMA = "ember_teacher_state_handoff_evaluation_v1"
EPISODE_SCHEMA = "ember_teacher_state_handoff_episode_v1"
PASSIVE_TAG = "ember_teacher_state_handoff_passive_capture_v1"
ARMS = ("T_replay", "T50_demo_NN")
CUT = 50
T_RESULTS = (
    ROOT.parent / "denoising_return_writer_20261006/readouts/parent/seen/ODE/evaluation/results.json",
    ROOT.parent / "query_conditioned_transition_read_20261005/parent/evaluation/seen12/results.json",
)


def _load_sealed(name: str, filename: str) -> Any:
    path = SEALED / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sealed = _load_sealed("_ember_teacher_handoff", "action_memory_controller.py")
SCENE_ROOT = _sealed.SCENE_ROOT
_sealed_validate_contract = _sealed.validate_contract


@lru_cache(maxsize=1)
def _registration() -> dict[str, Any]:
    return read_json(PRIOR_ROOT / "memory/demo_action_manifest.json")


@lru_cache(maxsize=1)
def _original_rows() -> dict[tuple[str, int, int], tuple[Path, dict[str, Any]]]:
    return {(row["suite"], row["task_id"], row["init_state_id"]): (path, row)
            for path in T_RESULTS for row in read_json(path)["rows"]}


def _cache_records(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    original = next((item for item in _registration()["conditions"]
                     if item["condition_id"] == row["condition_id"]), None)
    key = (row["suite"], row["task_id"], row["init_state_id"])
    reference = _original_rows().get(key)
    if (arm not in ARMS or original is None
            or any(row.get(name) != value for name, value in original.items())
            or reference is None or row.get("original_T_results_path") != str(reference[0])
            or row.get("original_T_row") != reference[1]):
        raise Pi05EvaluationError("handoff changed original teacher, geometry, scene or T trace")
    geometry = Path(row["geometry_path"]).resolve()
    trace = Path(reference[1]["continuous_control_trace"]["trace"]["path"]).resolve()
    if (not geometry.is_relative_to(PRIOR_ROOT) or not geometry.is_file()
            or not any(trace.is_relative_to(p.parent) for p in T_RESULTS) or not trace.is_file()):
        raise Pi05EvaluationError("handoff original cache/trace is absent or outside its source root")
    return {"geometry_path": _sealed._file_record(geometry),
            "original_T_trace": _sealed._file_record(trace)}


def _validate_study_metadata(value: Mapping[str, Any], path: Path,
                             source: Mapping[str, Any], role: str, formal: bool) -> None:
    from ember.operator_writer.scope import ROLE

    if (not path.is_relative_to(ROOT) or value.get("schema_version") != MANIFEST_SCHEMA
            or value.get("kind") != KIND or value.get("study_id") != STUDY
            or value.get("root") != str(ROOT) or value.get("arm") not in ARMS
            or role != ROLE or not formal or not source_reference_matches(value.get("source"), source)):
        raise Pi05EvaluationError("handoff source, study, role or fixed arm changed")
    original = _registration()
    preserved = ("geometry_definition_git", "action_offset", "position_scale_m", "gripper_scale_m",
                 "video_schedule_seed", "source", "tokenizer", "normalization", "policy",
                 "scene_root", "scene_manifest", "privileged_training_diagnostic",
                 "deployment_candidate", "held_or_test_reads", "gradient_updates", "policy_noise_used")
    fixed = {"cut_control_steps": CUT, "original_T_results": list(map(str, T_RESULTS)),
             "source_native_forwards": 0, "source_inference_steps": 0,
             "rollout_flow_steps": 0, "teacher_HDF_reads": 0,
             "geometry_registration": str(PRIOR_ROOT / "memory/demo_action_manifest.json")}
    if (any(value.get(key) != original[key] for key in preserved)
            or any(value.get(key) != wanted for key, wanted in fixed.items())):
        raise Pi05EvaluationError("handoff fixed physical, Value or information-wall contract changed")


def attach_contract(contract: dict[str, Any]) -> None:
    _sealed.attach_contract(contract)
    contract["policy"]["source_value_num_inference_steps"] = None
    validate_contract(contract)


def validate_contract(contract: Mapping[str, Any]) -> None:
    adapter = contract.get("adapter") or {}
    if adapter.get("schema_version") != EVAL_SCHEMA:
        raise Pi05EvaluationError("handoff evaluation schema changed")
    _validate_study_metadata({**adapter, "schema_version": MANIFEST_SCHEMA},
                             Path(adapter["manifest"]["path"]).resolve(), contract["model"],
                             str(contract["role"]), contract["mode"] == "formal")
    if (read_json(Path(contract["normalization"]["path"]))["stats"]["action"]
            != read_json(Path(adapter["normalization"]))["stats"]["action"]
            or contract["parallel"].get("cpu_controller_per_worker") is not True
            or contract["parallel"].get("one_policy_per_worker") is not False):
        raise Pi05EvaluationError("handoff normalization or model-free worker changed")
    _sealed_validate_contract(contract)


def _episode_evidence(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": EPISODE_SCHEMA, "study_id": STUDY, "arm": adapter["arm"],
            **{key: row[key] for key in ("condition_id", "global_task_id", "suite", "task_id",
                                        "init_state_id", "teacher_demo", "hdf5")},
            "cache_files": adapter["cache_files"][row["condition_id"]],
            "original_T_results_path": row["original_T_results_path"],
            "original_T_success": row["original_T_row"]["success"],
            "original_T_steps": row["original_T_row"]["steps"],
            "cut_control_steps": CUT, "privileged": True, "training_tasks_only": True,
            "ember_score": False, "controller": "recorded_T_prefix_then_replay_or_absolute_demo_1nn",
            "rollout_model_loaded": False, "rollout_flow_steps": 0, "policy_noise_consumed": False,
            "proposal_kind": "at_most_five_actual_commands_with_invalid_nonexecuted_padding"
            if adapter["arm"] == "T_replay" else "recorded_future_with_nonexecuted_last_action_padding",
            "action_origin": "recorded_T_actual_raw_commands" if adapter["arm"] == "T_replay"
            else "recorded_teacher_post_action_offset1",
            "action_origin_scope": "whole_episode" if adapter["arm"] == "T_replay" else "suffix_only",
            "prefix_action_origin": "recorded_T_actual_raw_commands",
            "actual_replan_metadata_authoritative": True}


def validate_episode(adapter: Mapping[str, Any], evidence: Any, *,
                     suite: str, task_id: int, init_state_id: int) -> bool:
    return (adapter.get("study_id") == STUDY and adapter.get("arm") in ARMS
            and _sealed.validate_episode(adapter, evidence, suite=suite,
                                         task_id=task_id, init_state_id=init_state_id))


@dataclass(frozen=True)
class PreparedHandoff(_sealed.PreparedActionMemory):
    recorded_actions: np.ndarray
    original_T_row: dict[str, Any]


class PrivilegedActionMemoryController(_sealed.PrivilegedActionMemoryController):
    """Own raw command replay and cut50; inherit the original NN computation."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        super().__init__(contract)
        sys.modules["ember.pi05_eval.action_memory_geometry"] = _load_sealed(
            "_ember_teacher_handoff_geometry", "action_memory_geometry.py")

    def _load_memory(self, row: Mapping[str, Any]) -> PreparedHandoff:
        reference = row["original_T_row"]
        path = reference["continuous_control_trace"]["trace"]["path"]
        with np.load(path, allow_pickle=False) as data:
            recorded = data["actions"].copy()
        if (recorded.shape != (reference["steps"], 7) or len(recorded) <= CUT
                or not np.isfinite(recorded).all()):
            raise Pi05EvaluationError("recorded T physical commands changed")
        with np.load(row["geometry_path"], allow_pickle=False) as data:
            frames, absolute = data["frames"], data["absolute"]
            actions, valid = data["demo_actions50"], data["demo_valid_lengths"]
        expected = np.minimum(50, int(row["length"]) - frames - 1)
        if (not np.array_equal(frames, row["frames"])
                or absolute.shape != (len(frames), 14 + 12 * len(row["signature"]))
                or actions.shape != (len(frames), 50, 7) or not np.array_equal(valid, expected)
                or (valid < 5).any() or not np.isfinite(absolute).all()
                or not np.isfinite(actions).all()):
            raise Pi05EvaluationError("sealed demo NN real future or signature changed")
        return PreparedHandoff(row["condition_id"], _episode_evidence(self.adapter, row),
                               tuple(tuple(p) for p in row["signature"]), frames,
                               absolute, actions, valid, recorded, reference)

    def _capture_boundary(self, slot: dict[str, Any]) -> None:
        if slot["steps"] != CUT or "handoff_boundary" in slot:
            return
        memory = slot["episode_adapter"]
        trace = slot["passive_trace"]
        names = [item["name"] for item in trace["body_registry"]]
        state = {key: np.asarray(trace[key][-1]).copy() for key in (
            "eef_pos", "eef_quat", "gripper_qpos", "body_positions", "predicates")}
        path = memory.original_T_row["continuous_control_trace"]["trace"]["path"]
        with np.load(path, allow_pickle=False) as original:
            body_names = original["body_names"].tolist()
            order = [body_names.index(name) for name in names]
            reference = {key: original[key][CUT].copy() for key in state}
            reference["body_positions"] = reference["body_positions"][order]
        slot["handoff_boundary"] = {
            "actual_control_step": CUT, "next_raw_action_index": CUT,
            "body_names": names, "state": {key: value.tolist() for key, value in state.items()},
            "original_T_state": {key: value.tolist() for key, value in reference.items()},
            "position_error_m": {key: float(np.linalg.norm(state[key] - reference[key], axis=-1).max())
                                 for key in ("eef_pos", "body_positions")},
            "eef_quaternion_component_max_abs_error": float(np.max(np.abs(state["eef_quat"] - reference["eef_quat"]))),
            "gripper_component_max_abs_error": float(np.max(np.abs(state["gripper_qpos"] - reference["gripper_qpos"]))),
            "native_predicates_match": bool(np.array_equal(state["predicates"], reference["predicates"])),
            "raw_prefix_matches_original": bool(np.array_equal(np.asarray(trace["actions"]), memory.recorded_actions[:CUT])),
            "runtime_reset_at_cut": False, "extra_settling_at_cut": 0,
            "state_clone_at_cut": False, "policy_noise_consumed": False}

    def _record_raw_plan(self, slot: dict[str, Any], task: Mapping[str, Any], root_seed: int) -> None:
        from ember.pi05_eval.trajectory_capture import record_replan
        from ember.pi05_eval_contract import policy_noise_seed
        from ember.pi05_processing import libero_policy_input
        import torch

        memory = slot["episode_adapter"]
        step = int(slot["steps"])
        end = len(memory.recorded_actions) if self.adapter["arm"] == "T_replay" else CUT
        actual = memory.recorded_actions[step:min(step + 5, end)].copy()
        if not len(actual):
            raise Pi05EvaluationError("recorded actions exhausted before its registered stop")
        plan = np.concatenate((actual, np.repeat(actual[-1:], 50 - len(actual), axis=0)))
        normalized = 2 * (plan - self.low) / (self.high - self.low + 1e-6) - 1
        raw_input = libero_policy_input(slot["obs"], str(task["language"]))
        processed = {key: value.unsqueeze(0) for key, value in raw_input.items() if isinstance(value, torch.Tensor)}
        evidence = {"condition_id": memory.key, "original_action_start": step,
                    "valid_length": len(actual), "action_origin": "recorded_T_actual_raw_commands",
                    "proposal_kind": "at_most_five_actual_commands_with_invalid_nonexecuted_padding",
                    "policy_noise_consumed": False, "rollout_flow_steps": 0,
                    "phase": "T_prefix" if step < CUT else "T_replay_suffix"}
        record_replan(slot, raw_input, processed, torch.from_numpy(normalized[None]), actual,
                      command_kind=KIND, raw_chunk=plan, valid_action_mask=np.arange(50) < len(actual),
                      replan_metadata=evidence)
        slot["action_plan"].extend(actual)
        slot["policy_noise_seeds"].append(policy_noise_seed(root_seed, str(task["suite"]),
            int(task["task_id"]), int(slot["init_state_id"]), int(slot["replan_index"])))
        slot["replan_index"] += 1

    def plan_slots(self, envs: Sequence[Any], slots: Sequence[dict[str, Any] | None], *,
                   task: Mapping[str, Any], root_seed: int, replan_steps: int) -> None:
        if replan_steps != 5:
            raise Pi05EvaluationError("handoff retains original five-action replans")
        nn_slots = []
        for slot in slots:
            if slot is None or slot["action_plan"]:
                nn_slots.append(None)
                continue
            self._capture_boundary(slot)
            if self.adapter["arm"] == "T_replay":
                slot["recorded_action_limit"] = len(slot["episode_adapter"].recorded_actions)
            if self.adapter["arm"] == "T_replay" or slot["steps"] < CUT:
                self._record_raw_plan(slot, task, root_seed)
                nn_slots.append(None)
            else:
                nn_slots.append(slot)
        super().plan_slots(envs, nn_slots, task=task, root_seed=root_seed, replan_steps=replan_steps)

    def finish_evidence(self, slot: Mapping[str, Any]) -> dict[str, Any]:
        memory = slot["episode_adapter"]
        exhausted = self.adapter["arm"] == "T_replay" and slot["steps"] >= len(memory.recorded_actions)
        return {"schema_version": "ember_teacher_state_handoff_outcome_v1", "cut_control_steps": CUT,
                "boundary": slot.get("handoff_boundary"), "actual_prefix_steps": min(CUT, int(slot["steps"])),
                "original_T_success": memory.original_T_row["success"],
                "original_T_steps": len(memory.recorded_actions), "actual_steps": int(slot["steps"]),
                "original_policy_noise_seeds_provenance": memory.original_T_row["policy_noise_seeds"],
                "termination": "native-success" if slot["episode_done"] else "replay-exhausted" if exhausted else "horizon",
                "original_success_reproduced": bool(slot["episode_done"]) == memory.original_T_row["success"],
                "no_online_policy_or_flow": True}


for _name in ("STUDY", "ROOT", "MANIFEST_SCHEMA", "EVAL_SCHEMA", "EPISODE_SCHEMA", "PASSIVE_TAG"):
    setattr(_sealed, _name, globals()[_name])
_sealed._cache_records = _cache_records
_sealed._validate_study_metadata = _validate_study_metadata
_sealed._episode_evidence = _episode_evidence
_sealed.validate_contract = validate_contract
inspect_manifest = _sealed.inspect_manifest
