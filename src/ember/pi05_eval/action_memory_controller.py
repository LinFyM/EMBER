"""Five-action frozen Values consumed by the sealed absolute-geometry controller.

This batch owns only the Value/schema boundary. Retrieval, live queries, paired
scenes, queue execution and capture remain the original frozen consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json, source_reference_matches


KIND = "privileged_action_memory_controller"
STUDY = "calibrated_action_memory_control_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
PRIOR_ROOT = ROOT.parent / "privileged_action_memory_control_20261007"
SEALED = PRIOR_ROOT / "frozen_controller/src/ember/pi05_eval"
MANIFEST_SCHEMA = "ember_calibrated_action_memory_manifest_v1"
EVAL_SCHEMA = "ember_calibrated_action_memory_evaluation_v1"
EPISODE_SCHEMA = "ember_calibrated_action_memory_episode_v1"
PASSIVE_TAG = "ember_calibrated_action_memory_passive_capture_v1"
ARMS = ("bare_endpoint", "calibrated_value")
GAMMA_CHECKPOINT = (ROOT.parent / "control_calibrated_read_write_20261003"
                    / "control_calibrated_read_write/train/attempts/fresh/checkpoints"
                    / "macro_00000450/ecp.safetensors")


def _load_sealed(name: str, filename: str) -> Any:
    path = SEALED / filename
    if not path.is_file():
        raise Pi05EvaluationError(f"sealed action-memory consumer is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sealed = _load_sealed("_ember_calibrated_memory_controller", "action_memory_controller.py")
SCENE_ROOT = _sealed.SCENE_ROOT
PreparedActionMemory = _sealed.PreparedActionMemory
_sealed_validate_contract = _sealed.validate_contract


@lru_cache(maxsize=1)
def _registration() -> dict[str, Any]:
    return read_json(PRIOR_ROOT / "memory/registration.json")


def _cache_records(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    original = next((item for item in _registration()["conditions"]
                     if item["condition_id"] == row["condition_id"]), None)
    if original is None or any(row.get(key) != value for key, value in original.items()):
        raise Pi05EvaluationError("calibrated memory changed the sealed teacher/geometry mapping")
    result = {}
    for name, root in (("geometry_path", PRIOR_ROOT), ("value_path", ROOT / "memory" / arm)):
        path = Path(row[name]).resolve()
        if arm not in ARMS or not path.is_relative_to(root) or not path.is_file():
            raise Pi05EvaluationError("calibrated memory cache is missing or outside its registered root")
        result[name] = _sealed._file_record(path)
    return result


def _validate_study_metadata(value: Mapping[str, Any], path: Path, source: Mapping[str, Any],
                             role: str, formal: bool) -> None:
    from ember.operator_writer.scope import ROLE

    if (not path.is_relative_to(ROOT) or value.get("schema_version") != MANIFEST_SCHEMA
            or value.get("kind") != KIND or value.get("study_id") != STUDY
            or value.get("root") != str(ROOT) or value.get("arm") not in ARMS
            or role != ROLE or not formal or source.get("optimizer_step") != 1000
            or not source_reference_matches(value.get("source"), source)):
        raise Pi05EvaluationError("calibrated action-memory study/source/role changed")
    original = _registration()
    preserved = ("geometry_definition_git", "action_offset", "position_scale_m", "gripper_scale_m",
                 "video_schedule_seed", "source", "tokenizer", "normalization", "policy",
                 "scene_root", "scene_manifest", "privileged_training_diagnostic",
                 "deployment_candidate", "held_or_test_reads", "gradient_updates", "policy_noise_used")
    fixed = {"source_noise_seed": 1729, "source_noise_shape": [50, 32],
             "source_inference_steps": 0, "source_value_mode": "existing_statefree_tau1_endpoint",
             "source_state_used": False, "source_flow_tau": 1, "source_native_forwards": 0,
             "rollout_flow_steps": 0, "value_shape": [5, 7], "valid_length": 5,
             "value_inverse_quantile_count": 1, "new_geometry_restores": 0,
             "geometry_registration": str(PRIOR_ROOT / "memory/registration.json")}
    gamma = value.get("gamma") or {}
    gamma_fixed = {"checkpoint": str(GAMMA_CHECKPOINT), "training_git": "2c630fb3",
                   "input": "existing_H0_mu0_only"}
    if (any(value.get(key) != original[key] for key in preserved)
            or any(value.get(key) != wanted for key, wanted in fixed.items())
            or any(gamma.get(key) != wanted for key, wanted in gamma_fixed.items())):
        raise Pi05EvaluationError("calibrated memory fixed Value/geometry/information-wall metadata changed")


def attach_contract(contract: dict[str, Any]) -> None:
    _sealed.attach_contract(contract)
    contract["policy"]["source_value_num_inference_steps"] = 0
    validate_contract(contract)


def validate_contract(contract: Mapping[str, Any]) -> None:
    adapter = contract.get("adapter") or {}
    if adapter.get("schema_version") != EVAL_SCHEMA:
        raise Pi05EvaluationError("calibrated memory evaluation schema changed")
    _validate_study_metadata(
        {**adapter, "schema_version": MANIFEST_SCHEMA}, Path(adapter["manifest"]["path"]).resolve(),
        contract["model"], str(contract["role"]), contract["mode"] == "formal")
    if (contract["policy"].get("source_value_num_inference_steps") != 0
            or read_json(Path(contract["normalization"]["path"]))["stats"]["action"]
            != read_json(Path(adapter["normalization"]))["stats"]["action"]
            or contract["parallel"].get("cpu_controller_per_worker") is not True
            or contract["parallel"].get("one_policy_per_worker") is not False
            or contract["parallel"].get("gpu_use") != "EGL_renderer_context_only"):
        raise Pi05EvaluationError("calibrated memory worker/Value execution contract changed")
    # The sealed validator's historical non-ODE arm is represented by None.
    _sealed_validate_contract({**contract, "policy": {
        **contract["policy"], "source_value_num_inference_steps": None}})


def _episode_evidence(adapter: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": EPISODE_SCHEMA, "study_id": STUDY, "arm": adapter["arm"],
            **{key: row[key] for key in ("condition_id", "global_task_id", "suite", "task_id",
                                        "init_state_id", "teacher_demo", "hdf5")},
            "cache_files": adapter["cache_files"][row["condition_id"]], "gamma": adapter["gamma"],
            "privileged": True, "training_tasks_only": True, "ember_score": False,
            "controller": "cpu_absolute_geometry_1nn", "rollout_model_loaded": False,
            "rollout_flow_steps": 0, "policy_noise_consumed": False,
            "source_state_used": False, "source_flow_tau": 1, "source_noise_seed": 1729,
            "value_shape": [5, 7], "valid_length": 5,
            "proposal_kind": "five_real_values_with_nonexecuted_fifth_action_padding",
            "action_origin": "frozen_source1000_statefree_tau1_probe1729_" + adapter["arm"]}


def validate_episode(adapter: Mapping[str, Any], evidence: Any, *,
                     suite: str, task_id: int, init_state_id: int) -> bool:
    return (adapter.get("study_id") == STUDY and adapter.get("schema_version") == EVAL_SCHEMA
            and adapter.get("arm") in ARMS and _sealed.validate_episode(
                adapter, evidence, suite=suite, task_id=task_id, init_state_id=init_state_id))


@dataclass(frozen=True)
class FiveActionValue:
    frames: np.ndarray
    normalized_actions5: np.ndarray
    raw_actions5: np.ndarray


class PrivilegedActionMemoryController(_sealed.PrivilegedActionMemoryController):
    """Change only offline Value loading; inherit the original plan_slots verbatim."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        super().__init__(contract)
        # The inherited method imports this historical qualified name at replan.
        geometry = _load_sealed("_ember_calibrated_memory_geometry", "action_memory_geometry.py")
        sys.modules["ember.pi05_eval.action_memory_geometry"] = geometry

    def _load_memory(self, row: Mapping[str, Any]) -> PreparedActionMemory:
        with np.load(row["geometry_path"], allow_pickle=False) as geometry:
            frames, absolute = geometry["frames"], geometry["absolute"]
        with np.load(row["value_path"], allow_pickle=False) as data:
            value = FiveActionValue(data["frames"], data["normalized_actions5"], data["raw_actions5"])
        count = len(frames)
        if (not np.issubdtype(frames.dtype, np.integer)
                or not np.issubdtype(value.frames.dtype, np.integer)
                or not np.array_equal(frames, row["frames"])
                or not np.array_equal(value.frames, frames)
                or absolute.shape != (count, 14 + 12 * len(row["signature"]))
                or value.normalized_actions5.shape != (count, 5, 7)
                or value.raw_actions5.shape != (count, 5, 7)
                or not all(np.isfinite(array).all() for array in (
                    absolute, value.normalized_actions5, value.raw_actions5))):
            raise Pi05EvaluationError("calibrated memory frame/finite/five-real-value shape changed")
        expected_raw = (value.normalized_actions5 + 1) * (self.high - self.low + 1e-6) / 2 + self.low
        if not np.allclose(value.raw_actions5, expected_raw, rtol=1e-5, atol=2e-6):
            raise Pi05EvaluationError("calibrated memory Values do not use the frozen single inverse quantile")
        actions = np.concatenate((value.raw_actions5,
                                  np.repeat(value.raw_actions5[:, 4:5], 45, axis=1)), axis=1)
        return PreparedActionMemory(
            row["condition_id"], _episode_evidence(self.adapter, row),
            tuple(tuple(point) for point in row["signature"]), frames, absolute, actions,
            np.full(count, 5, dtype=np.int64))


for _name in ("STUDY", "ROOT", "MANIFEST_SCHEMA", "EVAL_SCHEMA", "EPISODE_SCHEMA", "PASSIVE_TAG"):
    setattr(_sealed, _name, globals()[_name])
_sealed._cache_records = _cache_records
_sealed._validate_study_metadata = _validate_study_metadata
_sealed._episode_evidence = _episode_evidence
_sealed.validate_contract = validate_contract
inspect_manifest = _sealed.inspect_manifest
