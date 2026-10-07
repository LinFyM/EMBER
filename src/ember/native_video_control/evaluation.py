"""Finite M/V readout consumer for native_video_control_diagnostic_20261007.

The canonical evaluator owns queueing, scenes, flow execution and capture. This
batch owner only publishes frozen condition memory and installs its typed
controller; retire this module and its registrations when the batch closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.expert_manifold.video_schedule import condition_demo_index
from ember.lora import (copy_task_lora_state_, expected_lora_state_shapes,
                        inject_task_lora, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority


STUDY = "native_video_control_diagnostic_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
REFERENCE = Path("/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929"
                 "/attempts/scene_canonical144/MT/evaluation/correct144")
MANIFEST_SCHEMA = "ember_native_video_control_readout_v1"
ADAPTER_SCHEMA = "ember_native_video_control_eval_adapter_v1"
EPISODE_SCHEMA = "ember_native_video_control_episode_v1"
MEMORY_SCHEMA = "ember_native_video_control_memory_v1"
PASSIVE_TAG = "ember_native_video_control_passive_capture_v1"
MEMORY_KIND = "native_video_memory_controller"
SHARED_KIND = "native_shared_lora_reference"
KINDS = {MEMORY_KIND, SHARED_KIND}
STATES = (32, 33, 34, 35)
HOLDOUT = (12, 29, 32, 38)
SEEN_IDS = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25,
            28, 29, 32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64, 73,
            95, 96, 97, 101)
H_SOURCE = "dual_rgb_exact_language_native_beta_probe1729_tau1_no_execution_reader"


def _wall(arm: str) -> dict[str, Any]:
    return {"teacher_labels_runtime_reads": 0, "teacher_video_runtime_reads": 0,
            "rollout_optimization_steps": 0, "single_shared_beta": True,
            "memory_encodings_per_condition": int(arm == "V"),
            "final_ember_selection_qualification": False}


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _check_record(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_file() or file_record(path) != dict(record):
        raise Pi05EvaluationError(f"native control asset changed: {path}")
    return path


def _reference() -> dict[str, Any]:
    contract = read_json(REFERENCE / "run_contract.json")
    tasks = contract["adapter"]["tasks"]
    if (tuple(row["global_task_id"] for row in tasks) != SEEN_IDS
            or contract["rng"]["inference_seed"] != 7
            or contract["operator_read_write_scene"]["root"] != str(REFERENCE.parents[2] / "scenes")):
        raise Pi05EvaluationError("native control original MT144 reference changed")
    return contract


def evaluation_conditions(condition: str) -> list[dict[str, Any]]:
    """Return the original four-state mapping or its canonical +17 partner."""
    if condition not in {"correct", "same_task_other"}:
        raise Pi05EvaluationError("native control only registers correct/other")
    result = []
    for task in _reference()["adapter"]["tasks"]:
        gid = int(task["global_task_id"])
        if condition == "same_task_other" and gid not in HOLDOUT:
            continue
        if tuple(row["init_state_id"] for row in task["episodes"]) != STATES:
            raise Pi05EvaluationError("native control original MT state mapping changed")
        for episode in task["episodes"]:
            state = int(episode["init_state_id"])
            arguments = dict(demo_count=50, sampling_mode="without_replacement")
            correct = condition_demo_index(20260928, task["suite"], int(task["task_id"]),
                                           state, condition="correct", **arguments)
            demo = condition_demo_index(20260928, task["suite"], int(task["task_id"]),
                                        state, condition=condition, **arguments)
            if episode["teacher_demo_indices"] != [correct] or episode["video_ordinal"] != state:
                raise Pi05EvaluationError("native control original MT teacher pairing changed")
            result.append({"condition_id": f"task_{gid:02d}_demos_{demo}",
                           "global_task_id": gid, "suite": task["suite"],
                           "task_id": int(task["task_id"]), "language": task["language"],
                           "init_state_id": state, "video_ordinal": state,
                           "teacher_demo": demo, "paired_correct_demo": correct})
    return result


def _checkpoint(checkpoint: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from .checkpoint import inspect_checkpoint

    checkpoint = checkpoint.resolve()
    if arm not in {"M", "V"} or not checkpoint.is_relative_to(ROOT):
        raise Pi05EvaluationError("native control checkpoint is outside this batch")
    manifest = inspect_checkpoint(checkpoint, arm=arm, terminal=True)
    return manifest, file_record(checkpoint / "ecp.safetensors")


def _memory_identity(weights: Mapping, condition: Mapping, indices: Sequence[int]) -> dict[str, str]:
    import json

    return {"schema_version": MEMORY_SCHEMA, "study_id": STUDY,
            "condition_id": condition["condition_id"], "weights": json.dumps(dict(weights), sort_keys=True),
            "frame_indices": json.dumps(list(map(int, indices))), "H_source": H_SOURCE}


def save_memory(checkpoint: Path, condition: Mapping[str, Any], memory: torch.Tensor,
                frame_indices: Sequence[int], *, panel: str) -> dict[str, Any]:
    """Save C from exactly one legitimate native read, before any rollout."""
    _, weights = _checkpoint(checkpoint, "V")
    registered = {row["condition_id"]: row for row in evaluation_conditions(panel)}
    indices = tuple(map(int, frame_indices))
    if (registered.get(condition["condition_id"]) != dict(condition) or len(indices) < 2
            or indices[0] != 0 or any(b - a != 5 for a, b in zip(indices[:-2], indices[1:-1]))
            or not 0 < indices[-1] - indices[-2] <= 5
            or tuple(memory.shape) != (1, len(indices) * 50, 256)
            or not torch.isfinite(memory).all()):
        raise Pi05EvaluationError("native control C condition/full-slot/finite contract changed")
    path = ROOT / "readouts" / "V" / panel / "memories" / f"{condition['condition_id']}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Pi05EvaluationError("native control condition memory already published")
    save_file({"C": memory.detach().float().cpu().contiguous()}, str(path),
              metadata=_memory_identity(weights, condition, indices))
    return {**file_record(path), "frame_indices": list(indices)}


def publish_manifest(checkpoint: Path, *, arm: str, condition: str, asset_root: Path,
                     encoding_git: Mapping[str, Any],
                     memories: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Publish the sole terminal128 readout; M contains no video values."""
    checkpoint_manifest, weights = _checkpoint(checkpoint, arm)
    conditions = evaluation_conditions(condition)
    if (arm == "M" and (condition != "correct" or memories)
            or arm == "V" and set(memories or {}) != {row["condition_id"] for row in conditions}):
        raise Pi05EvaluationError("native control M/V condition registry changed")
    reference = _reference()
    keys = {(row["suite"], row["task_id"]) for row in conditions}
    tasks = [row for row in reference["tasks"] if (row["suite"], row["task_id"]) in keys]
    path = ROOT / "readouts" / arm / condition / "manifest.json"
    if path.exists():
        raise Pi05EvaluationError("native control readout manifest already published")
    value = {"schema_version": MANIFEST_SCHEMA, "study_id": STUDY, "status": "sealed",
             "kind": MEMORY_KIND if arm == "V" else SHARED_KIND, "arm": arm,
             "condition": condition, "checkpoint": str(checkpoint.resolve()),
             "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
             "weights": weights, "source": reference["model"],
             "lora_contract": file_record(asset_root / "configs/pi05_lora_rank128_aligned.json"),
             "training_git": checkpoint_manifest["training_git"], "encoding_git": dict(encoding_git),
             "reference_results": file_record(REFERENCE / "results.json"),
             "reference_contract": file_record(REFERENCE / "run_contract.json"),
             "scene": reference["operator_read_write_scene"], "tasks": tasks,
             "conditions": [{**row, **({"memory": dict(memories[row["condition_id"]])}
                                       if arm == "V" else {})} for row in conditions],
             "information_wall": _wall(arm)}
    _inspect_identity(value, path, reference["model"], tasks, checkpoint_manifest, weights,
                      "operator_seen_training36", True)
    if arm == "V":
        _inspect_memories(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value)
    return path


def select_tasks(args: Any, installed: Sequence[Any]) -> tuple[Any, ...]:
    manifest = read_json(args.native_video_control_manifest)
    expected = evaluation_conditions(str(manifest.get("condition")))
    keys = {(row["suite"], row["task_id"]) for row in expected}
    selected = tuple(task for task in installed if (task.suite, int(task.task_id)) in keys)
    if (args.role != "operator_seen_training36" or args.mode != "formal" or args.state_count != 4
            or tuple(args.init_state_ids or ()) != STATES or len(selected) * 4 != len(expected)
            or any(getattr(args, key, None) for key in ("trajectory_capture_selection",
                       "task_subset_selection", "occupancy_capture_selection", "capture_stage_predicates"))):
        raise Pi05EvaluationError("native control only permits its fixed seen144/other16 readouts")
    return selected


def _inspect_identity(value: Mapping, path: Path, source: Mapping, tasks: Sequence,
                      checkpoint: Mapping, weights: Mapping, role: str, formal: bool) -> dict:
    arm, condition = value["arm"], value["condition"]
    expected = evaluation_conditions(condition)
    plain = [{key: item[key] for key in expected[0]} for item in value["conditions"]]
    task_rows = [dict(vars(task)) if not isinstance(task, Mapping) else dict(task) for task in tasks]
    keys = {(row["suite"], int(row["task_id"])) for row in task_rows}
    reference = _reference()
    source_keys = ("source_run", "checkpoint", "model_path")
    identities = ((value.get("schema_version"), MANIFEST_SCHEMA), (value.get("study_id"), STUDY),
                  (value.get("status"), "sealed"), (formal, True), (role, "operator_seen_training36"),
                  (value.get("kind"), MEMORY_KIND if arm == "V" else SHARED_KIND), (plain, expected),
                  (path.resolve(), ROOT / "readouts" / arm / condition / "manifest.json"),
                  (keys, {(row["suite"], row["task_id"]) for row in expected}),
                  ({tuple(row["init_state_ids"]) for row in task_rows}, {STATES}),
                  (value["weights"], weights), (value["training_git"], checkpoint["training_git"]),
                  (value["source"], checkpoint["source"]),
                  (value["checkpoint_manifest"], file_record(Path(value["checkpoint"]) / "checkpoint_manifest.json")),
                  (value["reference_results"], file_record(REFERENCE / "results.json")),
                  (value["reference_contract"], file_record(REFERENCE / "run_contract.json")),
                  (value["scene"], reference["operator_read_write_scene"]),
                  (value["tasks"], [row for row in reference["tasks"] if (row["suite"], row["task_id"]) in keys]),
                  (value["information_wall"], _wall(arm)))
    sources = [cell.get(key) for key in source_keys for cell in (value["source"], source, reference["model"])]
    if (any(actual != wanted for actual, wanted in identities)
            or any(len(set(sources[start:start + 3])) != 1 for start in range(0, len(sources), 3))
            or value["encoding_git"].get("branch") != ""
            or not git_state_is_clean_pushed_or_frozen_authority(value["encoding_git"])):
        raise Pi05EvaluationError("native control readout authority/scope changed")
    if arm == "M" and (condition != "correct" or any("memory" in row for row in value["conditions"])):
        raise Pi05EvaluationError("native control M cannot consume video memory")
    return reference


def _inspect_memories(value: Mapping) -> None:
    for row in value["conditions"]:
        cell = row["memory"]
        path = _check_record({key: cell[key] for key in ("path", "bytes")})
        if not path.is_relative_to(ROOT / "readouts" / "V" / value["condition"] / "memories"):
            raise Pi05EvaluationError("native control memory outside the fixed readout")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            identities = ((list(reader.keys()), ["C"]), (reader.get_slice("C").get_dtype(), "F32"),
                          (reader.get_slice("C").get_shape(), [1, len(cell["frame_indices"]) * 50, 256]),
                          (reader.metadata(), _memory_identity(value["weights"], row, cell["frame_indices"])))
            if any(actual != wanted for actual, wanted in identities):
                raise Pi05EvaluationError("native control memory version/full slots changed")


def inspect_manifest(*, manifest_path: Path, source: Mapping[str, Any], tasks: Sequence[Any],
                     evaluation_role: str, require_formal: bool) -> dict[str, Any]:
    value = read_json(manifest_path)
    manifest, weights = _checkpoint(Path(value["checkpoint"]), value["arm"])
    reference = _inspect_identity(value, manifest_path, source, tasks, manifest, weights,
                                  evaluation_role, require_formal)
    for name in ("checkpoint_manifest", "lora_contract", "reference_results", "reference_contract"):
        _check_record(value[name])
    lora = load_pi05_lora_contract(Path(value["lora_contract"]["path"]))
    if lora.rank != 128 or manifest["lora"] != lora.to_dict():
        raise Pi05EvaluationError("native control complete rank128 changed")
    inspect_registered_scenes(Path(value["scene"]["root"]), reference["tasks"],
                              states=STATES, schema="ember_operator_seen_task_scenes_v1")
    if value["arm"] == "V":
        _inspect_memories(value)
    return {**value, "schema_version": ADAPTER_SCHEMA, "manifest": file_record(manifest_path),
            "evaluation_role": evaluation_role}


def capture_contract(adapter: Mapping, output_dir: Path) -> tuple[dict, dict]:
    output_dir = output_dir.resolve()
    if output_dir != ROOT / "readouts" / adapter["arm"] / adapter["condition"] / "evaluation":
        raise Pi05EvaluationError("native control evaluator output is outside its finite panel")
    full = [{"suite": row["suite"], "task_id": row["task_id"], "init_state_id": 32}
            for row in adapter["tasks"]]
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "selection_path": adapter["manifest"]["path"],
               "selection_bytes": adapter["manifest"]["bytes"], "mode": "compact",
               "full_conditions": full, "trajectory_root": str(output_dir / "trajectories"),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "trace_root": str(output_dir / "continuous_traces")},
               **{key: False for key in ("training_gradient_use", "checkpoint_selection_use",
                                        "validation_use", "test_use")}}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
             "training_gradient_use": False, "checkpoint_selection_use": False,
             "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
             "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def validate_capture_contract(contract: Mapping) -> None:
    capture, stage = capture_contract(contract["adapter"], Path(contract["output_dir"]))
    if (contract.get("diagnostic_occupancy_capture") != capture
            or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("native_video_control_scene") != contract["adapter"]["scene"]
            or contract["rng"]["inference_seed"] != 7):
        raise Pi05EvaluationError("native control capture/scene/RNG contract changed")


def episode_evidence(adapter: Mapping, row: Mapping) -> dict:
    return {"schema_version": EPISODE_SCHEMA, "study_id": STUDY, "arm": adapter["arm"],
            "condition": adapter["condition"], **{key: row[key] for key in (
                "condition_id", "global_task_id", "init_state_id", "teacher_demo", "video_ordinal")},
            "checkpoint_manifest": adapter["checkpoint_manifest"], "beta": adapter["weights"],
            **({"reader": adapter["weights"], "memory": row["memory"],
                "memory_encodings_per_condition": 1, "H_source": H_SOURCE} if adapter["arm"] == "V"
               else {"teacher_mapping_metadata_only": True})}


def validate_episode(adapter: Mapping, evidence: Mapping | None, *, suite: str,
                     task_id: int, init_state_id: int) -> bool:
    row = next((row for row in adapter["conditions"] if
                (row["suite"], row["task_id"], row["init_state_id"]) ==
                (suite, task_id, init_state_id)), None)
    return row is not None and evidence == episode_evidence(adapter, row)


@dataclass(frozen=True)
class PreparedNativeControl:
    key: str
    evidence: dict
    memory: torch.Tensor | None = None
    key_values: Any = None


class FrozenNativeControlAdapter:
    """One beta, with fixed C/KV consumed by V through every official flow step."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        del require_formal
        from .model import NativeVideoControl

        self.adapter, self.policy, self.device = evaluation_adapter, policy, device
        if (self.adapter["source"] != source or set(task_keys) !=
                {(row["suite"], row["task_id"]) for row in self.adapter["tasks"]}):
            raise Pi05EvaluationError("native control worker source/task scope changed")
        self.lora = load_pi05_lora_contract(Path(self.adapter["lora_contract"]["path"]))
        shapes = expected_lora_state_shapes(self.lora)
        with safe_open(str(_check_record(self.adapter["weights"])), framework="pt", device="cpu") as reader:
            template = {name: reader.get_tensor(f"common.values.{index}")
                        for index, name in enumerate(sorted(shapes))}
        validate_lora_state(template, self.lora)
        self.model = NativeVideoControl(self.lora, template, arm=self.adapter["arm"]).to(device).eval()
        self.model.load_state_dict(load_file(self.adapter["weights"]["path"], device=str(device)))
        self.model.requires_grad_(False)
        inject_task_lora(policy, self.lora)
        copy_task_lora_state_(policy, self.model.public_state(), self.lora)
        policy.requires_grad_(False).eval()

    @torch.no_grad()
    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedNativeControl:
        row = next(row for row in self.adapter["conditions"] if
                   (row["suite"], row["task_id"], row["init_state_id"]) == (suite, task_id, init_state_id))
        memory = kv = None
        if self.adapter["arm"] == "V":
            memory = load_file(row["memory"]["path"], device=str(self.device))["C"]
            if not torch.isfinite(memory).all():
                raise Pi05EvaluationError("native control fixed C is nonfinite")
            kv = self.model.prepare_memory(memory)
        return PreparedNativeControl(row["condition_id"], episode_evidence(self.adapter, row), memory, kv)

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if len(prepared) != noise.shape[0] or num_steps != 10:
            raise Pi05EvaluationError("native control official paired flow contract changed")
        if self.adapter["arm"] == "M":
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
        chunks = []
        for index, item in enumerate(prepared):
            inputs = {key: value[index:index + 1] for key, value in batch.items()}
            with self.model.execution_scope(self.policy, item.memory, key_values=item.key_values):
                chunks.append(self.policy.predict_action_chunk(inputs, noise=noise[index:index + 1],
                                                               num_steps=num_steps))
        return torch.cat(chunks)
