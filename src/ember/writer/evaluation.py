"""Inspect and execute independently compiled, strictly paired episode LoRAs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (copy_task_lora_state_, expected_lora_state_shapes, identity_lora_state,
                        inject_task_lora, task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, adapter_metadata, condition_id,
    file_record, frozen_authority, inspect_joint_checkpoint, method_metadata, planned_episodes,
    selection_contract, source_matches)


EVALUATION_SCHEMA = "ember_layered_writer_eval_adapter_v1"
EPISODE_SCHEMA = "ember_layered_writer_episode_v1"


def validate_task_scope(rows: Sequence[Mapping[str, Any]], role: str, asset_root: Path) -> None:
    if role not in {"development_train", "validation"}:
        raise ValueError("layered Writer evaluation excludes Test and non-target tasks")
    protocol = read_json(asset_root / "configs/libero_24_8_8_v1/protocol.json")
    manifest = read_json(asset_root / "configs/pi05_target_data_v1/manifest.json")
    canonical = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    split = "train" if role == "development_train" else "validation"
    expected = {(suite, task) for suite, roles in protocol["split"]["suites"].items() for task in roles[split]}
    keys = [(str(row["suite"]), int(row["task_id"])) for row in rows]
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= expected or (split == "validation" and set(keys) != expected):
        raise ValueError("task bank crosses the fixed split or omits validation8 tasks")
    for row in rows:
        actual = canonical.get(int(row["global_task_id"]))
        if actual is None or any(row.get(key) != actual[key] for key in ("suite", "task_id", "language", "split_role")):
            raise ValueError("task language/identity differs from the fixed target authority")
        source = asset_root / "data/datasets" / manifest["dataset"]["revision"] / actual["hdf5"]["relative_path"]
        if row["teacher_source"] != {"path": str(source.resolve()), "bytes": int(actual["hdf5"]["bytes"])}:
            raise ValueError("teacher source provenance differs from the fixed task")


def _selection(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = selection_contract(role=value["evaluation_role"], task_ids=value["task_ids"],
        cardinality=value["K"], arm=value["arm"], mode=value["mode"], seed=value["seed"],
        init_state_ids=value["init_state_ids"], video_pool=value["video_pool"], fixed_videos=value["fixed_videos"])
    if dict(value) != normalized:
        raise ValueError("video selection contract changed")
    return normalized


def _inspect_adapter_file(condition: Mapping[str, Any], checkpoint: Mapping[str, Any], lora) -> None:
    path = Path(condition["adapter"]["path"])
    if not path.is_file() or condition["adapter"] != file_record(path):
        raise ValueError("materialized adapter file changed")
    expected = expected_lora_state_shapes(lora)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if handle.metadata() != adapter_metadata(condition["condition_id"], checkpoint) or set(handle.keys()) != set(expected):
            raise ValueError("adapter condition/checkpoint identity or complete target set changed")
        for name, shape in expected.items():
            value = handle.get_slice(name)
            if tuple(value.get_shape()) != shape or value.get_dtype() != "F32":
                raise ValueError("materialized adapter must retain complete FP32 native shapes")


def _validate_video_frames(videos, demos, lengths) -> None:
    if [video["demo_index"] for video in videos] != demos:
        raise ValueError("actual teacher videos differ from the selected K-set")
    for video in videos:
        raw = int(video["raw_frame_count"])
        if raw <= 0 or raw != lengths[video["demo_index"]]:
            raise ValueError("teacher full-frame count differs from the data authority")
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        if video["frame_indices"] != indices or video["sampled_frame_count"] != len(indices):
            raise ValueError("teacher stride5/full-frame provenance changed")


def _inspect_conditions(manifest: Mapping[str, Any], root: Path, lora) -> None:
    conditions = {row["condition_id"]: row for row in manifest["conditions"]}
    tasks = {row["global_task_id"]: row for row in manifest["tasks"]}
    authority = read_json(Path(manifest["asset_root"]) / "configs/pi05_target_data_v1/manifest.json")
    lengths = {row["global_task_id"]: row["demonstrations"]["episode_lengths"] for row in authority["tasks"]}
    referenced = {episode["condition_id"] for row in tasks.values() for episode in row["episodes"]}
    if len(conditions) != len(manifest["conditions"]) or set(conditions) != referenced:
        raise ValueError("condition bank contains duplicates, missing rows, or unused adapters")
    for key, condition in conditions.items():
        task = tasks.get(condition["global_task_id"])
        demos = condition["teacher_demo_indices"]
        if (task is None or any(condition[field] != task[field] for field in ("suite", "task_id", "language"))
                or key != condition_id(task["global_task_id"], demos)
                or len(demos) != manifest["selection"]["K"] or demos != sorted(set(demos))
                or condition.get("writer_invocations") != 1 or condition.get("single_complete_rank16") is not True
                or Path(condition["adapter"]["path"]).resolve() != root / f"{key}.safetensors"):
            raise ValueError("materialized condition task/video/adapter provenance changed")
        _validate_video_frames(condition["teacher_videos"], demos, lengths[task["global_task_id"]])
        for episode in task["episodes"]:
            if episode["condition_id"] == key and episode["teacher_demo_indices"] != demos:
                raise ValueError("episode mapping changed its actual teacher K-set")
        _inspect_adapter_file(condition, manifest["writer_checkpoint"], lora)


def _inspect_scope(manifest, source, task_keys, evaluation_role, task_init_state_ids) -> None:
    role = manifest["evaluation_role"]
    selection = _selection(manifest["selection"])
    rows = manifest["tasks"]
    keys = [(str(row["suite"]), int(row["task_id"])) for row in rows]
    if (manifest.get("schema_version") != BANK_SCHEMA or manifest.get("kind") != BANK_KIND
            or manifest.get("status") != "sealed" or role != evaluation_role
            or manifest.get("arm") != selection["arm"] or role != selection["evaluation_role"]
            or len(task_keys) != len(set(task_keys)) or set(task_keys) != set(keys)
            or [row["global_task_id"] for row in rows] != selection["task_ids"]
            or manifest.get("single_complete_rank16") is not True
            or not frozen_authority(manifest["materialization_git"])
            or not source_matches(manifest["source"], source)):
        raise ValueError("layered Writer bank scope/source/commit changed")
    validate_task_scope(rows, role, Path(manifest["asset_root"]))
    for row in rows:
        if row["episodes"] != planned_episodes(selection, row["global_task_id"]):
            raise ValueError("episode video ordinal or deterministic pairing changed")
        requested = (task_init_state_ids or {}).get((row["suite"], row["task_id"]), ())
        if not set(requested) <= set(selection["init_state_ids"]):
            raise ValueError("bank does not cover the evaluator's fixed init states")


def inspect_layered_writer_bank(
    *, manifest_path: Path, source: Mapping[str, Any], task_keys: Sequence[tuple[str, int]],
    evaluation_role: str, require_formal: bool,
    task_init_state_ids: Mapping[tuple[str, int], Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Validate condition provenance and paired row coverage before workers start."""
    del require_formal  # This bank is always produced from a formal frozen checkpoint.
    try:
        path = manifest_path.resolve()
        manifest = read_json(path)
        _inspect_scope(manifest, source, task_keys, evaluation_role, task_init_state_ids)
        run, checkpoint = inspect_joint_checkpoint(Path(manifest["writer_checkpoint"]["path"]))
        if checkpoint != manifest["writer_checkpoint"] or manifest["method"] != method_metadata(run) or not source_matches(run["source"], source):
            raise ValueError("Writer checkpoint or method provenance changed")
        lora_path = Path(manifest["lora_contract"]["path"])
        if manifest["lora_contract"] != file_record(lora_path):
            raise ValueError("LoRA topology authority changed")
        lora = load_pi05_lora_contract(lora_path)
        if lora.rank != lora.alpha or lora.rank != 16 or len(lora.targets) != 38 or lora.dropout != 0:
            raise ValueError("evaluation requires one complete 38-target rank16 LoRA")
        wall = manifest["information_wall"]
        required = {"teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
                    "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
                    "writer_invocations_per_unique_condition": 1, "total_writer_invocations": len(manifest["conditions"]),
                    "outcome_dependent_video_selection": False, "shuffled_reversed_wrong_no_video": False}
        if any(wall.get(key) != value for key, value in required.items()):
            raise ValueError("layered Writer information wall changed")
        _inspect_conditions(manifest, path.parent, lora)
        return {**manifest, "schema_version": EVALUATION_SCHEMA, "manifest": file_record(path)}
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise Pi05EvaluationError(str(error)) from error


def episode_evidence(adapter: Mapping[str, Any], task: Mapping[str, Any], episode: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {row["condition_id"]: row for row in adapter["conditions"]}
    condition = conditions[episode["condition_id"]]
    return {"schema_version": EPISODE_SCHEMA, **dict(condition), **dict(episode),
            "selection_seed": adapter["selection"]["seed"], "selection_mode": adapter["selection"]["mode"],
            "K": adapter["selection"]["K"], "arm": adapter["arm"],
            "writer_checkpoint": dict(adapter["writer_checkpoint"]), "method": dict(adapter["method"]),
            "source_checkpoint": adapter["source"]["checkpoint"], "global_task_id": task["global_task_id"]}


def validate_layered_writer_episode(adapter, evidence, *, suite: str, task_id: int, init_state_id: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    for task in adapter.get("tasks", ()):
        if (task["suite"], task["task_id"]) == (suite, task_id):
            for episode in task["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return dict(evidence) == episode_evidence(adapter, task, episode)
    return False


@dataclass(frozen=True)
class PreparedLayeredLoRA:
    key: str
    evidence: dict[str, Any]


class FrozenLayeredWriterAdapter:
    """Execution only: no observer, video, Meta, or learned Writer is loaded."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal) -> None:
        del device, require_formal
        adapter = evaluation_adapter
        self.records = {(row["suite"], row["task_id"]): row for row in adapter["tasks"]}
        if (adapter.get("kind") != BANK_KIND or adapter.get("schema_version") != EVALUATION_SCHEMA
                or not source_matches(adapter["source"], source) or set(self.records) != set(task_keys)
                or adapter.get("single_complete_rank16") is not True):
            raise Pi05EvaluationError("layered Writer runtime bank changed")
        self.adapter, self.policy = adapter, policy
        self.conditions = {row["condition_id"]: row for row in adapter["conditions"]}
        self.lora = load_pi05_lora_contract(Path(adapter["lora_contract"]["path"]))
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.batched = BatchedLoRAInference(policy, self.lora)
        self.identity = identity_lora_state(self.lora)
        self._states: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._installed: str | None = None

    def _state(self, key: str) -> dict[str, torch.Tensor]:
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key]
        condition = self.conditions[key]
        _inspect_adapter_file(condition, self.adapter["writer_checkpoint"], self.lora)
        state = load_file(condition["adapter"]["path"], device="cpu")
        validate_lora_state(state, self.lora)
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise Pi05EvaluationError("runtime adapter has nonfinite or non-FP32 values")
        self._states[key] = state
        if len(self._states) > 16:
            self._states.popitem(last=False)
        return state

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedLayeredLoRA:
        task = self.records.get((str(suite), int(task_id)))
        if task is not None:
            for episode in task["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return PreparedLayeredLoRA(episode["condition_id"], episode_evidence(self.adapter, task, episode))
        raise Pi05EvaluationError("rollout task/init state is absent from the paired Writer bank")

    @torch.no_grad()
    def install(self, prepared: PreparedLayeredLoRA) -> None:
        if prepared.key != self._installed:
            copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)
            self._installed = prepared.key

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if not prepared or len(prepared) != noise.shape[0] or any(item.key not in self.conditions for item in prepared):
            raise Pi05EvaluationError("policy batch and paired Writer conditions differ")
        if self._installed is not None:
            copy_task_lora_state_(self.policy, self.identity, self.lora)
            self._installed = None
        with self.batched.activate([self._state(item.key) for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()
        self._states.clear()
