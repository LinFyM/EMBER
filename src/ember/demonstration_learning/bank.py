"""Compact P/I/M admission banks and their canonical evaluator adapter."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.expert_manifold.video_schedule import reference_demo_index
from ember.lora import (LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_,
                        expected_lora_state_shapes, identity_lora_state, inject_task_lora,
                        task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record, source_matches

from .data import TransferData
from .model import concatenate_factors
from .run import REPO, RUN_SCHEMA, STAGE, _frozen_git, build_runtime, specification


BANK_SCHEMA = "ember_demonstration_comparison_compact_bank_v1"
BANK_KIND = "demonstration_comparison_lora_bank"
EVAL_SCHEMA = "ember_demonstration_comparison_eval_adapter_v1"
EPISODE_SCHEMA = "ember_demonstration_comparison_episode_v1"
PASSIVE_TAG = "ember_demonstration_comparison_passive_capture_v1"
OLD_RUN_SCHEMA = "ember_demonstration_transfer_learning_closure_run_v2"
OLD_STAGE = "demonstration_transfer_learning_closure"


def _source_checkpoint(arm: str, spec: Mapping) -> Path:
    if arm in ("P", "I"):
        return Path(spec["bank"][f"{arm}_checkpoint"])
    if arm == "M":
        return Path(spec["runtime"]["run_root"]) / "M/fresh/checkpoints/macro_00000003"
    raise ValueError("bank arm is outside P/I/M")


def _inspect_checkpoint(arm: str, spec: Mapping, source: Mapping) -> tuple[Path, dict]:
    path = _source_checkpoint(arm, spec).resolve()
    run = read_json(path.parent.parent / "run_contract.json")
    manifest = read_json(path / "checkpoint_manifest.json")
    expected_schema = OLD_RUN_SCHEMA if arm in ("P", "I") else RUN_SCHEMA
    expected_stage = OLD_STAGE if arm in ("P", "I") else STAGE
    expected_macro = 6 if arm in ("P", "I") else 3
    expected_commit = (spec["bank"]["P_I_source_code"] if arm in ("P", "I")
                       else spec["bank"]["M_source_code"])
    required = (
        (run.get("schema_version"), expected_schema), (run.get("stage"), expected_stage),
        (run.get("arm"), arm), (run.get("git", {}).get("commit"), expected_commit),
        (run.get("git", {}).get("dirty_paths"), []), (run.get("source_trainable"), 0),
        (run.get("model"), spec["model"]),
        (manifest.get("schema_version"), "ember_ecp_checkpoint_v1"),
        (manifest.get("stage"), expected_stage),
        (manifest.get("run_contract_schema"), expected_schema),
        (manifest.get("next_macro"), expected_macro),
    )
    if not all(actual == wanted for actual, wanted in required) or not source_matches(run["source"], source):
        raise ValueError("bank checkpoint/source/model identity changed")
    if arm in ("P", "I"):
        old_spec = read_json(Path(run["spec"]))
        if (old_spec.get("schema_version") != "ember_demonstration_transfer_learning_engineering_spec_v2"
                or any(old_spec[key] != spec[key] for key in ("source", "operator", "model"))):
            raise ValueError("historical P/I engineering checkpoint is not compatible")
    else:
        if run.get("spec") != spec["bank"]["M_source_spec"]:
            raise ValueError("M3 checkpoint spec path changed")
        earlier = read_json(Path(run["spec"]))
        current_numeric = copy.deepcopy(spec)
        current_numeric["bank"].pop("M_source_code")
        current_numeric["bank"].pop("M_source_spec")
        if earlier != current_numeric:
            raise ValueError("M3 checkpoint numerical and evaluation contract changed")
    for name, row in manifest["files"].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != int(row["bytes"]):
            raise ValueError("bank checkpoint file missing or truncated")
    return path, run


def _split_complete(state: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    common, conditional = {}, {}
    if len(state) != 76:
        raise ValueError("bank full LoRA is not 38-target A/B")
    for name, value in state.items():
        axis = 0 if name.endswith(LORA_A_SUFFIX) else 1 if name.endswith(LORA_B_SUFFIX) else -1
        if axis < 0 or value.ndim != 2 or value.shape[axis] != 144:
            raise ValueError("bank complete factor rank changed")
        common[name] = value.narrow(axis, 0, 128).detach().float().cpu().contiguous()
        conditional[name] = value.narrow(axis, 128, 16).detach().float().cpu().contiguous()
    return common, conditional


def _checked_cpu(state: Mapping[str, torch.Tensor], shapes: Mapping[str, tuple]) -> dict:
    if set(state) != set(shapes):
        raise ValueError("bank factor names differ from complete LoRA contract")
    result = {name: value.detach().float().cpu().contiguous() for name, value in state.items()}
    if any(tuple(result[name].shape) != tuple(shape) or not torch.isfinite(result[name]).all()
           for name, shape in shapes.items()):
        raise ValueError("bank factor shape or finite check failed")
    return result


def _task_rows(spec: Mapping, arm: str) -> tuple[list, list]:
    manifest = read_json(REPO / spec["data"]["manifest"])
    authority = {row["global_task_id"]: row for row in manifest["tasks"]}
    tasks, conditions = [], {}
    for global_id in spec["bank"]["task_ids"]:
        task = authority[global_id]
        episodes = []
        for state in spec["bank"]["init_state_ids"]:
            demo = reference_demo_index(7, task["suite"], task["task_id"], state,
                                        demo_count=50, sampling_mode="without_replacement")
            key = "shared" if arm == "M" else f"task_{global_id:03d}_demo_{demo:02d}"
            episodes.append({"init_state_id": state, "condition_id": key,
                             "teacher_demo": None if arm == "M" else demo,
                             "video_ordinal": demo})
            if arm != "M":
                conditions[key] = {"condition_id": key, "global_task_id": global_id,
                                   "teacher_demo": demo, "language": task["language"]}
        tasks.append({"global_task_id": global_id, "suite": task["suite"],
                      "task_id": task["task_id"], "language": task["language"],
                      "split_role": task["split_role"], "episodes": episodes})
    return tasks, list(conditions.values())


@torch.no_grad()
def materialize(arm: str, asset_root: Path, device: torch.device) -> dict:
    _frozen_git()
    spec = specification()
    output = Path(spec["runtime"]["run_root"]) / "banks" / arm
    if output.exists():
        raise ValueError("registered bank already exists")
    started = time.perf_counter()
    runtime = build_runtime(asset_root, spec, device, arm=arm)
    checkpoint, run = _inspect_checkpoint(arm, spec, runtime.source)
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.eval()
    tasks, conditions = _task_rows(spec, arm)
    output.mkdir(parents=True)
    shared_path = output / "shared.safetensors"
    if arm == "M":
        shared = _checked_cpu(runtime.compile(None), expected_lora_state_shapes(runtime.lora))
        save_file(shared, str(shared_path))
    else:
        common_shapes = expected_lora_state_shapes(derive_pi05_lora_rank(runtime.lora, rank=128))
        conditional_shapes = expected_lora_state_shapes(derive_pi05_lora_rank(runtime.lora, rank=16))
        data = TransferData(asset_root, spec, arm=arm)
        try:
            first = True
            for row in conditions:
                condition, raw, sampled = runtime.condition(data, row["global_task_id"], row["teacher_demo"])
                state = runtime.compile(condition)
                validate_lora_state(state, runtime.lora)
                common, conditional = _split_complete(state)
                if first:
                    save_file(_checked_cpu(common, common_shapes), str(shared_path))
                    first = False
                path = output / f"{row['condition_id']}.safetensors"
                save_file(_checked_cpu(conditional, conditional_shapes), str(path))
                row.update({"factors": file_record(path), "raw_frames": raw,
                            "sampled_frames": sampled})
        finally:
            data.close()
    bank = {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "arm": arm,
            "source": runtime.source, "checkpoint": str(checkpoint),
            "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
            "training_git": run["git"]["commit"],
            "spec": file_record(REPO / "configs/demonstration_transfer_v1/learning_engineering_spec.json"),
            "asset_root": str(asset_root.resolve()), "lora": runtime.lora.to_dict(),
            "base_lora_contract": str((asset_root / read_json(asset_root / "configs/pi05_writer_data_v1.json")
                                       ["authorities"]["lora_contract"]).resolve()),
            "shared": file_record(shared_path), "conditions": conditions, "tasks": tasks,
            "scene_root": str(Path(spec["runtime"]["run_root"]) / "scenes"),
            "information_wall": {"teacher_video_values_read": 0 if arm == "M" else len(conditions),
                                 "teacher_runtime_reads": 0, "deployment_adapters": 1,
                                 "validation_test_gradients": False}}
    write_json_atomic(output / "manifest.json", bank)
    seconds = time.perf_counter() - started
    result = {"arm": arm, "conditions": len(conditions), "seconds": seconds,
              "lora_per_second": len(conditions) / seconds if conditions else 1 / seconds,
              "bank_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
              "manifest": str(output / "manifest.json"),
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    write_json_atomic(output / "completion.json", result)
    return result


def _tensor_file(record: Mapping, shapes: Mapping) -> None:
    path = Path(record["path"])
    if record != file_record(path):
        raise ValueError("compact factor file changed")
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(shapes) or any(
                tuple(reader.get_slice(name).get_shape()) != tuple(shape)
                for name, shape in shapes.items()):
            raise ValueError("compact factor shape changed")
        if any(not torch.isfinite(reader.get_tensor(name)).all() for name in shapes):
            raise ValueError("compact factor is nonfinite")


def _inspect_bank_scope(bank: Mapping, spec: Mapping, arm: str, path: Path,
                        source: Mapping, task_keys: tuple, states: Mapping | None,
                        role: str, checkpoint: Path, run: Mapping, tasks: list) -> None:
    required = (
        (bank.get("schema_version"), BANK_SCHEMA), (bank.get("kind"), BANK_KIND),
        (role, "development_train"),
        (bank["spec"], file_record(REPO / "configs/demonstration_transfer_v1/learning_engineering_spec.json")),
        (bank["checkpoint"], str(checkpoint)), (bank["training_git"], run["git"]["commit"]),
        (bank["checkpoint_manifest"], file_record(checkpoint / "checkpoint_manifest.json")),
        (bank["tasks"], tasks), (set(task_keys), {(r["suite"], r["task_id"]) for r in tasks}),
        (bank["asset_root"], "/data1/user/ymdai/projects/EMBER"),
        (bank["scene_root"], str(Path(spec["runtime"]["run_root"]) / "scenes")),
    )
    if not all(actual == wanted for actual, wanted in required) or not source_matches(bank["source"], source):
        raise ValueError("comparison bank task/source/checkpoint/scene scope changed")
    if states is not None and any(tuple(states.get((r["suite"], r["task_id"]), ())) != (0, 1, 2, 3)
                                  for r in tasks):
        raise ValueError("comparison bank official init states changed")
    if path != Path(spec["runtime"]["run_root"]) / "banks" / arm / "manifest.json":
        raise ValueError("comparison bank is outside its registered root")
    authority = read_json(Path(bank["asset_root"]) / "configs/pi05_writer_data_v1.json")
    base = (Path(bank["asset_root"]) / authority["authorities"]["lora_contract"]).resolve()
    if bank["base_lora_contract"] != str(base):
        raise ValueError("comparison bank source LoRA authority changed")
    wall = {"teacher_video_values_read": 0 if arm == "M" else 8,
            "teacher_runtime_reads": 0, "deployment_adapters": 1,
            "validation_test_gradients": False}
    if bank["information_wall"] != wall:
        raise ValueError("comparison bank information wall changed")


def _inspect_bank_factors(bank: Mapping, base: Any, arm: str, conditions: list) -> None:
    rank = 128 if arm == "M" else 144
    if bank["lora"] != derive_pi05_lora_rank(base, rank=rank).to_dict():
        raise ValueError("comparison bank LoRA topology changed")
    _tensor_file(bank["shared"], expected_lora_state_shapes(derive_pi05_lora_rank(base, rank=128)))
    if arm == "M":
        if bank["conditions"] or bank["information_wall"]["teacher_video_values_read"] != 0:
            raise ValueError("M bank accessed teacher video")
        return
    stripped = [{k: v for k, v in row.items() if k not in ("factors", "raw_frames", "sampled_frames")}
                for row in bank["conditions"]]
    if len(bank["conditions"]) != 8 or stripped != conditions:
        raise ValueError("P/I bank condition mapping changed")
    shapes = expected_lora_state_shapes(derive_pi05_lora_rank(base, rank=16))
    for row in bank["conditions"]:
        _tensor_file(row["factors"], shapes)
        if row["sampled_frames"] <= 0 or row["raw_frames"] < row["sampled_frames"]:
            raise ValueError("teacher frame provenance invalid")


def _inspect_scenes(root: Path) -> dict:
    registered = read_json(root / "manifest.json")
    if registered.get("schema_version") != "ember_demonstration_comparison_scenes_v1" or len(registered["scenes"]) != 8:
        raise ValueError("eight shared canonical scenes missing")
    expected = {(suite, task, state) for suite, task in (("libero_spatial", 2), ("libero_10", 8))
                for state in range(4)}
    actual = set()
    for row in registered["scenes"]:
        actual.add((row["suite"], row["task_id"], row["state"]))
        if file_record(Path(row["path"])) != {"path": row["path"], "bytes": row["bytes"]}:
            raise ValueError("registered canonical scene missing")
    if actual != expected:
        raise ValueError("canonical scene task/state coverage changed")
    return registered


def inspect_bank(*, manifest_path: Path, source: Mapping, task_keys: tuple,
                 evaluation_role: str, require_formal: bool,
                 task_init_state_ids: Mapping | None = None) -> dict:
    del require_formal
    try:
        path = manifest_path.resolve()
        bank = read_json(path)
        spec = specification()
        arm = bank["arm"]
        tasks, conditions = _task_rows(spec, arm)
        checkpoint, run = _inspect_checkpoint(arm, spec, source)
        _inspect_bank_scope(bank, spec, arm, path, source, task_keys, task_init_state_ids,
                            evaluation_role, checkpoint, run, tasks)
        base = load_pi05_lora_contract(Path(bank["base_lora_contract"]))
        _inspect_bank_factors(bank, base, arm, conditions)
        scene = Path(bank["scene_root"])
        _inspect_scenes(scene)
        return {**bank, "schema_version": EVAL_SCHEMA, "manifest": file_record(path),
                "scene_manifest": file_record(scene / "manifest.json")}
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise Pi05EvaluationError(str(error)) from error


@dataclass(frozen=True)
class PreparedComparisonLoRA:
    key: str
    evidence: dict


class FrozenComparisonAdapter:
    """Load only compact LoRA factors; batch official policy calls without Writer."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        del device, require_formal
        bank = evaluation_adapter
        if bank.get("kind") != BANK_KIND or bank.get("schema_version") != EVAL_SCHEMA or not source_matches(bank["source"], source):
            raise Pi05EvaluationError("comparison execution bank changed")
        self.bank, self.policy = bank, policy
        self.tasks = {(row["suite"], row["task_id"]): row for row in bank["tasks"]}
        if set(self.tasks) != set(task_keys):
            raise Pi05EvaluationError("comparison worker task keys changed")
        base = load_pi05_lora_contract(Path(bank["base_lora_contract"]))
        self.lora = derive_pi05_lora_rank(base, rank=128 if bank["arm"] == "M" else 144)
        inject_task_lora(policy, self.lora)
        for value in task_lora_state_dict(policy).values():
            value.requires_grad_(False)
        policy.eval()
        self.batched = BatchedLoRAInference(policy, self.lora)
        self.identity = identity_lora_state(self.lora)
        self.shared = load_file(bank["shared"]["path"], device="cpu")
        self.conditions = {row["condition_id"]: row for row in bank["conditions"]}
        self.states: OrderedDict[str, dict] = OrderedDict()

    def _state(self, key: str) -> dict:
        if key in self.states:
            self.states.move_to_end(key)
            return self.states[key]
        if self.bank["arm"] == "M":
            if key != "shared":
                raise Pi05EvaluationError("M has no task-routed condition")
            state = self.shared
        else:
            row = self.conditions[key]
            if row["factors"] != file_record(Path(row["factors"]["path"])):
                raise Pi05EvaluationError("comparison condition changed during evaluation")
            state = concatenate_factors(self.shared, load_file(row["factors"]["path"], device="cpu"))
        validate_lora_state(state, self.lora)
        self.states[key] = state
        if len(self.states) > 8:
            self.states.popitem(last=False)
        return state

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedComparisonLoRA:
        task = self.tasks[(suite, task_id)]
        episode = next(row for row in task["episodes"] if row["init_state_id"] == init_state_id)
        return PreparedComparisonLoRA(episode["condition_id"], {
            "schema_version": EPISODE_SCHEMA, "arm": self.bank["arm"],
            "global_task_id": task["global_task_id"], "init_state_id": init_state_id,
            "condition_id": episode["condition_id"], "video_ordinal": episode["video_ordinal"],
            "teacher_demo": episode["teacher_demo"], "shared": self.bank["shared"],
            "checkpoint": self.bank["checkpoint"], "scene_manifest": self.bank["scene_manifest"]})

    @torch.no_grad()
    def install(self, prepared: PreparedComparisonLoRA) -> None:
        copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if not prepared or len(prepared) != noise.shape[0]:
            raise Pi05EvaluationError("comparison LoRA batch lost paired episodes")
        copy_task_lora_state_(self.policy, self.identity, self.lora)
        with self.batched.activate([self._state(item.key) for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()
        self.states.clear()


def validate_episode(bank: Mapping, evidence: Any, *, suite: str, task_id: int, init_state_id: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    task = next((row for row in bank["tasks"] if (row["suite"], row["task_id"]) == (suite, task_id)), None)
    if task is None:
        return False
    episode = next((row for row in task["episodes"] if row["init_state_id"] == init_state_id), None)
    if episode is None:
        return False
    expected = {"schema_version": EPISODE_SCHEMA, "arm": bank["arm"],
                "global_task_id": task["global_task_id"], "init_state_id": init_state_id,
                "condition_id": episode["condition_id"], "video_ordinal": episode["video_ordinal"],
                "teacher_demo": episode["teacher_demo"], "shared": bank["shared"],
                "checkpoint": bank["checkpoint"], "scene_manifest": bank["scene_manifest"]}
    return dict(evidence) == expected


def registered_capture(args, tasks, output_dir: Path, path: Path, manifest: Mapping,
                       task_subset: Mapping | None) -> tuple[dict, dict]:
    bank_path = Path(args.static_task_lora_manifest).resolve()
    bank = read_json(bank_path)
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": 0} for task in tasks]
    if (bank.get("kind") != BANK_KIND or manifest.get("schema_version") != "ember_pi05_registered_trajectory_capture_v1"
            or manifest.get("study_id") != "demonstration_transfer_comparison_admission_20260927"
            or manifest.get("task_subset_selection") != task_subset["selection_path"]
            or manifest.get("full_conditions") != full or manifest.get("mode") != "compact"
            or manifest.get("passive_control_trace") != PASSIVE_TAG
            or manifest.get("stage_predicates") is not True
            or args.role != "development_train" or args.mode != "screen"
            or tuple(tuple(task.init_state_ids) for task in tasks) != ((0, 1, 2, 3),) * 2
            or output_dir.resolve() != bank_path.parent.parent.parent / "evaluation" / bank["arm"]
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("comparison official capture scope changed")
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "selection_path": str(path), "selection_bytes": path.stat().st_size,
               "mode": "compact", "full_conditions": full,
               "trajectory_root": str((output_dir / "trajectories").resolve()),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "trace_root": str((output_dir / "continuous_traces").resolve())},
               "training_gradient_use": False, "checkpoint_selection_use": False,
               "validation_use": False, "test_use": False}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
             "full_conditions_only": False, "training_gradient_use": False,
             "checkpoint_selection_use": False, "validation_action_reads": 0,
             "validation_reward_reads": 0, "held_data_use": False,
             "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def attach_capture_provenance(contract: dict, repo_root: Path) -> None:
    del repo_root
    adapter = contract.get("adapter") or {}
    scene = contract.get("demonstration_comparison_scene") or {}
    if adapter.get("kind") != BANK_KIND or scene.get("manifest") != adapter.get("scene_manifest"):
        raise Pi05EvaluationError("comparison scene and bank adapter are not paired")
    contract["passive_capture_provenance"] = {
        "schema_version": PASSIVE_TAG, "bank": adapter["manifest"],
        "scene": adapter["scene_manifest"], "checkpoint": adapter["checkpoint"],
        "evaluation_commit": contract["git"]["commit"]}


def validate_capture_contract(contract: Mapping, repo_root: Path) -> None:
    adapter = contract.get("adapter") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    path = Path(capture["selection_path"])
    args = SimpleNamespace(static_task_lora_manifest=Path(adapter["manifest"]["path"]),
                           role=contract["role"], mode=contract["mode"])
    tasks = [SimpleNamespace(**row) for row in contract["tasks"]]
    expected, stage = registered_capture(args, tasks, Path(contract["output_dir"]),
                                         path, read_json(path), contract["diagnostic_task_subset"])
    regenerated = dict(contract)
    regenerated.pop("passive_capture_provenance", None)
    attach_capture_provenance(regenerated, repo_root)
    if (capture != expected or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("passive_capture_provenance") != regenerated["passive_capture_provenance"]):
        raise Pi05EvaluationError("comparison passive capture or scene provenance changed")


def prepare_selectors(asset_root: Path) -> dict:
    """Seal exactly two train tasks, four states and two state0 full captures."""
    from ember.pi05_eval_contract import inspect_installed_target_tasks, load_evaluation_authorities

    spec = specification()
    root = Path(spec["runtime"]["run_root"])
    output = root / "selectors"
    authorities = load_evaluation_authorities(
        asset_root / spec["source"]["evaluation_config"], asset_root)
    installed, _ = inspect_installed_target_tasks(
        authorities, role="development_train", state_count=4,
        libero_config_dir=output / "libero_config")
    selected = [(ordinal, task) for ordinal, task in enumerate(installed)
                if (task.suite, task.task_id) in {("libero_spatial", 2), ("libero_10", 8)}]
    if len(selected) != 2 or [2 if task.suite == "libero_spatial" else 38
                                  for _, task in selected] != [2, 38]:
        raise ValueError("registered official train task order changed")
    output.mkdir(parents=True, exist_ok=True)
    subset_path = output / "tasks.json"
    if subset_path.exists():
        raise ValueError("comparison selectors already exist")
    subset = {"schema_version": "ember_pi05_task_subset_selection_v1",
              "role": "development_train", "mode": "screen", "state_count": 4,
              "task_ordinals": [ordinal for ordinal, _ in selected],
              "global_task_ids": [2, 38],
              "tasks": [{"global_task_id": global_id, "suite": task.suite,
                         "task_id": task.task_id} for global_id, (_, task) in zip((2, 38), selected)],
              "init_state_ids": [0, 1, 2, 3], "outcome_dependence": False,
              "validation_use": False, "test_use": False}
    write_json_atomic(subset_path, subset)
    for arm in ("P", "I", "M"):
        capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
                   "study_id": "demonstration_transfer_comparison_admission_20260927",
                   "task_subset_selection": str(subset_path), "mode": "compact",
                   "full_conditions": [{"suite": task.suite, "task_id": task.task_id,
                                        "init_state_id": 0} for _, task in selected],
                   "passive_control_trace": PASSIVE_TAG, "stage_predicates": True,
                   "training_gradient_use": False, "checkpoint_selection_use": False,
                   "validation_use": False, "test_use": False}
        write_json_atomic(output / f"{arm}_capture.json", capture)
    return {"subset": str(subset_path), "captures": [str(output / f"{arm}_capture.json")
                                                    for arm in ("P", "I", "M")]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("materialize", "freeze-scenes", "selectors"))
    parser.add_argument("--arm", choices=("P", "I", "M"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int)
    args = parser.parse_args()
    if args.phase == "materialize":
        if args.arm is None or torch.cuda.device_count() != 1:
            raise ValueError("materialization requires one specified arm and one visible GPU")
        print(json.dumps(materialize(args.arm, args.asset_root, torch.device("cuda:0")), sort_keys=True))
    elif args.phase == "selectors":
        if args.arm is not None or args.physical_gpu_id is not None:
            raise ValueError("selectors are CPU-only and shared by all arms")
        print(json.dumps(prepare_selectors(args.asset_root), sort_keys=True))
    else:
        if args.arm is not None or args.physical_gpu_id is None:
            raise ValueError("scene freezing requires one physical rendering GPU and no arm")
        from ember.pi05_eval.scene import freeze_registered_scenes

        _frozen_git()
        spec = specification()
        root = Path(spec["runtime"]["run_root"]) / "scenes"
        print(json.dumps(freeze_registered_scenes(args.asset_root, root,
                                                   physical_gpu_id=args.physical_gpu_id), sort_keys=True))


if __name__ == "__main__":
    main()
