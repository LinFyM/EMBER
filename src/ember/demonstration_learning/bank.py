"""Read sealed formal P/I banks through the canonical evaluator."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.expert_manifold.video_schedule import reference_demo_index
from ember.lora import (LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_,
                        expected_lora_state_shapes, identity_lora_state, inject_task_lora,
                        task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.task_protocol import load_task_authorities


BANK_SCHEMA = "ember_demonstration_comparison_compact_bank_v1"
BANK_KIND = "demonstration_comparison_lora_bank"
EVAL_SCHEMA = "ember_demonstration_comparison_eval_adapter_v1"
EPISODE_SCHEMA = "ember_demonstration_comparison_episode_v1"
PASSIVE_TAG = "ember_demonstration_comparison_passive_capture_v1"
STUDY = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927")
ASSET_ROOT = Path("/data1/user/ymdai/projects/EMBER")
SPEC_RELATIVE = Path("configs/demonstration_transfer_v1/learning_spec.json")
FORMAL_STAGES = {
    "stage1": {"macro": 288, "attempt": "fresh", "git": "f4a80cd564843bb487bef55e3196d0597ed6cdd5",
               "freeze": Path("/data1/user/ymdai/projects/EMBER-demonstration-stage1-formal")},
    "stage2": {"macro": 576, "attempt": "from288", "git": "bc729e869c5b37f807e386ab11aa5b0b508ed86f",
               "freeze": Path("/data1/user/ymdai/projects/EMBER-demonstration-stage2-formal")},
}


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def source_matches(left: Mapping, right: Mapping) -> bool:
    return all(left.get(key) and right.get(key) and
               Path(left[key]).resolve() == Path(right[key]).resolve()
               for key in ("source_run", "checkpoint", "model_path"))


def _formal_origin(bank: Mapping, path: Path) -> tuple[str, Mapping]:
    arm = bank.get("arm")
    if arm not in ("P", "I"):
        raise ValueError("only sealed formal P/I banks are readable")
    for stage, origin in FORMAL_STAGES.items():
        expected = STUDY / stage / arm / "banks" / str(origin["macro"]) / "manifest.json"
        if path == expected:
            return stage, origin
    raise ValueError("bank is outside the four sealed formal P/I sources")


def inspect_formal_provenance(bank: Mapping, path: Path, source: Mapping) -> tuple[dict, Path, dict]:
    """Read the bank's own frozen spec and completed ECP metadata, never current Git."""
    path = path.resolve()
    stage, origin = _formal_origin(bank, path)
    arm, macro = bank["arm"], origin["macro"]
    spec_path = origin["freeze"] / SPEC_RELATIVE
    checkpoint = (STUDY / stage / arm / "train" / "attempts" / origin["attempt"]
                  / "checkpoints" / f"macro_{macro:08d}")
    if (bank.get("spec") != file_record(spec_path) or bank.get("training_git") != origin["git"]
            or bank.get("checkpoint") != str(checkpoint)
            or bank.get("checkpoint_manifest") != file_record(checkpoint / "checkpoint_manifest.json")
            or not source_matches(bank["source"], source)):
        raise ValueError("sealed bank spec/checkpoint/Git/source identity changed")
    if (bank.get("shared", {}).get("path") != str(path.parent / "shared.safetensors")
            or len(bank.get("conditions", ())) != 400
            or any(row.get("factors", {}).get("path") != str(path.parent / f"{row['condition_id']}.safetensors")
                   for row in bank["conditions"])):
        raise ValueError("sealed bank factor paths or condition count changed")
    spec = read_json(spec_path)
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    completion = read_json(checkpoint.parent.parent / "completion.json")
    parent = None if stage == "stage1" else {
        "git": FORMAL_STAGES["stage1"]["git"],
        "checkpoint": str(STUDY / "stage1" / arm / "train" / "attempts" / "fresh"
                          / "checkpoints" / "macro_00000288"),
    }
    expected = (
        (spec.get("schema_version"), "ember_demonstration_transfer_learning_spec_v1"),
        (spec.get("runtime", {}).get("study_root"), str(STUDY)),
        (spec.get("runtime", {}).get("run_root"), str(STUDY / stage)),
        (spec.get("bank", {}).get("source_macro"), macro),
        (spec.get("bank", {}).get("arms"), ["P", "I"]),
        (spec.get("bank", {}).get("task_ids"), [3, 6, 11, 16, 23, 26, 31, 39]),
        (spec.get("bank", {}).get("init_state_ids"), list(range(50))),
        (spec.get("bank", {}).get("video_schedule", {}).get("mode"), "correct"),
        (spec.get("bank", {}).get("video_schedule", {}).get("seed"), 7),
        (spec.get("bank", {}).get("video_schedule", {}).get("cardinality"), 1),
        (spec.get("bank", {}).get("video_schedule", {}).get("demos"), list(range(50))),
        (run.get("schema_version"), f"ember_demonstration_transfer_formal_{stage}_run_v1"),
        (run.get("stage"), f"demonstration_transfer_learning_{stage}"),
        (run.get("arm"), arm), (run.get("git", {}).get("commit"), origin["git"]),
        (run.get("git", {}).get("dirty_paths"), []),
        (run.get("git", {}).get("pushed_ref"), "origin/main"),
        (run.get("spec"), str(spec_path)), (run.get("source_trainable"), 0),
        (run.get("source"), bank["source"]), (run.get("model"), spec["model"]),
        (run.get("optimizer"), spec["optimization"]),
        (run.get("source_identity_before_every_compile"), True),
        (run.get("topology", {}).get("world_size"), 2),
        (run.get("qualification"), f"formal_{stage}_macro{macro}"),
        (run.get("stage1_parent"), parent),
        (manifest.get("schema_version"), "ember_ecp_checkpoint_v1"),
        (manifest.get("stage"), f"demonstration_transfer_learning_{stage}"),
        (manifest.get("run_contract_schema"), run["schema_version"]),
        (manifest.get("next_macro"), macro), (manifest.get("world_size"), 2),
        (completion.get("status"), "formal_training_complete"),
        (completion.get("updates"), macro), (completion.get("checkpoint"), str(checkpoint)),
    )
    if not all(actual == wanted for actual, wanted in expected):
        raise ValueError("sealed bank formal training provenance changed")
    if spec["source"]["checkpoint"] != str(Path(bank["source"]["checkpoint"]).relative_to(ASSET_ROOT)):
        raise ValueError("sealed bank source checkpoint differs from frozen spec")
    for name, record in manifest["files"].items():
        file = checkpoint / name
        if not file.is_file() or file.stat().st_size != int(record["bytes"]):
            raise ValueError("sealed checkpoint file missing or truncated")
    return spec, checkpoint, run


def concatenate_factors(common: Mapping[str, torch.Tensor],
                        conditional: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One rank144 factor set computes Bc Ac + Bv Av without cross terms."""
    if len(common) != 76 or set(common) != set(conditional):
        raise ValueError("complete 38-target shared/conditional factor set changed")
    result = {}
    for name, shared in common.items():
        value = conditional[name]
        axis = 0 if name.endswith(LORA_A_SUFFIX) else 1 if name.endswith(LORA_B_SUFFIX) else -1
        if (axis < 0 or shared.ndim != 2 or value.ndim != 2
                or shared.shape[axis] != 128 or value.shape[axis] != 16
                or shared.shape[1 - axis] != value.shape[1 - axis]):
            raise ValueError(f"sealed bank LoRA shapes differ: {name}")
        result[name] = torch.cat((shared, value), dim=axis)
    return result


def _task_rows(spec: Mapping, asset_root: Path) -> tuple[list, list]:
    """Use only official protocol and task metadata; no held episode labels."""
    _, manifest = load_task_authorities(asset_root, spec["data"]["protocol"])
    authorities = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    tasks, conditions = [], {}
    for global_id in spec["bank"]["task_ids"]:
        task = authorities[global_id]
        if task["split_role"] != "validation":
            raise ValueError("sealed bank task crosses validation authority")
        suite, local, language = task["suite"], int(task["task_id"]), str(task["language"])
        episodes = []
        for state in spec["bank"]["init_state_ids"]:
            demo = reference_demo_index(spec["bank"]["video_schedule"]["seed"],
                                        suite, local, state,
                                        demo_count=50, sampling_mode="without_replacement")
            key = f"task_{global_id:03d}_demo_{demo:02d}"
            episodes.append({"init_state_id": state, "condition_id": key,
                             "teacher_demo": demo,
                             "video_ordinal": demo})
            conditions[key] = {"condition_id": key, "global_task_id": global_id,
                               "teacher_demo": demo, "language": language}
        tasks.append({"global_task_id": global_id, "suite": suite,
                      "task_id": local, "language": language,
                      "split_role": "validation", "episodes": episodes})
    return tasks, list(conditions.values())


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


def _inspect_bank_scope(bank: Mapping, source: Mapping,
                        task_keys: tuple, states: Mapping | None, role: str,
                        checkpoint: Path, run: Mapping, tasks: list) -> None:
    required = (
        (bank.get("schema_version"), BANK_SCHEMA), (bank.get("kind"), BANK_KIND),
        (role, "validation"),
        (bank["spec"], file_record(Path(run["spec"]))),
        (bank["checkpoint"], str(checkpoint)), (bank["training_git"], run["git"]["commit"]),
        (bank["checkpoint_manifest"], file_record(checkpoint / "checkpoint_manifest.json")),
        (bank["tasks"], tasks), (set(task_keys), {(r["suite"], r["task_id"]) for r in tasks}),
        (bank["asset_root"], str(ASSET_ROOT)),
        (bank["scene_root"], str(STUDY / "scenes")),
    )
    if not all(actual == wanted for actual, wanted in required) or not source_matches(bank["source"], source):
        raise ValueError("comparison bank task/source/checkpoint/scene scope changed")
    if states is not None and any(tuple(states.get((r["suite"], r["task_id"]), ())) != tuple(range(50))
                                  for r in tasks):
        raise ValueError("comparison bank official init states changed")
    authority = read_json(Path(bank["asset_root"]) / "configs/pi05_writer_data_v1.json")
    base = (Path(bank["asset_root"]) / authority["authorities"]["lora_contract"]).resolve()
    if bank["base_lora_contract"] != str(base):
        raise ValueError("comparison bank source LoRA authority changed")
    wall = {"teacher_video_values_read": 400,
            "teacher_runtime_reads": 0, "deployment_adapters": 1,
            "validation_test_gradients": False}
    if bank["information_wall"] != wall:
        raise ValueError("comparison bank information wall changed")


def _inspect_bank_factors(bank: Mapping, base: Any, conditions: list) -> None:
    if bank["lora"] != derive_pi05_lora_rank(base, rank=144).to_dict():
        raise ValueError("comparison bank LoRA topology changed")
    _tensor_file(bank["shared"], expected_lora_state_shapes(derive_pi05_lora_rank(base, rank=128)))
    stripped = [{k: v for k, v in row.items() if k not in ("factors", "raw_frames", "sampled_frames")}
                for row in bank["conditions"]]
    if len(bank["conditions"]) != 400 or stripped != conditions:
        raise ValueError("P/I bank condition mapping changed")
    shapes = expected_lora_state_shapes(derive_pi05_lora_rank(base, rank=16))
    for row in bank["conditions"]:
        _tensor_file(row["factors"], shapes)
        if row["sampled_frames"] <= 0 or row["raw_frames"] < row["sampled_frames"]:
            raise ValueError("teacher frame provenance invalid")


def _inspect_scenes(root: Path, tasks: list) -> dict:
    registered = read_json(root / "manifest.json")
    if registered.get("schema_version") != "ember_demonstration_formal_scenes_v1" or len(registered["scenes"]) != 400:
        raise ValueError("400 shared canonical scenes missing")
    expected = {(task["suite"], task["task_id"], state) for task in tasks for state in range(50)}
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
    try:
        path = manifest_path.resolve()
        bank = read_json(path)
        if not require_formal:
            raise ValueError("formal correct400 admits only P/I banks")
        spec, checkpoint, run = inspect_formal_provenance(bank, path, source)
        tasks, conditions = _task_rows(spec, Path(bank["asset_root"]))
        _inspect_bank_scope(bank, source, task_keys, task_init_state_ids,
                            evaluation_role, checkpoint, run, tasks)
        base = load_pi05_lora_contract(Path(bank["base_lora_contract"]))
        _inspect_bank_factors(bank, base, conditions)
        scene = Path(bank["scene_root"])
        _inspect_scenes(scene, tasks)
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
        if bank["arm"] not in ("P", "I"):
            raise Pi05EvaluationError("formal worker admits only P/I")
        self.lora = derive_pi05_lora_rank(base, rank=144)
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
    try:
        _formal_origin(bank, bank_path)
    except ValueError as error:
        raise Pi05EvaluationError(str(error)) from error
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": 0} for task in tasks]
    if (bank.get("kind") != BANK_KIND or manifest.get("schema_version") != "ember_pi05_registered_trajectory_capture_v1"
            or manifest.get("study_id") != "demonstration_transfer_learning_20260927"
            or task_subset is not None or manifest.get("task_subset_selection") is not None
            or manifest.get("full_conditions") != full or manifest.get("mode") != "compact"
            or manifest.get("passive_control_trace") != PASSIVE_TAG
            or manifest.get("stage_predicates") is not True
            or args.role != "validation" or args.mode != "formal" or len(tasks) != 8
            or tuple(tuple(task.init_state_ids) for task in tasks) != (tuple(range(50)),) * 8
            or output_dir.resolve() != bank_path.parent.parent.parent / "evaluation" / "correct400"
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
