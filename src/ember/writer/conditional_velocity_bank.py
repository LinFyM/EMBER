"""Read sealed velocity banks through the existing PI0.5 evaluator interface."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA
from ember.lora import (LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_, expected_lora_state_shapes,
                        identity_lora_state, inject_task_lora,
                        task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.task_protocol import load_task_authorities
from ember.writer.materialization import file_record, planned_episodes, selection_contract, source_matches


BANK_SCHEMA = "ember_conditional_velocity_compact_bank_v1"
BANK_KIND = "conditional_velocity_lora_bank"
EVAL_SCHEMA = "ember_conditional_velocity_eval_adapter_v1"
EPISODE_SCHEMA = "ember_conditional_velocity_episode_v1"
PASSIVE_TAG = "ember_conditional_velocity_passive_capture_v1"
RUN_SCHEMA = "ember_conditional_velocity_learning_run_v1"
STAGE = "conditional_velocity_learning"
SPEC_SCHEMA = "ember_conditional_velocity_learning_v1"
STUDY_ROOT = Path("/data1/user/ymdai/ember_runs/conditional_velocity_operator_learning_20260927")


def compile_velocity_state(
    common: Mapping[str, torch.Tensor], coefficient: torch.Tensor, projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rebuild the single rank-135 state from sealed common and condition factors."""
    if coefficient.shape != (7, 256) or projection.shape != (256, 1024):
        raise ValueError("conditional velocity coefficient or shared projection changed")
    if len(common) != 76 or any(not name.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)) for name in common):
        raise ValueError("common actor lost the complete 38-target LoRA")
    result = {}
    for name, value in common.items():
        is_a = name.endswith(LORA_A_SUFFIX)
        if value.ndim != 2 or value.shape[0 if is_a else 1] != 128:
            raise ValueError(f"common actor rank changed: {name}")
        if is_a:
            extra = (coefficient.float() @ projection.float() if name.startswith("model.action_out_proj")
                     else value.new_zeros(7, value.shape[1]))
            if extra.shape[1] != value.shape[1]:
                raise ValueError("conditional action_out input width changed")
            result[name] = torch.cat((value, extra.to(value.dtype)), dim=0)
        else:
            extra = value.new_zeros(value.shape[0], 7)
            if name.startswith("model.action_out_proj"):
                if value.shape[0] != 32:
                    raise ValueError("native action_out output width changed")
                extra[:7] = torch.eye(7, device=value.device, dtype=value.dtype)
            result[name] = torch.cat((value, extra), dim=1)
    return result


def canonical_selection(spec: Mapping) -> dict:
    return selection_contract(role="validation", task_ids=spec["validation_tasks"], cardinality=1,
                              arm="correct", mode="per_init_ordinal",
                              seed=spec["evaluation"]["video_schedule_seed"],
                              init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))


def _bank_spec(bank: Mapping) -> tuple[dict, Path]:
    """Use the sealed frozen-tree spec named by a historical bank."""
    path = Path(bank["spec"]["path"]).resolve()
    if bank["spec"] != file_record(path):
        raise ValueError("velocity bank frozen spec changed")
    spec = read_json(path)
    if spec.get("schema_version") != SPEC_SCHEMA or Path(spec.get("run_root", "")) != STUDY_ROOT:
        raise ValueError("velocity bank study/spec identity changed")
    return spec, path.parents[2]


def _expected_checkpoint(spec: Mapping, mode: str, update: int) -> tuple[Path, dict | None]:
    if mode not in ("V", "L"):
        raise ValueError("unknown velocity bank mode")
    if update == spec["first_stop_update"] == 270:
        return STUDY_ROOT / mode / "checkpoints" / "macro_00000270", None
    continuation = spec.get("continuation_450") or {}
    if update == continuation.get("qualification_update") == 450:
        expected = {**continuation, "parent_checkpoint": str(
            STUDY_ROOT / mode / "checkpoints" / "macro_00000270")}
        path = STUDY_ROOT / continuation["subdir"] / mode / "checkpoints" / "macro_00000450"
        return path, expected
    raise ValueError("bank checkpoint is outside the sealed 270/450 decisions")


def _checkpoint(bank: Mapping, spec: Mapping, frozen_root: Path) -> tuple[dict, dict]:
    mode, update = bank["mode"], bank["checkpoint"]["macro"]
    expected, continuation = _expected_checkpoint(spec, mode, update)
    checkpoint = Path(bank["checkpoint"]["path"]).resolve()
    if checkpoint != expected.resolve():
        raise ValueError("velocity bank checkpoint path changed")
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = read_json(run_path)
    record = read_json(checkpoint / "checkpoint_manifest.json")
    expected_files = {"ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt"}
    frozen_git = git_state(frozen_root)
    if (run.get("schema_version") != RUN_SCHEMA or run.get("stage") != STAGE
            or run.get("mode") != mode or run.get("source_trainable") != 0
            or run.get("scientific_qualification") is not True
            or run.get("spec") != str(Path(bank["spec"]["path"]).resolve())
            or run.get("continuation") != continuation
            or run.get("git", {}).get("commit") != frozen_git["commit"]
            or frozen_git.get("branch") != "" or frozen_git.get("dirty_paths")
            or record.get("schema_version") != ECP_CHECKPOINT_SCHEMA
            or record.get("stage") != STAGE or record.get("run_contract_schema") != RUN_SCHEMA
            or record.get("next_macro") != update or record.get("world_size") != 2
            or set(record.get("files", {})) != expected_files):
        raise ValueError("velocity bank checkpoint/frozen authority changed")
    for name, item in record["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"checkpoint file changed: {name}")
    _inspect_training_cursor(checkpoint, run, update)
    return run, {"path": str(checkpoint), "manifest": file_record(checkpoint / "checkpoint_manifest.json"),
                 "weights": file_record(checkpoint / "ecp.safetensors"), "macro": update}


def _inspect_training_cursor(checkpoint: Path, run: Mapping, update: int) -> None:
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta",
                         mmap=True, weights_only=True)
    sampler = trainer.get("sampler_state", {})
    if (trainer.get("schema_version") != ECP_CHECKPOINT_SCHEMA
            or trainer.get("stage") != STAGE or trainer.get("next_macro") != update
            or trainer.get("training_state") != {"updates": update}
            or trainer.get("metrics_rows") != update
            or sampler.get("next_step") != update
            or {key: value for key, value in sampler.items() if key != "next_step"} != run["sampler"]):
        raise ValueError("velocity trainer/sampler cursor changed")
    metrics = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()[:update]
    if len(metrics) != update or [json.loads(row)["update"] for row in metrics] != list(range(1, update + 1)):
        raise ValueError("velocity checkpoint metrics prefix changed")


def _expected_episodes(selection: Mapping, task: int, mode: str) -> list[dict]:
    rows = planned_episodes(selection, task)
    if mode == "L":
        for row in rows:
            row["condition_id"] = f"task_{task:02d}_language"
            row["scheduled_video_demo_indices"] = row.pop("teacher_demo_indices")
    return rows


def _inspect_bank_identity(bank, spec, frozen_root, source, evaluation_role, require_formal):
    mode = bank["mode"]
    if (bank.get("schema_version") != BANK_SCHEMA or bank.get("kind") != BANK_KIND
            or bank.get("status") != "sealed" or evaluation_role != "validation"
            or not require_formal or bank.get("arm") != "correct"
            or bank.get("evaluation_role") != "validation"
            or bank.get("selection") != canonical_selection(spec)
            or bank.get("protocol") != spec["protocol"]
            or not source_matches(bank["source"], source)):
        raise ValueError("velocity bank protocol, source or official selection changed")
    run, checkpoint = _checkpoint(bank, spec, frozen_root)
    if (bank["checkpoint"] != checkpoint or bank["training_run"] != file_record(
            Path(checkpoint["path"]).parent.parent / "run_contract.json")
            or run["mode"] != mode):
        raise ValueError("velocity bank training/checkpoint identity changed")
    _inspect_shared_file(bank, mode)


def _inspect_shared_file(bank, mode):
    base = Path(bank["base_lora_contract"]["path"])
    if bank["base_lora_contract"] != file_record(base):
        raise ValueError("bank LoRA authority changed")
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(base), rank=135)
    if (bank["lora"] != lora.to_dict() or lora.alpha != 135 or len(lora.targets) != 38
            or bank["shared"] != file_record(Path(bank["shared"]["path"]))):
        raise ValueError("compact bank common rank/topology changed")
    with safe_open(bank["shared"]["path"], framework="pt", device="cpu") as handle:
        expected_common = expected_lora_state_shapes(
            derive_pi05_lora_rank(load_pi05_lora_contract(base), rank=128))
        if (handle.metadata() != {"schema_version": BANK_SCHEMA, "mode": mode,
                                  "macro": str(bank["checkpoint"]["macro"])}
                or set(handle.keys()) != set(expected_common) | {"velocity_U"}
                or tuple(handle.get_slice("velocity_U").get_shape()) != (256, 1024)
                or handle.get_slice("velocity_U").get_dtype() != "F32"
                or any(tuple(handle.get_slice(name).get_shape()) != shape or
                       handle.get_slice(name).get_dtype() != "F32"
                       for name, shape in expected_common.items())):
            raise ValueError("compact common β/U is incomplete")


def _inspect_bank_tasks(bank, spec, task_keys, task_init_state_ids):
    rows, mode = bank["tasks"], bank["mode"]
    selected = [(row["suite"], row["task_id"]) for row in rows]
    if (len(rows) != 8 or len(set(selected)) != 8 or set(selected) != set(task_keys)
            or [row["global_task_id"] for row in rows] != spec["validation_tasks"]
            or any(row["episodes"] != _expected_episodes(bank["selection"], row["global_task_id"], mode)
                   for row in rows)):
        raise ValueError("velocity official400 episode pairing changed")
    if task_init_state_ids is not None and any(
            tuple(task_init_state_ids.get(key, range(50))) != tuple(range(50)) for key in selected):
        raise ValueError("velocity official400 init states changed")
    _, authority = load_task_authorities(Path(bank["asset_root"]), spec["protocol"])
    canonical = {row["global_task_id"]: row for row in authority["tasks"]}
    if any((row["suite"], row["task_id"], row["language"], row["split_role"]) !=
           tuple(canonical[row["global_task_id"]][key]
                 for key in ("suite", "task_id", "language", "split_role")) for row in rows):
        raise ValueError("velocity task/language authority changed")
    data_root = Path(bank["asset_root"]) / "data/datasets" / authority["dataset"]["revision"]
    if any(row["teacher_source"] != {
            "path": str((data_root / canonical[row["global_task_id"]]["hdf5"]["relative_path"]).resolve()),
            "bytes": int(canonical[row["global_task_id"]]["hdf5"]["bytes"])} for row in rows):
        raise ValueError("velocity teacher provenance changed")


def _inspect_bank_conditions(bank):
    rows, mode = bank["tasks"], bank["mode"]
    expected = {episode["condition_id"] for row in rows for episode in row["episodes"]}
    conditions = {row["condition_id"]: row for row in bank["conditions"]}
    if len(conditions) != len(bank["conditions"]) or set(conditions) != expected:
        raise ValueError("velocity condition bank is incomplete")
    languages = {row["global_task_id"]: row["language"] for row in rows}
    for key, row in conditions.items():
        item = row["coefficient"]
        if item != file_record(Path(item["path"])):
            raise ValueError("velocity R file changed")
        with safe_open(item["path"], framework="pt", device="cpu") as handle:
            if (handle.metadata() != {"schema_version": BANK_SCHEMA, "condition_id": key, "mode": mode}
                    or handle.keys() != ["R"] or tuple(handle.get_slice("R").get_shape()) != (7, 256)
                    or handle.get_slice("R").get_dtype() != "F32"):
                raise ValueError("velocity R file shape/identity changed")
        if mode == "L":
            if row["source"] != {"kind": "exact_language_only",
                                  "language": languages[row["global_task_id"]], "video_values_read": 0}:
                raise ValueError("language condition provenance changed")
        elif (row["source"].get("kind") != "ordered_agentview_rgb" or
              row["source"].get("sampled_frames") != len(row["source"].get("frame_indices", []))):
            raise ValueError("video condition provenance changed")
    wall = bank["information_wall"]
    if (wall.get("teacher_video_values_read") != (len(conditions) if mode == "V" else 0)
            or wall.get("execution_adapters") != 1 or wall.get("teacher_video_runtime_reads") != 0
            or wall.get("validation_test_gradients") is not False):
        raise ValueError("velocity information wall changed")


def inspect_velocity_bank(*, manifest_path: Path, source: Mapping[str, Any],
                          task_keys: Sequence[tuple[str, int]], evaluation_role: str,
                          require_formal: bool, task_init_state_ids: Mapping | None = None) -> dict:
    try:
        path = manifest_path.resolve()
        bank = read_json(path)
        spec, frozen_root = _bank_spec(bank)
        _inspect_bank_identity(bank, spec, frozen_root, source, evaluation_role, require_formal)
        _inspect_bank_tasks(bank, spec, task_keys, task_init_state_ids)
        _inspect_bank_conditions(bank)
        return {**bank, "schema_version": EVAL_SCHEMA, "manifest": file_record(path)}
    except (KeyError, TypeError, ValueError, OSError, StopIteration) as error:
        raise Pi05EvaluationError(str(error)) from error


def episode_evidence(adapter: Mapping, task: Mapping, episode: Mapping) -> dict:
    condition = next(row for row in adapter["conditions"] if row["condition_id"] == episode["condition_id"])
    return {"schema_version": EPISODE_SCHEMA, "condition_id": episode["condition_id"],
            "global_task_id": task["global_task_id"], "init_state_id": episode["init_state_id"],
            "mode": adapter["mode"], "coefficient": condition["coefficient"],
            "shared": adapter["shared"], "checkpoint": adapter["checkpoint"],
            "paired_correct_demos": episode["paired_correct_demos"],
            "paired_other_demos": episode["paired_other_demos"],
            "video_ordinal": episode["video_ordinal"],
            "selection_seed": adapter["selection"]["seed"],
            "selection_mode": adapter["selection"]["mode"], "K": 1}


def validate_velocity_episode(adapter: Mapping, evidence: Any, *, suite: str, task_id: int,
                              init_state_id: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    for task in adapter.get("tasks", ()):
        if (task["suite"], task["task_id"]) == (suite, task_id):
            for episode in task["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return dict(evidence) == episode_evidence(adapter, task, episode)
    return False


def validate_velocity_adapter_fields(adapter: Mapping, row: Mapping, *, suite: str,
                                     task_id: int, init_state_id: int) -> bool:
    if any(row.get(name) is not None for name in (
            "horizon_writer_lora", "static_task_lora", "task_expert", "policy_adapter_sha256")):
        return False
    return validate_velocity_episode(adapter, row.get("conditional_velocity_lora"),
                                     suite=suite, task_id=task_id, init_state_id=init_state_id)


@dataclass(frozen=True)
class PreparedVelocityLoRA:
    key: str
    evidence: dict


class FrozenVelocityAdapter:
    """Rebuild one exact LoRA before each episode; batch through the existing hooks."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        del device, require_formal
        adapter = evaluation_adapter
        self.records = {(row["suite"], row["task_id"]): row for row in adapter["tasks"]}
        if (adapter.get("kind") != BANK_KIND or adapter.get("schema_version") != EVAL_SCHEMA
                or set(self.records) != set(task_keys) or not source_matches(adapter["source"], source)):
            raise Pi05EvaluationError("velocity execution bank changed")
        self.adapter, self.policy = adapter, policy
        self.conditions = {row["condition_id"]: row for row in adapter["conditions"]}
        self.lora = derive_pi05_lora_rank(load_pi05_lora_contract(
            Path(adapter["base_lora_contract"]["path"])), rank=135)
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.batched = BatchedLoRAInference(policy, self.lora)
        self.identity = identity_lora_state(self.lora)
        if adapter["shared"] != file_record(Path(adapter["shared"]["path"])):
            raise Pi05EvaluationError("velocity shared file changed after bank inspection")
        shared = load_file(adapter["shared"]["path"], device="cpu")
        self.U = shared.pop("velocity_U")
        self.common = shared
        self._states: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._installed: str | None = None

    def _state(self, key: str) -> dict[str, torch.Tensor]:
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key]
        row = self.conditions[key]
        if row["coefficient"] != file_record(Path(row["coefficient"]["path"])):
            raise Pi05EvaluationError("velocity coefficient changed after bank inspection")
        R = load_file(row["coefficient"]["path"], device="cpu")["R"]
        state = compile_velocity_state(self.common, R, self.U)
        validate_lora_state(state, self.lora)
        self._states[key] = state
        if len(self._states) > 16:
            self._states.popitem(last=False)
        return state

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedVelocityLoRA:
        row = self.records.get((suite, task_id))
        if row is not None:
            for episode in row["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return PreparedVelocityLoRA(episode["condition_id"], episode_evidence(self.adapter, row, episode))
        raise Pi05EvaluationError("rollout state is outside the velocity official400 bank")

    @torch.no_grad()
    def install(self, prepared: PreparedVelocityLoRA) -> None:
        if prepared.key != self._installed:
            copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)
            self._installed = prepared.key

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if not prepared or len(prepared) != noise.shape[0] or any(item.key not in self.conditions for item in prepared):
            raise Pi05EvaluationError("velocity batch and paired bank conditions differ")
        if self._installed is not None:
            copy_task_lora_state_(self.policy, self.identity, self.lora)
            self._installed = None
        with self.batched.activate([self._state(item.key) for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()
        self._states.clear()


def registered_capture(args, tasks, output_dir, path, manifest, task_subset, repo_root, schema):
    """Reconstruct the sealed full/compact capture contract for old official rows."""
    del repo_root
    full = tuple((str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
                 for row in manifest.get("full_conditions", ()))
    expected = {(str(task.suite), int(task.task_id), 0) for task in tasks}
    bank_path = getattr(args, "static_task_lora_manifest", None)
    if bank_path is None:
        raise Pi05EvaluationError("velocity capture requires a sealed bank")
    bank = read_json(bank_path)
    spec, frozen_root = _bank_spec(bank)
    mode = bank.get("mode")
    if (path != (frozen_root / "configs/conditional_velocity_operator_v1/official_capture.json").resolve()
            or mode not in ("V", "L")
            or Path(spec["run_root"]) != STUDY_ROOT
            or output_dir.resolve() != (Path(bank["training_run"]["path"]).parent /
                                        "evaluation/correct400").resolve()
            or args.role != "validation" or task_subset is not None or len(tasks) != 8
            or bank.get("kind") != BANK_KIND
            or manifest.get("schema_version") != schema
            or manifest.get("mode") != "compact" or set(full) != expected or len(full) != 8
            or manifest.get("passive_control_trace") != PASSIVE_TAG
            or manifest.get("stage_predicates") is not True
            or any(tuple(task.init_state_ids) != tuple(range(50)) for task in tasks)
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("velocity official400 full/compact capture contract changed")
    capture = {"schema_version": schema, "selection_path": str(path),
               "selection_bytes": path.stat().st_size, "mode": "compact",
               "full_conditions": [{"suite": suite, "task_id": task_id, "init_state_id": state}
                                   for suite, task_id, state in full],
               "trajectory_root": str((output_dir / "trajectories").resolve()),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "spec_path": bank["spec"]["path"],
                                 "spec_bytes": bank["spec"]["bytes"],
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


def attach_capture_provenance(contract_row: dict, repo_root: Path) -> None:
    del repo_root
    capture = contract_row.get("diagnostic_occupancy_capture") or {}
    trace = capture.get("passive_trace") or {}
    if trace.get("schema_version") != PASSIVE_TAG:
        return
    adapter = contract_row.get("adapter") or {}
    if adapter.get("kind") != BANK_KIND:
        raise Pi05EvaluationError("velocity passive capture requires the sealed bank")
    bank = read_json(Path(adapter["manifest"]["path"]))
    run = read_json(Path(bank["training_run"]["path"]))
    if contract_row["git"]["commit"] != run["git"]["commit"]:
        raise Pi05EvaluationError("velocity passive capture bank/code identity changed")
    contract_row["passive_capture_provenance"] = {
        "schema_version": PASSIVE_TAG, "bank_manifest": adapter["manifest"],
        "checkpoint": adapter["checkpoint"],
        "evaluation_commit": contract_row["git"]["commit"],
    }


def validate_capture_contract(contract_row: Mapping, repo_root: Path) -> None:
    capture = contract_row.get("diagnostic_occupancy_capture") or {}
    trace = capture.get("passive_trace") or {}
    if trace.get("schema_version") != PASSIVE_TAG:
        raise Pi05EvaluationError("velocity passive capture tag changed")
    adapter = contract_row.get("adapter") or {}
    path = Path(capture["selection_path"]).resolve()
    args = SimpleNamespace(role=contract_row["role"],
                           static_task_lora_manifest=Path(adapter["manifest"]["path"]))
    tasks = tuple(SimpleNamespace(**row) for row in contract_row["tasks"])
    expected, stage = registered_capture(
        args, tasks, Path(contract_row["output_dir"]), path, read_json(path), None,
        repo_root, "ember_pi05_registered_trajectory_capture_v1",
    )
    fresh = dict(contract_row)
    fresh.pop("passive_capture_provenance", None)
    attach_capture_provenance(fresh, repo_root)
    if (capture != expected or contract_row.get("diagnostic_stage_predicates") != stage
            or contract_row.get("passive_capture_provenance") != fresh["passive_capture_provenance"]):
        raise Pi05EvaluationError("velocity passive all-row capture or provenance changed")


if __name__ == "__main__":
    raise SystemExit("conditional-velocity bank materialization is retired")
