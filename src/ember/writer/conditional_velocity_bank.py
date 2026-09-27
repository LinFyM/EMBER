"""Compact rank-135 velocity bank and the existing PI0.5 batch evaluator interface."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA
from ember.lora import (copy_task_lora_state_, expected_lora_state_shapes,
                        identity_lora_state, inject_task_lora,
                        task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.task_protocol import load_task_authorities
from ember.writer.conditional_velocity import compile_velocity_state
from ember.writer.conditional_velocity_training import (ROOT, SCHEMA as RUN_SCHEMA, SPEC, STAGE,
                                                        build_runtime, continuation_checkpoint,
                                                        continuation_record, continuation_root,
                                                        contract, require_frozen)
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import (file_record, planned_episodes, selection_contract,
                                          source_matches)


BANK_SCHEMA = "ember_conditional_velocity_compact_bank_v1"
BANK_KIND = "conditional_velocity_lora_bank"
EVAL_SCHEMA = "ember_conditional_velocity_eval_adapter_v1"
EPISODE_SCHEMA = "ember_conditional_velocity_episode_v1"
PASSIVE_TAG = "ember_conditional_velocity_passive_capture_v1"


def canonical_selection(spec: Mapping) -> dict:
    return selection_contract(role="validation", task_ids=spec["validation_tasks"], cardinality=1,
                              arm="correct", mode="per_init_ordinal",
                              seed=spec["evaluation"]["video_schedule_seed"],
                              init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))


def _checkpoint(checkpoint: Path, mode: str) -> tuple[dict, dict]:
    spec, _ = contract()
    update = spec["continuation_450"]["qualification_update"]
    checkpoint = checkpoint.resolve()
    if checkpoint != continuation_checkpoint(spec, mode, update).resolve():
        raise ValueError("450 bank requires the registered mode450 checkpoint")
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    record = read_json(checkpoint / "checkpoint_manifest.json")
    expected_files = {"ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt"}
    if (run.get("schema_version") != RUN_SCHEMA or run.get("stage") != STAGE
            or run.get("mode") != mode or run.get("source_trainable") != 0
            or run.get("scientific_qualification") is not True
            or run.get("spec") != str(SPEC)
            or run.get("continuation") != continuation_record(spec, mode)
            or record.get("schema_version") != ECP_CHECKPOINT_SCHEMA
            or record.get("stage") != STAGE or record.get("run_contract_schema") != RUN_SCHEMA
            or record.get("next_macro") != update or record.get("world_size") != 2
            or set(record.get("files", {})) != expected_files):
        raise ValueError("velocity checkpoint or optimizer cursor is outside the formal batch")
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


def materialize(*, asset_root: Path, checkpoint: Path, output: Path, mode: str,
                device: torch.device) -> Path:
    spec, reference = contract()
    root = continuation_root(spec, mode)
    require_frozen(root, mode)
    run, checkpoint_record = _checkpoint(checkpoint, mode)
    update = spec["continuation_450"]["qualification_update"]
    if (output.resolve() != (root / "banks" / str(update)).resolve()
            or device.type != "cuda" or run["git"]["commit"] != git_state(ROOT)["commit"]):
        raise ValueError("bank requires clean frozen GPU materialization of its exact checkpoint")
    tasks = load_learning_tasks(asset_root, spec["validation_tasks"], role="validation",
                                protocol_path=spec["protocol"])
    selection = canonical_selection(spec)
    runtime = build_runtime(asset_root, reference, device, mode=mode)
    if not source_matches(runtime.source, run["source"]):
        raise ValueError("bank source differs from training source")
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.requires_grad_(False).eval()
    runtime.policy.eval()
    base_path = asset_root / read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]["lora_contract"]
    common = {key: value.detach().float().cpu().contiguous() for key, value in runtime.state.common().items()}
    U = runtime.state.U.weight.detach().float().cpu().contiguous()
    output.mkdir(parents=True, exist_ok=False)
    (output / "coefficients").mkdir()
    shared_path = output / "shared.safetensors"
    save_file({**common, "velocity_U": U}, str(shared_path),
              metadata={"schema_version": BANK_SCHEMA, "mode": mode, "macro": str(update)})
    store = None
    if mode == "V":
        from ember.writer.data import RawTeacherVideoStore

        store = RawTeacherVideoStore(tuple(row.authority for row in tasks.values()),
                                    frame_stride=5, camera_view="agentview")
    conditions, task_rows = [], []
    try:
        for task_id in spec["validation_tasks"]:
            task = tasks[task_id]
            episodes = planned_episodes(selection, task_id)
            for episode in episodes:
                if mode == "L":
                    episode["condition_id"] = f"task_{task_id:02d}_language"
                    episode["scheduled_video_demo_indices"] = episode.pop("teacher_demo_indices")
            for episode in episodes:
                name = episode["condition_id"]
                if mode == "L" and any(row["condition_id"] == name for row in conditions):
                    continue
                demo = episode["teacher_demo_indices"][0] if mode == "V" else -1
                with torch.no_grad():
                    condition, raw_frames, sampled_frames = runtime.condition(
                        store, task_id, demo, task.authority.language,
                    )
                    _, R = runtime.compile(condition)
                if mode == "V" and raw_frames != task.episode_lengths[demo]:
                    raise ValueError("bank video length differs from the registered episode")
                R = R.detach().float().cpu().contiguous()
                if R.shape != (7, 256) or not torch.isfinite(R).all():
                    raise ValueError("condition coefficient is incomplete")
                path = output / "coefficients" / f"{name}.safetensors"
                save_file({"R": R}, str(path), metadata={"schema_version": BANK_SCHEMA,
                    "condition_id": name, "mode": mode})
                provenance = ({"kind": "ordered_agentview_rgb", "teacher_demo": demo,
                               "raw_frames": raw_frames, "sampled_frames": sampled_frames,
                               "frame_indices": list(map(int, condition[1].cpu().tolist()))}
                              if mode == "V" else {"kind": "exact_language_only",
                                                    "language": task.authority.language,
                                                    "video_values_read": 0})
                conditions.append({"condition_id": name, "global_task_id": task_id,
                                   "coefficient": file_record(path), "source": provenance})
            task_rows.append({"global_task_id": task_id, "suite": task.suite,
                              "task_id": task.suite_task_id, "language": task.authority.language,
                              "split_role": "validation",
                              "teacher_source": {"path": str(task.authority.path.resolve()),
                                                 "bytes": task.authority.expected_bytes},
                              "episodes": episodes})
    finally:
        if store is not None:
            store.close()
    manifest = {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "status": "sealed",
                "mode": mode, "arm": "correct", "evaluation_role": "validation",
                "source": runtime.source, "checkpoint": checkpoint_record,
                "training_run": file_record(checkpoint.parent.parent / "run_contract.json"),
                "spec": file_record(SPEC), "protocol": spec["protocol"],
                "asset_root": str(asset_root.resolve()),
                "lora": runtime.lora.to_dict(), "base_lora_contract": file_record(base_path),
                "shared": file_record(shared_path), "selection": selection,
                "conditions": conditions, "tasks": task_rows,
                "information_wall": {"deployment_inputs": (["exact task language", "ordered RGB videos",
                    "original frame indices"] if mode == "V" else ["exact task language"]),
                    "teacher_action_state_reward_terminal_reads": 0,
                    "teacher_video_values_read": len(conditions) if mode == "V" else 0,
                    "validation_test_gradients": False, "execution_adapters": 1,
                    "teacher_video_runtime_reads": 0, "deployment_loss_or_optimizer": False}}
    path = output / "bank.json"
    write_json_atomic(path, manifest)
    return path


def _expected_episodes(selection: Mapping, task: int, mode: str) -> list[dict]:
    rows = planned_episodes(selection, task)
    if mode == "L":
        for row in rows:
            row["condition_id"] = f"task_{task:02d}_language"
            row["scheduled_video_demo_indices"] = row.pop("teacher_demo_indices")
    return rows


def _inspect_bank_identity(bank, spec, source, evaluation_role, require_formal):
    mode = bank["mode"]
    if (bank.get("schema_version") != BANK_SCHEMA or bank.get("kind") != BANK_KIND
            or bank.get("status") != "sealed" or evaluation_role != "validation"
            or not require_formal or bank.get("arm") != "correct"
            or bank.get("evaluation_role") != "validation"
            or bank.get("selection") != canonical_selection(spec)
            or bank.get("spec") != file_record(SPEC)
            or bank.get("protocol") != spec["protocol"]
            or not source_matches(bank["source"], source)):
        raise ValueError("velocity bank protocol, source or official selection changed")
    run, checkpoint = _checkpoint(Path(bank["checkpoint"]["path"]), mode)
    if (bank["checkpoint"] != checkpoint or bank["training_run"] != file_record(
            Path(checkpoint["path"]).parent.parent / "run_contract.json")
            or run["git"]["commit"] != git_state(ROOT)["commit"] or run["mode"] != mode):
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
        spec, _ = contract()
        path = manifest_path.resolve()
        bank = read_json(path)
        _inspect_bank_identity(bank, spec, source, evaluation_role, require_formal)
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
    """Use the existing compact/full trajectory sink for the fixed official panel."""
    full = tuple((str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
                 for row in manifest.get("full_conditions", ()))
    expected = {(str(task.suite), int(task.task_id), 0) for task in tasks}
    mode = read_json(args.static_task_lora_manifest).get("mode") if getattr(
        args, "static_task_lora_manifest", None) is not None else None
    spec, _ = contract()
    if (path != (repo_root / "configs/conditional_velocity_operator_v1/official_capture.json").resolve()
            or mode not in ("V", "L")
            or output_dir.resolve() != (continuation_root(spec, mode) / "evaluation/correct400").resolve()
            or args.role != "validation" or task_subset is not None or len(tasks) != 8
            or getattr(args, "static_task_lora_manifest", None) is None
            or read_json(args.static_task_lora_manifest).get("kind") != BANK_KIND
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
                                 "spec_path": str(SPEC.resolve()), "spec_bytes": SPEC.stat().st_size,
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
    capture = contract_row.get("diagnostic_occupancy_capture") or {}
    trace = capture.get("passive_trace") or {}
    if trace.get("schema_version") != PASSIVE_TAG:
        return
    adapter = contract_row.get("adapter") or {}
    if adapter.get("kind") != BANK_KIND or contract_row["git"]["commit"] != git_state(repo_root)["commit"]:
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
    path = (repo_root / "configs/conditional_velocity_operator_v1/official_capture.json").resolve()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("V", "L"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(materialize(asset_root=args.asset_root, checkpoint=args.checkpoint,
                      output=args.output, mode=args.mode, device=torch.device(args.device)))


if __name__ == "__main__":
    main()
