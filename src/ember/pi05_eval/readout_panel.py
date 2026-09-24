"""Registered two-task four-cell evaluation scope for the original C bank."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.frozen_prefix import POLICY_PANEL, TASK_KEYS, _original_panel, _read
from ember.pi05_eval.readout_state import GROUPS, SCHEMA as STATE_SCHEMA
from ember.pi05_eval_contract import load_run_contract


PANEL_SCHEMA = "ember_readout_realization_panel_v1"
CONTRACT_SCHEMA = "ember_readout_realization_intervention_v1"
SPEC_RELATIVE = Path("configs/readout_realization_causality_v1/experiment_spec.json")


def _scope(panel: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...], tuple[tuple[str, int], ...]]:
    if (spec.get("schema_version") != "ember_readout_realization_causality_spec_v1"
            or [(row["global_task_id"], row["suite"], row["task_id"])
                for row in spec["tasks"]] != [(14, *TASK_KEYS[0]), (21, *TASK_KEYS[1])]
            or spec.get("state_ids") != list(range(50))
            or spec["rollouts"]["count"] != 400):
        raise Pi05EvaluationError("readout frozen specification changed")
    group, phase = panel.get("group"), panel.get("phase")
    states = tuple(panel.get("state_ids", ()))
    expected = (0,) if phase in ("pilot", "smoke") else tuple(range(1, 50)) if phase == "remaining" else ()
    keys = (TASK_KEYS[1],) if phase == "smoke" else TASK_KEYS
    if (panel.get("schema_version") != PANEL_SCHEMA
            or panel.get("study_id") != spec["study_id"]
            or group not in GROUPS or states != expected
            or panel.get("task_ids") != ([21] if phase == "smoke" else [14, 21])
            or any(panel.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("readout panel changed registered scope")
    return str(group), str(phase), states, keys


def _validate_request(args: Any, *, study: Path, panel_path: Path,
                      output_dir: Path, bank_path: Path, phase: str) -> None:
    if (not panel_path.is_relative_to(study) or not output_dir.resolve().is_relative_to(study)
            or args.role != "development_train"
            or args.mode != ("smoke" if phase == "smoke" else "formal")
            or args.state_count != 50
            or Path(args.static_task_lora_manifest).resolve() != bank_path
            or any(getattr(args, name, None) for name in (
                "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
                "frozen_replay_registration", "frozen_prefix_panel", "approach_channel_panel",
                "capture_stage_predicates", "exploration_sigma", "init_state_ids"))):
        raise Pi05EvaluationError("readout evaluator request changed")


def _original_adapter(bank_path: Path, installed_tasks: Sequence[Any],
                      model: Mapping[str, Any], keys: tuple[tuple[str, int], ...]
                      ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[tuple[str, int], Any]]:
    original_root = bank_path.parents[2]
    original, _, _ = _original_panel(
        root=original_root, name=POLICY_PANEL["C_correct"], model=model,
        keys=tuple((row["suite"], int(row["task_id"])) for row in
                   load_run_contract(original_root / "evaluation" / POLICY_PANEL["C_correct"] /
                                     "run_contract.json")["tasks"]),
    )
    if Path(original["adapter"]["manifest"]["path"]).resolve() != bank_path:
        raise Pi05EvaluationError("readout original C bank identity changed")
    installed = {(task.suite, int(task.task_id)): task for task in installed_tasks}
    bank_keys = tuple((row["suite"], int(row["task_id"])) for row in original["tasks"])
    if len(bank_keys) != 8 or not set(keys) <= set(bank_keys):
        raise Pi05EvaluationError("readout original held8 scope changed")
    inspected = inspect_static_task_lora_adapter(
        manifest_path=bank_path, source=model,
        tasks=tuple(installed[key] for key in bank_keys),
        evaluation_role="development_train", require_formal=True,
    )
    if inspected != original["adapter"]:
        raise Pi05EvaluationError("readout original bank differs from sealed C panel")
    selected = dict(inspected)
    selected["tasks"] = [row for row in inspected["tasks"]
                         if (row["suite"], int(row["task_id"])) in keys]
    return original_root, inspected, selected, installed


def _derived_authority(panel: Mapping[str, Any], *, study: Path, phase: str,
                       bank_path: Path, selected: Mapping[str, Any],
                       inspected: Mapping[str, Any], states: tuple[int, ...]) -> Path:
    derived_path = Path(panel.get("derived_manifest_path",
                                  study / "derived_states" / "manifest.json")).resolve()
    if not derived_path.is_relative_to(study) or (phase != "smoke" and
            derived_path != study / "derived_states" / "manifest.json"):
        raise Pi05EvaluationError("readout derived manifest path changed")
    derived = _read(derived_path)
    expected_conditions = {episode["condition_id"] for row in selected["tasks"]
                           for episode in row["episodes"] if episode["init_state_id"] in states}
    if (derived.get("schema_version") != STATE_SCHEMA
            or not expected_conditions <= set(derived["conditions"])
            or phase != "smoke" and len(derived["conditions"]) != 100
            or Path(derived["original_bank_manifest"]["path"]).resolve() != bank_path
            or derived["original_bank_manifest"]["bytes"] != bank_path.stat().st_size):
        raise Pi05EvaluationError("readout derived state provenance changed")
    for condition_id in expected_conditions:
        original_condition = next(row for row in inspected["conditions"]
                                  if row["condition_id"] == condition_id)
        if derived["conditions"][condition_id]["original"] != original_condition["adapter"]:
            raise Pi05EvaluationError("readout derived state original condition changed")
    return derived_path


def _capture_contract(output_dir: Path, keys: tuple[tuple[str, int], ...],
                      states: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = {
        "schema_version": "ember_pi05_readout_realization_capture_v1",
        "mode": "compact",
        "full_conditions": [{"suite": suite, "task_id": task_id, "init_state_id": state}
                            for suite, task_id in keys for state in (0, 25) if state in states],
        "trajectory_root": str((output_dir / "trajectories").resolve()),
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "post_settling_then_every_executed_action_change_points",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_action_reads": 0, "validation_reward_reads": 0,
        "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return capture, stage


def prepare_payload(args: Any, *, authorities: Any, installed_tasks: Sequence[Any],
                    model: Mapping[str, Any], tokenizer: Mapping[str, Any],
                    libero_paths: Mapping[str, str], output_dir: Path, repo_root: Path,
                    command: Sequence[str]) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    from ember.pi05_eval.preparation import parse_gpu_indices, shards_from_contract
    from ember.pi05_eval_contract import build_run_contract

    spec_path = (repo_root / SPEC_RELATIVE).resolve()
    panel_path = Path(args.readout_realization_panel).resolve()
    spec, panel = _read(spec_path), _read(panel_path)
    group, phase, states, keys = _scope(panel, spec)
    study = Path(spec["run_root"]).resolve()
    bank_path = Path(spec["bank_root"]).resolve() / "manifest.json"
    _validate_request(args, study=study, panel_path=panel_path,
                      output_dir=output_dir, bank_path=bank_path, phase=phase)
    original_root, inspected, selected, installed = _original_adapter(
        bank_path, installed_tasks, model, keys)
    tasks = tuple(replace(installed[key], init_state_ids=states) for key in keys)
    derived_path = _derived_authority(
        panel, study=study, phase=phase, bank_path=bank_path,
        selected=selected, inspected=inspected, states=states)
    capture, stage = _capture_contract(output_dir, keys, states)
    intervention = {
        "schema_version": CONTRACT_SCHEMA,
        "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
        "panel_path": str(panel_path), "panel_bytes": panel_path.stat().st_size,
        "derived_manifest": {"path": str(derived_path), "bytes": derived_path.stat().st_size},
        "group": group, "phase": phase, "state_ids": list(states),
        "task_ids": panel["task_ids"],
        "original_contract_path": str(original_root / "evaluation" /
                                      POLICY_PANEL["C_correct"] / "run_contract.json"),
        "trace_root": str((output_dir / "object_traces").resolve()),
        "contact_sampling": "after_settling_and_each_control_step; no substep exclusion claim",
    }
    contract = build_run_contract(
        authorities=authorities, tasks=tasks, libero_paths=libero_paths,
        model=model, tokenizer=tokenizer, output_dir=output_dir,
        role=args.role, mode=args.mode, replicas_per_gpu=args.replicas_per_gpu,
        physical_gpu_ids=parse_gpu_indices(args.gpu_indices), command=command,
        adapter=selected,
    )
    if (contract["policy"]["replan_steps"] != 5
            or contract["policy"]["num_inference_steps"] != 10
            or contract["rng"]["inference_seed"] != 7
            or contract["environment"]["dummy_settling_steps"] != 10):
        raise Pi05EvaluationError("readout canonical policy clock changed")
    contract["readout_realization_intervention"] = intervention
    contract["diagnostic_occupancy_capture"] = capture
    contract["diagnostic_stage_predicates"] = stage
    contract["diagnostic_task_subset"] = None
    shards = shards_from_contract(contract)
    return contract, shards, {
        "event": "prepared", "contract_reference": contract["contract_reference"],
        "tasks": len(tasks), "states": sum(len(task.init_state_ids) for task in tasks),
        "shards": len(shards), "replicas_per_gpu": args.replicas_per_gpu,
        "physical_gpu_ids": contract["parallel"]["physical_gpu_ids"],
        "arm": contract["arm"], "output_dir": str(output_dir),
        "readout_realization_intervention": str(panel_path),
    }


def reinspect_adapter(contract: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    intervention = contract["readout_realization_intervention"]
    if intervention.get("schema_version") != CONTRACT_SCHEMA:
        raise Pi05EvaluationError("readout intervention schema changed")
    for field in ("spec", "panel"):
        path = Path(intervention[f"{field}_path"])
        if not path.is_file() or path.stat().st_size != intervention[f"{field}_bytes"]:
            raise Pi05EvaluationError("readout panel authority changed")
    manifest = Path(intervention["derived_manifest"]["path"])
    if not manifest.is_file() or manifest.stat().st_size != intervention["derived_manifest"]["bytes"]:
        raise Pi05EvaluationError("readout derived manifest changed")
    spec = _read(Path(intervention["spec_path"]))
    panel = _read(Path(intervention["panel_path"]))
    group, phase, states, keys = _scope(panel, spec)
    if (group, phase, list(states)) != (intervention["group"], intervention["phase"],
                                        intervention["state_ids"]):
        raise Pi05EvaluationError("readout panel changed after prepare")
    original = load_run_contract(Path(intervention["original_contract_path"]))
    bank_tasks = tuple(SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                       init_state_ids=row["init_state_ids"])
                       for row in original["tasks"])
    inspected = inspect_static_task_lora_adapter(
        manifest_path=Path(contract["adapter"]["manifest"]["path"]),
        source=model, tasks=bank_tasks, evaluation_role=str(contract["role"]),
        require_formal=True,
    )
    selected = dict(inspected)
    selected["tasks"] = [row for row in inspected["tasks"]
                         if (row["suite"], int(row["task_id"])) in keys]
    return selected
