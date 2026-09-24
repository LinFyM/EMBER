"""Sealed authority for the two-task, saved-action channel intervention."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.frozen_prefix import (
    POLICY_PANEL, TASK_KEYS, _original_panel, _read, _reference_index,
)
from ember.pi05_eval_contract import load_run_contract


PANEL_SCHEMA = "ember_approach_channel_panel_v1"
CONTRACT_SCHEMA = "ember_approach_channel_intervention_v1"
SPEC_RELATIVE = Path("configs/approach_channel_causality_v1/experiment_spec.json")
PREFIX_DONORS = {
    "C_all": ("C_correct",) * 7,
    "B_xy_C_rest": ("B", "B") + ("C_correct",) * 5,
    "B_z_C_rest": ("C_correct", "C_correct", "B") + ("C_correct",) * 4,
    "B_xyz_C_rest": ("B", "B", "B") + ("C_correct",) * 4,
    "B_all": ("B",) * 7,
}
PURE = frozenset(("C_all", "B_all"))
PILOT = (0, 25)
REMAINDER = tuple(state for state in range(50) if state not in PILOT)


def _validate_spec(spec: Mapping[str, Any]) -> None:
    if (spec.get("schema_version") != "ember_approach_channel_causality_spec_v1"
            or spec.get("study_id") != "approach_channel_causality_20260924"
            or spec.get("task_ids") != [14, 21]
            or spec.get("state_ids") != list(range(50))
            or spec.get("cut_control_steps") != 25
            or spec.get("followers") != ["B", "C_correct"]
            or {name: tuple(value.get("channel_donors", ()))
                for name, value in spec.get("prefixes", {}).items()} != PREFIX_DONORS):
        raise Pi05EvaluationError("approach-channel specification changed")


def _panel_scope(panel: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, str, str, tuple[int, ...], tuple[tuple[str, int], ...]]:
    _validate_spec(spec)
    prefix, follower, phase = panel.get("prefix"), panel.get("follower"), panel.get("phase")
    states = tuple(panel.get("state_ids", ()))
    task_ids = tuple(panel.get("task_ids", ()))
    expected_states = (PILOT if prefix in PURE and phase == "pilot" else
                       REMAINDER if prefix in PURE and phase == "remaining" else
                       tuple(range(50)) if prefix not in PURE and phase == "mixed" else
                       (0,) if prefix == "B_xy_C_rest" and follower == "C_correct" and phase == "smoke" else ())
    expected_task_ids = (21,) if phase == "smoke" else (14, 21)
    if (panel.get("schema_version") != PANEL_SCHEMA
            or panel.get("study_id") != spec["study_id"]
            or prefix not in PREFIX_DONORS or follower not in spec["followers"]
            or any(type(state) is not int for state in states)
            or states != expected_states or task_ids != expected_task_ids
            or any(panel.get(name) is not False for name in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("approach-channel panel changed its registered scope")
    keys = (TASK_KEYS[1],) if phase == "smoke" else TASK_KEYS
    return str(prefix), str(follower), str(phase), states, keys


def _validate_request(args: Any, spec: Mapping[str, Any], *, phase: str,
                      panel_path: Path, output_dir: Path, bank_path: Path) -> None:
    observed = (spec.get("role"), args.role, args.mode, args.state_count,
                Path(args.static_task_lora_manifest).resolve() if args.static_task_lora_manifest else None)
    expected = ("development_train", "development_train", "smoke" if phase == "smoke" else "formal",
                50, bank_path.resolve())
    prohibited = ("task_subset_selection", "trajectory_capture_selection",
                  "occupancy_capture_selection", "frozen_replay_registration",
                  "capture_stage_predicates", "exploration_sigma", "init_state_ids",
                  "frozen_prefix_panel")
    if (observed != expected or any(getattr(args, name, None) for name in prohibited)
            or not panel_path.is_relative_to(Path(spec["run_root"]).resolve())
            or not output_dir.resolve().is_relative_to(Path(spec["run_root"]).resolve())):
        raise Pi05EvaluationError("approach-channel evaluator request changed")


def prepare_scope(args: Any, *, repo_root: Path, installed_tasks: Sequence[Any],
                  model: Mapping[str, Any], output_dir: Path) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    panel_path = Path(args.approach_channel_panel).resolve()
    spec_path = (repo_root / SPEC_RELATIVE).resolve()
    spec, panel = _read(spec_path), _read(panel_path)
    prefix, follower, phase, states, task_keys = _panel_scope(panel, spec)
    root = Path(spec["reference_root"]).resolve()
    bank_path = root / "materialization" / POLICY_PANEL[follower] / "manifest.json"
    _validate_request(args, spec, phase=phase, panel_path=panel_path,
                      output_dir=output_dir, bank_path=bank_path)
    installed = {(task.suite, int(task.task_id)): task for task in installed_tasks}
    source_contracts = {donor: load_run_contract(
        root / "evaluation" / POLICY_PANEL[donor] / "run_contract.json")
        for donor in ("B", "C_correct")}
    bank_keys = tuple((row["suite"], int(row["task_id"]))
                      for row in source_contracts["B"]["tasks"])
    if (len(bank_keys) != 8 or len(set(bank_keys)) != 8
            or any(key not in installed for key in bank_keys)
            or tuple((row["suite"], int(row["task_id"]))
                     for row in source_contracts["C_correct"]["tasks"]) != bank_keys
            or not set(task_keys) <= set(bank_keys)):
        raise Pi05EvaluationError("approach-channel original bank scope changed")
    references = {}
    donor_contracts = {}
    for donor in ("B", "C_correct"):
        donor_contract, original_rows, results_path = _original_panel(
            root=root, name=POLICY_PANEL[donor], model=model, keys=bank_keys)
        contract_path = root / "evaluation" / POLICY_PANEL[donor] / "run_contract.json"
        references[donor] = _reference_index(
            rows=original_rows, results_path=results_path,
            task_keys=task_keys, states=states)
        donor_contracts[donor] = {"path": str(contract_path), "bytes": contract_path.stat().st_size}
    follower_contract = source_contracts[follower]
    if Path(follower_contract["adapter"]["manifest"]["path"]).resolve() != bank_path.resolve():
        raise Pi05EvaluationError("approach-channel follower bank changed")
    adapter = inspect_static_task_lora_adapter(
        manifest_path=bank_path, source=model,
        tasks=tuple(installed[key] for key in bank_keys),
        evaluation_role="development_train", require_formal=True,
    )
    if adapter != follower_contract["adapter"]:
        raise Pi05EvaluationError("approach-channel adapter differs from original panel")
    selected = dict(adapter)
    selected["tasks"] = [row for row in adapter["tasks"]
                         if (row["suite"], int(row["task_id"])) in task_keys]
    tasks = tuple(replace(installed[key], init_state_ids=states) for key in task_keys)
    full = [{"suite": suite, "task_id": task_id, "init_state_id": state_id}
            for suite, task_id in task_keys for state_id in PILOT if state_id in states]
    capture = {
        "schema_version": "ember_pi05_approach_channel_capture_v1",
        "mode": "compact", "full_conditions": full,
        "trajectory_root": str((output_dir / "trajectories").resolve()),
        "external_prefix_commands": True,
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
    intervention = {
        "schema_version": CONTRACT_SCHEMA,
        "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
        "panel_path": str(panel_path), "panel_bytes": panel_path.stat().st_size,
        "prefix": prefix, "follower": follower, "phase": phase,
        "channel_donors": list(PREFIX_DONORS[prefix]),
        "cut_control_steps": 25, "state_ids": list(states),
        "task_ids": list(panel["task_ids"]),
        "donor_contracts": donor_contracts,
        "donor_references": references,
        "trace_root": str((output_dir / "object_traces").resolve()),
        "full_state_ids": list(PILOT),
        "contact_sampling": "after_settling_and_each_control_step; no substep exclusion claim",
    }
    return tasks, selected, intervention, capture, stage


def prepare_payload(args: Any, *, authorities: Any, installed_tasks: Sequence[Any],
                    model: Mapping[str, Any], tokenizer: Mapping[str, Any],
                    libero_paths: Mapping[str, str], output_dir: Path, repo_root: Path,
                    command: Sequence[str]) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    from ember.pi05_eval.preparation import parse_gpu_indices, shards_from_contract
    from ember.pi05_eval_contract import build_run_contract

    tasks, adapter, intervention, capture, stage = prepare_scope(
        args, repo_root=repo_root, installed_tasks=installed_tasks,
        model=model, output_dir=output_dir,
    )
    contract = build_run_contract(
        authorities=authorities, tasks=tasks, libero_paths=libero_paths,
        model=model, tokenizer=tokenizer, output_dir=output_dir,
        role=args.role, mode=args.mode, replicas_per_gpu=args.replicas_per_gpu,
        physical_gpu_ids=parse_gpu_indices(args.gpu_indices), command=command,
        adapter=adapter,
    )
    if (contract["policy"]["replan_steps"] != 5
            or contract["policy"]["num_inference_steps"] != 10
            or contract["rng"]["inference_seed"] != 7
            or contract["environment"]["dummy_settling_steps"] != 10):
        raise Pi05EvaluationError("approach-channel canonical policy clock changed")
    contract["approach_channel_intervention"] = intervention
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
        "approach_channel_intervention": intervention["panel_path"],
    }


def reinspect_adapter(contract: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    intervention = contract["approach_channel_intervention"]
    if intervention.get("schema_version") != CONTRACT_SCHEMA:
        raise Pi05EvaluationError("approach-channel contract schema changed")
    for name in ("spec", "panel"):
        path = Path(intervention[f"{name}_path"])
        if not path.is_file() or path.stat().st_size != intervention[f"{name}_bytes"]:
            raise Pi05EvaluationError(f"approach-channel {name} authority changed")
    spec = _read(Path(intervention["spec_path"]))
    panel = _read(Path(intervention["panel_path"]))
    scope = _panel_scope(panel, spec)
    if scope != (intervention["prefix"], intervention["follower"], intervention["phase"],
                 tuple(intervention["state_ids"]),
                 (TASK_KEYS[1],) if intervention["phase"] == "smoke" else TASK_KEYS):
        raise Pi05EvaluationError("approach-channel panel changed after prepare")
    reference = {}
    for donor, asset in intervention["donor_contracts"].items():
        path = Path(asset["path"])
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise Pi05EvaluationError(f"approach-channel donor contract changed: {donor}")
        reference[donor] = load_run_contract(path)
    bank_tasks = tuple(SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                       init_state_ids=row["init_state_ids"])
                       for row in reference["B"]["tasks"])
    inspected = inspect_static_task_lora_adapter(
        manifest_path=Path(contract["adapter"]["manifest"]["path"]),
        source=model, tasks=bank_tasks, evaluation_role=str(contract["role"]),
        require_formal=True,
    )
    selected = dict(inspected)
    keys = (TASK_KEYS[1],) if intervention["phase"] == "smoke" else TASK_KEYS
    selected["tasks"] = [row for row in inspected["tasks"]
                         if (row["suite"], int(row["task_id"])) in keys]
    return selected
