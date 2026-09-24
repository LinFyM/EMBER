"""Registered, read-only scope for the two-task frozen prefix intervention."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ember.eval_adapters import inspect_static_task_lora_adapter, validate_episode_adapter_fields
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import load_run_contract


PANEL_SCHEMA = "ember_frozen_prefix_panel_v1"
CONTRACT_SCHEMA = "ember_frozen_prefix_intervention_v1"
SPEC_RELATIVE = Path("configs/frozen_prefix_causality_v1/experiment_spec.json")
POLICY_PANEL = {
    "B": "B_language_630_held",
    "C_correct": "C_video_fm_420_held",
    "C_other": "C_video_fm_420_other",
    "C_wrong": "C_video_fm_420_wrong",
}
TASK_KEYS = (("libero_object", 4), ("libero_goal", 1))
PILOT = (0, 25)
REMAINDER = tuple(state for state in range(50) if state not in PILOT)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Pi05EvaluationError(f"invalid frozen-prefix authority: {path}") from error
    if not isinstance(value, dict):
        raise Pi05EvaluationError(f"frozen-prefix authority is not an object: {path}")
    return value


def _panel_scope(panel: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[str, str, int, tuple[int, ...]]:
    anchor, follower = panel.get("anchor"), panel.get("follower")
    cut = panel.get("cut_control_steps")
    states = tuple(panel.get("state_ids", ()))
    if (panel.get("schema_version") != PANEL_SCHEMA
            or panel.get("study_id") != spec.get("study_id")
            or anchor not in spec["anchors"] or follower not in spec["followers"]
            or cut not in spec["cut_control_steps"]
            or any(type(state) is not int for state in states)
            or tuple(sorted(set(states))) != states
            or not set(states) <= set(spec["state_ids"])):
        raise Pi05EvaluationError("frozen-prefix panel changed its registered scope")
    expected = PILOT if anchor == follower and panel.get("phase") == "pilot" else (
        REMAINDER if anchor == follower and panel.get("phase") == "remaining" else
        tuple(spec["state_ids"]) if anchor != follower and panel.get("phase") == "cross" else
        (0,) if anchor == follower == "B" and cut == 25 and panel.get("phase") == "smoke" else ()
    )
    if states != expected:
        raise Pi05EvaluationError("frozen-prefix panel state coverage changed")
    if any(panel.get(key) is not False for key in (
        "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")):
        raise Pi05EvaluationError("frozen-prefix panel information wall changed")
    return str(anchor), str(follower), int(cut), states


def _original_panel(
    *, root: Path, name: str, model: Mapping[str, Any], keys: Sequence[tuple[str, int]],
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]], Path]:
    run = root / "evaluation" / name
    contract_path, results_path = run / "run_contract.json", run / "results.json"
    contract = load_run_contract(contract_path)
    results = _read(results_path)
    if (not (run / "launcher_completion.json").is_file()
            or not (run / "run_summary.json").is_file()
            or contract["model"] != model
            or contract["role"] != "development_train"
            or contract["mode"] != "formal"
            or contract["git"]["commit"] != "43d801b16ee3ed0bc80f963c89ad003b7cc40732"
            or {(row["suite"], int(row["task_id"])) for row in contract["tasks"]} != set(keys)
            or int(results.get("overall", {}).get("episodes", -1)) != 400):
        raise Pi05EvaluationError(f"original completed panel identity changed: {name}")
    rows = {(row["suite"], int(row["task_id"]), int(row["init_state_id"])): row
            for row in results.get("rows", ())}
    if len(rows) != 400 or len(results.get("rows", ())) != 400:
        raise Pi05EvaluationError(f"original panel row coverage changed: {name}")
    for (suite, task_id, state_id), row in rows.items():
        if (not 0 <= state_id < 50
                or not validate_episode_adapter_fields(contract["adapter"], row,
                    suite=suite, task_id=task_id, init_state_id=state_id)):
            raise Pi05EvaluationError(f"original panel adapter evidence changed: {name}")
    return contract, rows, results_path


def _validate_request(
    args: Any, spec: Mapping[str, Any], *, smoke: bool, panel_path: Path,
    output_dir: Path, bank_path: Path,
) -> None:
    observed = (spec.get("schema_version"), spec.get("task_ids"), spec.get("role"),
                args.role, args.mode, args.state_count,
                Path(args.static_task_lora_manifest).resolve() if args.static_task_lora_manifest else None)
    expected = ("ember_frozen_prefix_causality_spec_v1", [14, 21], "development_train",
                "development_train", "smoke" if smoke else "formal", 50, bank_path.resolve())
    prohibited = (
        "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
        "frozen_replay_registration", "capture_stage_predicates", "exploration_sigma",
        "init_state_ids",
    )
    if (observed != expected or any(getattr(args, name, None) for name in prohibited)
            or not panel_path.is_relative_to(Path(spec["run_root"]).resolve())
            or not output_dir.resolve().is_relative_to(Path(spec["run_root"]).resolve())):
        raise Pi05EvaluationError("frozen-prefix evaluator request changed")


def _reference_index(
    *, rows: Mapping[tuple[str, int, int], Mapping[str, Any]], results_path: Path,
    task_keys: Sequence[tuple[str, int]], states: Sequence[int],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for suite, task_id in task_keys:
        for state_id in states:
            row = rows[(suite, task_id, state_id)]
            trajectory = row["occupancy_trajectory"]
            path = Path(trajectory["path"]).resolve()
            if (not path.is_relative_to(results_path.parent / "trajectories")
                    or not path.is_file() or path.stat().st_size != int(trajectory["bytes"])
                    or row["policy_seed_root"] != 7 or row["env_seed"] != 7):
                raise Pi05EvaluationError("frozen-prefix original trajectory identity changed")
            references[f"{suite}:{task_id}:{state_id}"] = {
                "original_results_path": str(results_path),
                "original_results_bytes": results_path.stat().st_size,
                "trajectory": dict(trajectory),
                "success": bool(row["success"]), "steps": int(row["steps"]),
                "policy_noise_seeds": list(row["policy_noise_seeds"]),
                "anchor_condition_id": row["horizon_writer_lora"]["condition_id"],
                "anchor_teacher_demo_indices": row["horizon_writer_lora"]["teacher_demo_indices"],
            }
    return references


def prepare_scope(
    args: Any, *, repo_root: Path, installed_tasks: Sequence[Any],
    model: Mapping[str, Any], output_dir: Path,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    panel_path = Path(args.frozen_prefix_panel).resolve()
    spec_path = (repo_root / SPEC_RELATIVE).resolve()
    spec, panel = _read(spec_path), _read(panel_path)
    anchor, follower, cut, states = _panel_scope(panel, spec)
    smoke = panel["phase"] == "smoke"
    task_keys = (TASK_KEYS[1],) if smoke else TASK_KEYS
    root = Path(spec["reference_root"]).resolve()
    bank_path = root / "materialization" / POLICY_PANEL[follower] / "manifest.json"
    _validate_request(args, spec, smoke=smoke, panel_path=panel_path,
                      output_dir=output_dir, bank_path=bank_path)
    installed = {(task.suite, int(task.task_id)): task for task in installed_tasks}
    anchor_contract_path = root / "evaluation" / POLICY_PANEL[anchor] / "run_contract.json"
    anchor_contract = load_run_contract(anchor_contract_path)
    bank_keys = tuple((row["suite"], int(row["task_id"])) for row in anchor_contract["tasks"])
    if (len(bank_keys) != 8 or len(set(bank_keys)) != 8
            or any(key not in installed for key in bank_keys)
            or not set(task_keys) <= set(bank_keys)):
        raise Pi05EvaluationError("frozen-prefix original held8 bank scope changed")
    follower_contract, _, _ = _original_panel(root=root, name=POLICY_PANEL[follower],
                                               model=model, keys=bank_keys)
    if Path(follower_contract["adapter"]["manifest"]["path"]).resolve() != bank_path.resolve():
        raise Pi05EvaluationError("frozen-prefix follower bank differs from original panel")
    bank_tasks = tuple(installed[key] for key in bank_keys)
    adapter = inspect_static_task_lora_adapter(
        manifest_path=bank_path, source=model, tasks=bank_tasks,
        evaluation_role="development_train", require_formal=True,
    )
    if adapter != follower_contract["adapter"]:
        raise Pi05EvaluationError("frozen-prefix follower adapter differs from original panel")
    selected = dict(adapter)
    selected["tasks"] = [row for row in adapter["tasks"]
                         if (row["suite"], int(row["task_id"])) in task_keys]
    if len(selected["tasks"]) != len(task_keys):
        raise Pi05EvaluationError("frozen-prefix selected adapter is incomplete")
    tasks = tuple(replace(installed[key], init_state_ids=states) for key in task_keys)
    _, original_rows, original_results_path = _original_panel(
        root=root, name=POLICY_PANEL[anchor], model=model, keys=bank_keys)
    references = _reference_index(rows=original_rows, results_path=original_results_path,
                                  task_keys=task_keys, states=states)
    full = [{"suite": suite, "task_id": task_id, "init_state_id": state_id}
            for suite, task_id in task_keys for state_id in PILOT if state_id in states]
    capture = {
        "schema_version": "ember_pi05_frozen_prefix_capture_v1",
        "mode": "compact", "full_conditions": full,
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
    intervention = {
        "schema_version": CONTRACT_SCHEMA,
        "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
        "panel_path": str(panel_path), "panel_bytes": panel_path.stat().st_size,
        "anchor": anchor, "follower": follower, "cut_control_steps": cut,
        "phase": panel["phase"], "state_ids": list(states),
        "reference_contract_path": str(anchor_contract_path),
        "reference_contract_bytes": anchor_contract_path.stat().st_size,
        "reference_bank_keys": [list(key) for key in bank_keys],
        "references": references,
        "trace_root": str((output_dir / "object_traces").resolve()),
        "full_state_ids": list(PILOT),
    }
    return tasks, selected, intervention, capture, stage


def prepare_payload(
    args: Any, *, authorities: Any, installed_tasks: Sequence[Any],
    model: Mapping[str, Any], tokenizer: Mapping[str, Any], libero_paths: Mapping[str, str],
    output_dir: Path, repo_root: Path, command: Sequence[str],
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
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
    contract["frozen_prefix_intervention"] = intervention
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
        "frozen_prefix_intervention": intervention["panel_path"],
    }


def reinspect_adapter(contract: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    intervention = contract["frozen_prefix_intervention"]
    if intervention.get("schema_version") != CONTRACT_SCHEMA:
        raise Pi05EvaluationError("frozen-prefix contract schema changed")
    for field in ("spec", "panel", "reference_contract"):
        path = Path(intervention[f"{field}_path"])
        if not path.is_file() or path.stat().st_size != intervention[f"{field}_bytes"]:
            raise Pi05EvaluationError(f"frozen-prefix {field} authority changed")
    spec = _read(Path(intervention["spec_path"]))
    panel = _read(Path(intervention["panel_path"]))
    if _panel_scope(panel, spec) != (
            intervention["anchor"], intervention["follower"],
            intervention["cut_control_steps"], tuple(intervention["state_ids"])):
        raise Pi05EvaluationError("frozen-prefix panel changed after prepare")
    reference = load_run_contract(Path(intervention["reference_contract_path"]))
    bank_tasks = tuple(SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                       init_state_ids=row["init_state_ids"])
                       for row in reference["tasks"])
    inspected = inspect_static_task_lora_adapter(
        manifest_path=Path(contract["adapter"]["manifest"]["path"]),
        source=model, tasks=bank_tasks, evaluation_role=str(contract["role"]),
        require_formal=True,
    )
    selected = dict(inspected)
    task_keys = (TASK_KEYS[1],) if intervention["phase"] == "smoke" else TASK_KEYS
    selected["tasks"] = [row for row in inspected["tasks"]
                         if (row["suite"], int(row["task_id"])) in task_keys]
    return selected
