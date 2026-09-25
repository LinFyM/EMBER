"""Registered two-stage evaluation scope for the four frozen native/W cells."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.writer.native_reader_transfer import SPEC, authority, cell_spec, expected_bank


TAG = "ember_native_reader_transfer_passive_capture_v1"


def _stage(output: Path, cell: str) -> str:
    root = Path(authority()["resources"]["study_root"]).resolve() / "evaluation"
    for stage in ("pilot", "remaining"):
        if output.resolve() == root / stage / cell:
            return stage
    raise Pi05EvaluationError("native transfer output is outside its exact pilot/remaining cell")


def _cases(stage: str) -> tuple[int, ...]:
    return (0,) if stage == "pilot" else tuple(range(1, 50))


def select_tasks(args: Any, tasks: Sequence[Any], repo_root: Path) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    cell = str(args.native_reader_transfer_cell)
    cell_spec(cell)
    stage = _stage(Path(args.output_dir), cell)
    spec = authority()
    config = repo_root / "configs/relational_support_causality_v1/evaluation.json"
    if (args.role != "development_train" or args.mode != "formal" or args.state_count != 50
            or Path(args.config).resolve() != config.resolve()
            or Path(args.static_task_lora_manifest).resolve() != expected_bank(cell)
            or any(getattr(args, field, None) for field in (
                "task_subset_selection", "trajectory_capture_selection", "occupancy_capture_selection",
                "frozen_replay_registration", "frozen_prefix_panel", "approach_channel_panel",
                "readout_realization_panel", "source_sft_checkpoint", "task_expert_config"))
            or getattr(args, "capture_stage_predicates", False)
            or getattr(args, "exploration_sigma", False)):
        raise Pi05EvaluationError("native transfer evaluation arguments changed")
    wanted = {(row["suite"], row["task_id"]) for row in spec["evaluation"]["tasks"]}
    selected = tuple(replace(task, init_state_ids=_cases(stage)) for task in tasks
                     if (task.suite, task.task_id) in wanted)
    if len(selected) != 2 or {(task.suite, task.task_id) for task in selected} != wanted:
        raise Pi05EvaluationError("native transfer target task identity changed")
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": state}
            for task in selected for state in spec["evaluation"]["full_states"]
            if state in _cases(stage)]
    root = Path(args.output_dir).resolve()
    spec_path = (repo_root / SPEC).resolve()
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "mode": "compact", "full_conditions": full,
        "trajectory_root": str(root / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(root / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    predicates = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_action_reads": 0,
        "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return selected, capture, predicates


def attach(contract: dict[str, Any], *, cell: str, repo_root: Path) -> None:
    spec = authority()
    stage = _stage(Path(contract["output_dir"]), cell)
    adapter = contract.get("adapter") or {}
    bank = adapter.get("manifest") or {}
    path = expected_bank(cell)
    actual = {(task["suite"], int(task["task_id"]), int(state))
              for task in contract["tasks"] for state in task["init_state_ids"]}
    expected = {(task["suite"], task["task_id"], state)
                for task in spec["evaluation"]["tasks"] for state in _cases(stage)}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    full = {(row["suite"], int(row["task_id"]), int(row["init_state_id"]))
            for row in capture.get("full_conditions", [])}
    expected_full = {case for case in expected if case[2] in (0, 25)}
    spec_path = (repo_root / SPEC).resolve()
    if (contract.get("role") != "development_train" or contract.get("mode") != "formal"
            or contract.get("arm") != "correct" or actual != expected or full != expected_full
            or bank != {"path": str(path), "bytes": path.stat().st_size}
            or capture.get("passive_trace") != {
                "schema_version": TAG, "spec_path": str(spec_path), "spec_bytes": spec_path.stat().st_size,
                "trace_root": str(Path(contract["output_dir"]) / "continuous_traces")}
            or contract.get("diagnostic_stage_predicates", {}).get("full_conditions_only") is not False):
        raise Pi05EvaluationError("native transfer rollout scope, bank or passive trace changed")
    contract["native_reader_transfer"] = {
        "schema_version": "ember_native_reader_transfer_evaluation_v1",
        "cell": cell, "stage": stage, "study_spec": {"path": str(spec_path), "bytes": spec_path.stat().st_size},
        "bank_manifest": bank, "N_parent": cell_spec(cell)["N_parent"],
        "W_parent": cell_spec(cell)["W_parent"],
        "evaluation_commit": contract["git"]["commit"],
    }


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    registration = contract.get("native_reader_transfer") or {}
    if not registration:
        return
    expected = dict(contract)
    expected.pop("native_reader_transfer", None)
    attach(expected, cell=registration.get("cell"), repo_root=repo_root)
    if expected["native_reader_transfer"] != registration:
        raise Pi05EvaluationError("native transfer evaluation registration changed after prepare")


def validate_bank(manifest: Mapping[str, Any], path: Path, cell: str, current_commit: str) -> None:
    from ember.writer.materialization import selection_contract

    spec = authority()
    definition = cell_spec(cell)
    expected_selection = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=20260911,
        init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    if path.resolve() != expected_bank(cell):
        raise ValueError("native transfer bank path is outside its registered cell")
    if (manifest.get("selection") != expected_selection or manifest.get("arm") != "correct"
            or manifest.get("evaluation_role") != "development_train"
            or len(manifest.get("conditions", [])) != 100):
        raise ValueError("native transfer bank task/video mapping changed")
    parent = spec["parents"][definition["W_parent"]]
    if (Path(manifest["writer_checkpoint"]["path"]).resolve() != Path(parent["checkpoint"]).resolve()
            or manifest["writer_checkpoint"]["training_commit"] != parent["training_commit"]):
        raise ValueError("native transfer W checkpoint identity changed")
    if definition["endpoint_bank_reuse"]:
        _validate_endpoint_bank(manifest, parent)
    else:
        _validate_mixed_bank(manifest, cell, definition, parent, spec, current_commit)


def _validate_endpoint_bank(manifest: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    if (manifest.get("native_reader_transfer") is not None
            or manifest.get("materialization_git", {}).get("commit") != parent["bank_commit"]
            or manifest.get("registered_stage1_panel_id") != f"{parent['arm']}_1260_core_correct"):
        raise ValueError("native transfer self cell must use its sealed E3 endpoint bank")


def _validate_mixed_bank(manifest: Mapping[str, Any], cell: str, definition: Mapping[str, Any],
                         parent: Mapping[str, Any], spec: Mapping[str, Any], current_commit: str) -> None:
    transfer = manifest.get("native_reader_transfer") or {}
    n_parent = spec["parents"][definition["N_parent"]]
    roles = transfer.get("partition") or {}
    rule = spec["parameter_partition"]
    origin = transfer.get("N_checkpoint") or {}
    identity = {
        "schema_version": "ember_native_reader_transfer_bank_v1", "cell": cell,
        "N_parent": definition["N_parent"], "W_parent": definition["W_parent"],
        "W_checkpoint": str(Path(parent["checkpoint"]).resolve()), "study_spec": str(SPEC),
    }
    if (any(transfer.get(key) != value for key, value in identity.items())
            or Path(origin.get("path", "")).resolve() != Path(n_parent["checkpoint"]).resolve()
            or origin.get("training_commit") != n_parent["training_commit"]
            or {key: len(roles.get(key, [])) for key in ("N_keys", "W_keys", "fixed_keys")} != {
                "N_keys": rule["N_tensors"], "W_keys": rule["W_tensors"],
                "fixed_keys": rule["fixed_buffer_tensors"]}
            or roles.get("N_elements") != rule["N_elements"]
            or manifest.get("materialization_git", {}).get("commit") != current_commit
            or manifest.get("compilation", {}).get("new_conditions") != 100
            or manifest.get("compilation", {}).get("reused_conditions") != 0):
        raise ValueError("native transfer mixed bank provenance changed")
