"""Registered ten-panel scope for the existing passive PI05 capture owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.language_content_contract import SPEC_PATH, evaluation_panel, spec


TAG = "ember_language_content_path_passive_capture_v1"


def _cases(panel: Mapping[str, Any]) -> tuple[set[tuple[str, int, int]], set[tuple[str, int, int]]]:
    suite_key = lambda task: (SUITE_ORDER[task // 10], task % 10)
    cases = {(*suite_key(task), state) for task in panel["task_ids"] for state in panel["state_ids"]}
    capture = spec()["evaluation"]["capture"]
    full_tasks = capture["held_full_task_ids"] if panel["kind"].startswith("held") else capture["seen_full_task_ids"]
    full_states = capture["held_full_state_ids"] if panel["kind"].startswith("held") else capture["seen_full_state_ids"]
    full = {(*suite_key(task), state) for task in full_tasks for state in full_states}
    if len(cases) != panel["rows"] or not full <= cases:
        raise Pi05EvaluationError("language-content cases or fixed capture changed")
    return cases, full


def prepare_from_manifest(
    args: Any, *, repo_root: Path, output_dir: Path, task_subset: Mapping[str, Any],
    tasks: Sequence[Any], manifest: Mapping[str, Any], selection_path: Path,
    full: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel = evaluation_panel(output_dir)
    if panel is None:
        raise Pi05EvaluationError("language-content passive capture requires one registered panel")
    study = Path(spec()["outputs"]["planned_run_root"]).resolve()
    expected, expected_full = _cases(panel)
    actual = {(str(task.suite), int(task.task_id), int(state))
              for task in tasks for state in task.init_state_ids}
    selectors = study / "launch" / "selectors"
    subset_manifest = read_json(Path(task_subset["selection_path"]))
    if (manifest.get("schema_version") != "ember_pi05_registered_trajectory_capture_v1"
            or manifest.get("passive_control_trace") != TAG
            or manifest.get("study_spec") != SPEC_PATH
            or manifest.get("panel_id") != panel["id"]
            or subset_manifest.get("study_spec") != SPEC_PATH
            or subset_manifest.get("panel_id") != panel["id"]
            or task_subset.get("diagnostic_subset") != "registered_train_subset"
            or manifest.get("task_subset_selection") != task_subset["selection_path"]
            or manifest.get("mode") != "compact" or manifest.get("stage_predicates") is not True
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))
            or args.role != "development_train" or args.mode != "screen"
            or args.state_count != len(panel["state_ids"])
            or Path(args.config).resolve() != (repo_root / "configs/libero_24_8_8_coverage_v1/evaluation.json").resolve()
            or actual != expected or set(full) != expected_full or len(full) != len(expected_full)
            or not selection_path.resolve().is_relative_to(selectors)
            or not Path(task_subset["selection_path"]).resolve().is_relative_to(selectors)):
        raise Pi05EvaluationError("language-content panel, selector, or passive capture changed")
    spec_path = (repo_root / SPEC_PATH).resolve()
    capture = {
        "schema_version": "ember_pi05_registered_trajectory_capture_v1",
        "selection_path": str(selection_path.resolve()), "selection_bytes": selection_path.stat().st_size,
        "mode": "compact",
        "full_conditions": [{"suite": suite, "task_id": task, "init_state_id": state}
                            for suite, task, state in full],
        "trajectory_root": str(output_dir.resolve() / "trajectories"),
        "passive_trace": {"schema_version": TAG, "spec_path": str(spec_path),
                          "spec_bytes": spec_path.stat().st_size,
                          "trace_root": str(output_dir.resolve() / "continuous_traces")},
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_use": False, "test_use": False,
    }
    stage = {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "full_conditions_only": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_action_reads": 0, "validation_reward_reads": 0,
        "held_data_use": False, "claim_boundary": "BDDL predicates are partial progress signals",
    }
    return capture, stage


def attach_requested_capture(args: Any, contract: dict[str, Any], repo_root: Path,
                             output_dir: Path) -> None:
    panel = evaluation_panel(output_dir)
    if panel is None:
        return
    capture = contract.get("diagnostic_occupancy_capture") or {}
    adapter = contract.get("adapter")
    if ((panel["model"] == "Source") != (adapter is None)
            or capture.get("passive_trace", {}).get("schema_version") != TAG):
        raise Pi05EvaluationError("language-content panel lacks its exact bank or all-row passive trace")
    if adapter is not None:
        from ember.writer.language_content_contract import validate_evaluation_bank
        from ember.writer.materialization import inspect_writer_checkpoint
        from ember.pi05_eval_contract import git_state

        path = Path(adapter["manifest"]["path"])
        run, _ = inspect_writer_checkpoint(Path(adapter["writer_checkpoint"]["path"]))
        validate_evaluation_bank(panel, path, adapter, run, git_state(repo_root)["commit"])
    contract["language_content_provenance"] = {
        "panel": panel["id"], "study_spec": SPEC_PATH,
        "evaluation_commit": contract["git"]["commit"],
        "bank_manifest": adapter["manifest"] if adapter is not None else None,
        "bank_training_commit": adapter["writer_checkpoint"]["training_commit"] if adapter else None,
        "bank_materialization_commit": adapter["materialization_git"]["commit"] if adapter else None,
    }


def validate_contract(contract: Mapping[str, Any], repo_root: Path) -> None:
    output = Path(contract["output_dir"]).resolve()
    panel = evaluation_panel(output)
    if panel is None:
        return
    capture = contract.get("diagnostic_occupancy_capture") or {}
    path = Path(str(capture.get("selection_path", "")))
    if (not path.is_file() or path.stat().st_size != capture.get("selection_bytes")
            or capture.get("passive_trace", {}).get("schema_version") != TAG
            or capture["passive_trace"].get("spec_path") != str((repo_root / SPEC_PATH).resolve())
            or capture["passive_trace"].get("spec_bytes") != (repo_root / SPEC_PATH).stat().st_size
            or capture["passive_trace"].get("trace_root") != str(output / "continuous_traces")
            or contract.get("diagnostic_stage_predicates", {}).get("full_conditions_only") is not False):
        raise Pi05EvaluationError("language-content passive trace or spec changed after prepare")
    manifest = read_json(path)
    if (manifest.get("passive_control_trace") != TAG
            or manifest.get("study_spec") != SPEC_PATH
            or manifest.get("panel_id") != panel["id"]
            or manifest.get("mode") != "compact" or manifest.get("stage_predicates") is not True
            or manifest.get("task_subset_selection") !=
               (contract.get("diagnostic_task_subset") or {}).get("selection_path")
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))
            or contract.get("diagnostic_stage_predicates", {}).get("capture") !=
               "all_rows_post_settling_then_every_executed_control_step"):
        raise Pi05EvaluationError("language-content capture selector identity changed")
    full = tuple((row["suite"], int(row["task_id"]), int(row["init_state_id"]))
                 for row in manifest["full_conditions"])
    cases, expected_full = _cases(panel)
    actual = {(row["suite"], int(row["task_id"]), int(state))
              for row in contract["tasks"] for state in row["init_state_ids"]}
    if actual != cases or set(full) != expected_full or capture["full_conditions"] != manifest["full_conditions"]:
        raise Pi05EvaluationError("language-content panel rows or full cases changed after prepare")
    expected = dict(contract)
    expected.pop("language_content_provenance", None)
    attach_requested_capture(None, expected, repo_root, output)
    if contract.get("language_content_provenance") != expected.get("language_content_provenance"):
        raise Pi05EvaluationError("language-content bank or evaluation provenance changed")
