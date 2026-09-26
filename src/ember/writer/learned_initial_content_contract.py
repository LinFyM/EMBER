"""Exact S0 training, bank and evaluation authority for the learned first-frame study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.language_content_contract import validate_config as validate_c0_config


STUDY = "learned_initial_content_causality_20260926"
CONFIG_SCHEMA = "ember_learned_initial_content_causality_config_v1"
SPEC_PATH = "configs/learned_initial_content_causality_v1/experiment_spec.json"
ROOT = Path(__file__).resolve().parents[3]
FROZEN_GPU_COMMIT = "4f6c75e419eeb785a7ee38941df19ba958ab5c21"


def _allowed_s0_bank_commit(panel_id: str, root: Path, bank_commit: str,
                            evaluation_commit: str) -> bool:
    if bank_commit == evaluation_commit:
        return True
    if panel_id != "S0_630_held_other" or bank_commit != FROZEN_GPU_COMMIT:
        return False
    record_path = root / "launch" / "held_other_cpu_stage_exception.json"
    if not record_path.is_file():
        return False
    return read_json(record_path) == {
        "schema_version": "ember_initial_content_held_other_cpu_stage_exception_v1",
        "study_id": STUDY, "frozen_training_and_bank_commit": FROZEN_GPU_COMMIT,
        "evaluation_commit": evaluation_commit,
        "panels": ["C0_630_held_other", "S0_630_held_other"],
        "scope": "registered_arm_and_exact_source_validation_only_no_model_or_rollout_change",
    }


def spec() -> dict[str, Any]:
    value = read_json(ROOT / SPEC_PATH)
    panels = value["evaluation"]["panels"]
    if (value.get("schema_version") != "ember_learned_initial_content_causality_spec_v1"
            or value.get("study_id") != STUDY
            or value["training_reference_commit"] != "dca1b5500ac0f912d56cc1c76382004457e807a4"
            or value["optimization"]["updates"] != 630
            or value["training_execution"]["world_size"] != 2
            or [(p["id"], p["rows"]) for p in panels] != [
                (f"{model}_630_{kind}", 64 if kind == "seen_correct" else 80)
                for model in ("C0", "S0")
                for kind in ("held_correct", "held_other", "seen_correct")]
            or value["evaluation"]["new_rows"] != 448
            or value["evaluation"]["new_S0_banks"] != 224):
        raise ValueError("learned initial-content registration changed")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    study = spec()
    expected_experiment = {
        "kind": STUDY, "arm_id": "S0", "parameterization": "video_writer",
        "objective": "matched_full_fm", "extra_endpoint_prefix": False,
        "language_content_path": False, "initial_content_only": True,
    }
    if (config.get("schema_version") != CONFIG_SCHEMA
            or config.get("study_spec") != SPEC_PATH
            or config.get("update_version") != STUDY
            or config.get("experiment") != expected_experiment):
        raise ValueError("S0 explicit first-content training identity changed")
    parent = read_json(ROOT / study["base_training_config"])
    validate_c0_config(parent)
    expected = dict(parent)
    expected.update(schema_version=CONFIG_SCHEMA, study_spec=SPEC_PATH,
                    update_version=STUDY, experiment=expected_experiment)
    expected["evidence"] = {**parent["evidence"],
                            "profile_registration": config.get("evidence", {}).get("profile_registration")}
    if config != expected:
        raise ValueError("S0 changed C0 task/data/model/loss/optimizer or 630 prefix")
    profile = config["evidence"]["profile_registration"]
    if profile.get("status") == "complete":
        from ember.writer.conditional_contract import _validate_profile

        _validate_profile(profile)
        if profile["world_size"] != 2:
            raise ValueError("S0 profile requires world2")
    elif profile != {"status": "pending"}:
        raise ValueError("S0 profile registration changed")
    return config


def panel(panel_id: str) -> dict[str, Any]:
    study = spec()
    row = next((p for p in study["evaluation"]["panels"] if p["id"] == panel_id), None)
    if row is None:
        raise ValueError("S0 study panel is unregistered")
    held = row["kind"] != "seen_correct"
    return {**row, "condition": "same_task_other" if row["kind"] == "held_other" else "correct",
            "task_ids": study["evaluation"]["held_task_ids" if held else "seen_task_ids"],
            "state_ids": study["evaluation"]["held_state_ids" if held else "seen_state_ids"],
            "study_id": STUDY}


def selection_for(panel_id: str) -> dict[str, Any]:
    from ember.writer.materialization import selection_contract

    row, evaluation = panel(panel_id), spec()["evaluation"]
    return selection_contract(
        role="development_train", task_ids=row["task_ids"], cardinality=1,
        arm=row["condition"], mode="per_init_ordinal", seed=evaluation["video_schedule_seed"],
        init_state_ids=row["state_ids"],
        video_pool=evaluation["held_video_pool"] if row["kind"] != "seen_correct" else range(46, 50),
    )


def bank_panel(config: Mapping[str, Any], selection: Mapping[str, Any], *,
               checkpoint: Path, output: Path) -> dict[str, Any]:
    validate_config(dict(config))
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    output = output.resolve()
    row = panel(output.name)
    if (row["model"] != "S0" or output.parent != root / "materialization"
            or checkpoint.resolve() != root / "training/S0/checkpoints/macro_00000630"
            or dict(selection) != selection_for(row["id"])):
        raise ValueError("S0 bank checkpoint, output, or exact video map changed")
    return row


def evaluation_panel(output: Path) -> dict[str, Any] | None:
    root = Path(spec()["outputs"]["planned_run_root"]).resolve() / "evaluation"
    output = output.resolve()
    if output.parent.parent != root:
        return None
    stage = output.parent.name
    if stage not in {"pilot", "remaining", "complete"}:
        raise ValueError("S0 study evaluation stage is unregistered")
    row = panel(output.name)
    if (stage != "complete") != (row["kind"] == "held_correct"):
        raise ValueError("S0 study staged evaluation panel changed")
    return {**row, "stage": stage, "rows": (2 if stage == "pilot" else
                                             78 if stage == "remaining" else row["rows"])}


def validate_evaluation_bank(panel_row: Mapping[str, Any], path: Path,
                             manifest: Mapping[str, Any], run: Mapping[str, Any],
                             commit: str) -> None:
    study = spec()
    row = evaluation_panel(Path(study["outputs"]["planned_run_root"]) /
                           "evaluation" / panel_row["stage"] / panel_row["id"])
    if row != dict(panel_row) or manifest["selection"] != selection_for(row["id"]):
        raise ValueError("S0 study evaluation or video map changed")
    if row["model"] == "C0":
        from ember.writer.language_content_contract import bank_panel as c0_bank_panel

        reference = study["frozen_references"]["C0"]
        key = {"held_correct": "held_correct_bank", "held_other": "held_other_bank",
               "seen_correct": "seen_correct_bank"}[row["kind"]]
        if (path.resolve() != Path(reference[key]).resolve()
                or run["git"]["commit"] != reference["training_commit"]
                or manifest["materialization_git"]["commit"] != reference["training_commit"]
                or Path(manifest["writer_checkpoint"]["path"]).resolve() !=
                   Path(reference["checkpoint"]).resolve()
                or run["config"]["experiment"]["arm_id"] != "C0"
                or manifest["writer_checkpoint"]["macro"] != 630):
            raise ValueError("S0 comparison must retain exact frozen C0 bank")
        c0_bank_panel(run["config"], manifest["selection"],
                      checkpoint=Path(reference["checkpoint"]), output=path.parent)
        return
    root = Path(study["outputs"]["planned_run_root"]).resolve()
    trace = root / "features" / row["id"] / "index.json"
    bank_commit = manifest["materialization_git"]["commit"]
    if (path.resolve() != root / "materialization" / row["id"] / "manifest.json"
            or run["git"]["commit"] != FROZEN_GPU_COMMIT
            or bank_commit != FROZEN_GPU_COMMIT
            or not _allowed_s0_bank_commit(row["id"], root, bank_commit, commit)
            or manifest.get("learned_initial_content") != {
                "study_id": STUDY, "panel": row["id"], "spec_path": SPEC_PATH,
                "feature_index": str(trace)}
            or not trace.is_file()
            or run["config"]["experiment"]["arm_id"] != "S0"
            or manifest["writer_checkpoint"]["macro"] != 630):
        raise ValueError("S0 bank must use its frozen E and same-forward feature trace")
    bank_panel(run["config"], manifest["selection"],
               checkpoint=Path(manifest["writer_checkpoint"]["path"]), output=path.parent)
