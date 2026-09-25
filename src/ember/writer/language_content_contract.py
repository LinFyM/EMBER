"""Frozen two-arm language-content-path diagnostic registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.conditional_contract import validate_config as validate_parent_config


CONFIG_SCHEMA = "ember_language_content_path_causality_config_v1"
EXPERIMENT = "language_content_path_causality_20260926"
SPEC_PATH = "configs/language_content_path_causality_v1/experiment_spec.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def spec() -> dict[str, Any]:
    value = read_json(REPO_ROOT / SPEC_PATH)
    if (value.get("schema_version") != "ember_language_content_path_causality_spec_v1"
            or value.get("study_id") != EXPERIMENT
            or value["optimization"]["updates"] != 630
            or value["sampling"]["event_plan_capacity_updates"] != 1260
            or value["training_execution"]["world_size"] != 2
            or len(value["evaluation"]["panels"]) != 10
            or sum(row["rows"] for row in value["evaluation"]["panels"]) != 736):
        raise ValueError("language-content-path scientific registration changed")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    study = spec()
    arm_id = config.get("experiment", {}).get("arm_id")
    arms = {arm["id"]: arm for arm in study["arms"]}
    if (config.get("schema_version") != CONFIG_SCHEMA or config.get("study_spec") != SPEC_PATH
            or arm_id not in arms or config.get("experiment") != {
                "kind": EXPERIMENT, "arm_id": arm_id, "parameterization": "video_writer",
                "objective": "matched_full_fm", "extra_endpoint_prefix": False,
                "language_content_path": arms[arm_id]["language_content_path"],
            }):
        raise ValueError("language-content-path arm or scalar switch changed")
    parent = read_json(REPO_ROOT / study["base_training_config"])
    expected = dict(parent)
    expected.update(schema_version=CONFIG_SCHEMA, study_spec=SPEC_PATH,
                    update_version=EXPERIMENT, experiment=config["experiment"])
    expected["evidence"] = {**parent["evidence"],
        "checkpoint_updates": [105, 210, 315, 420, 525, 630],
        "evaluation_updates": [630],
        "profile_registration": config.get("evidence", {}).get("profile_registration")}
    if config != expected:
        raise ValueError("language-content-path source, data, loss, optimizer, or 630 prefix changed")
    profile = config["evidence"]["profile_registration"]
    if profile.get("status") == "complete":
        from ember.writer.conditional_contract import _validate_profile

        _validate_profile(profile)
        if profile["world_size"] != 2:
            raise ValueError("language-content-path full profile must use world2")
    elif profile != {"status": "pending"}:
        raise ValueError("language-content-path profile registration changed")
    # The original four-arm validator remains sealed and independently checked.
    inherited = dict(parent)
    validate_parent_config(inherited)
    return config


def bank_panel(config: Mapping[str, Any], selection: Mapping[str, Any], *, checkpoint: Path,
               output: Path) -> dict[str, Any]:
    validate_config(dict(config))
    study = spec()
    arm_id = config["experiment"]["arm_id"]
    if checkpoint.name != "macro_00000630" or checkpoint.parent.parent.name != arm_id:
        raise ValueError("language-content-path bank requires this arm's complete 630 checkpoint")
    root = Path(study["outputs"]["planned_run_root"]).resolve() / "materialization"
    output = output.resolve()
    panel = next((row for row in study["evaluation"]["panels"]
                  if row["id"] == output.name and row["model"] == arm_id), None)
    if panel is None or output.parent != root:
        raise ValueError("language-content-path bank is outside its exact registered panel")
    held = panel["kind"] != "seen_correct"
    expected = {
        "evaluation_role": "development_train", "task_ids": panel["task_ids"],
        "K": 1, "arm": "same_task_other" if panel["condition"] == "same_task_other" else "correct",
        "mode": "per_init_ordinal", "seed": study["evaluation"]["video_schedule_seed"],
        "init_state_ids": panel["state_ids"],
        "video_pool": study["evaluation"]["held_video_pool"] if held else list(range(46, 50)),
        "fixed_videos": {}, "outcome_dependence": False, "gradient_use": False,
        "without_replacement": True,
        "schedule": ("expert_manifold_canonical_permutation_v1" if held
                     else "conditional_compilation_reserved_seen_v1"),
        "without_replacement_scope": "per_task_per_arm_round", "schedule_state_origin": 0,
        "video_ordinal_rule": "init_state_id",
    }
    if dict(selection) != expected:
        raise ValueError("language-content-path bank task, state, video, or control map changed")
    return panel


def evaluation_panel(output: Path) -> dict[str, Any] | None:
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve() / "evaluation"
    output = output.resolve()
    if output.parent != root:
        return None
    panel = next((row for row in study["evaluation"]["panels"] if row["id"] == output.name), None)
    if panel is None:
        raise ValueError("language-content-path evaluation is outside ten registered panels")
    return panel


def validate_evaluation_bank(panel: Mapping[str, Any], manifest_path: Path,
                             manifest: Mapping[str, Any], run: Mapping[str, Any],
                             current_commit: str) -> bool:
    """Return whether the one registered B630 held bank may be evaluated on states0..9."""
    study = spec()
    registered = next((row for row in study["evaluation"]["panels"]
                       if row["id"] == panel.get("id")), None)
    if registered != dict(panel) or manifest["selection"]["task_ids"] != panel["task_ids"]:
        raise ValueError("language-content evaluation panel identity changed")
    model = panel["model"]
    manifest_path = manifest_path.resolve()
    if model == "B630":
        reference = study["frozen_references"]["B630"]
        root = reference["held_bank_root" if panel["kind"] == "held_correct" else "seen_bank_root"]
        if (manifest_path != Path(root).resolve() / "manifest.json"
                or run["git"]["commit"] != reference["training_commit"]
                or manifest["materialization_git"]["commit"] != reference["training_commit"]
                or run["config"]["experiment"]["arm_id"] != "B_language"
                or manifest["writer_checkpoint"]["macro"] != 630
                or manifest["arm"] != "correct" or panel["condition"] != "correct"):
            raise ValueError("language-content B630 reference bank or checkpoint changed")
        return panel["kind"] == "held_correct"
    if model not in {"C0", "Cplus"}:
        raise ValueError("only registered C models have generated banks")
    expected = Path(study["outputs"]["planned_run_root"]).resolve() / "materialization" / panel["id"] / "manifest.json"
    if (manifest_path != expected or run["git"]["commit"] != current_commit
            or manifest["materialization_git"]["commit"] != current_commit
            or run["config"]["experiment"]["arm_id"] != model
            or manifest["writer_checkpoint"]["macro"] != 630):
        raise ValueError("language-content generated bank must use the one frozen training/materialization commit")
    bank_panel(run["config"], manifest["selection"],
               checkpoint=Path(manifest["writer_checkpoint"]["path"]), output=manifest_path.parent)
    return False
