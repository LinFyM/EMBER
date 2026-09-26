"""Frozen two-arm language-content-path diagnostic registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.conditional_contract import validate_config as validate_parent_config


CONFIG_SCHEMA = "ember_language_content_path_causality_config_v1"
EXPERIMENT = "language_content_path_causality_20260926"
SPEC_PATH = "configs/language_content_path_causality_v1/experiment_spec.json"
FIXED400_SPEC_PATH = "configs/language_content_path_fixed400_v1/experiment_spec.json"
FIXED400_STUDY = "language_content_path_fixed400_20260926"
NATIVE_FEATURE_STUDY = "native_feature_change_causality_20260926"
NATIVE_FEATURE_SPEC_PATH = "configs/native_feature_change_causality_v1/experiment_spec.json"
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


def fixed400_spec() -> dict[str, Any]:
    value = read_json(REPO_ROOT / FIXED400_SPEC_PATH)
    evaluation = value["evaluation"]
    if (value.get("schema_version") != "ember_language_content_fixed400_v1"
            or value.get("study_id") != FIXED400_STUDY
            or evaluation["task_ids"] != [0, 1, 14, 15, 20, 21, 36, 38]
            or evaluation["new_state_ids"] != list(range(10, 50))
            or evaluation["prior_state_ids"] != list(range(10))
            or [(row["id"], row["model"], row["condition"], row["rows"])
                for row in evaluation["panels"]] != [
                    ("Source_held_correct", "Source", "correct", 320),
                    ("B630_held_correct", "B630", "correct", 320),
                    ("C0_630_held_correct", "C0", "correct", 320),
                    ("C0_630_held_other", "C0", "same_task_other", 320),
                ]
            or evaluation["video_schedule_seed"] != 20260911
            or evaluation["other_demo_offset"] != 17
            or value["frozen_inputs"]["C0_training_commit"] !=
               "dca1b5500ac0f912d56cc1c76382004457e807a4"):
        raise ValueError("fixed400 scientific registration changed")
    return value


def fixed400_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    study = fixed400_spec()
    if dict(panel) not in study["evaluation"]["panels"]:
        raise ValueError("fixed400 panel is outside its registration")
    return {**panel, "kind": ("held_other" if panel["condition"] == "same_task_other"
                               else "held_correct"),
            "task_ids": study["evaluation"]["task_ids"],
            "state_ids": study["evaluation"]["new_state_ids"],
            "study_id": FIXED400_STUDY}


def fixed400_explicit_states(args: Any) -> bool:
    """Admit only this registered output-root selector pair before task selection."""
    study = fixed400_spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve() / "launch" / "selectors"
    subset = getattr(args, "task_subset_selection", None)
    capture = getattr(args, "trajectory_capture_selection", None)
    if subset is None or capture is None:
        return False
    subset, capture = Path(subset).resolve(), Path(capture).resolve()
    panels = {row["id"] for row in study["evaluation"]["panels"]}
    return (getattr(args, "role", None) == "development_train"
            and getattr(args, "mode", None) == "screen"
            and getattr(args, "state_count", None) == 40
            and tuple(getattr(args, "init_state_ids", ()) or ()) == tuple(range(10, 50))
            and not getattr(args, "exploration_sigma", False)
            and not getattr(args, "capture_stage_predicates", False)
            and getattr(args, "occupancy_capture_selection", None) is None
            and getattr(args, "frozen_replay_registration", None) is None
            and subset.parent == capture.parent == root
            and any(subset.name == f"{name}_subset.json"
                    and capture.name == f"{name}_capture.json" for name in panels))


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
    arm_id = config["experiment"]["arm_id"]
    if checkpoint.name != "macro_00000630" or checkpoint.parent.parent.name != arm_id:
        raise ValueError("language-content-path bank requires this arm's complete 630 checkpoint")
    fixed = fixed400_spec()
    fixed_root = Path(fixed["outputs"]["planned_run_root"]).resolve() / "materialization"
    output = output.resolve()
    if output.parent == fixed_root:
        panel = next((row for row in fixed["evaluation"]["panels"]
                      if row["id"] == output.name and row["model"] == "C0"), None)
        if (panel is None or arm_id != "C0"
                or checkpoint.resolve() != Path(fixed["frozen_inputs"]["C0_checkpoint"]).resolve()):
            raise ValueError("fixed400 bank requires the one frozen C0 checkpoint and registered panel")
        from ember.writer.materialization import selection_contract

        expected = selection_contract(
            role="development_train", task_ids=fixed["evaluation"]["task_ids"], cardinality=1,
            arm=panel["condition"], mode="per_init_ordinal",
            seed=fixed["evaluation"]["video_schedule_seed"],
            init_state_ids=fixed["evaluation"]["new_state_ids"], video_pool=range(50))
        if dict(selection) != expected:
            raise ValueError("fixed400 bank task, state, video, or control map changed")
        return fixed400_panel(panel)
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve() / "materialization"
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


def evaluation_scope(output: Path) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    from ember.writer.native_feature_change import panel as native_panel, spec as native_spec

    native = native_spec()
    native_root = Path(native["outputs"]["planned_run_root"]).resolve() / "evaluation"
    output = output.resolve()
    if output.parent.parent == native_root:
        return native, NATIVE_FEATURE_SPEC_PATH, native_panel(output.name, output.parent.name)
    fixed = fixed400_spec()
    fixed_root = Path(fixed["outputs"]["planned_run_root"]).resolve() / "evaluation"
    output = output.resolve()
    if output.parent == fixed_root:
        panel = next((row for row in fixed["evaluation"]["panels"] if row["id"] == output.name), None)
        if panel is None:
            raise ValueError("fixed400 evaluation is outside four registered panels")
        return fixed, FIXED400_SPEC_PATH, fixed400_panel(panel)
    study = spec()
    root = Path(study["outputs"]["planned_run_root"]).resolve() / "evaluation"
    if output.parent != root:
        return None
    panel = next((row for row in study["evaluation"]["panels"] if row["id"] == output.name), None)
    if panel is None:
        raise ValueError("language-content-path evaluation is outside ten registered panels")
    return study, SPEC_PATH, panel


def evaluation_panel(output: Path) -> dict[str, Any] | None:
    scope = evaluation_scope(output)
    return scope[2] if scope is not None else None


def validate_evaluation_bank(panel: Mapping[str, Any], manifest_path: Path,
                             manifest: Mapping[str, Any], run: Mapping[str, Any],
                             current_commit: str) -> bool:
    """Validate the registered panel's frozen Writer bank and subset provenance."""
    if panel.get("study_id") == NATIVE_FEATURE_STUDY:
        from ember.writer.native_feature_change import (
            panel as native_panel, registered_selection, spec as native_spec,
        )
        study = native_spec()
        cell = panel.get("id")
        root = Path(study["outputs"]["planned_run_root"]).resolve()
        if (dict(panel) != native_panel(cell, panel.get("stage"))
                or manifest_path.resolve() != root / "materialization" / cell / "manifest.json"
                or manifest.get("native_feature_change") != {
                    "study_id": NATIVE_FEATURE_STUDY, "cell": cell,
                    "spec_path": NATIVE_FEATURE_SPEC_PATH, "native_read_conditions": 80,
                    "feature_index": str(root / "features" / "index.json"),
                    "source_training_commit": study["frozen_input"]["training_commit"],
                }
                or manifest.get("selection") != registered_selection(study)
                or run["git"]["commit"] != study["frozen_input"]["training_commit"]
                or manifest["materialization_git"]["commit"] != current_commit
                or Path(manifest["writer_checkpoint"]["path"]).resolve() !=
                   Path(study["frozen_input"]["checkpoint"]).resolve()
                or manifest["writer_checkpoint"]["macro"] != 630
                or run["config"]["experiment"]["arm_id"] != "C0"
                or manifest["arm"] != "correct"
                or len(manifest["conditions"]) != 80
                or Path(run["source"]["checkpoint"]).resolve() !=
                   Path(study["frozen_input"]["source_checkpoint"]).resolve()):
            raise ValueError("native-feature bank, source, schedule, or single-commit origin changed")
        return False
    if panel.get("study_id") == FIXED400_STUDY:
        fixed = fixed400_spec()
        registered = next((fixed400_panel(row) for row in fixed["evaluation"]["panels"]
                           if row["id"] == panel.get("id")), None)
        if registered != dict(panel) or manifest["selection"]["task_ids"] != panel["task_ids"]:
            raise ValueError("fixed400 evaluation panel identity changed")
        frozen = fixed["frozen_inputs"]
        manifest_path = manifest_path.resolve()
        if panel["model"] == "B630":
            if (manifest_path != Path(frozen["B630_bank_root"]).resolve() / "manifest.json"
                    or run["git"]["commit"] != frozen["B630_bank_commit"]
                    or manifest["materialization_git"]["commit"] != frozen["B630_bank_commit"]
                    or run["config"]["experiment"]["arm_id"] != "B_language"
                    or manifest["writer_checkpoint"]["macro"] != 630
                    or manifest["arm"] != "correct"
                    or manifest["selection"]["init_state_ids"] != list(range(50))):
                raise ValueError("fixed400 B630 reference bank or checkpoint changed")
            return True
        if panel["model"] != "C0":
            raise ValueError("only frozen C0 and B630 have fixed400 banks")
        expected = Path(fixed["outputs"]["planned_run_root"]).resolve() / "materialization" / panel["id"] / "manifest.json"
        if (manifest_path != expected
                or run["git"]["commit"] != frozen["C0_training_commit"]
                or manifest["materialization_git"]["commit"] != current_commit
                or run["config"]["experiment"]["arm_id"] != "C0"
                or manifest["writer_checkpoint"]["macro"] != 630):
            raise ValueError("fixed400 C0 training, materialization, or bank identity changed")
        bank_panel(run["config"], manifest["selection"],
                   checkpoint=Path(manifest["writer_checkpoint"]["path"]), output=manifest_path.parent)
        return False
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
