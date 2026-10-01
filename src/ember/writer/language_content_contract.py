"""Read-only identity of sealed language-content Writer checkpoints."""
from __future__ import annotations
from pathlib import Path
from typing import Any
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
