"""Read-only identity of sealed learned-first-frame Writer checkpoints."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ember.pi05_source_checkpoint import read_json
from ember.writer.language_content_contract import validate_config as validate_c0_config

STUDY = "learned_initial_content_causality_20260926"

CONFIG_SCHEMA = "ember_learned_initial_content_causality_config_v1"

SPEC_PATH = "configs/learned_initial_content_causality_v1/experiment_spec.json"

ROOT = Path(__file__).resolve().parents[3]

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
