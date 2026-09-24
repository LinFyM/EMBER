"""Frozen configuration authority for the relation-support learning intervention."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.task_protocol import load_task_authorities
from ember.writer.conditional_contract import _validate_profile
from ember.writer.learning_data import RELATIONAL_EVENT_SCHEMA, query_allocation


CONFIG_SCHEMA = "ember_relational_support_causality_config_v1"
EXPERIMENT = "relational_support_causality_20260924"
UPDATE_VERSION = "relational_support_causality_v1"
SPEC_PATH = "configs/relational_support_causality_v1/experiment_spec.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _matches_stage1_selection(selection: dict[str, Any], panel: dict[str, Any], seed: int) -> bool:
    support = panel["kind"] == "support_correct"
    wanted = {
        "evaluation_role": "nonheld_meta" if support else "development_train",
        "arm": "same_task_other" if panel["kind"] == "target_other" else "correct",
        "task_ids": panel["task_ids"], "init_state_ids": panel["state_ids"],
        "video_pool": list(range(50)), "seed": seed, "K": 1,
        "mode": "per_init_ordinal", "schedule": "expert_manifold_canonical_permutation_v1",
        "fixed_videos": {}, "without_replacement": True,
    }
    return all(selection.get(key) == value for key, value in wanted.items())


def registered_stage1_bank_panel(
    config: dict[str, Any], selection: dict[str, Any], *,
    checkpoint: Path | None = None, output: Path | None = None,
) -> dict[str, Any]:
    """Admit only one exact materialization panel from the current study stage."""
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("study_spec") != SPEC_PATH:
        raise ValueError("stage1 bank requires the registered relation-support training schema")
    spec = read_json(REPO_ROOT / SPEC_PATH)
    stage = spec["evaluation"]["stage1"]
    amendment = spec["execution"].get("stage1_subset_materialization_amendment", {})
    arms = {row["id"]: row for row in spec["arms"]}
    arm_id = config.get("experiment", {}).get("arm_id")
    if (amendment.get("id") != "stage1_subset_materialization_in_E_20260925"
            or spec["evaluation"].get("active_stage") != "mechanism_core_v1"
            or len(stage["panels"]) != 18 or stage.get("expected_rows") != 1500
            or arm_id not in arms):
        raise ValueError("stage1 bank authority or trained arm changed")
    matching = [row for row in stage["panels"] if row["arm"] == arm_id and row["kind"] in
                {"target_correct", "target_other", "support_correct"}]
    for panel in matching:
        if not _matches_stage1_selection(selection, panel, spec["evaluation"]["video_schedule_seed"]):
            continue
        root = Path(spec["outputs"]["planned_run_root"]).resolve()
        if (checkpoint is not None and checkpoint.resolve()
                != root/"training"/arm_id/"checkpoints"/"macro_00001260"):
            raise ValueError("stage1 bank checkpoint is not this arm's completed1260")
        if output is not None and output.resolve() != root/"materialization"/panel["id"]:
            raise ValueError("stage1 bank output does not identify its exact panel")
        return panel
    raise ValueError("bank selection is outside registered stage1 task/state/condition scope")


def stage1_bank_materialization_commit(
    *, panel_id: str, manifest_path: Path, evaluation_commit: str,
    spec: Mapping[str, Any] | None = None,
) -> str:
    """Admit only the two sealed E2 banks retained by the E2/E3 decision."""
    spec = spec if spec is not None else read_json(REPO_ROOT / SPEC_PATH)
    exception = spec['execution'].get('passive_arena_region_exception', {})
    study = Path(spec['outputs']['planned_run_root']).resolve()
    expected = (study / 'materialization' / panel_id / 'manifest.json').resolve()
    if (exception.get('id') != 'retain_unaffected_E2_and_fix_arena_registry_E3_20260925'
            or manifest_path.resolve() != expected):
        raise ValueError('stage1 bank manifest is outside its exact registered output')
    if panel_id in exception['retained_E2_banks']:
        if (panel_id not in {'C_S01_1260_core_correct', 'C_S01_1260_support_core'}
                or evaluation_commit == exception['E2_commit']):
            raise ValueError('retained E2 bank requires the separate E3 evaluation authority')
        return exception['E2_commit']
    return evaluation_commit


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compare every scientific field with a frozen B/C recipe and the new spec."""
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("study_spec") != SPEC_PATH:
        raise ValueError("relation-support configuration identity changed")
    spec = read_json(REPO_ROOT / SPEC_PATH)
    if spec.get("study_id") != EXPERIMENT or spec.get("status") != "registered_for_implementation_and_bounded_execution":
        raise ValueError("relation-support study authority changed")
    arms = {row["id"]: row for row in spec["arms"]}
    arm = arms.get(config.get("experiment", {}).get("arm_id"))
    if arm is None or len(arms) != 6:
        raise ValueError("relation-support arm is not registered")
    base = read_json(REPO_ROOT / arm["base_config"])
    expected = deepcopy(base)
    expected.update(schema_version=CONFIG_SCHEMA, study_spec=SPEC_PATH, update_version=UPDATE_VERSION)
    expected["source"]["evaluation_config"] = spec["protocol"]["evaluation"]
    expected["data"].update(
        version=RELATIONAL_EVENT_SCHEMA, event_schema_version=RELATIONAL_EVENT_SCHEMA,
        protocol=spec["protocol"]["path"], task_ids=arm["fit28"],
        study_spec=SPEC_PATH, pool_id=arm["pool"],
    )
    expected["experiment"] = {
        "kind": EXPERIMENT, "arm_id": arm["id"], "pool": arm["pool"],
        "parameterization": arm["parameterization"], "objective": arm["objective"],
        "extra_endpoint_prefix": arm["extra_endpoint_prefix"],
    }
    expected["evidence"]["profile_registration"] = config.get("evidence", {}).get("profile_registration")
    expected["runtime"] = config.get("runtime")
    if config != expected:
        raise ValueError("relation-support scientific recipe differs from registered B/C and pool")
    if (spec["protocol"]["pools"].get(arm["pool"]) != arm["fit28"]
            or len(arm["fit28"]) != 28
            or set(arm["fit28"]) & set(spec["protocol"]["diagnostic_held8"])
            or set(spec["protocol"]["common26"]) - set(arm["fit28"])):
        raise ValueError("relation-support gradient whitelist or common tasks changed")
    _, manifest = load_task_authorities(REPO_ROOT, spec["protocol"]["path"])
    by_id = {row["global_task_id"]: row for row in manifest["tasks"]}
    if any(by_id[task]["split_role"] != "train" for task in arm["fit28"]):
        raise ValueError("relation-support gradients cross a task split")
    profile = config["evidence"]["profile_registration"]
    _validate_profile(profile)
    if profile["status"] == "complete":
        choices = spec["execution"]["C_world_size_choices_after_profile"] if arm["id"].startswith("C_") else spec["execution"]["B_world_size_choices_after_profile"]
        if profile["world_size"] not in choices:
            raise ValueError("relation-support profile topology is outside its registered choices")
    if (config["evidence"]["checkpoint_updates"] != list(range(105, 1261, 105))
            or config["evidence"]["evaluation_updates"] != spec["optimization"]["evaluation_updates"]
            or config["evidence"]["supervised_validation"] != {"optimizer_updates": []}
            or any(type(value) is not int or value <= 0 for value in config["runtime"].values())):
        raise ValueError("relation-support checkpoint, action wall or physical runtime changed")
    query_allocation(config["data"], 0)
    return config
