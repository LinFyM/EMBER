"""Fail-closed loader for the predeclared Gate 0 source-only pilot."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any


class GateZeroContractError(RuntimeError):
    """Raised when the Gate 0 pilot no longer matches its sealed authority."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inclusive(bounds: list[int], label: str) -> set[int]:
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(value, int) for value in bounds)
        or bounds[0] < 0
        or bounds[1] < bounds[0]
    ):
        raise GateZeroContractError(f"invalid {label} episode bounds")
    return set(range(bounds[0], bounds[1] + 1))


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroContractError(f"{label} changed: {actual!r} != {expected!r}")


def _validate_episode_partitions(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    declared = {
        "source_base_fit": spec["access"]["source_base_fit"],
        "oracle_support": spec["access"]["oracle_support"],
        "functional_query": spec["access"]["functional_query"],
        "locked_source_report": spec["access"]["locked_source_report"],
    }
    surfaces = {name: _inclusive(bounds, name) for name, bounds in declared.items()}
    if len(set().union(*surfaces.values())) != sum(len(values) for values in surfaces.values()):
        raise GateZeroContractError("episode partitions overlap")
    for name, bounds in declared.items():
        _require_equal(bounds, phase0["episode_authority"][name], f"episode authority {name}")
    _require_equal(spec["base_fit"]["episode_bounds"], declared["source_base_fit"], "base-fit bounds")
    _require_equal(spec["oracle"]["support_episode_bounds"], declared["oracle_support"], "support bounds")
    _require_equal(
        spec["oracle"]["selection"]["episode_bounds"], declared["functional_query"], "query bounds"
    )
    _require_equal(
        spec["report"]["offline_episode_bounds"], declared["locked_source_report"], "report bounds"
    )


def _validate_scientific_surface(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    source = set(phase0["splits"]["source"])
    validation = set(phase0["splits"]["validation"])
    held = set(phase0["splits"]["held_out"])
    task_ids = spec["data"]["task_ids"]
    _require_equal(task_ids, spec["oracle"]["task_ids"], "oracle pilot tasks")
    _require_equal(task_ids, spec["base_competence"]["task_ids"], "competence pilot tasks")
    if len(task_ids) != len(set(task_ids)) or not set(task_ids) <= source:
        raise GateZeroContractError("Gate 0 pilot tasks must be unique resealed source tasks")
    if set(task_ids) & (validation | held):
        raise GateZeroContractError("Gate 0 pilot may not use validation or held tasks")
    _require_equal(spec["access"]["forbidden_numeric_splits"], ["validation", "held_out"], "forbidden splits")
    _require_equal(spec["access"]["fully_pristine_all_fields"], [48, 49], "fully pristine rows")
    _require_equal(spec["access"]["action_reward_policy_outcome_locked"], [46, 49], "modality lock")
    _require_equal(spec["access"]["prior_rgb_only_exposure"], [40, 47], "prior RGB disclosure")
    if spec["oracle"]["selection"]["report_access_before_selection_freeze"]:
        raise GateZeroContractError("report access is forbidden before oracle selection freeze")
    if not spec["report"]["no_selection_after_report_access"]:
        raise GateZeroContractError("report access must permanently stop model selection")


def _validate_oracle(spec: dict[str, Any]) -> None:
    oracle = spec["oracle"]
    _require_equal(oracle["rank"], 8, "primary oracle rank")
    _require_equal(oracle["alpha"], 8, "primary oracle alpha")
    _require_equal(oracle["dropout"], 0.0, "primary oracle dropout")
    _require_equal(oracle["init_lora_weights"], True, "functional-zero LoRA initialization")
    _require_equal(
        oracle["initial_physical_delta_required_exact_zero"],
        True,
        "initial physical-delta requirement",
    )
    _require_equal(
        oracle["initial_fixed_loss_required_exact_base"],
        True,
        "initial fixed-loss requirement",
    )
    _require_equal(oracle["modules_to_save"], [], "modules_to_save")
    targets = oracle["target_modules"]
    expected_targets = [
        "model.vlm_with_expert.lm_expert.layers.14.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.14.self_attn.v_proj",
        "model.vlm_with_expert.lm_expert.layers.15.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.15.self_attn.v_proj",
    ]
    _require_equal(targets, expected_targets, "primary oracle target matrices")
    _require_equal(oracle["expected_trainable_parameters"], 40320, "oracle parameter count")
    if oracle["checkpoint_candidates"][0] != 0 or oracle["checkpoint_candidates"][-1] != oracle["steps"]:
        raise GateZeroContractError("oracle checkpoint candidates must include base and final steps")


def _validate_thresholds(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    gate = phase0["gate_zero"]["thresholds"]
    threshold = spec["thresholds"]
    _require_equal(threshold["median_success_gain_pp"], gate["median_success_gain_pp"], "success threshold")
    _require_equal(
        threshold["median_locked_action_loss_reduction_fraction"],
        gate["median_query_action_loss_reduction_fraction"],
        "action-loss threshold",
    )
    _require_equal(threshold["positive_task_fraction"], gate["positive_task_fraction"], "positive-task threshold")
    _require_equal(threshold["median_action_kl_proxy_max"], gate["median_action_kl_max"], "drift threshold")
    if spec["diagnostics"]["exact_policy_likelihood_claimed"]:
        raise GateZeroContractError("SmolVLA pilot must not claim an exact policy likelihood")


def _validate_resources(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    resources = spec["resources"]
    _require_equal(resources["maximum_concurrent_gpus"], phase0["resources"]["max_concurrent_gpus"], "GPU ceiling")
    if resources["maximum_concurrent_gpus"] > 4 or resources["minimum_free_memory_mib"] < 10240:
        raise GateZeroContractError("Gate 0 resource headroom contract weakened")
    if resources["pilot_gpus"] > resources["maximum_concurrent_gpus"]:
        raise GateZeroContractError("Gate 0 pilot exceeds the GPU ceiling")


def load_gate_zero_contract(
    path_or_text: Path | str,
    phase0_path: Path,
    *,
    from_text: bool = False,
) -> dict[str, Any]:
    """Load and cross-check the Gate 0 contract against the permanent Phase 0 seal."""

    try:
        if from_text:
            spec = tomllib.loads(str(path_or_text))
        else:
            spec = tomllib.loads(Path(path_or_text).read_text(encoding="utf-8"))
        phase0 = tomllib.loads(phase0_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroContractError("invalid Gate 0 or Phase 0 TOML") from error
    if spec.get("schema_version") != 1:
        raise GateZeroContractError("unsupported Gate 0 contract schema")
    if spec.get("status") != "predeclared_before_source_policy_outcomes":
        raise GateZeroContractError("Gate 0 contract is not frozen before outcomes")
    _require_equal(spec["authority"]["phase0_contract_sha256"], _sha256(phase0_path), "Phase 0 SHA256")
    _validate_episode_partitions(spec, phase0)
    _validate_scientific_surface(spec, phase0)
    _validate_oracle(spec)
    _validate_thresholds(spec, phase0)
    _validate_resources(spec, phase0)
    if spec["recovery"]["threshold_changes_forbidden"] is not True:
        raise GateZeroContractError("Gate 0 recovery may not change thresholds")
    if spec["recovery"]["held_access_forbidden"] is not True:
        raise GateZeroContractError("Gate 0 recovery may not inspect held data")
    if spec["recovery"]["writer_authorized_by_pilot"] is not False:
        raise GateZeroContractError("the two-task pilot cannot authorize Writer training")
    return spec
