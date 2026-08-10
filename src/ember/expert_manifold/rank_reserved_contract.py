"""Fail-closed load-only authority for the active q/v rank-reserved compiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.rank_reserved_authority import (
    RANK_RESERVED_ADAPTER_SCHEMA,
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_EPISODE_SCHEMA,
    RANK_RESERVED_FAMILY,
    RANK_RESERVED_PROGRAM_REFERENCE,
    RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA,
    _ASSETS,
    _BASE_AUTHORITY,
    _COMPILER,
    _DESIGN_AUTHORITY,
    _EVALUATION_BASE,
    _GENERATION_EVIDENCE,
    _METHOD,
    _TOP_LEVEL,
    _repo_file,
    load_rank_reserved_program_reference,
    rank_reserved_output_path,
)
from ember.expert_manifold.rank_reserved_deployment import (
    _deployment_evidence_matches,
    build_rank_reserved_deployment_evidence,
    load_rank_reserved_profile_evidence,
)
from ember.expert_manifold.v6_prior_contract import REPO_ROOT, load_v6_prior_config
from ember.pi05_source_checkpoint import write_json_atomic


__all__ = (
    "RANK_RESERVED_ADAPTER_SCHEMA",
    "RANK_RESERVED_CANONICAL_CONFIG",
    "RANK_RESERVED_CONFIG_SCHEMA",
    "RANK_RESERVED_EPISODE_SCHEMA",
    "RANK_RESERVED_FAMILY",
    "RANK_RESERVED_PROGRAM_REFERENCE",
    "RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA",
    "load_rank_reserved_config",
    "load_rank_reserved_profile_evidence",
    "load_rank_reserved_program_reference",
    "rank_reserved_asset",
    "rank_reserved_output_path",
    "seal_rank_reserved_deployment",
)


def _evaluation_matches(evaluation: object) -> bool:
    if not isinstance(evaluation, Mapping):
        return False
    fixed = {name: evaluation.get(name) for name in _EVALUATION_BASE}
    state = (evaluation.get("formal_status"), evaluation.get("online_smoke_evidence"))
    return (
        set(evaluation)
        == set(_EVALUATION_BASE) | {"formal_status", "online_smoke_evidence"}
        and fixed == _EVALUATION_BASE
        and (
            state == ("awaiting_live_a40_rank_reserved_deployment_profile", None)
            or (
                state[0] == "sealed_from_live_a40_rank_reserved_deployment_profile"
                and _deployment_evidence_matches(state[1])
            )
        )
    )


def _generation_evidence_matches(value: object) -> bool:
    if value != _GENERATION_EVIDENCE:
        return False
    try:
        path = rank_reserved_output_path(
            value["path"],
            label="rank-reserved generation evidence",
            expected_bytes=int(value["bytes"]),
            require_file=True,
        )
        result = json.loads(path.read_text(encoding="utf-8"))
        structural = result["structural"]
        aggregate = result["aggregate"]
        native = aggregate["native_dynamic"]["task_common_video_specific"]["expert"]
        scope = result["evidence_scope"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == value["schema"]
        and result.get("passed") is True
        and result.get("status") == "valid_generation_only_qv_pivot_rank14_plus2_gate"
        and result.get("execution", {}).get("device") == "NVIDIA A40"
        and int(result.get("execution", {}).get("content_hashes", -1)) == 0
        and structural.get("finite") is True
        and structural.get("kept_b_bit_exact") is True
        and structural.get("residual_b_zero_exact") is True
        and int(structural.get("qv_pairs", -1)) == 288
        and int(structural.get("action_pairs", -1)) == 16
        and int(structural.get("carrier_a_nonzero_rows", -1))
        == int(structural.get("carrier_a_rows", -2))
        == 2_880
        and float(aggregate.get("qv_tangent_rank2_capture_mean", 0.0)) >= 0.999
        and float(native.get("energy_weighted_cosine", 0.0)) >= 0.99
        and float(native.get("error_to_zero_ratio", 1.0)) <= 0.10
        and int(scope.get("held_rows", -1)) == 80
        and int(scope.get("held_tasks", -1)) == 8
        and int(scope.get("policy_action_forwards", -1)) == 0
        and int(scope.get("rollouts", -1)) == 0
        and int(scope.get("training_updates", -1)) == 0
        and result.get("compiler", {}).get("public_rank") == 16
        and result.get("compiler", {}).get("single_lora") is True
        and result.get("compiler", {}).get("expert_runtime_bank") is False
        and result.get("compiler", {}).get("language_only_bypass") is False
    )


def seal_rank_reserved_deployment(
    path: Path,
    *,
    require_run_commit: str,
) -> dict[str, Any]:
    """Atomically move the canonical config from awaiting to live-sealed."""

    path = path.resolve()
    config = load_rank_reserved_config(path)
    if (
        path != RANK_RESERVED_CANONICAL_CONFIG.resolve()
        or config.get("status") != "awaiting_live_a40_rank_reserved_deployment_profile"
        or config.get("evaluation", {}).get("formal_status")
        != "awaiting_live_a40_rank_reserved_deployment_profile"
        or config.get("evaluation", {}).get("online_smoke_evidence") is not None
        or len(require_run_commit) != 40
    ):
        raise ExpertManifoldError("rank-reserved deployment seal state changed")
    evidence = build_rank_reserved_deployment_evidence(
        config,
        require_run_commit=require_run_commit,
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid rank-reserved Writer config JSON") from error
    raw["status"] = "sealed_from_live_a40_rank_reserved_deployment_profile"
    raw["evaluation"][
        "formal_status"
    ] = "sealed_from_live_a40_rank_reserved_deployment_profile"
    raw["evaluation"]["online_smoke_evidence"] = evidence
    if not _evaluation_matches(raw.get("evaluation")):
        raise ExpertManifoldError("rank-reserved sealed evaluation config changed")
    write_json_atomic(path, raw)
    load_rank_reserved_config(path)
    return evidence


def load_rank_reserved_config(path: Path) -> dict[str, Any]:
    """Load the sole active rank-reserved config and inherit only frozen v6 assets."""

    path = path.resolve()
    if path != RANK_RESERVED_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("non-canonical rank-reserved Writer config")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid rank-reserved Writer config JSON") from error
    if (
        not isinstance(config, dict)
        or set(config) != _TOP_LEVEL
        or config.get("schema_version") != RANK_RESERVED_CONFIG_SCHEMA
        or config.get("status")
        not in {
            "awaiting_live_a40_rank_reserved_deployment_profile",
            "sealed_from_live_a40_rank_reserved_deployment_profile",
        }
        or config.get("method") != _METHOD
        or config.get("base_writer_authority") != _BASE_AUTHORITY
        or config.get("design_authority") != _DESIGN_AUTHORITY
        or not _generation_evidence_matches(config.get("generation_evidence"))
        or config.get("compiler") != _COMPILER
        or config.get("assets") != _ASSETS
        or not _evaluation_matches(config.get("evaluation"))
        or config.get("status") != config["evaluation"]["formal_status"]
        or config.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("rank-reserved Writer config changed")
    base_path = _repo_file(
        str(_BASE_AUTHORITY["path"]),
        label="rank-reserved base Writer authority",
        expected_bytes=int(_BASE_AUTHORITY["bytes"]),
    )
    _repo_file(str(_DESIGN_AUTHORITY["path"]), label="rank-reserved design authority")
    base = load_v6_prior_config(base_path)
    load_rank_reserved_program_reference(RANK_RESERVED_PROGRAM_REFERENCE)
    effective = dict(config)
    effective["writer"] = {
        **dict(base["writer"]),
        "architecture": "frozen_v6_qv_rank14_plus_condition_local_rank2_reward_v1",
    }
    effective["initialization"] = dict(base["initialization"])
    effective["authorities"] = dict(base["authorities"])
    effective["information_wall"] = dict(base["information_wall"])
    effective["condition_feature"] = dict(base["condition_feature"])
    effective["program_residual"] = dict(base["program_residual"])
    return effective


def rank_reserved_asset(config: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Resolve exactly macro0 or the sealed cycle1 Program reference."""

    checkpoint = checkpoint.resolve()
    macro0 = (REPO_ROOT / str(config["assets"]["macro0"]["checkpoint"])).resolve()
    cycle1 = (REPO_ROOT / str(config["assets"]["cycle1"]["checkpoint"])).resolve()
    if checkpoint == macro0:
        return dict(config["assets"]["macro0"])
    if checkpoint == cycle1:
        reference = load_rank_reserved_program_reference(cycle1)
        return {
            **dict(config["assets"]["cycle1"]),
            "program_reference": reference,
        }
    raise ExpertManifoldError("rank-reserved evaluation asset is not registered")
