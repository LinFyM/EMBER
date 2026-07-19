"""Audit whether fixed-flow query gains predict generated source-query actions."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST, validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    restore_trainable_state,
    sha256_file,
    validate_candidate_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_session import open_oracle_model_session
from ember.gate_zero_support.mature_contract import load_mature_lora_positive_control_spec
from ember.gate_zero_support.mature_lora_lr_contract import load_mature_lora_lr_recovery_spec
from ember.gate_zero_support.screen_runtime import (
    _close_parallel,
    _gather,
    _initialize_parallel,
)


EXPECTED_NAME = "smolvla_libero90_gate_zero_query_action_alignment_audit_v1"
EXPECTED_STATUS = (
    "predeclared_after_action_expert_capacity_failure_before_query_action_alignment_outcomes"
)
RESULT_NAME = "query_action_alignment_result.json"


class GateZeroQueryActionAlignmentError(RuntimeError):
    """Raised when the offline alignment audit escapes its frozen authority."""


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroQueryActionAlignmentError(f"{label} changed")


def _load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroQueryActionAlignmentError(f"invalid {label} TOML") from error


def validate_query_action_alignment_spec(
    spec: Mapping[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    lora_fit_path: Path,
    action_fit_path: Path,
    capacity_path: Path,
) -> None:
    for key, expected in (
        ("schema_version", 1),
        ("name", EXPECTED_NAME),
        ("status", EXPECTED_STATUS),
        ("surface", "source_only_offline_query_action_alignment_diagnostic"),
        ("task_ids", [3, 4]),
        (
            "conditions",
            [
                "frozen_base",
                "mature_official_default_r32_lr25e6_recovery",
                "mature_action_expert_lr25e6_recovery",
            ],
        ),
    ):
        _require(spec.get(key), expected, key)
    authority = spec.get("authority", {})
    for key, path in (
        ("gate_zero_contract_sha256", gate_zero_path),
        ("phase0_contract_sha256", phase0_path),
        ("source_competence_contract_sha256", competence_path),
        ("lora_fit_contract_sha256", lora_fit_path),
        ("action_expert_fit_contract_sha256", action_fit_path),
        ("capacity_contract_sha256", capacity_path),
    ):
        _require(authority.get(key), sha256_file(path), key)
    _require(
        authority.get("capacity_result_sha256"),
        "9a91fbb8d53bff90a1c6bcb58bef1270f076f14212ca846c369ca8017bf170ad",
        "capacity result",
    )
    for key in ("validation_numeric_access", "held_numeric_access", "locked_report_numeric_access"):
        _require(authority.get(key), False, key)
    evidence = spec.get("candidate_evidence", {})
    expected_hashes = {
        "task3_lora_manifest_sha256": "06127175cfecc8305a8ff13f2b0f085773e7ab02365b8f2153b043e2f29c47cf",
        "task3_lora_state_sha256": "b20d60b19d83198eb26fa6a1827a910403390c0757bb1208f8af46a400faa68c",
        "task4_lora_manifest_sha256": "9e76ed1d8e1cb1e1dfcc7ca2c27307cc6a5fbd824a40b6f844840f6c846cb3cc",
        "task4_lora_state_sha256": "ba0f22683f04cc1cf6e8cbf287d8a5af9bc039d9f07300da77d6fe553067c053",
        "task3_action_expert_manifest_sha256": "a1eaaf1d7e81b6d602743b5c943eb65a142d07fad9ec66f038fde3326fce24ab",
        "task3_action_expert_state_sha256": "67a2ffe2054e0cd1985211bcfe6fbf929d7203ab4572f56fc12b1cbc089962b2",
        "task4_action_expert_manifest_sha256": "ca5f1586b9b735014e00c778e003246235ef4707eb8030d62e78684d2d0dbe1c",
        "task4_action_expert_state_sha256": "d5a94c0f8b34b06d939bfea3e1b62fc9db48db648ab8827f65d705709736226c",
    }
    for key, expected in expected_hashes.items():
        _require(evidence.get(key), expected, key)
    for key, expected in (
        ("lora_variant", "mature_official_default_r32_lr25e6_recovery"),
        ("action_expert_variant", "mature_action_expert_lr25e6_recovery"),
        ("candidate_step", 1_000),
        ("lora_trainable_parameters", 1_485_312),
        ("action_expert_trainable_parameters", 99_880_992),
    ):
        _require(evidence.get(key), expected, key)
    query = spec.get("query", {})
    for key, expected in (
        ("episode_bounds", [40, 45]),
        ("anchor_frames_per_demo", 8),
        ("anchor_count_per_task", 48),
        ("action_chunk_size", 50),
        ("action_dimension", 7),
        ("time_partition_count", 4),
        ("fixed_inference_noise_seed", 2026071835),
        ("new_environment_rollout_episodes", 0),
    ):
        _require(query.get(key), expected, f"query {key}")
    decision = spec.get("decision", {})
    for key in (
        "may_change_gate_threshold",
        "may_authorize_gate_zero",
        "may_authorize_writer",
        "may_seal_writer_targets",
        "may_access_validation_or_held",
    ):
        _require(decision.get(key), False, f"decision {key}")
    _require(decision.get("no_new_closed_loop_outcome"), True, "closed-loop boundary")
    resources = spec.get("resources", {})
    for key, expected in (
        ("maximum_concurrent_gpus", 4),
        ("diagnostic_gpus", 2),
        ("minimum_free_memory_mib", 10_240),
    ):
        _require(resources.get(key), expected, f"resource {key}")
    load_mature_lora_lr_recovery_spec(
        lora_fit_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )
    load_mature_lora_positive_control_spec(
        action_fit_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )


def load_query_action_alignment_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    lora_fit_path: Path,
    action_fit_path: Path,
    capacity_path: Path,
) -> dict[str, Any]:
    spec = _load_toml(path, "query-action alignment")
    validate_query_action_alignment_spec(
        spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        lora_fit_path=lora_fit_path,
        action_fit_path=action_fit_path,
        capacity_path=capacity_path,
    )
    return spec


def classify_action_alignment(
    base_mse: Mapping[int, float], candidate_mse: Mapping[str, Mapping[int, float]]
) -> dict[str, Any]:
    """Classify deterministic action-error signs without inventing a Gate threshold."""

    tasks = {3, 4}
    if set(base_mse) != tasks or set(candidate_mse) != {"lora", "action_expert"}:
        raise GateZeroQueryActionAlignmentError("alignment classification identity changed")
    reductions: dict[str, dict[str, float]] = {}
    signs: list[bool] = []
    for condition in ("lora", "action_expert"):
        values = candidate_mse[condition]
        if set(values) != tasks:
            raise GateZeroQueryActionAlignmentError("alignment candidate task identity changed")
        reductions[condition] = {}
        for task in sorted(tasks):
            base, candidate = float(base_mse[task]), float(values[task])
            if not (base > 0 and candidate >= 0):
                raise GateZeroQueryActionAlignmentError("alignment MSE is invalid")
            reduction = (base - candidate) / base
            reductions[condition][str(task)] = reduction
            signs.append(reduction > 0)
    if all(signs):
        status = "generated_action_error_improves_without_closed_loop_conversion"
    elif not any(signs):
        status = "fixed_flow_query_surrogate_misaligned"
    else:
        status = "aggregate_query_hides_action_error_heterogeneity"
    return {"status": status, "relative_action_mse_reduction": reductions}


def _candidate(
    spec: Mapping[str, Any], *, root: Path, task_id: int, kind: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    evidence = spec["candidate_evidence"]
    variant = evidence[f"{kind}_variant"]
    candidate = root / f"{variant}_task{task_id}" / "candidates" / "001000"
    manifest = validate_candidate_artifact(
        candidate, expected={"variant": variant, "task_id": task_id, "step": 1_000}
    )
    if (
        sha256_file(candidate / "candidate_manifest.json")
        != evidence[f"task{task_id}_{kind}_manifest_sha256"]
        or sha256_file(candidate / "trainable_state.safetensors")
        != evidence[f"task{task_id}_{kind}_state_sha256"]
        or manifest.get("trainable_parameters") != evidence[f"{kind}_trainable_parameters"]
    ):
        raise GateZeroQueryActionAlignmentError(f"{kind} candidate authority changed")
    metrics = manifest.get("metrics", {})
    if not (
        float(metrics.get("base_query_flow_mse", -1))
        > float(metrics.get("query_flow_mse", -1))
        >= 0
    ):
        raise GateZeroQueryActionAlignmentError(f"{kind} candidate lost positive query evidence")
    return load_file(candidate / "trainable_state.safetensors"), manifest


def _evaluate_variant(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    phase0: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    fit: Mapping[str, Any],
    task_id: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    kind = (
        "lora"
        if fit["variants"][0] == spec["candidate_evidence"]["lora_variant"]
        else "action_expert"
    )
    root = args.lora_fit_root if kind == "lora" else args.action_expert_fit_root
    variant = spec["candidate_evidence"][f"{kind}_variant"]
    session = open_oracle_model_session(
        spec=fit, parent=parent, phase0=phase0, checkpoint=checkpoint,
        manifest=args.manifest, dataset_root=args.dataset_root,
        source_base_checkpoint=args.source_base_checkpoint, variant=variant,
        task_id=task_id, variant_spec=fit["fit"][variant],
    )
    try:
        base_context = session.model.disable_adapter() if kind == "lora" else contextlib.nullcontext()
        with base_context:
            base = session.evaluator.evaluate_action_chunk_errors(session.model)
        state, manifest = _candidate(spec, root=root, task_id=task_id, kind=kind)
        restore_trainable_state(session.model, state)
        candidate = session.evaluator.evaluate_action_chunk_errors(session.model)
        metrics = manifest["metrics"]
        candidate.update(
            query_flow_mse=metrics["query_flow_mse"],
            base_query_flow_mse=metrics["base_query_flow_mse"],
            query_flow_mse_reduction=(metrics["base_query_flow_mse"] - metrics["query_flow_mse"])
            / metrics["base_query_flow_mse"],
        )
        return base, {variant: candidate}, session.task_authorities
    finally:
        session.close()
        del session
        gc.collect()
        torch.cuda.empty_cache()


def _evaluate_task(
    args: argparse.Namespace, spec: Mapping[str, Any], parent: Mapping[str, Any],
    phase0: Mapping[str, Any], checkpoint: Mapping[str, Any],
    lora_fit: Mapping[str, Any], action_fit: Mapping[str, Any], task_id: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    authority = None
    for fit in (lora_fit, action_fit):
        base, candidate, current_authority = _evaluate_variant(
            args, spec, parent, phase0, checkpoint, fit, task_id
        )
        if "frozen_base" in summaries and base != summaries["frozen_base"]:
            raise GateZeroQueryActionAlignmentError("base action metrics changed across wrappers")
        summaries.setdefault("frozen_base", base)
        summaries.update(candidate)
        authority = authority or current_authority
    return {
        "task_id": task_id,
        "task_authorities": authority,
        "conditions": summaries,
    }


def _validate_capacity_result(path: Path, spec: Mapping[str, Any]) -> None:
    if sha256_file(path) != spec["authority"]["capacity_result_sha256"]:
        raise GateZeroQueryActionAlignmentError("capacity result hash changed")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroQueryActionAlignmentError("invalid capacity result") from error
    if (
        result.get("status") != "nonmatched_action_expert_capacity_behavioral_signal_absent"
        or result.get("gate_zero_authorized") is not False
        or result.get("writer_authorized") is not False
    ):
        raise GateZeroQueryActionAlignmentError("capacity failure boundary changed")


def _load_runtime_inputs(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    parent = load_gate_zero_contract(args.gate_zero_contract, args.phase0_contract)
    phase0 = _load_toml(args.phase0_contract, "Phase 0")
    common = dict(
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
    )
    lora_fit = load_mature_lora_lr_recovery_spec(args.lora_fit_contract, **common)
    action_fit = load_mature_lora_positive_control_spec(args.action_expert_fit_contract, **common)
    checkpoint = validate_source_base_checkpoint(
        args.source_base_checkpoint,
        expected={
            "step": spec["authority"]["source_base_checkpoint_step"],
            "checkpoint_role": spec["authority"]["source_base_checkpoint_role"],
        },
    )
    expected = spec["authority"]["source_base_checkpoint_manifest_sha256"]
    if sha256_file(args.source_base_checkpoint / CHECKPOINT_MANIFEST) != expected:
        raise GateZeroQueryActionAlignmentError("source checkpoint hash changed")
    return parent, phase0, checkpoint, lora_fit, action_fit


def _classify_records(records: list[dict[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    by_task = {record["task_id"]: record for record in records}
    base = {
        task: record["conditions"]["frozen_base"]["mean_squared_error"]
        for task, record in by_task.items()
    }
    candidates = {}
    for short, key in (("lora", "lora_variant"), ("action_expert", "action_expert_variant")):
        variant = spec["candidate_evidence"][key]
        candidates[short] = {
            task: record["conditions"][variant]["mean_squared_error"]
            for task, record in by_task.items()
        }
    return classify_action_alignment(base, candidates)


def _build_result(
    args: argparse.Namespace, spec: Mapping[str, Any], records: list[dict[str, Any]],
    decision: Mapping[str, Any], wall_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1, "status": decision["status"], "surface": spec["surface"],
        "contract_sha256": sha256_file(args.config),
        "capacity_result_sha256": sha256_file(args.capacity_result), "records": records,
        "decision": {**decision, "interpretation": spec["decision"],
                     "gate_zero_authorized": False, "writer_authorized": False},
        "gate_zero_authorized": False, "writer_authorized": False,
        "validation_numeric_access": False, "held_numeric_access": False,
        "new_environment_rollout_episodes": 0,
        "resources": {"physical_gpus": args.physical_gpus, "gpu_count": 2,
                      "wall_seconds": wall_seconds},
        "tracking": {"backend": "trackio", "project": spec["resources"]["tracking_project"],
                     "run": args.output_dir.name,
                     "dashboard_command": "trackio show --project EMBER_gate0"},
    }


def _track_records(args: argparse.Namespace, spec: Mapping[str, Any], records: list[dict[str, Any]]) -> None:
    import trackio

    trackio.init(
        project=spec["resources"]["tracking_project"], name=args.output_dir.name,
        group=spec["resources"]["tracking_group"],
        config={"surface": spec["surface"], "task_ids": spec["task_ids"]},
    )
    for record in records:
        for index, condition in enumerate(spec["conditions"]):
            trackio.log({"task_id": record["task_id"], "condition_index": index,
                         "action_chunk_mse": record["conditions"][condition]["mean_squared_error"]})
    trackio.finish()


def run_query_action_alignment(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_query_action_alignment_spec(
        args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        lora_fit_path=args.lora_fit_contract,
        action_fit_path=args.action_expert_fit_contract,
        capacity_path=args.capacity_contract,
    )
    _validate_capacity_result(args.capacity_result, spec)
    parent, phase0, checkpoint, lora_fit, action_fit = _load_runtime_inputs(args, spec)
    context = _initialize_parallel()
    try:
        if context.world_size != 2 or context.rank not in (0, 1):
            raise GateZeroQueryActionAlignmentError("alignment audit requires two task ranks")
        started = time.perf_counter()
        local = _evaluate_task(
            args,
            spec,
            parent,
            phase0,
            checkpoint,
            lora_fit,
            action_fit,
            spec["task_ids"][context.rank],
        )
        gathered = _gather(context, local)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        records = sorted(gathered, key=lambda value: value["task_id"])
        decision = _classify_records(records, spec)
        result = _build_result(args, spec, records, decision, time.perf_counter() - started)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / RESULT_NAME, result)
        _track_records(args, spec, records)
        write_output_checksums(args.output_dir)
        update_latest_link(args.output_dir, args.latest_link)
        return result
    finally:
        _close_parallel(context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "gate-zero-contract",
        "phase0-contract",
        "source-competence-contract",
        "lora-fit-contract",
        "action-expert-fit-contract",
        "capacity-contract",
        "capacity-result",
        "manifest",
        "dataset-root",
        "source-base-checkpoint",
        "lora-fit-root",
        "action-expert-fit-root",
        "output-dir",
        "latest-link",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    try:
        result = run_query_action_alignment(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output_dir / f"failure_packet_rank_{os.environ.get('RANK', '0')}.json",
            {
                "schema_version": 1,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "gate_zero_authorized": False,
                "writer_authorized": False,
                "validation_numeric_access": False,
                "held_numeric_access": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
