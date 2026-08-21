"""Frozen all-panel functional audit for an ECP Stage 1 projection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.stage1_data import load_stage1_evidence_bank, load_stage1_tasks
from ember.ecp.stage1_materialization import PROJECTION_SCHEMA
from ember.ecp.stage1_support import (
    cache_policy_support_panels,
    load_policy_support_bank,
    policy_support_distillation_loss,
    policy_support_loss_from_response,
)
from ember.ecp.stage1_training import (
    REPO_ROOT,
    load_stage1_authorities,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.lora import validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything


AUDIT_CONFIG_SCHEMA = "ember_ecp_stage1_policy_support_audit_config_v1"
AUDIT_SCHEMA = "ember_ecp_stage1_policy_support_audit_v1"
AUDIT_SHARD_SCHEMA = "ember_ecp_stage1_policy_support_audit_shard_v1"


def _file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size}


def load_audit_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    thresholds = config.get("thresholds", {})
    required = {
        "minimum_fit_tasks_better_than_source",
        "minimum_fit_tasks_better_than_shared",
        "minimum_held_tasks_better_than_source",
        "minimum_held_tasks_better_than_shared",
    }
    if (
        config.get("schema_version") != AUDIT_CONFIG_SCHEMA
        or config.get("status") != "active_frozen_support_audit"
        or set(thresholds) != required
        or any(int(thresholds[name]) < 0 for name in required)
        or config.get("information_wall", {}).get("validation_action_or_reward_reads")
        != 0
        or config.get("information_wall", {}).get("test_action_or_reward_reads")
        != 0
    ):
        raise ValueError("unsupported ECP policy-support audit contract")
    return config


def _authority_path(config: Mapping[str, Any], name: str, asset_root: Path) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else asset_root / path


def _load_projection(
    *, manifest_path: Path, support_manifest: Path, tasks: Sequence[Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = read_json(manifest_path)
    rows = {int(row["ordinal"]): dict(row) for row in manifest.get("tasks", ())}
    expected = {int(task.ordinal): task for task in tasks}
    support = manifest.get("policy_support_bank", {})
    if (
        manifest.get("schema_version") != PROJECTION_SCHEMA
        or manifest.get("repository", {}).get("dirty_paths") != []
        or set(rows) != set(expected)
        or Path(str(support.get("path", ""))).resolve() != support_manifest.resolve()
        or int(support.get("bytes", -1)) != support_manifest.stat().st_size
        or manifest.get("information_wall", {}).get("validation_action_or_reward_reads")
        != 0
        or manifest.get("information_wall", {}).get("test_action_or_reward_reads")
        != 0
    ):
        raise ValueError("ECP policy-support projection authority changed")
    for ordinal, row in rows.items():
        task = expected[ordinal]
        adapter = Path(str(row.get("projected_adapter", ""))).resolve()
        if (
            row.get("fold_role") != task.fold_role
            or int(row.get("global_task_id", -1)) != int(task.global_task_id)
            or not adapter.is_file()
            or adapter.stat().st_size != int(row.get("projected_adapter_bytes", -1))
        ):
            raise ValueError("ECP policy-support projected adapter changed")
    return manifest, rows


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(float(row[name]) for row in rows) / len(rows)


def _panel_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    candidate = _mean(rows, "candidate_response")
    source = _mean(rows, "source_response")
    shared = _mean(rows, "shared_response")
    consensus = _mean(rows, "consensus_response")
    return {
        "panels": len(rows),
        "candidate_response": candidate,
        "source_response": source,
        "shared_response": shared,
        "consensus_response": consensus,
        "candidate_to_source_ratio": candidate / max(source, 1e-12),
        "candidate_to_shared_ratio": candidate / max(shared, 1e-12),
        "candidate_panel_wins_over_source": sum(
            float(row["candidate_response"]) < float(row["source_response"])
            for row in rows
        ),
        "candidate_panel_wins_over_shared": sum(
            float(row["candidate_response"]) < float(row["shared_response"])
            for row in rows
        ),
    }


def _task_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "all": _panel_summary(rows),
        "successful": _panel_summary(
            [row for row in rows if row["kind"] == "successful"]
        ),
        "learner": _panel_summary(
            [row for row in rows if row["kind"] == "learner"]
        ),
    }


def _aggregate_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def aggregate(kind: str) -> dict[str, Any] | None:
        summaries = [task["summary"][kind] for task in tasks]
        summaries = [summary for summary in summaries if summary is not None]
        if not summaries:
            return None
        candidate = _mean(summaries, "candidate_response")
        source = _mean(summaries, "source_response")
        shared = _mean(summaries, "shared_response")
        return {
            "tasks": len(summaries),
            "panels": sum(int(summary["panels"]) for summary in summaries),
            "candidate_response": candidate,
            "source_response": source,
            "shared_response": shared,
            "consensus_response": _mean(summaries, "consensus_response"),
            "candidate_to_source_ratio": candidate / max(source, 1e-12),
            "candidate_to_shared_ratio": candidate / max(shared, 1e-12),
            "candidate_tasks_better_than_source": sum(
                float(summary["candidate_response"])
                < float(summary["source_response"])
                for summary in summaries
            ),
            "candidate_tasks_better_than_shared": sum(
                float(summary["candidate_response"])
                < float(summary["shared_response"])
                for summary in summaries
            ),
        }

    return {
        "all": aggregate("all"),
        "successful": aggregate("successful"),
        "learner": aggregate("learner"),
    }


def summarize_policy_support_audit(
    *, tasks: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = [task for task in tasks if task["fold_role"] == "fit"]
    held = [task for task in tasks if task["fold_role"] == "held_transform_only"]
    aggregates = {"fit19": _aggregate_tasks(fit), "held5": _aggregate_tasks(held)}
    fit_all = aggregates["fit19"]["all"]
    held_all = aggregates["held5"]["all"]
    conditions = {
        "fit_aggregate_better_than_source": fit_all["candidate_response"]
        < fit_all["source_response"],
        "fit_aggregate_better_than_shared": fit_all["candidate_response"]
        < fit_all["shared_response"],
        "held_aggregate_better_than_source": held_all["candidate_response"]
        < held_all["source_response"],
        "held_aggregate_better_than_shared": held_all["candidate_response"]
        < held_all["shared_response"],
        "fit_task_breadth_over_source": fit_all[
            "candidate_tasks_better_than_source"
        ]
        >= int(thresholds["minimum_fit_tasks_better_than_source"]),
        "fit_task_breadth_over_shared": fit_all[
            "candidate_tasks_better_than_shared"
        ]
        >= int(thresholds["minimum_fit_tasks_better_than_shared"]),
        "held_task_breadth_over_source": held_all[
            "candidate_tasks_better_than_source"
        ]
        >= int(thresholds["minimum_held_tasks_better_than_source"]),
        "held_task_breadth_over_shared": held_all[
            "candidate_tasks_better_than_shared"
        ]
        >= int(thresholds["minimum_held_tasks_better_than_shared"]),
    }
    return aggregates, {
        "thresholds": {name: int(value) for name, value in thresholds.items()},
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def _evaluate_task(
    *,
    task: Any,
    support_bank: Any,
    support_task: Any,
    projection_row: Mapping[str, Any],
    authorities: Any,
) -> dict[str, Any]:
    adapter = load_file(
        str(Path(str(projection_row["projected_adapter"])).resolve()),
        device=str(next(authorities.policy.parameters()).device),
    )
    validate_lora_state(adapter, authorities.contract)
    requests = {(int(task.ordinal), int(panel.panel_id)) for panel in support_task.panels}
    cached = cache_policy_support_panels(
        bank=support_bank,
        requests=requests,
        device=next(authorities.policy.parameters()).device,
    )
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for panel in support_task.panels:
            item = cached[(int(task.ordinal), int(panel.panel_id))]
            candidate = policy_support_distillation_loss(
                policy=authorities.policy,
                candidate_state=adapter,
                contract=authorities.contract,
                cached=item,
            )
            source = policy_support_loss_from_response(
                candidate=item.panel.source_response, panel=item.panel
            )
            shared = policy_support_loss_from_response(
                candidate=item.panel.shared_response, panel=item.panel
            )
            weights = item.panel.expert_weights.float().clamp_min(1e-4)
            weights = weights / weights.sum()
            consensus_response = torch.einsum(
                "m,mbhd->bhd", weights, item.panel.expert_responses.float()
            )
            consensus = policy_support_loss_from_response(
                candidate=consensus_response, panel=item.panel
            )
            rows.append(
                {
                    "panel_id": int(panel.panel_id),
                    "kind": str(panel.kind),
                    "learner_success": panel.learner_success,
                    "outcome_weight": float(panel.outcome_weight),
                    "candidate_response": float(candidate.response),
                    "source_response": float(source.response),
                    "shared_response": float(shared.response),
                    "consensus_response": float(consensus.response),
                }
            )
    return {
        "ordinal": int(task.ordinal),
        "global_task_id": int(task.global_task_id),
        "suite": str(task.suite),
        "task_id": int(task.task_id),
        "fold_role": str(task.fold_role),
        "projected_adapter": _file(
            Path(str(projection_row["projected_adapter"]))
        ),
        "summary": _task_summary(rows),
        "panels": rows,
    }


def build_audit_shard(args: Any) -> None:
    audit_config = load_audit_config(args.config)
    output = args.output_dir / f"shard_{args.shard_index:02d}.json"
    if output.exists():
        raise ValueError("policy-support audit shard output already exists")
    stage1_config_path = _authority_path(
        audit_config, "stage1_config", REPO_ROOT
    )
    stage1_config = load_stage1_config(stage1_config_path)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("policy-support audit requires clean pushed authority")
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("each policy-support audit shard owns one GPU")
    seed_everything(int(stage1_config["optimization"]["seed"]), context)
    base_authorities = load_stage1_authorities(args, stage1_config, context)
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(stage1_config, "target_manifest"),
        selection_path=stage1_repo_authority(
            stage1_config, "successful_member_selection"
        ),
        data_root=args.data_root,
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(
            stage1_config, "successful_member_selection"
        ),
        phase_analysis_path=stage1_asset_authority(
            stage1_config, "phase_analysis", args.asset_root
        ),
        phase_code_root=stage1_asset_authority(
            stage1_config, "phase_code_root", args.asset_root
        ),
        asset_root=args.asset_root,
        contract=base_authorities.contract,
        device=context.device,
    )
    selected = tuple(
        task for task in tasks if task.ordinal % args.shard_count == args.shard_index
    )
    if not selected:
        raise ValueError("policy-support audit shard has no tasks")
    support_manifest = stage1_asset_authority(
        stage1_config, "policy_support_bank", args.asset_root
    )
    support = load_policy_support_bank(
        manifest_path=support_manifest,
        evidence_bank=evidence,
        task_ordinals={int(task.ordinal) for task in selected},
        device=context.device,
    )
    projection_path = _authority_path(
        audit_config, "projection_manifest", args.asset_root
    )
    projection, projection_rows = _load_projection(
        manifest_path=projection_path,
        support_manifest=support_manifest,
        tasks=tasks,
    )
    rows = []
    for task in selected:
        rows.append(
            _evaluate_task(
                task=task,
                support_bank=support,
                support_task=support.task(task.ordinal),
                projection_row=projection_rows[task.ordinal],
                authorities=base_authorities,
            )
        )
        print({"ordinal": task.ordinal, **rows[-1]["summary"]["all"]}, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output,
        {
            "schema_version": AUDIT_SHARD_SCHEMA,
            "repository": repository,
            "audit_config": _file(args.config),
            "stage1_config": _file(stage1_config_path),
            "projection_manifest": _file(projection_path),
            "projection_training_commit": projection["repository"]["commit"],
            "support_manifest": _file(support_manifest),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "tasks": rows,
        },
    )


def assemble_audit(args: Any) -> dict[str, Any]:
    audit_config = load_audit_config(args.config)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("policy-support audit assembly requires clean authority")
    rows = []
    authority = None
    for index in range(args.shard_count):
        shard = read_json(args.output_dir / f"shard_{index:02d}.json")
        if (
            shard.get("schema_version") != AUDIT_SHARD_SCHEMA
            or int(shard.get("shard_index", -1)) != index
            or int(shard.get("shard_count", -1)) != args.shard_count
            or shard.get("repository", {}).get("commit") != repository["commit"]
        ):
            raise ValueError("policy-support audit shard authority changed")
        current = {
            name: shard[name]
            for name in (
                "audit_config",
                "stage1_config",
                "projection_manifest",
                "projection_training_commit",
                "support_manifest",
            )
        }
        authority = current if authority is None else authority
        if current != authority:
            raise ValueError("policy-support audit shards used different authorities")
        rows.extend(dict(row) for row in shard["tasks"])
    rows.sort(key=lambda row: int(row["ordinal"]))
    if [int(row["ordinal"]) for row in rows] != list(range(24)):
        raise ValueError("policy-support audit does not cover train24")
    aggregates, gate = summarize_policy_support_audit(
        tasks=rows, thresholds=audit_config["thresholds"]
    )
    assert authority is not None
    result = {
        "schema_version": AUDIT_SCHEMA,
        "repository": repository,
        **authority,
        "task_equal": True,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
        "tasks": rows,
        "aggregates": aggregates,
        "gate": gate,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(args.output_dir / "audit.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "assemble"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-frames-per-call", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "data_root",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid policy-support audit shard")
    if args.mode == "build" and any(
        getattr(args, name) is None
        for name in ("asset_root", "source_run", "checkpoint", "data_root")
    ):
        raise ValueError("policy-support audit build requires source authorities")
    return args
