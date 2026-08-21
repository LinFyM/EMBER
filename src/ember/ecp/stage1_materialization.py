"""Materialize one privileged ECP Stage 1 consensus LoRA per train24 task."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.compiler import select_compiled_state
from ember.ecp.stage1_data import (
    build_stage1_video_store,
    load_stage1_evidence_bank,
    load_stage1_tasks,
    pack_stage1_videos,
    tokenize_stage1_languages,
)
from ember.ecp.stage1_objective import (
    effective_update_cosine_matrix,
    exact_effective_update_loss,
)
from ember.ecp.stage1_training import (
    REPO_ROOT,
    RUN_SCHEMA,
    STAGE,
    load_stage1_authorities,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.lora import validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything


PROJECTION_SCHEMA = "ember_ecp_stage1_privileged_projection_v4"
PROJECTION_KIND = "ecp_stage1_privileged_content_compiler"


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size}


def _load_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    repository: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    task_visits = checkpoint_macro(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    weights = checkpoint / "ecp.safetensors"
    run_contract = read_json(checkpoint.parent.parent / "run_contract.json")
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != STAGE
        or int(manifest.get("next_macro", -1)) != task_visits
        or manifest.get("run_contract_schema") != RUN_SCHEMA
        or int(manifest.get("world_size", -1)) != 6
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
        or run_contract.get("schema_version") != RUN_SCHEMA
        or run_contract.get("stage") != STAGE
        or run_contract.get("mode") != "formal"
        or run_contract.get("git", {}).get("commit") != repository.get("commit")
    ):
        raise ValueError("ECP Stage 1 materialization checkpoint changed")
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    model.requires_grad_(False).eval()
    return task_visits, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest": _file(manifest_path),
        "weights": _file(weights),
        "training_commit": str(run_contract["git"]["commit"]),
    }


def _base_rows(
    path: Path, *, expert_bank_root: Path, expert_step: int
) -> dict[int, dict[str, Any]]:
    manifest = read_json(path)
    rows = {int(row["ordinal"]): dict(row) for row in manifest.get("tasks", ())}
    if (
        manifest.get("schema_version")
        != "ember_phase_aligned_functional_decoder_train24_projection_v1"
        or manifest.get("projection_kind") != "stable_shared_prior_baseline"
        or Path(str(manifest.get("expert_bank_root", ""))).resolve()
        != expert_bank_root.resolve()
        or int(manifest.get("expert_step", -1)) != expert_step
        or len(rows) != 24
        or set(rows) != set(range(24))
    ):
        raise ValueError("ECP Stage 1 base evaluation surface changed")
    return rows


def _materialize_task(
    *,
    task: Any,
    visit: int,
    config: Mapping[str, Any],
    authorities: Any,
    evidence_bank: Any,
    video_store: Any,
    language_tokens: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    packed = pack_stage1_videos(
        store=video_store,
        ordinal=task.ordinal,
        visit=visit,
        seed=int(config["data"]["pair_seed"]),
        k=int(config["data"]["visible_videos_per_visit"]),
        device=next(authorities.model.parameters()).device,
    )
    tokens, mask = language_tokens[task.ordinal]
    expert = authorities.policy.model.paligemma_with_expert.gemma_expert.model
    with torch.no_grad(), authorities.observer.action_meta.installed(expert):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = authorities.observer.model.encoder(
                policy=authorities.policy,
                frames=packed.frames,
                video_offsets=packed.video_offsets,
                frame_condition_ids=packed.frame_condition_ids,
                language_tokens=tokens,
                language_mask=mask,
            )
            evidence = evidence_bank.evidence(task.ordinal)
            output = authorities.model(
                encoded, evidence, packed.video_group_ids
            )
            candidate = select_compiled_state(
                output.consensus_compilation.state, 0
            )
            member_loss = exact_effective_update_loss(
                candidate, evidence.member_states, authorities.contract
            )
            prior_loss = exact_effective_update_loss(
                candidate, authorities.prior_state, authorities.contract
            )
    stored = {
        name: value.detach().cpu().contiguous() for name, value in candidate.items()
    }
    validate_lora_state(stored, authorities.contract)
    path = output_dir / (
        f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}.safetensors"
    )
    save_file(stored, str(path))
    diagnostics = {
        "anchor": output.anchors.process[0].detach().float().flatten(),
        "teacher": output.teacher.program.process[0].detach().float().flatten(),
        "correction": (
            output.teacher.program.process[0] - output.anchors.process[0]
        ).detach().float().flatten(),
    }
    row = {
        "projected_adapter": str(path.resolve()),
        "projected_adapter_bytes": path.stat().st_size,
        "video_demo_indices": list(packed.demo_indices),
        "visible_video_count": len(packed.demo_indices),
        "successful_member_count": len(evidence_bank.member_indices(task.ordinal)),
        "member_effective_update_loss": float(member_loss.detach()),
        "stable_prior_effective_update_loss": float(prior_loss.detach()),
        "exact_owner_attention": float(
            output.consensus_compilation.exact_owner_attention.detach()
        ),
        "active_event_count": int((output.teacher.program.presence > 0.5).sum()),
        "q_pi_evidence_gate_mean": float(output.teacher.evidence_gate.mean()),
        "q_pi_evidence_gate_min": float(output.teacher.evidence_gate.min()),
        "q_pi_evidence_gate_max": float(output.teacher.evidence_gate.max()),
    }
    return row, {name: value.detach() for name, value in candidate.items()}, diagnostics


def _stack_states(
    rows: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([row[name] for row in rows]) for name in rows[0]
    }


def _off_diagonal_summary(matrix: torch.Tensor) -> dict[str, float]:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    values = matrix[mask]
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _feature_pair_summary(rows: list[torch.Tensor]) -> dict[str, float]:
    features = torch.stack(rows)
    features = torch.nn.functional.normalize(features, dim=1)
    return _off_diagonal_summary(features @ features.transpose(0, 1))


def _cross_task_geometry(
    *,
    candidates: list[Mapping[str, torch.Tensor]],
    directs: list[Mapping[str, torch.Tensor]],
    program_rows: list[Mapping[str, torch.Tensor]],
    contract: Any,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _stack_states(candidates)
    direct = _stack_states(directs)
    candidate_pair, candidate_energy, _ = effective_update_cosine_matrix(
        candidate, candidate, contract
    )
    direct_pair, direct_energy, _ = effective_update_cosine_matrix(
        direct, direct, contract
    )
    candidate_direct, _, _ = effective_update_cosine_matrix(
        candidate, direct, contract
    )
    own = candidate_direct.diagonal()
    other = candidate_direct.masked_fill(
        torch.eye(
            candidate_direct.shape[0],
            dtype=torch.bool,
            device=candidate_direct.device,
        ),
        float("-inf"),
    ).max(dim=1).values
    norm_ratio = (
        candidate_energy.clamp_min(0).sqrt()
        / direct_energy.clamp_min(1e-20).sqrt()
    )
    thresholds = gate["pre_rollout_geometry"]
    candidate_summary = _off_diagonal_summary(candidate_pair)
    own_retrieval = int((own > other).sum())
    passed = (
        candidate_summary["mean"]
        <= float(thresholds["maximum_candidate_pair_cosine_mean"])
        and float(own.mean()) >= float(thresholds["minimum_mean_own_direct_cosine"])
        and float(own.mean()) > float(other.mean())
        and own_retrieval >= int(thresholds["minimum_own_retrieval_count"])
        and float(norm_ratio.mean())
        >= float(thresholds["minimum_mean_candidate_to_direct_norm_ratio"])
    )
    return {
        "task_count": len(candidates),
        "candidate_pair_cosine": candidate_summary,
        "direct_pair_cosine": _off_diagonal_summary(direct_pair),
        "candidate_to_direct": {
            "mean_own_cosine": float(own.mean()),
            "mean_nearest_other_cosine": float(other.mean()),
            "own_retrieval_count": own_retrieval,
            "mean_effective_norm_ratio": float(norm_ratio.mean()),
        },
        "program_pair_cosine": {
            name: _feature_pair_summary([row[name] for row in program_rows])
            for name in ("anchor", "teacher", "correction")
        },
        "thresholds": dict(thresholds),
        "passed": passed,
    }


def _projection_manifest(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    task_visits: int,
    checkpoint_asset: Mapping[str, Any],
    base_manifest: Path,
    rank: int,
    rows: list[dict[str, Any]],
    cross_task_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROJECTION_SCHEMA,
        "projection_kind": PROJECTION_KIND,
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "stage1_config": _file(args.config),
        "stage1_checkpoint": checkpoint_asset["weights"],
        "stage1_checkpoint_authority": dict(checkpoint_asset),
        "base_projection_manifest": _file(base_manifest),
        "expert_bank_root": str(args.expert_bank_root.resolve()),
        "expert_step": int(args.expert_step),
        "optimization": {
            "task_visits": task_visits,
            "fold": int(config["roles"]["fold"]),
            "fit_task_count": 19,
            "held_task_count": 5,
            "held_shared_gradient_steps": 0,
            "compiler_frozen_for_materialization": True,
            "single_complete_lora": True,
            "final_lora_averaging": False,
            "rank": rank,
            "all_ranks_writable": True,
            "parameterization": "prior-only exact template; full-process absolute factors",
            "content_address_separated": True,
            "functional_start_task_visits": int(
                config["objective"]["functional_start_task_visits"]
            ),
            "coordinate_bootstrap_end_task_visits": int(
                config["objective"]["coordinate_bootstrap"]["end_task_visits"]
            ),
            "objective_phase": (
                "coordinate_bootstrap"
                if task_visits
                <= int(
                    config["objective"]["coordinate_bootstrap"][
                        "end_task_visits"
                    ]
                )
                else "policy_functional"
            ),
        },
        "information_wall": {
            "role": "development_train_oracle_only",
            "deployment_carrier": False,
            "privileged_q_pi": True,
            "teacher_action_deployment_reads": 0,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "second_adapter_deployed": False,
        },
        "tasks": rows,
        "cross_task_geometry": dict(cross_task_geometry),
        "content_hash_policy": "disabled_by_owner",
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
    if repository["dirty_paths"]:
        raise ValueError("formal ECP Stage 1 materialization requires a clean worktree")
    if args.output_dir.exists():
        raise ValueError("ECP Stage 1 materialization output already exists")
    config = load_stage1_config(args.config)
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("ECP Stage 1 materialization uses one GPU")
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities = load_stage1_authorities(args, config, context)
    task_visits, checkpoint_asset = _load_checkpoint(
        args.stage1_checkpoint,
        model=authorities.model,
        device=context.device,
        repository=repository,
    )
    if (
        task_visits
        not in set(
            int(value)
            for value in config["optimization"]["checkpoint_task_visits"]
        )
        or int(config["materialization"]["visible_video_count"])
        != int(config["data"]["visible_videos_per_visit"])
    ):
        raise ValueError("ECP Stage 1 materialization contract changed")
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(config, "target_manifest"),
        selection_path=stage1_repo_authority(
            config, "successful_member_selection"
        ),
        data_root=args.data_root,
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(
            config, "successful_member_selection"
        ),
        phase_analysis_path=stage1_asset_authority(
            config, "phase_analysis", args.asset_root
        ),
        phase_code_root=stage1_asset_authority(
            config, "phase_code_root", args.asset_root
        ),
        asset_root=args.asset_root,
        contract=authorities.contract,
        device=context.device,
    )
    base_manifest = stage1_asset_authority(
        config, "base_projection_manifest", args.asset_root
    )
    base_rows = _base_rows(
        base_manifest,
        expert_bank_root=args.expert_bank_root,
        expert_step=args.expert_step,
    )
    languages = tokenize_stage1_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(
            authorities.source_config["features"]["tokenizer_max_length"]
        ),
        device=context.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    store = build_stage1_video_store(
        tasks, frame_stride=int(config["data"]["frame_stride"])
    )
    rows = []
    candidate_states = []
    direct_states = []
    program_rows = []
    try:
        visit = int(config["materialization"]["video_visit"])
        for task in tasks:
            generated, candidate, program = _materialize_task(
                task=task,
                visit=visit,
                config=config,
                authorities=authorities,
                evidence_bank=evidence,
                video_store=store,
                language_tokens=languages,
                output_dir=args.output_dir,
            )
            base = base_rows[task.ordinal]
            direct = load_file(
                str(Path(str(base["expert_checkpoint"])) / "adapter.safetensors"),
                device=str(context.device),
            )
            validate_lora_state(direct, authorities.contract)
            candidate_states.append(candidate)
            direct_states.append(direct)
            program_rows.append(program)
            rows.append(
                {
                    "suite": task.suite,
                    "task_id": task.task_id,
                    "ordinal": task.ordinal,
                    "global_task_id": task.global_task_id,
                    "expert_checkpoint": base["expert_checkpoint"],
                    "fold_role": task.fold_role,
                    **generated,
                }
            )
    finally:
        store.close()
    geometry = _cross_task_geometry(
        candidates=candidate_states,
        directs=direct_states,
        program_rows=program_rows,
        contract=authorities.contract,
        gate=config["gate2"],
    )
    result = _projection_manifest(
        args=args,
        config=config,
        repository=repository,
        task_visits=task_visits,
        checkpoint_asset=checkpoint_asset,
        base_manifest=base_manifest,
        rank=int(authorities.contract.rank),
        rows=rows,
        cross_task_geometry=geometry,
    )
    write_json_atomic(args.output_dir / "projection_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_stage1_privileged_compiler_v4.json",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--expert-step", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames-per-call", type=int)
    return parser


def finalize_materialization_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "stage1_checkpoint",
        "tokenizer_path",
        "data_root",
        "expert_bank_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.expert_step != 2000:
        raise ValueError("ECP Stage 1 held5 evaluation uses the fixed step2000 bank")
    return args
