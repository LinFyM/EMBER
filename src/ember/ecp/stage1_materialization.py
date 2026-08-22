"""Materialize the ECP direct-absolute reachability surface over train24."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.compiler import select_compiled_state
from ember.ecp.stage1_data import (
    build_stage1_video_store,
    gauge_canonicalize_factors,
    load_stage1_evidence_bank,
    load_stage1_tasks,
    pack_stage1_videos,
    stage1_demo_indices,
    tokenize_stage1_languages,
)
from ember.ecp.stage1_free_program import TaskLocalFreeProgramTable
from ember.ecp.stage1_objective import (
    effective_update_cosine_matrix,
    exact_effective_update_loss,
)
from ember.ecp.stage1_support import load_policy_support_bank
from ember.ecp.stage1_config import (
    REPO_ROOT,
    RUN_SCHEMA,
    STAGE,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.ecp.stage1_training import load_stage1_authorities
from ember.ecp.program import ECPProgram
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything


PROJECTION_SCHEMA = "ember_ecp_stage1_direct_absolute_free_program_projection_v22"
PROJECTION_KIND = "ecp_stage1_privileged_direct_absolute_reachability"


@dataclass(frozen=True)
class Stage1MaterializationConfig:
    base: dict[str, Any]
    stage: str
    run_schema: str
    seed: int
    checkpoint_cursors: tuple[int, ...]
    cursor_name: str
    settings: Mapping[str, Any]
    projection_schema: str
    projection_kind: str
    objective_phase: str


def resolve_stage1_materialization_config(
    path: Path,
) -> Stage1MaterializationConfig:
    base = load_stage1_config(path)
    return Stage1MaterializationConfig(
        base=base,
        stage=STAGE,
        run_schema=RUN_SCHEMA,
        seed=int(base["optimization"]["seed"]),
        checkpoint_cursors=tuple(
            int(value)
            for value in base["materialization"]["allowed_checkpoint_task_visits"]
        ),
        cursor_name="task_visits",
        settings=base["materialization"],
        projection_schema=PROJECTION_SCHEMA,
        projection_kind=PROJECTION_KIND,
        objective_phase="task_balanced_direct_absolute_free_program_reachability",
    )


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size}


def _load_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    repository: Mapping[str, Any],
    expected_stage: str,
    expected_run_schema: str,
) -> tuple[int, dict[str, Any]]:
    cursor = checkpoint_macro(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    weights = checkpoint / "ecp.safetensors"
    run_contract = read_json(checkpoint.parent.parent / "run_contract.json")
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != expected_stage
        or int(manifest.get("next_macro", -1)) != cursor
        or manifest.get("run_contract_schema") != expected_run_schema
        or int(manifest.get("world_size", -1)) != 6
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
        or run_contract.get("schema_version") != expected_run_schema
        or run_contract.get("stage") != expected_stage
        or run_contract.get("mode") != "formal"
        or run_contract.get("git", {}).get("commit") != repository.get("commit")
    ):
        raise ValueError("ECP Stage 1 materialization checkpoint changed")
    state = load_file(str(weights), device=str(device))
    ordinal_key = "free_programs.task_ordinals"
    if ordinal_key not in state:
        raise ValueError("ECP Stage 1 checkpoint has no free Program table")
    programs = {}
    for index, ordinal_value in enumerate(state[ordinal_key].tolist()):
        prefix = f"free_programs.rows.{index}."
        programs[int(ordinal_value)] = ECPProgram(
            language=state[prefix + "language"][None],
            scene=state[prefix + "scene"][None],
            process=state[prefix + "base_process"][None],
            presence=state[prefix + "presence"][None],
            uncertainty=state[prefix + "base_uncertainty"][None],
        )
    cell = config["free_program_oracle"]
    model.add_module(
        "free_programs",
        TaskLocalFreeProgramTable(
            programs,
            process_delta_scale=float(cell["process_delta_scale"]),
            uncertainty_log_scale_bound=float(cell["uncertainty_log_scale_bound"]),
        ).to(device),
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return cursor, {
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
    support_bank: Any,
    video_store: Any,
    language_tokens: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    free_programs = authorities.model.free_programs
    is_free = task.ordinal in free_programs.ordinals
    evidence = evidence_bank.evidence(task.ordinal, support_bank.task(task.ordinal))
    if is_free:
        row = free_programs.row(task.ordinal)
        demo_indices = stage1_demo_indices(
            ordinal=task.ordinal,
            visit=visit,
            seed=int(config["data"]["pair_seed"]),
            k=int(config["data"]["visible_videos_per_visit"]),
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            program = row()
            anchor = row.base_program()
            compilation = authorities.model.compiler(program)
    else:
        packed = pack_stage1_videos(
            store=video_store,
            ordinal=task.ordinal,
            visit=visit,
            seed=int(config["data"]["pair_seed"]),
            k=int(config["data"]["visible_videos_per_visit"]),
            device=next(authorities.model.parameters()).device,
        )
        demo_indices = packed.demo_indices
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
                anchor = authorities.model.visible_program(
                    encoded, packed.video_group_ids, group_count=1
                )
                teacher = authorities.model.policy_teacher(anchor, evidence)
                program = teacher.program
                compilation = authorities.model.compiler(program)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        candidate = select_compiled_state(compilation.state, 0)
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
        "anchor": anchor.process[0].detach().float().flatten(),
        "teacher": program.process[0].detach().float().flatten(),
        "correction": (program.process[0] - anchor.process[0])
        .detach()
        .float()
        .flatten(),
    }
    row = {
        "projected_adapter": str(path.resolve()),
        "projected_adapter_bytes": path.stat().st_size,
        "video_demo_indices": list(demo_indices),
        "visible_video_count": len(demo_indices),
        "program_route": (
            "fit_task_local_free_program" if is_free else "held_frozen_q_pi"
        ),
        "successful_member_count": len(evidence_bank.member_indices(task.ordinal)),
        "member_effective_update_loss": float(member_loss.detach()),
        "stable_prior_effective_update_loss": float(prior_loss.detach()),
        "exact_owner_attention": float(compilation.exact_owner_attention.detach()),
        "active_event_count": int((program.presence > 0.5).sum()),
    }
    if not is_free:
        row.update(
            {
                "q_pi_evidence_gate_mean": float(teacher.evidence_gate.mean()),
                "q_pi_evidence_gate_min": float(teacher.evidence_gate.min()),
                "q_pi_evidence_gate_max": float(teacher.evidence_gate.max()),
                "support_attention_entropy": float(
                    teacher.support_attention_entropy.mean()
                ),
            }
        )
    else:
        row.update(
            {
                name: float(value)
                for name, value in free_programs.row(task.ordinal).diagnostics().items()
            }
        )
    return row, {name: value.detach() for name, value in candidate.items()}, diagnostics


def _stack_states(
    rows: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {name: torch.stack([row[name] for row in rows]) for name in rows[0]}


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


def _effective_rank_summary(
    rows: list[Mapping[str, torch.Tensor]], contract: Any
) -> dict[str, float]:
    participation = []
    top_one = []
    for state in rows:
        for owner in contract.targets:
            _, canonical_b = gauge_canonicalize_factors(
                state[owner.name + LORA_A_SUFFIX],
                state[owner.name + LORA_B_SUFFIX],
            )
            singular = canonical_b.float().square().sum(dim=0)
            energy = singular.square()
            probability = energy / energy.sum().clamp_min(1e-20)
            participation.append(1.0 / probability.square().sum())
            top_one.append(probability.max())
    participation_tensor = torch.stack(participation)
    top_one_tensor = torch.stack(top_one)
    return {
        "mean_participation_rank": float(participation_tensor.mean()),
        "minimum_participation_rank": float(participation_tensor.min()),
        "mean_top1_energy_fraction": float(top_one_tensor.mean()),
        "maximum_top1_energy_fraction": float(top_one_tensor.max()),
    }


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
    candidate_direct, _, _ = effective_update_cosine_matrix(candidate, direct, contract)
    own = candidate_direct.diagonal()
    other = (
        candidate_direct.masked_fill(
            torch.eye(
                candidate_direct.shape[0],
                dtype=torch.bool,
                device=candidate_direct.device,
            ),
            float("-inf"),
        )
        .max(dim=1)
        .values
    )
    norm_ratio = (
        candidate_energy.clamp_min(0).sqrt() / direct_energy.clamp_min(1e-20).sqrt()
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
        "effective_rank": {
            "candidate": _effective_rank_summary(candidates, contract),
            "direct": _effective_rank_summary(directs, contract),
        },
        "thresholds": dict(thresholds),
        "passed": passed,
    }


def _projection_manifest(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    checkpoint_cursor: int,
    checkpoint_asset: Mapping[str, Any],
    base_manifest: Path,
    support_manifest: Path,
    rank: int,
    rows: list[dict[str, Any]],
    cross_task_geometry: Mapping[str, Any],
    materialization: Stage1MaterializationConfig,
) -> dict[str, Any]:
    return {
        "schema_version": materialization.projection_schema,
        "projection_kind": materialization.projection_kind,
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "stage1_config": _file(args.config),
        "stage1_checkpoint": checkpoint_asset["weights"],
        "stage1_checkpoint_authority": dict(checkpoint_asset),
        "base_projection_manifest": _file(base_manifest),
        "policy_support_bank": _file(support_manifest),
        "expert_bank_root": str(args.expert_bank_root.resolve()),
        "expert_step": int(args.expert_step),
        "optimization": {
            materialization.cursor_name: checkpoint_cursor,
            "fold": int(config["roles"]["fold"]),
            "fit_task_count": 19,
            "held_task_count": 5,
            "held_shared_gradient_steps": 0,
            "compiler_trainable_during_training": False,
            "fit_task_local_free_program_trainable_during_training": True,
            "visible_program_frozen_during_training": True,
            "policy_teacher_frozen_during_training": True,
            "compiler_frozen_for_materialization": True,
            "single_complete_lora": True,
            "final_lora_averaging": False,
            "rank": rank,
            "all_ranks_writable": True,
            "parameterization": "prior-only exact template; full-process direct absolute family-specific A/B generation",
            "content_address_separated": True,
            "query_content_modulated": True,
            "policy_support_teacher": True,
            "raw_factor_amplitude_retained": True,
            "fixed_rank_partition": False,
            "second_adapter_deployed": False,
            "objective_phase": materialization.objective_phase,
        },
        "information_wall": {
            "role": "development_train_oracle_only",
            "deployment_carrier": False,
            "privileged_q_pi": "fit initialization and held inference only",
            "fit_task_local_free_program_deployed": False,
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
    materialization = resolve_stage1_materialization_config(args.config)
    config = materialization.base
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("ECP Stage 1 materialization uses one GPU")
    seed_everything(materialization.seed, context)
    authorities = load_stage1_authorities(args, config, context)
    checkpoint_cursor, checkpoint_asset = _load_checkpoint(
        args.stage1_checkpoint,
        model=authorities.model,
        config=config,
        device=context.device,
        repository=repository,
        expected_stage=materialization.stage,
        expected_run_schema=materialization.run_schema,
    )
    if checkpoint_cursor not in set(materialization.checkpoint_cursors) or int(
        materialization.settings["visible_video_count"]
    ) != int(config["data"]["visible_videos_per_visit"]):
        raise ValueError("ECP Stage 1 materialization contract changed")
    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(config, "target_manifest"),
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
        data_root=args.data_root,
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
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
    support_manifest = stage1_asset_authority(
        config, "policy_support_bank", args.asset_root
    )
    support = load_policy_support_bank(
        manifest_path=support_manifest,
        evidence_bank=evidence,
        contract=authorities.contract,
        task_ordinals=set(range(24)),
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
        max_length=int(authorities.source_config["features"]["tokenizer_max_length"]),
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
        visit = int(materialization.settings["video_visit"])
        for task in tasks:
            generated, candidate, program = _materialize_task(
                task=task,
                visit=visit,
                config=config,
                authorities=authorities,
                evidence_bank=evidence,
                support_bank=support,
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
        checkpoint_cursor=checkpoint_cursor,
        checkpoint_asset=checkpoint_asset,
        base_manifest=base_manifest,
        support_manifest=support_manifest,
        rank=int(authorities.contract.rank),
        rows=rows,
        cross_task_geometry=geometry,
        materialization=materialization,
    )
    write_json_atomic(args.output_dir / "projection_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_stage1_direct_absolute_free_program_v22.json",
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
