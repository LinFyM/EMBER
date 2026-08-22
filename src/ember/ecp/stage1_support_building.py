"""Build the frozen full-layer policy-support bank used by ECP Stage 1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.observer import TargetOwnerProjector
from ember.ecp.policy_response import (
    CapturedPolicyResponse,
    capture_policy_response,
)
from ember.ecp.stage1_data import ECPStage1EvidenceBank
from ember.ecp.stage1_support import (
    SUPPORT_BANK_SCHEMA,
    SUPPORT_CHANNELS,
    SUPPORT_TASK_SCHEMA,
    LearnerOccupancySource,
    load_learner_occupancy_sources,
)
from ember.ecp.stage1_config import (
    REPO_ROOT,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.ecp.stage1_training import load_stage1_authorities
from ember.functional_adaptation.phase_alignment import arc_length_phase_indices
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything
def _trajectory_panels(
    *,
    path: Path,
    expected_bytes: int,
    expected_success: bool,
    selected_indices: Sequence[int] | None,
    device: torch.device,
) -> tuple[tuple[tuple[int, int], dict[str, torch.Tensor]], ...]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    observations = tuple(value.get("observations", ()))
    actions = tuple(value.get("action_chunks", ()))
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or value.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or bool(value.get("success")) != expected_success
        or len(observations) != len(actions)
        or len(observations) < 8
    ):
        raise ValueError("policy-support trajectory authority changed")
    if selected_indices is None:
        action_sequence = torch.cat(actions).float().flatten(1)
        selected = tuple(
            int(index)
            for index in arc_length_phase_indices(action_sequence, count=8).tolist()
        )
    else:
        selected = tuple(int(index) for index in selected_indices)
    if len(selected) != 8 or max(selected) >= len(observations):
        raise ValueError("policy-support trajectory lost eight phase states")
    result = []
    for start in range(0, 8, 2):
        pair = (selected[start], selected[start + 1])
        keys = set(observations[pair[0]])
        if any(set(observations[index]) != keys for index in pair):
            raise ValueError("policy-support observation keys changed")
        batch = {
            name: torch.cat([observations[index][name] for index in pair]).to(
                device, non_blocking=True
            )
            for name in sorted(keys)
        }
        from lerobot.utils.constants import ACTION

        batch[ACTION] = torch.cat([actions[index] for index in pair]).to(
            device, non_blocking=True
        )
        result.append((pair, batch))
    return tuple(result)


def _selected_state(
    states: Mapping[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in states.items()}


def _support_weights(
    *,
    source: torch.Tensor,
    shared: torch.Tensor,
    experts: torch.Tensor,
    expert_weights: torch.Tensor,
    learner_success: bool | None,
    agreement_temperature: float,
    failed_learner_base_weight: float,
) -> tuple[float, float, float]:
    weights = expert_weights.float().clamp_min(1e-4)
    weights = weights / weights.sum()
    consensus = torch.einsum("m,mbhd->bhd", weights, experts.float())
    adapter_energy = (consensus - source.float()).square().mean().clamp_min(1e-8)
    response_scale = (
        consensus.square().mean() + source.float().square().mean()
    ).clamp_min(1e-8)
    disagreement = torch.einsum(
        "m,mbhd->bhd", weights, (experts.float() - consensus[None]).square()
    ).mean()
    agreement = torch.exp(
        -agreement_temperature * disagreement / adapter_energy
    )
    outcome = (
        1.0 if learner_success is not False else failed_learner_base_weight
    )
    response_weight = float((agreement * outcome).clamp(1e-3, 1.0))
    source_distance = (source.float() - consensus).square().mean() / response_scale
    shared_distance = (shared.float() - consensus).square().mean() / response_scale
    source_weight = float(torch.exp(-agreement_temperature * source_distance).clamp(0, 1))
    shared_weight = float(torch.exp(-agreement_temperature * shared_distance).clamp(0, 1))
    return response_weight, source_weight, shared_weight


def _panel_payload(
    *,
    panel_id: int,
    kind: str,
    trajectory_path: Path,
    trajectory_bytes: int,
    selected_indices: tuple[int, int],
    policy_seed: int,
    responses: Mapping[str, CapturedPolicyResponse],
    member_keys: Sequence[str],
    expert_weights: torch.Tensor,
    learner_success: bool | None,
    agreement_temperature: float,
    failed_learner_base_weight: float,
) -> dict[str, Any]:
    expert = torch.stack([responses[key].flow for key in member_keys])
    expert_owner = torch.stack(
        [responses[key].owner_basis for key in member_keys]
    )
    outcome, source_weight, shared_weight = _support_weights(
        source=responses["source"].flow,
        shared=responses["shared"].flow,
        experts=expert,
        expert_weights=expert_weights,
        learner_success=learner_success,
        agreement_temperature=agreement_temperature,
        failed_learner_base_weight=failed_learner_base_weight,
    )
    return {
        "panel_id": panel_id,
        "kind": kind,
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_bytes": trajectory_bytes,
        "selected_indices": list(selected_indices),
        "policy_seed": policy_seed,
        "source_response": responses["source"].flow.to(
            device="cpu", dtype=torch.bfloat16
        ),
        "shared_response": responses["shared"].flow.to(
            device="cpu", dtype=torch.bfloat16
        ),
        "expert_responses": expert.to(device="cpu", dtype=torch.bfloat16),
        "source_owner_response": responses["source"].owner_basis.to(
            device="cpu", dtype=torch.bfloat16
        ),
        "expert_owner_responses": expert_owner.to(
            device="cpu", dtype=torch.bfloat16
        ),
        "expert_weights": expert_weights.detach().cpu().float(),
        "outcome_weight": outcome,
        "source_support_weight": source_weight,
        "shared_support_weight": shared_weight,
        "learner_success": learner_success,
    }


def _capture_panel(
    *,
    policy: torch.nn.Module,
    states: Mapping[str, Mapping[str, torch.Tensor]],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    projector: TargetOwnerProjector,
    policy_seed: int,
    horizon_basis: int,
) -> dict[str, CapturedPolicyResponse]:
    return {
        name: capture_policy_response(
            policy=policy,
            state=state,
            contract=contract,
            batch=batch,
            projector=projector,
            policy_seed=policy_seed,
            horizon_basis=horizon_basis,
        )
        for name, state in states.items()
    }


def build_task_support(
    *,
    task: Any,
    evidence_bank: ECPStage1EvidenceBank,
    learner_sources: Sequence[LearnerOccupancySource],
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    shared_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    projector: TargetOwnerProjector,
    output_path: Path,
    base_policy_seed: int,
    horizon_basis: int,
    agreement_temperature: float,
    failed_learner_base_weight: float,
) -> dict[str, Any]:
    indices = evidence_bank.member_indices(task.ordinal)
    members = tuple(evidence_bank.members[index] for index in indices)
    member_states = tuple(_selected_state(evidence_bank.member_states, index) for index in indices)
    member_keys = tuple(f"expert_{local}" for local in range(len(indices)))
    expert_weights = evidence_bank.reliability[
        torch.tensor(indices, device=evidence_bank.reliability.device)
    ].float()
    response = torch.zeros(
        len(indices), 8, 38, len(SUPPORT_CHANNELS), horizon_basis, 128
    )
    response_weights = torch.zeros(len(indices), 8, len(SUPPORT_CHANNELS))
    panels: list[dict[str, Any]] = []
    panel_id = 0

    for local, member in enumerate(members):
        trajectory_panels = _trajectory_panels(
            path=member.trajectory_path,
            expected_bytes=member.trajectory_path.stat().st_size,
            expected_success=True,
            selected_indices=member.selected_replan_indices,
            device=next(policy.parameters()).device,
        )
        for panel_index, (selected, batch) in enumerate(trajectory_panels):
            seed = base_policy_seed + task.ordinal * 100_000 + local * 1_000 + panel_index
            states = {
                "source": identity_state,
                "shared": shared_state,
                **{key: state for key, state in zip(member_keys, member_states, strict=True)},
            }
            captured = _capture_panel(
                policy=policy,
                states=states,
                contract=contract,
                batch=batch,
                projector=projector,
                policy_seed=seed,
                horizon_basis=horizon_basis,
            )
            event_slice = slice(2 * panel_index, 2 * panel_index + 2)
            source_basis = captured["source"].owner_basis.cpu()
            response[local, event_slice, :, 0] = (
                captured[member_keys[local]].owner_basis.cpu() - source_basis
            )
            response[local, event_slice, :, 1] = (
                captured["shared"].owner_basis.cpu() - source_basis
            )
            response_weights[local, event_slice, :2] = 1.0
            panels.append(
                _panel_payload(
                    panel_id=panel_id,
                    kind="successful",
                    trajectory_path=member.trajectory_path,
                    trajectory_bytes=member.trajectory_path.stat().st_size,
                    selected_indices=selected,
                    policy_seed=seed,
                    responses=captured,
                    member_keys=member_keys,
                    expert_weights=expert_weights,
                    learner_success=None,
                    agreement_temperature=agreement_temperature,
                    failed_learner_base_weight=failed_learner_base_weight,
                )
            )
            panel_id += 1

    learner_accumulator = torch.zeros_like(response[:, :, :, 2:])
    learner_weight = torch.zeros_like(response_weights[:, :, 2:])
    projected_cache: dict[Path, Mapping[str, torch.Tensor]] = {}
    for source_index, source in enumerate(learner_sources):
        projected = projected_cache.get(source.projected_adapter)
        if projected is None:
            projected = load_file(
                str(source.projected_adapter),
                device=str(next(policy.parameters()).device),
            )
            validate_lora_state(projected, contract)
            projected_cache[source.projected_adapter] = projected
        trajectory_panels = _trajectory_panels(
            path=source.trajectory_path,
            expected_bytes=source.trajectory_bytes,
            expected_success=source.success,
            selected_indices=None,
            device=next(policy.parameters()).device,
        )
        for panel_index, (selected, batch) in enumerate(trajectory_panels):
            seed = (
                base_policy_seed
                + task.ordinal * 100_000
                + 50_000
                + source_index * 1_000
                + panel_index
            )
            states = {
                "source": identity_state,
                "shared": shared_state,
                "learner": projected,
                **{key: state for key, state in zip(member_keys, member_states, strict=True)},
            }
            captured = _capture_panel(
                policy=policy,
                states=states,
                contract=contract,
                batch=batch,
                projector=projector,
                policy_seed=seed,
                horizon_basis=horizon_basis,
            )
            payload = _panel_payload(
                panel_id=panel_id,
                kind="learner",
                trajectory_path=source.trajectory_path,
                trajectory_bytes=source.trajectory_bytes,
                selected_indices=selected,
                policy_seed=seed,
                responses=captured,
                member_keys=member_keys,
                expert_weights=expert_weights,
                learner_success=source.success,
                agreement_temperature=agreement_temperature,
                failed_learner_base_weight=failed_learner_base_weight,
            )
            panels.append(payload)
            weight = float(payload["outcome_weight"])
            event_slice = slice(2 * panel_index, 2 * panel_index + 2)
            source_basis = captured["source"].owner_basis.cpu()
            learner_delta = captured["learner"].owner_basis.cpu() - source_basis
            shared_delta = captured["shared"].owner_basis.cpu() - source_basis
            for local, key in enumerate(member_keys):
                learner_accumulator[local, event_slice, :, 0] += weight * (
                    captured[key].owner_basis.cpu() - source_basis
                )
                learner_accumulator[local, event_slice, :, 1] += weight * learner_delta
                learner_accumulator[local, event_slice, :, 2] += weight * shared_delta
                learner_weight[local, event_slice] += weight
            panel_id += 1
    nonzero = learner_weight > 0
    normalized = learner_accumulator / learner_weight[:, :, None, :, None, None].clamp_min(1e-8)
    response[:, :, :, 2:] = torch.where(
        nonzero[:, :, None, :, None, None], normalized, learner_accumulator
    )
    response_weights[:, :, 2:] = learner_weight / max(len(learner_sources), 1)
    payload = {
        "schema_version": SUPPORT_TASK_SCHEMA,
        "ordinal": int(task.ordinal),
        "global_task_id": int(task.global_task_id),
        "suite": str(task.suite),
        "task_id": int(task.task_id),
        "fold_role": str(task.fold_role),
        "member_indices": list(indices),
        "policy_response": response.to(dtype=torch.bfloat16),
        "policy_response_weights": response_weights.float(),
        "panels": panels,
        "successful_panel_count": sum(panel["kind"] == "successful" for panel in panels),
        "learner_panel_count": sum(panel["kind"] == "learner" for panel in panels),
        "learner_successful_trajectory_count": sum(source.success for source in learner_sources),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return {
        "ordinal": int(task.ordinal),
        "fold_role": str(task.fold_role),
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
        "member_count": len(indices),
        "successful_panels": payload["successful_panel_count"],
        "learner_panels": payload["learner_panel_count"],
    }


def build_support_shard(args: Any) -> None:
    config = load_stage1_config(args.config)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("policy-support bank requires a clean pushed authority")
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("each policy-support shard owns one GPU")
    seed_everything(int(config["optimization"]["seed"]), context)
    authorities = load_stage1_authorities(args, config, context)
    from ember.ecp.stage1_data import load_stage1_evidence_bank, load_stage1_tasks

    tasks = load_stage1_tasks(
        target_manifest=stage1_repo_authority(config, "target_manifest"),
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
        data_root=args.data_root,
    )
    evidence = load_stage1_evidence_bank(
        selection_path=stage1_repo_authority(config, "successful_member_selection"),
        phase_analysis_path=stage1_asset_authority(config, "phase_analysis", args.asset_root),
        phase_code_root=stage1_asset_authority(config, "phase_code_root", args.asset_root),
        asset_root=args.asset_root,
        contract=authorities.contract,
        device=context.device,
    )
    learner = load_learner_occupancy_sources(
        root=stage1_asset_authority(config, "learner_occupancy", args.asset_root),
        tasks=tasks,
    )
    selected = tuple(
        task for task in tasks if task.ordinal % args.shard_count == args.shard_index
    )
    if not selected:
        raise ValueError("policy-support shard has no tasks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    projector = authorities.observer.model.encoder.observer.projector
    support = config["policy_support"]
    for task in selected:
        output = args.output_dir / f"task_{task.ordinal:02d}.pt"
        if output.exists():
            raise ValueError(f"policy-support task output exists: {output}")
        rows.append(
            build_task_support(
                task=task,
                evidence_bank=evidence,
                learner_sources=learner.get(task.ordinal, ()),
                policy=authorities.policy,
                identity_state=authorities.identity_state,
                shared_state=authorities.prior_state,
                contract=authorities.contract,
                projector=projector,
                output_path=output,
                base_policy_seed=int(support["capture_policy_seed"]),
                horizon_basis=int(support["horizon_basis"]),
                agreement_temperature=float(support["agreement_temperature"]),
                failed_learner_base_weight=float(
                    support["failed_learner_base_weight"]
                ),
            )
        )
        print(rows[-1], flush=True)
    write_json_atomic(
        args.output_dir / f"shard_{args.shard_index:02d}.json",
        {
            "schema_version": "ember_ecp_stage1_policy_support_shard_v2",
            "repository": repository,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "tasks": rows,
        },
    )


def assemble_support_bank(args: Any) -> dict[str, Any]:
    config = load_stage1_config(args.config)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("policy-support assembly requires a clean pushed authority")
    rows = []
    for shard_index in range(args.shard_count):
        shard = read_json(args.output_dir / f"shard_{shard_index:02d}.json")
        if (
            shard.get("schema_version")
            != "ember_ecp_stage1_policy_support_shard_v2"
            or int(shard.get("shard_index", -1)) != shard_index
            or int(shard.get("shard_count", -1)) != args.shard_count
            or shard.get("repository", {}).get("commit") != repository.get("commit")
        ):
            raise ValueError("policy-support shard authority changed")
        rows.extend(dict(row) for row in shard["tasks"])
    rows.sort(key=lambda row: int(row["ordinal"]))
    if [int(row["ordinal"]) for row in rows] != list(range(24)):
        raise ValueError("policy-support bank does not cover train24")
    for row in rows:
        path = args.output_dir / str(row["file"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError("policy-support task file changed during assembly")
    result = {
        "schema_version": SUPPORT_BANK_SCHEMA,
        "repository": repository,
        "config": {"path": str(args.config), "bytes": args.config.stat().st_size},
        "support_channels": list(SUPPORT_CHANNELS),
        "event_slots": 8,
        "owners": 38,
        "horizon_basis": int(config["policy_support"]["horizon_basis"]),
        "program_width": int(config["model"]["program_width"]),
        "source_policy_frozen": True,
        "observer_projection_frozen": True,
        "owner_resolved_panels": True,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
        "tasks": rows,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(args.output_dir / "manifest.json", result)
    return result
