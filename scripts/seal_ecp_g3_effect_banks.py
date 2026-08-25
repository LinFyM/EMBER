#!/usr/bin/env python3
"""Seal compact verified policy-effect banks for G3 fit tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.g1_objective import family_balanced_sensitivity_weights
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
    prepare_execution_policy_prefix,
    prepare_policy_effect_prefix_cache,
)
from ember.ecp.shared_compiler_assets import (
    SharedCompilerMember,
    SharedTaskMembers,
    authority_path,
    load_shared_compiler_config,
    load_shared_task_members,
    project_member_to_mobile_rank4,
)
from ember.ecp.shared_compiler_effects import (
    G3_EFFECT_BANK_SCHEMA,
    G3_EFFECT_ROOT_SCHEMA,
)
from ember.ecp.stage0_training import load_stage0_config
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[1]


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda" or device.index is None:
        raise ValueError("G3 effect sealing requires an explicit CUDA device")
    torch.cuda.set_device(device)
    return device


def _occupancy_rows(
    path: Path, *, target: bool
) -> dict[tuple[int, int], dict[str, Any]]:
    results = read_json(path.resolve())
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for row in sorted(
        results.get("rows", ()), key=lambda value: int(value["init_state_id"])
    ):
        if row.get("success") is not True or "occupancy_trajectory" not in row:
            continue
        expert = row.get("task_expert", {})
        task_id = int(
            expert.get("global_task_id", -1)
            if target
            else expert.get("task_id", -1)
        )
        key = (task_id, int(expert.get("step", -1)))
        selected.setdefault(key, dict(row))
    return selected


def _trajectory(
    row: Mapping[str, Any], member: SharedCompilerMember
) -> tuple[dict[str, torch.Tensor], ...]:
    record = row["occupancy_trajectory"]
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError("G3 verified occupancy trajectory changed")
    observed = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = Path(str(row["task_expert"]["checkpoint"])).resolve()
    observations = tuple(observed.get("observations", ()))
    if (
        observed.get("schema_version")
        not in {
            "ember_writer_occupancy_trajectory_v1",
            "ember_pi05_occupancy_trajectory_v1",
        }
        or observed.get("success") is not True
        or str(observed.get("suite")) != str(row["suite"])
        or int(observed.get("task_id", -1)) != int(row["task_id"])
        or int(observed.get("init_state_id", -1)) != int(row["init_state_id"])
        or checkpoint / "adapter.safetensors" != member.adapter
        or len(observations) < 4
    ):
        raise ValueError("G3 member occupancy is not its verified success trajectory")
    indices = torch.linspace(0, len(observations) - 1, 4).round().long().tolist()
    if len(set(indices)) != 4:
        raise ValueError("G3 verified trajectory has too few distinct replans")
    return tuple(dict(observations[index]) for index in indices)


def _observation_batch(
    rows: Sequence[Mapping[str, torch.Tensor]], device: torch.device
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("G3 occupancy observation batch is empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("G3 occupancy observation batch changed")
    return {
        name: torch.cat([row[name] for row in rows]).to(
            device=device, non_blocking=True
        )
        for name in sorted(keys)
    }


def _slice_prefix(
    prefix: ExecutionPolicyPrefix, start: int, stop: int
) -> ExecutionPolicyPrefix:
    return ExecutionPolicyPrefix(
        embeddings=prefix.embeddings[start:stop], padding=prefix.padding[start:stop]
    )


def _capture_states(
    *,
    policy: torch.nn.Module,
    observer: torch.nn.Module,
    lora: BatchedLoRAInference,
    states: Sequence[Mapping[str, torch.Tensor]],
    prefix: ExecutionPolicyPrefix,
    suffix_noise: torch.Tensor,
    microbatch: int,
    denoising_steps: int,
) -> tuple[PolicyEffectResponse, ...]:
    rows: list[list[PolicyEffectResponse]] = [[] for _ in states]
    for start in range(0, suffix_noise.shape[0], microbatch):
        stop = min(start + microbatch, suffix_noise.shape[0])
        selected_prefix = _slice_prefix(prefix, start, stop)
        cache = prepare_policy_effect_prefix_cache(policy, selected_prefix)
        for index, state in enumerate(states):
            rows[index].append(
                capture_policy_effect_response(
                    policy=policy,
                    observer=observer,
                    lora=lora,
                    state=state,
                    prefix=selected_prefix,
                    suffix_noise=suffix_noise[start:stop],
                    denoising_steps=denoising_steps,
                    prepared_prefix_cache=cache,
                )
            )
    return tuple(
        PolicyEffectResponse(
            owner=torch.cat([value.owner for value in values]),
            flow=torch.cat([value.flow for value in values]),
            action=torch.cat([value.action for value in values]),
        )
        for values in rows
    )


def _tensor_values(
    *,
    prefix: ExecutionPolicyPrefix,
    suffix_noise: torch.Tensor,
    validity: torch.Tensor,
    carrier: PolicyEffectResponse,
    members: Sequence[PolicyEffectResponse],
    reliability: torch.Tensor,
    family_weights: torch.Tensor,
    projections: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    values = {
        "prefix_embeddings": prefix.embeddings,
        "prefix_padding": prefix.padding,
        "suffix_noise": suffix_noise,
        "validity": validity,
        "trajectory_ids": torch.arange(
            len(members), device=validity.device
        ).repeat_interleave(4),
        "reliability": reliability,
        "family_weights": family_weights,
        "carrier_owner": carrier.owner,
        "carrier_flow": carrier.flow,
        "carrier_action": carrier.action,
        "members_owner": torch.stack([value.owner for value in members]),
        "members_flow": torch.stack([value.flow for value in members]),
        "members_action": torch.stack([value.action for value in members]),
    }
    for member, state in enumerate(projections):
        for name, value in state.items():
            values[f"projection.{member}.{name}"] = value
    return {
        name: value.detach().cpu().contiguous() for name, value in values.items()
    }


def _seal_task(
    *,
    output_dir: Path,
    task: SharedTaskMembers,
    occupancy: Mapping[tuple[str, int, int], Mapping[str, Any]],
    policy: torch.nn.Module,
    observer: torch.nn.Module,
    lora: BatchedLoRAInference,
    carrier: Mapping[str, torch.Tensor],
    contract: Any,
    owners: Sequence[Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    task_dir = output_dir / f"task_{task.task.authority_id:03d}"
    partial = output_dir / f".task_{task.task.authority_id:03d}.partial"
    if task_dir.exists() or partial.exists():
        raise ValueError(f"G3 effect task output already exists: {task_dir}")
    partial.mkdir(parents=True)
    observations = []
    anchors = []
    for member_index, member in enumerate(task.members):
        key = (task.task.domain, task.task.domain_task_id, member.step)
        row = occupancy.get(key)
        if row is None:
            raise ValueError(f"G3 verified occupancy missing member {key}")
        selected = _trajectory(row, member)
        observations.extend(selected)
        anchors.extend(
            {
                "member": member.name,
                "member_index": member_index,
                "member_step": member.step,
                "occupancy_path": str(row["occupancy_trajectory"]["path"]),
                "init_state_id": int(row["init_state_id"]),
                "selected_replan": replan,
            }
            for replan in torch.linspace(
                0, int(row["occupancy_trajectory"]["replans"]) - 1, 4
            ).round().long().tolist()
        )
    batch = _observation_batch(observations, device)
    prefix = prepare_execution_policy_prefix(policy, batch)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["optimization"]["seed"]) + task.task.authority_id * 1_000_003
    )
    suffix_noise = torch.randn(
        len(observations), 50, 32, generator=generator, dtype=torch.float32
    ).to(device)
    member_states = [
        load_file(str(member.adapter), device=str(device)) for member in task.members
    ]
    states = (carrier, *member_states)
    responses = _capture_states(
        policy=policy,
        observer=observer,
        lora=lora,
        states=states,
        prefix=prefix,
        suffix_noise=suffix_noise,
        microbatch=int(config["optimization"]["effect_microbatch_states"]),
        denoising_steps=int(config["optimization"]["denoising_steps"]),
    )
    carrier_response = responses[0]
    member_responses = responses[1:]
    validity = torch.zeros(
        len(task.members), len(observations), dtype=torch.bool, device=device
    )
    for member in range(len(task.members)):
        validity[member, member * 4 : (member + 1) * 4] = True
    sensitivity = []
    for member, response in enumerate(member_responses):
        mask = validity[member]
        sensitivity.append(
            (
                response.owner[mask].float()
                - carrier_response.owner[mask].float()
            ).square().mean((0, 2, 3)).sqrt()
        )
    family_weights = family_balanced_sensitivity_weights(
        torch.stack(sensitivity), owners
    )
    reliability = torch.tensor(
        [max(member.successes, 1) for member in task.members],
        dtype=torch.float32,
        device=device,
    )
    reliability = reliability / reliability.sum()
    projections = [
        project_member_to_mobile_rank4(
            member=state, carrier=carrier, contract=contract
        )
        for state in member_states
    ]
    tensor_path = partial / "effect_bank.safetensors"
    save_file(
        _tensor_values(
            prefix=prefix,
            suffix_noise=suffix_noise,
            validity=validity,
            carrier=carrier_response,
            members=member_responses,
            reliability=reliability,
            family_weights=family_weights,
            projections=projections,
        ),
        str(tensor_path),
    )
    manifest = {
        "schema_version": G3_EFFECT_BANK_SCHEMA,
        "status": "complete",
        "task": {
            "authority_id": task.task.authority_id,
            "domain": task.task.domain,
            "domain_task_id": task.task.domain_task_id,
            "role": task.task.role,
            "language": task.task.language,
        },
        "state_count": len(observations),
        "member_count": len(task.members),
        "tensor_file": {
            "path": str((task_dir / "effect_bank.safetensors").resolve()),
            "bytes": tensor_path.stat().st_size,
        },
        "metadata": {
            "members": [
                {
                    "name": member.name,
                    "step": member.step,
                    "adapter": str(member.adapter),
                    "adapter_bytes": member.adapter_bytes,
                    "successes": member.successes,
                }
                for member in task.members
            ],
            "anchors": anchors,
            "action_meta_installed": False,
            "held_gradient_use": False,
            "global_member_only": True,
        },
    }
    write_json_atomic(partial / "manifest.json", manifest)
    os.replace(partial, task_dir)
    return manifest


def _task_ids(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item)
    if not rows or len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError(
            "task IDs must be unique comma-separated integers"
        )
    return rows


def _load_tasks(args: argparse.Namespace, config: Mapping[str, Any]):
    from ember.ecp.natural_program_data import load_natural_program_tasks

    fold = config["fold"]
    tasks = load_natural_program_tasks(
        meta_protocol_path=authority_path(
            config, "meta_protocol", asset_root=args.asset_root
        ),
        source_manifest_path=authority_path(
            config, "source_manifest", asset_root=args.asset_root
        ),
        target_manifest_path=authority_path(
            config, "target_manifest", asset_root=args.asset_root
        ),
        data_root=args.data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )
    return load_shared_task_members(config, tasks, asset_root=args.asset_root)


def worker(args: argparse.Namespace) -> None:
    device = _device(args.device)
    config = load_shared_compiler_config(args.config)
    panel = _load_tasks(args, config)
    selected = [row for row in panel if row.task.authority_id in args.task_ids]
    if (
        len(selected) != len(args.task_ids)
        or any(row.task.role not in {"meta_fit", "target_fit"} for row in selected)
    ):
        raise ValueError("G3 effect worker received a held or unknown task")
    meta = _occupancy_rows(
        authority_path(config, "meta_verified_occupancy", asset_root=args.asset_root),
        target=False,
    )
    target1000 = _occupancy_rows(args.target_step1000_results, target=True)
    target2000 = _occupancy_rows(args.target_step2000_results, target=True)
    occupancy = {
        **{("libero90_nonheld", key[0], key[1]): row for key, row in meta.items()},
        **{("target_train24", key[0], key[1]): row for key, row in target1000.items()},
        **{("target_train24", key[0], key[1]): row for key, row in target2000.items()},
    }
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    policy = load_policy(source_checkpoint / "policy", source_config, device)
    policy.requires_grad_(False).eval()
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", asset_root=args.asset_root)
    )
    owners = build_target_owners(contract)
    observer = load_frozen_native_observer(
        stage0_config=load_stage0_config(
            authority_path(config, "stage0_config", asset_root=args.asset_root)
        ),
        owners=owners,
        native_checkpoint=authority_path(
            config, "native_observer_checkpoint", asset_root=args.asset_root
        ),
        device=device,
        max_frames_per_call=int(config["model"]["frame_chunk_size"]),
    )
    prepare_frozen_writer_policy(policy, contract)
    lora = BatchedLoRAInference(policy, contract)
    action_meta = [
        module
        for root in (policy, observer)
        for module in root.modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    if action_meta or any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError(
            "G3 effect sealer loaded Action Meta or trainable policy state"
        )
    carrier = load_file(
        str(authority_path(config, "stable_carrier", asset_root=args.asset_root)),
        device=str(device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in selected:
        records.append(
            _seal_task(
                output_dir=args.output_dir,
                task=task,
                occupancy=occupancy,
                policy=policy,
                observer=observer.encoder.observer,
                lora=lora,
                carrier=carrier,
                contract=contract,
                owners=owners,
                config=config,
                device=device,
            )
        )
    lora.close()
    completion = args.output_dir / f"worker_{args.worker_id}_completion.json"
    if completion.exists():
        raise ValueError("G3 effect worker completion already exists")
    write_json_atomic(
        completion,
        {
            "schema_version": "ember_ecp_g3_effect_worker_v1",
            "worker_id": args.worker_id,
            "task_ids": list(args.task_ids),
            "completed": len(records),
            "device": str(device),
            "git": git_state(REPO_ROOT),
        },
    )


def aggregate(args: argparse.Namespace) -> None:
    config = load_shared_compiler_config(args.config)
    panel = _load_tasks(args, config)
    fit = [row for row in panel if row.task.role in {"meta_fit", "target_fit"}]
    records = []
    for row in fit:
        path = args.output_dir / f"task_{row.task.authority_id:03d}" / "manifest.json"
        manifest = read_json(path)
        if (
            manifest.get("schema_version") != G3_EFFECT_BANK_SCHEMA
            or manifest.get("status") != "complete"
            or int(manifest.get("task", {}).get("authority_id", -1))
            != row.task.authority_id
        ):
            raise ValueError("G3 effect-bank aggregation found an invalid task")
        records.append(
            {
                "authority_id": row.task.authority_id,
                "role": row.task.role,
                "manifest": str(path.resolve()),
                "manifest_bytes": path.stat().st_size,
                "member_count": len(row.members),
            }
        )
    if len(records) != 75 or sum(row["member_count"] for row in records) != 93:
        raise ValueError("G3 fit task/member panel changed")
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("G3 effect-bank root manifest already exists")
    write_json_atomic(
        manifest_path,
        {
            "schema_version": G3_EFFECT_ROOT_SCHEMA,
            "status": "complete",
            "task_count": len(records),
            "member_count": sum(row["member_count"] for row in records),
            "roles": {"meta_fit": 56, "target_fit": 19},
            "records": records,
            "information_wall": {
                "held_gradient_tasks": 0,
                "action_meta_installed": False,
                "deployment_use": False,
            },
            "config": {
                "path": str(args.config.resolve()),
                "bytes": args.config.resolve().stat().st_size,
            },
            "git": git_state(REPO_ROOT),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-step1000-results", type=Path)
    parser.add_argument("--target-step2000-results", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--task-ids", type=_task_ids)
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    if args.command == "worker" and (
        args.target_step1000_results is None
        or args.target_step2000_results is None
        or args.device is None
        or args.task_ids is None
        or args.worker_id is None
    ):
        parser.error(
            "worker requires target occupancies, device, task IDs, and worker ID"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
