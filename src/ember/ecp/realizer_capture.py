"""GPU capture of probe-preserving ECP effect-code evidence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectParticles,
    capture_policy_effect_particles,
    prepare_execution_policy_prefix,
    prepare_policy_effect_prefix_cache,
)
from ember.ecp.realizer_evidence import (
    EFFECT_PARTICLE_SHARD_SCHEMA,
    balanced_member_shards,
    load_effect_member_rows,
    load_member_anchors,
    resolve_asset,
    save_effect_member,
)
from ember.ecp.stage0_training import stage0_source_authority
from ember.ecp.stage1_equivalence import stack_observations
from ember.ecp.stage1_parameterization import (
    project_expert_onto_rank_reserved_residual,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


def _authority_path(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    return resolve_asset(asset_root, config["authorities"][name])


def _prepare_prefix(
    policy: torch.nn.Module,
    observations: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    microbatch: int,
) -> ExecutionPolicyPrefix:
    batch = stack_observations(observations)
    embeddings, padding = [], []
    for start in range(0, len(observations), microbatch):
        stop = min(start + microbatch, len(observations))
        encoded = prepare_execution_policy_prefix(
            policy,
            {
                name: value[start:stop].to(device, non_blocking=True)
                for name, value in batch.items()
            },
        )
        embeddings.append(encoded.embeddings)
        padding.append(encoded.padding)
    return ExecutionPolicyPrefix(torch.cat(embeddings), torch.cat(padding))


def _concat_particles(values: Sequence[PolicyEffectParticles]) -> PolicyEffectParticles:
    return PolicyEffectParticles(
        owner=torch.cat([value.owner for value in values]),
        flow=torch.cat([value.flow for value in values]),
        action=torch.cat([value.action for value in values]),
    )


def _capture_pair(
    *,
    policy: torch.nn.Module,
    observer: torch.nn.Module,
    lora: BatchedLoRAInference,
    carrier: Mapping[str, torch.Tensor],
    member: Mapping[str, torch.Tensor],
    prefix: ExecutionPolicyPrefix,
    suffix_noise: torch.Tensor,
    device: torch.device,
    microbatch: int,
) -> tuple[PolicyEffectParticles, PolicyEffectParticles]:
    carrier_rows, member_rows = [], []
    for start in range(0, int(suffix_noise.shape[0]), microbatch):
        stop = min(start + microbatch, int(suffix_noise.shape[0]))
        cell = ExecutionPolicyPrefix(
            prefix.embeddings[start:stop], prefix.padding[start:stop]
        )
        noise = suffix_noise[start:stop].to(device, non_blocking=True)
        cache = prepare_policy_effect_prefix_cache(policy, cell)
        try:
            carrier_rows.append(
                capture_policy_effect_particles(
                    policy=policy,
                    observer=observer,
                    lora=lora,
                    state=carrier,
                    prefix=cell,
                    suffix_noise=noise,
                    denoising_steps=10,
                    prepared_prefix_cache=cache,
                ).to("cpu")
            )
            member_rows.append(
                capture_policy_effect_particles(
                    policy=policy,
                    observer=observer,
                    lora=lora,
                    state=member,
                    prefix=cell,
                    suffix_noise=noise,
                    denoising_steps=10,
                    prepared_prefix_cache=cache,
                ).to("cpu")
            )
        finally:
            del cache
    return _concat_particles(carrier_rows), _concat_particles(member_rows)


def _particle_order(
    owner: torch.Tensor, *, trajectory_count: int
) -> torch.Tensor:
    expected_states = 8 * trajectory_count
    if owner.shape != (expected_states, 2, 38, 4, 128):
        raise ValueError("effect particle capture changed shape")
    return (
        owner.reshape(trajectory_count, 8, 2, 38, 4, 128)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(2 * trajectory_count, 8, 38, 4, 128)
        .contiguous()
    )


def _residual_tail(
    state: Mapping[str, torch.Tensor], contract: LoRAContract, carrier_rank: int
) -> dict[str, torch.Tensor]:
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        result[a_name] = state[a_name][carrier_rank:].detach().cpu()
        result[b_name] = state[b_name][:, carrier_rank:].detach().cpu()
    return result


def capture_effect_shard(args: Any) -> Path:
    asset_root = args.asset_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if (
        config.get("schema_version") != "ember_ecp_fixed_effect_realizer_v1"
        or config.get("status") != "preregistered_before_effect_particle_capture"
    ):
        raise ValueError("fixed effect realizer config changed")
    repository = git_state(Path(__file__).resolve().parents[3])
    if args.mode == "formal" and not git_state_is_clean_pushed_or_frozen_authority(
        repository
    ):
        raise ValueError("formal effect capture requires clean pushed authority")

    rows = load_effect_member_rows(
        _authority_path(config, "task_evidence_manifest", asset_root)
    )
    assignments = balanced_member_shards(rows, int(args.shard_count))
    selected = (
        (int(args.member_index),)
        if args.member_index is not None
        else assignments[int(args.shard_index)]
    )
    if not selected or min(selected) < 0 or max(selected) >= len(rows):
        raise ValueError("effect capture selected an invalid member shard")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    stage1_config = read_json(_authority_path(config, "stage1_config", asset_root))
    source = stage0_source_authority(args)
    source_config = load_config(
        _authority_path(stage1_config, "source_base_config", asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, device)
    contract = load_pi05_lora_contract(
        _authority_path(config, "lora_contract", asset_root)
    )
    prepare_frozen_writer_policy(policy, contract)
    carrier = load_file(
        str(_authority_path(config, "stable_carrier", asset_root)),
        device=str(device),
    )
    validate_lora_state(carrier, contract)
    stage0_config = read_json(
        _authority_path(stage1_config, "stage0_config", asset_root)
    )
    native = load_frozen_native_observer(
        stage0_config=stage0_config,
        owners=build_target_owners(contract),
        native_checkpoint=_authority_path(
            config, "native_observer_checkpoint", asset_root
        ),
        device=device,
    )
    lora = BatchedLoRAInference(policy, contract)
    microbatch = int(config["effect_particles"]["capture_microbatch_size"])
    carrier_rank = int(config["output"]["carrier_rank"])
    members = []
    started = time.monotonic()
    try:
        for index in selected:
            row = rows[index]
            member_started = time.monotonic()
            anchors = load_member_anchors(row, asset_root=asset_root)
            prefix = _prepare_prefix(
                policy,
                anchors.observations,
                device=device,
                microbatch=microbatch,
            )
            checkpoint = resolve_asset(asset_root, row["checkpoint"])
            member = load_file(
                str(checkpoint / "adapter.safetensors"), device=str(device)
            )
            validate_lora_state(member, contract)
            carrier_effect, member_effect = _capture_pair(
                policy=policy,
                observer=native.encoder.observer,
                lora=lora,
                carrier=carrier,
                member=member,
                prefix=prefix,
                suffix_noise=anchors.suffix_noise,
                device=device,
                microbatch=microbatch,
            )
            owner_delta = _particle_order(
                member_effect.owner - carrier_effect.owner,
                trajectory_count=anchors.trajectory_count,
            )
            projected, metrics = project_expert_onto_rank_reserved_residual(
                carrier=carrier,
                expert=member,
                contract=contract,
                carrier_rank=carrier_rank,
            )
            residual = _residual_tail(projected, contract, carrier_rank)
            tensor_path = output_dir / f"member_{index:03d}.safetensors"
            tensor_bytes = save_effect_member(
                path=tensor_path,
                owner_delta=owner_delta,
                residual=residual,
                trajectory_count=anchors.trajectory_count,
                contract=contract,
            )
            members.append(
                {
                    "index": index,
                    "ordinal": int(row["ordinal"]),
                    "global_task_id": int(row["global_task_id"]),
                    "asset_key": str(row["asset_key"]),
                    "domain": (
                        "libero90_nonheld"
                        if str(row["asset_key"]).startswith("source90:")
                        else "target_train"
                    ),
                    "member": str(row["member"]),
                    "reliability": float(row["reliability"]),
                    "trajectory_count": anchors.trajectory_count,
                    "particle_count": 2 * anchors.trajectory_count,
                    "tensor_path": str(tensor_path),
                    "tensor_bytes": tensor_bytes,
                    "required_correction_energy": float(
                        sum(value.required_correction_energy for value in metrics)
                    ),
                    "captured_seconds": time.monotonic() - member_started,
                }
            )
            del prefix, member, carrier_effect, member_effect, projected, residual
    finally:
        lora.close()
    manifest = output_dir / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": EFFECT_PARTICLE_SHARD_SCHEMA,
            "mode": str(args.mode),
            "repository": repository,
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size},
            "source": source,
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "assigned_member_indices": list(assignments[int(args.shard_index)]),
            "selected_member_indices": list(selected),
            "members": members,
            "runtime": {
                "device": str(device),
                "capture_microbatch_size": microbatch,
                "elapsed_seconds": time.monotonic() - started,
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            },
            "information_wall": {
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "held_optimizer_steps": 0,
            },
        },
    )
    return manifest
