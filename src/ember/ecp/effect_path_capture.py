"""GPU capture for known-success ECP effect-path calibration."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.effect_path_calibration import (
    REPO_ROOT,
    TASK_RESULT_SCHEMA,
    build_verified_member_objective,
    build_verified_member_validity,
    carrier_drift_by_category,
    global_particle_loss,
    summarize_task_paths,
    verified_member_losses,
)
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
    prepare_policy_effect_prefix_cache,
)
from ember.ecp.stage0_training import stage0_source_authority
from ember.ecp.stage1_equivalence import Stage1EffectBank, load_effect_bank
from ember.ecp.stage1_objective import (
    build_particle_objective,
    candidate_snapshot,
    realization_config_from_mapping,
)
from ember.ecp.stage1_parameterization import (
    interpolate_rank_reserved_endpoint,
    rank_reserved_relative_distance,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


def _authority_path(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def _adapter_file(value: str | Path) -> Path:
    path = Path(value).resolve()
    return path / "adapter.safetensors" if path.is_dir() else path


def _concat_response(values: Sequence[PolicyEffectResponse]) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=torch.cat([value.owner for value in values]),
        flow=torch.cat([value.flow for value in values]),
        action=torch.cat([value.action for value in values]),
    )


def _result_rows(path: Path) -> dict[tuple[str, int, int], bool]:
    rows = read_json(path.resolve()).get("rows", ())
    indexed = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): bool(
            row["success"]
        )
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("effect-path direct result rows overlap")
    return indexed


def _initial_success(
    *,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    asset_root: Path,
) -> dict[str, dict[int, bool]]:
    global_id = int(task["global_task_id"])
    step500 = {
        int(value) for value in config["tasks"]["earliest_step500_global_task_ids"]
    }
    paths = {
        "latest": _authority_path(config, "latest_results", asset_root),
        "independent": _authority_path(config, "independent_results", asset_root),
        "earliest": _authority_path(
            config,
            (
                "earliest_step500_results"
                if global_id in step500
                else "earliest_step250_results"
            ),
            asset_root,
        ),
    }
    init_ids = [
        int(row["init_state_id"])
        for row in anchors
        if str(row["category"]) == "initial"
    ]
    key = (str(task["suite"]), int(task["task_id"]))
    result = {}
    for member, path in paths.items():
        rows = _result_rows(path)
        result[member] = {
            init_id: bool(rows[(*key, init_id)]) for init_id in init_ids
        }
    return result


def _projection_records(
    *,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    asset_root: Path,
) -> dict[str, Path]:
    analysis = read_json(_authority_path(config, "projection_analysis", asset_root))
    records = {
        str(row["member"]): row
        for row in analysis.get("records", ())
        if int(row["task"]["ordinal"]) == int(task["ordinal"])
    }
    expected = tuple(str(value) for value in config["tasks"]["members"])
    if set(records) != set(expected):
        raise ValueError("known-success projection panel changed")
    metadata = {str(row["name"]): row for row in members}
    for name in expected:
        if _adapter_file(records[name]["expert_adapter"]) != _adapter_file(
            metadata[name]["adapter"]
        ):
            raise ValueError("known-success projection member identity changed")
    return {
        name: Path(str(records[name]["projected_adapter"])).resolve()
        for name in expected
    }


def _residual_tail(
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    carrier_rank: int,
) -> dict[str, torch.Tensor]:
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        result[a_name] = state[a_name][carrier_rank:]
        result[b_name] = state[b_name][:, carrier_rank:]
    return result


def _capture_paths(
    *,
    policy: torch.nn.Module,
    observer: torch.nn.Module,
    lora: BatchedLoRAInference,
    bank: Stage1EffectBank,
    states: Mapping[tuple[str, float], Mapping[str, torch.Tensor]],
    device: torch.device,
    microbatch: int,
) -> dict[tuple[str, float], PolicyEffectResponse]:
    captured: dict[tuple[str, float], list[PolicyEffectResponse]] = {
        key: [] for key in states
    }
    for start in range(0, bank.state_count, microbatch):
        stop = min(start + microbatch, bank.state_count)
        prefix = ExecutionPolicyPrefix(
            bank.prefix.embeddings[start:stop].to(device),
            bank.prefix.padding[start:stop].to(device),
        )
        noise = bank.suffix_noise[start:stop].to(device)
        prefix_cache = prepare_policy_effect_prefix_cache(policy, prefix)
        for key, state in states.items():
            captured[key].append(
                capture_policy_effect_response(
                    policy=policy,
                    observer=observer,
                    lora=lora,
                    state=state,
                    prefix=prefix,
                    suffix_noise=noise,
                    denoising_steps=10,
                    prepared_prefix_cache=prefix_cache,
                ).to("cpu")
            )
        del prefix_cache
    return {key: _concat_response(values) for key, values in captured.items()}


def run_task(
    *,
    args: Any,
    config: Mapping[str, Any],
    asset_root: Path,
    output: Path,
    device: torch.device,
) -> Path:
    ordinals = tuple(int(value) for value in config["tasks"]["ordinals"])
    ordinal = int(args.task_ordinal)
    if ordinal not in ordinals:
        raise ValueError("effect-path task is outside the preregistered panel")
    stage1_config = read_json(_authority_path(config, "stage1_config", asset_root))
    bank_manifest = (
        _authority_path(config, "effect_bank_root", asset_root)
        / f"task_{ordinal}/manifest.json"
    )
    manifest = read_json(bank_manifest)
    task = dict(manifest["metadata"]["task"])
    if int(task["ordinal"]) != ordinal:
        raise ValueError("effect-path bank belongs to another task")
    bank = load_effect_bank(bank_manifest, "cpu")
    member_records = tuple(manifest["metadata"]["members"])
    member_names = tuple(str(value) for value in config["tasks"]["members"])
    if tuple(str(row["name"]) for row in member_records) != member_names:
        raise ValueError("effect-path member ordering changed")
    initial_success = _initial_success(
        config=config,
        task=task,
        anchors=manifest["metadata"]["anchors"],
        asset_root=asset_root,
    )
    validity = build_verified_member_validity(
        anchors=manifest["metadata"]["anchors"],
        member_names=member_names,
        initial_success=initial_success,
    )

    source = stage0_source_authority(args)
    source_config = load_config(
        _authority_path(stage1_config, "source_base_config", asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, device)
    contract = load_pi05_lora_contract(
        _authority_path(stage1_config, "lora_contract", asset_root)
    )
    prepare_frozen_writer_policy(policy, contract)
    carrier = load_file(
        str(_authority_path(stage1_config, "stable_carrier", asset_root)),
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
            stage1_config, "native_observer_checkpoint", asset_root
        ),
        device=device,
    )
    projected = _projection_records(
        config=config,
        task=task,
        members=member_records,
        asset_root=asset_root,
    )
    carrier_rank = int(config["path"]["carrier_rank"])
    alphas = tuple(float(value) for value in config["path"]["alphas"])
    states = {}
    endpoint_trust = {}
    endpoint_paths = {}
    for member in member_names:
        endpoint = load_file(str(projected[member]), device=str(device))
        validate_lora_state(endpoint, contract)
        endpoint_paths[member] = str(projected[member])
        endpoint_trust[member] = float(
            rank_reserved_relative_distance(
                _residual_tail(endpoint, contract, carrier_rank),
                carrier,
                contract,
                carrier_rank,
            )
        )
        for alpha in alphas[1:]:
            states[(member, alpha)] = interpolate_rank_reserved_endpoint(
                carrier=carrier,
                endpoint=endpoint,
                alpha=alpha,
                contract=contract,
                carrier_rank=carrier_rank,
            )

    lora = BatchedLoRAInference(policy, contract)
    started = time.monotonic()
    try:
        responses = _capture_paths(
            policy=policy,
            observer=native.encoder.observer,
            lora=lora,
            bank=bank,
            states=states,
            device=device,
            microbatch=int(config["path"]["capture_microbatch_size"]),
        )
    finally:
        lora.close()

    realization = realization_config_from_mapping(stage1_config["solver"])
    legacy = build_particle_objective(bank, realization)
    verified = build_verified_member_objective(bank, validity, realization)
    member_index = {name: index for index, name in enumerate(member_names)}
    rows = []
    for member in member_names:
        for alpha in alphas:
            response = bank.carrier if alpha == 0.0 else responses[(member, alpha)]
            trust = endpoint_trust[member] * alpha * alpha
            legacy_snapshot, _, _ = candidate_snapshot(
                response,
                bank,
                legacy,
                realization,
                torch.tensor(trust),
            )
            losses = verified_member_losses(response, bank, verified, realization)
            global_loss, responsibilities = global_particle_loss(losses, verified)
            rows.append(
                {
                    "member": member,
                    "alpha": alpha,
                    "trust_distance": trust,
                    "legacy": asdict(legacy_snapshot),
                    "verified_member_losses": {
                        name: float(losses[index])
                        for name, index in member_index.items()
                    },
                    "matching_verified_loss": float(losses[member_index[member]]),
                    "global_particle_loss": float(global_loss),
                    "global_member_responsibilities": {
                        name: float(responsibilities[index])
                        for name, index in member_index.items()
                    },
                    "carrier_drift_by_category": carrier_drift_by_category(
                        response, bank, verified, realization
                    ),
                }
            )
    payload = {
        "schema_version": TASK_RESULT_SCHEMA,
        "mode": "formal",
        "repository": git_state(REPO_ROOT),
        "task": task,
        "effect_bank_manifest": str(bank_manifest),
        "members": list(member_names),
        "member_reliability": {
            name: float(bank.member_reliability[index])
            for index, name in enumerate(member_names)
        },
        "validity_counts": {
            name: {
                "initial": int((validity[index] & (bank.category_ids == 0)).sum()),
                "successful": int((validity[index] & (bank.category_ids == 1)).sum()),
                "candidate": int((validity[index] & (bank.category_ids == 2)).sum()),
                "recovery": int((validity[index] & (bank.category_ids == 3)).sum()),
            }
            for index, name in enumerate(member_names)
        },
        "projected_endpoints": endpoint_paths,
        "endpoint_trust": endpoint_trust,
        "rows": rows,
        "summary": summarize_task_paths(rows, member_names),
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "optimizer_steps": 0,
        "new_closed_loop_rows": 0,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
        "action_meta_installed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("effect-path task output already exists")
    write_json_atomic(output, payload)
    return output
