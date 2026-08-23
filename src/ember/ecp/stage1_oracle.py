"""Run and publish one occupancy-complete ECP Stage 1B realization oracle."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
)
from ember.ecp.stage0_training import stage0_source_authority
from ember.ecp.stage1_equivalence import load_effect_bank
from ember.ecp.stage1_realization import (
    RealizationConfig,
    solve_rank_reserved_particle_effects,
)
from ember.lora import validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


RESULT_SCHEMA = "ember_ecp_stage1b_mobile_rank4_oracle_task_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _authority_path(config: Mapping[str, Any], name: str, asset_root: Path) -> Path:
    path = Path(str(config["authorities"][name]))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def _solver_config(value: Mapping[str, Any]) -> RealizationConfig:
    return RealizationConfig(
        steps=int(value["steps"]),
        step_rms=float(value["step_rms"]),
        step_decay_power=float(value["step_decay_power"]),
        temperature=float(value["temperature"]),
        owner_weight=float(value["owner_weight"]),
        flow_weight=float(value["flow_weight"]),
        action_weight=float(value["action_weight"]),
        carrier_barrier_weight=float(value["carrier_barrier_weight"]),
        preservation_weight=float(value["preservation_weight"]),
        signal_floor_fraction=float(value["signal_floor_fraction"]),
        minimum_confidence=float(value["minimum_confidence"]),
        trust_region=float(value["trust_region"]),
        trust_weight=float(value["trust_weight"]),
        microbatch_size=int(value["microbatch_size"]),
    )


def solve_stage1_task(
    *,
    args: Any,
    config: Mapping[str, Any],
    effect_bank_manifest: Path,
    asset_root: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    held = tuple(int(value) for value in config["roles"]["held_task_ordinals"])
    profile = int(config["roles"]["profile_fit_task_ordinal"])
    ordinal = int(args.task_ordinal)
    if ordinal != profile and ordinal not in held:
        raise ValueError("ECP Stage 1B formal solver is outside profile/held roles")
    bank_manifest = read_json(effect_bank_manifest.resolve())
    task = bank_manifest.get("metadata", {}).get("task", {})
    if int(task.get("ordinal", -1)) != int(args.task_ordinal):
        raise ValueError("ECP Stage 1B effect bank belongs to another task")
    bank = load_effect_bank(effect_bank_manifest, device)

    source = stage0_source_authority(args)
    source_config = load_config(
        _authority_path(config, "source_base_config", asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, device)
    contract = load_pi05_lora_contract(
        _authority_path(config, "lora_contract", asset_root)
    )
    carrier_rank = int(config["solver"]["carrier_rank"])
    residual_rank = int(config["solver"]["residual_rank"])
    if carrier_rank + residual_rank != int(contract.rank):
        raise ValueError("ECP Stage 1 rank reservation changed")
    prepare_frozen_writer_policy(policy, contract)
    carrier = load_file(
        str(_authority_path(config, "stable_carrier", asset_root)),
        device=str(device),
    )
    validate_lora_state(carrier, contract)
    stage0_config = read_json(_authority_path(config, "stage0_config", asset_root))
    native = load_frozen_native_observer(
        stage0_config=stage0_config,
        owners=build_target_owners(contract),
        native_checkpoint=_authority_path(
            config, "native_observer_checkpoint", asset_root
        ),
        device=device,
    )
    lora = BatchedLoRAInference(policy, contract)

    def response(
        state: Mapping[str, torch.Tensor], indices: torch.Tensor
    ) -> PolicyEffectResponse:
        return capture_policy_effect_response(
            policy=policy,
            observer=native.encoder.observer,
            lora=lora,
            state=state,
            prefix=ExecutionPolicyPrefix(
                bank.prefix.embeddings.index_select(0, indices),
                bank.prefix.padding.index_select(0, indices),
            ),
            suffix_noise=bank.suffix_noise.index_select(0, indices),
            denoising_steps=10,
        )

    started = time.monotonic()
    try:
        candidate, history, final = solve_rank_reserved_particle_effects(
            carrier=carrier,
            bank=bank,
            contract=contract,
            response=response,
            config=_solver_config(config["solver"]),
            carrier_rank=carrier_rank,
        )
    finally:
        lora.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / (
        f"task_{int(task['ordinal']):03d}_global_{int(task['global_task_id']):02d}.safetensors"
    )
    result_path = output_dir / f"task_{int(task['ordinal']):03d}.json"
    if adapter_path.exists() or result_path.exists():
        raise ValueError("ECP Stage 1B task output already exists")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in candidate.items()},
        str(adapter_path),
    )
    initial = history[0].snapshot.total
    payload = {
        "schema_version": RESULT_SCHEMA,
        "mode": "formal",
        "scientific_role": (
            "fit_task_numerical_resource_profile"
            if ordinal == profile
            else "held5_mobile_rank4_privileged_realization_oracle"
        ),
        "repository": git_state(REPO_ROOT),
        "task": task,
        "effect_bank": {
            "path": str(effect_bank_manifest.resolve()),
            "bytes": effect_bank_manifest.stat().st_size,
        },
        "solver": dict(config["solver"]),
        "history": [
            {
                "step": row.step,
                "snapshot": asdict(row.snapshot),
                "gradient_rms": row.gradient_rms,
                "a_gradient_rms": row.a_gradient_rms,
                "b_gradient_rms": row.b_gradient_rms,
                "applied_step_rms": row.applied_step_rms,
            }
            for row in history
        ],
        "final": asdict(final),
        "initial_to_final_total_ratio": final.total / max(initial, 1e-12),
        "adapter": {
            "path": str(adapter_path.resolve()),
            "bytes": adapter_path.stat().st_size,
        },
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "action_meta_installed": False,
        "parameterization": {
            "carrier_rank": carrier_rank,
            "residual_rank": residual_rank,
            "zero_residual_is_exact_carrier": True,
            "effective_update_additive": True,
            "single_complete_rank16": True,
        },
        "held_shared_gradient_steps": 0,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }
    write_json_atomic(result_path, payload)
    return result_path
