"""Materialize complete LoRAs from the fixed phase-aligned decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.functional_adaptation.decoder import FunctionalAdapterDecoder
from ember.functional_adaptation.phase_decoder_codes import (
    PhaseDecoderCodeAuthority,
)
from ember.functional_adaptation.phase_decoder_panels import PhaseMemberSource
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_source_checkpoint import write_json_atomic


PROJECTION_SCHEMA = "ember_phase_aligned_functional_decoder_train24_projection_v1"


def phase_decoder_asset(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing phase decoder asset: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size}


def save_phase_decoder(
    decoder: FunctionalAdapterDecoder, output_dir: Path
) -> Path:
    path = output_dir / "decoder.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in decoder.state_dict().items()
        },
        str(path),
    )
    return path


def save_stable_shared_prior(
    decoder: FunctionalAdapterDecoder,
    contract: LoRAContract,
    output_dir: Path,
) -> Path:
    """Materialize the one task-independent adapter learned by the prior stage."""

    path = output_dir / "shared_prior.safetensors"
    with torch.no_grad():
        code = torch.zeros(
            decoder.code_width,
            device=next(decoder.parameters()).device,
        )
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in decoder(code).items()
        }
    validate_lora_state(state, contract)
    save_file(state, str(path))
    return path


def _adapter_rows(
    *,
    family: str,
    base_bank: Mapping[str, Any],
    codes: PhaseDecoderCodeAuthority,
    decoder: FunctionalAdapterDecoder,
    contract: LoRAContract,
    output_dir: Path,
    functional_rows: Sequence[Mapping[str, Any]],
    shared_adapter_path: Path | None,
    decoder_mode: str,
) -> list[dict[str, Any]]:
    by_ordinal = {int(row["ordinal"]): dict(row) for row in base_bank["tasks"]}
    losses = {
        int(row["member_index"]): float(row["mean_loss"])
        for row in functional_rows
    }
    fit_groups = codes.member_groups(codes.fit_ordinals)
    fit_codes = {
        ordinal: codes.fit_task_codes[index]
        for index, ordinal in enumerate(codes.fit_ordinals)
    }
    held_indices = {
        member.ordinal: index
        for index, member in enumerate(codes.members)
        if member.fold_role == "held_transform_only" and member.member == family
    }
    all_groups = {
        ordinal: tuple(
            index
            for index, member in enumerate(codes.members)
            if member.ordinal == ordinal
        )
        for ordinal in range(24)
    }
    rows = []
    with torch.no_grad():
        for ordinal in range(24):
            base = by_ordinal[ordinal]
            if family == "shared":
                code = torch.zeros(
                    decoder.code_width,
                    device=next(decoder.parameters()).device,
                )
                role = "fit19_stable_shared_prior_zero_code"
                indices = all_groups[ordinal]
                if shared_adapter_path is None:
                    raise ValueError("stable shared prior adapter is missing")
                path = shared_adapter_path
            elif ordinal in fit_codes:
                code = fit_codes[ordinal]
                role = (
                    "fit_task_stable_prior_residual_consensus"
                    if decoder_mode == "shared_prior_residual"
                    else "fit_task_successful_member_consensus"
                )
                indices = fit_groups[codes.fit_ordinals.index(ordinal)]
                path = output_dir / "projected_adapters" / "fit" / f"task_{ordinal:02d}.safetensors"
            else:
                member_index = held_indices[ordinal]
                code = codes.member_codes[member_index]
                role = (
                    f"held_transform_only_stable_prior_residual_{family}_member"
                    if decoder_mode == "shared_prior_residual"
                    else f"held_transform_only_{family}_member"
                )
                indices = (member_index,)
                path = output_dir / "projected_adapters" / f"held_{family}" / f"task_{ordinal:02d}.safetensors"
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                candidate = decoder(code)
                reference = load_file(
                    str(Path(base["checkpoint"]) / "adapter.safetensors"),
                    device="cpu",
                )
                stored = {
                    name: value.detach().cpu().to(reference[name].dtype).contiguous()
                    for name, value in candidate.items()
                }
                validate_lora_state(stored, contract)
                save_file(stored, str(path))
            rows.append(
                {
                    "suite": base["suite"],
                    "task_id": int(base["task_id"]),
                    "ordinal": ordinal,
                    "global_task_id": int(base["global_task_id"]),
                    "expert_checkpoint": str(Path(base["checkpoint"]).resolve()),
                    "projected_adapter": str(path.resolve()),
                    "projected_adapter_bytes": path.stat().st_size,
                    "code_role": role,
                    "functional_target_member_indices": list(indices),
                    "functional_flow_eval_relative_loss": sum(losses[index] for index in indices) / len(indices),
                }
            )
    return rows


def _manifest(
    *,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    repository: Mapping[str, Any],
    config_path: Path,
    decoder_path: Path,
    code_root: Path,
    result_path: Path,
    expert_bank_root: Path,
    expert_step: int,
    decoder_mode: str,
    shared_prior_adapter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECTION_SCHEMA,
        "projection_kind": {
            "stable_shared_prior": "stable_shared_prior_baseline",
            "shared_prior_residual": "stable_shared_prior_task_residual_decoder",
            "phase_decoder": "phase_aligned_success_equivalence_decoder",
            "state_aggregation": "phase_aligned_success_equivalence_decoder",
        }[decoder_mode],
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "functional_config": phase_decoder_asset(config_path),
        "decoder_checkpoint": phase_decoder_asset(decoder_path),
        "code_artifact": phase_decoder_asset(code_root / "phase_codes.safetensors"),
        "training_result": phase_decoder_asset(result_path),
        "shared_prior_adapter": shared_prior_adapter,
        "expert_bank_root": str(expert_bank_root.resolve()),
        "expert_step": expert_step,
        "optimization": {
            "fit_task_count": 19,
            "held_task_count": 5,
            "decoder_frozen": True,
            "held_code_gradient_steps": 0,
            "code_member": family,
            "final_lora_averaging": False,
            "single_complete_lora": True,
            "second_adapter_deployed": False,
            "rank_partition": (
                {
                    "shared": [0, 12],
                    "task_residual": [12, 16],
                    "merge": "exact_effective_delta_sum",
                }
                if decoder_mode
                in {"stable_shared_prior", "shared_prior_residual"}
                else None
            ),
        },
        "information_wall": {
            "role": "development_train_oracle_only",
            "deployment_carrier": False,
            "validation_experts": 0,
            "test_experts": 0,
        },
        "tasks": list(rows),
        "content_hash_policy": "disabled_by_owner",
    }


def materialize_phase_decoder_projections(
    *,
    config_path: Path,
    task_expert_config_path: Path,
    codes: PhaseDecoderCodeAuthority,
    member_sources: Sequence[PhaseMemberSource],
    decoder: FunctionalAdapterDecoder,
    contract: LoRAContract,
    repository: Mapping[str, Any],
    source: Mapping[str, Any],
    expert_bank_root: Path,
    output_dir: Path,
    functional_rows: Sequence[Mapping[str, Any]],
    decoder_path: Path,
    decoder_mode: str = "phase_decoder",
    shared_adapter_path: Path | None = None,
    shared_prior_authority: Mapping[str, Any] | None = None,
) -> None:
    """Publish earliest/latest full banks without averaging any final LoRA."""

    task_keys = tuple(
        (item.member.suite, item.member.task_id)
        for item in member_sources
        if item.member.member in {"earliest", "only"}
    )
    if len(task_keys) != 24:
        raise ValueError("phase decoder did not recover the complete train24 family")
    base_bank = inspect_task_expert_bank(
        config_path=task_expert_config_path,
        bank_root=expert_bank_root,
        step=2000,
        source=source,
        task_keys=task_keys,
        evaluation_role="development_train",
        require_formal=True,
    )
    result_path = output_dir / "result.json"
    manifests = {}
    families = ("shared",) if decoder_mode == "stable_shared_prior" else ("earliest", "latest")
    for family in families:
        rows = _adapter_rows(
            family=family,
            base_bank=base_bank,
            codes=codes,
            decoder=decoder,
            contract=contract,
            output_dir=output_dir,
            functional_rows=functional_rows,
            shared_adapter_path=shared_adapter_path,
            decoder_mode=decoder_mode,
        )
        path = output_dir / f"projection_{family}.json"
        write_json_atomic(
            path,
            _manifest(
                family=family,
                rows=rows,
                repository=repository,
                config_path=config_path,
                decoder_path=decoder_path,
                code_root=codes.root,
                result_path=result_path,
                expert_bank_root=expert_bank_root,
                expert_step=int(base_bank["step"]),
                decoder_mode=decoder_mode,
                shared_prior_adapter=(
                    phase_decoder_asset(shared_adapter_path)
                    if shared_adapter_path is not None
                    else dict(shared_prior_authority["adapter"])
                    if shared_prior_authority is not None
                    else None
                ),
            ),
        )
        manifests[family] = phase_decoder_asset(path)
    write_json_atomic(
        output_dir / "completion.json",
        {
            "schema_version": "ember_phase_aligned_functional_decoder_completion_v1",
            "result": phase_decoder_asset(result_path),
            "decoder": phase_decoder_asset(decoder_path),
            "projections": manifests,
        },
    )
