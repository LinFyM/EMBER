"""Fit-only kernel mapping from frozen language features to rank4 updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from safetensors import safe_open

from ember.ecp.native_materialization import low_rank_balanced_svd
from ember.ecp.shared_compiler_effects import (
    G3_EFFECT_BANK_SCHEMA,
    G3_EFFECT_ROOT_SCHEMA,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_source_checkpoint import read_json


def language_feature(
    *,
    policy: torch.nn.Module,
    program: torch.nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    embedded = program.encoder.embed_language_conditions(policy, tokens)
    p_lang = program.language_reader(embedded, mask)[0].float()
    rows = functional.normalize(p_lang, dim=-1)
    return functional.normalize(rows.flatten(), dim=0)


def load_language_effect_records(
    root_manifest: Path, expected_ids: set[int]
) -> dict[int, dict[str, Any]]:
    root = read_json(root_manifest)
    records = {
        int(row.get("authority_id", -1)): dict(row)
        for row in root.get("records", ())
    }
    if (
        root.get("schema_version") != G3_EFFECT_ROOT_SCHEMA
        or root.get("status") != "complete"
        or int(root.get("task_count", -1)) != 75
        or int(root.get("member_count", -1)) != 93
        or root.get("roles") != {"meta_fit": 56, "target_fit": 19}
        or set(records) != expected_ids
    ):
        raise ValueError("language baseline effect authority changed")
    return records


def load_task_rank4_target(
    *,
    record: Mapping[str, Any],
    contract: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    manifest_path = Path(str(record["manifest"])).resolve()
    manifest = read_json(manifest_path)
    tensor_cell = manifest.get("tensor_file", {})
    tensor_path = Path(str(tensor_cell.get("path", ""))).resolve()
    members = tuple(manifest.get("metadata", {}).get("members", ()))
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != int(record["manifest_bytes"])
        or manifest.get("schema_version") != G3_EFFECT_BANK_SCHEMA
        or manifest.get("status") != "complete"
        or len(members) != int(record["member_count"])
        or len(members) not in (1, 2)
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != int(tensor_cell.get("bytes", -1))
        or manifest.get("metadata", {}).get("held_gradient_use") is not False
    ):
        raise ValueError("language baseline task effect authority changed")
    output: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        reliability = handle.get_tensor("reliability").float()
        if reliability.shape != (len(members),) or not torch.allclose(
            reliability.sum(), torch.ones(())
        ):
            raise ValueError("language baseline member reliability changed")
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            a = torch.cat(
                [
                    handle.get_tensor(f"projection.{index}.{a_name}").float()
                    for index in range(len(members))
                ]
            ).to(device)
            b = torch.cat(
                [
                    handle.get_tensor(f"projection.{index}.{b_name}").float()
                    * reliability[index]
                    for index in range(len(members))
                ],
                dim=1,
            ).to(device)
            a4, b4 = low_rank_balanced_svd(a, b, output_rank=4)
            output[a_name] = a4.cpu()
            output[b_name] = b4.cpu()
    validate_lora_state(output, contract)
    return output


def kernel_ridge_weights(
    fit: torch.Tensor, held: torch.Tensor, *, relative_ridge: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = fit.double().mean(0)
    fit_centered = functional.normalize(fit.double() - mean, dim=-1)
    held_centered = functional.normalize(held.double() - mean, dim=-1)
    kernel = fit_centered @ fit_centered.T
    scale = kernel.diagonal().mean().clamp_min(1e-8)
    system = kernel + relative_ridge * scale * torch.eye(
        kernel.shape[0], dtype=kernel.dtype
    )
    coefficients = torch.linalg.solve(system, fit_centered @ held_centered.T).T
    weights = coefficients + (1.0 - coefficients.sum(-1, keepdim=True)) / len(fit)
    return weights.float(), mean.float(), fit_centered.float()


def mix_rank4_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: torch.Tensor,
    *,
    contract: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    weights = weights.to(device=device, dtype=torch.float32)
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = torch.cat([state[a_name] for state in states]).to(device)
        b = torch.cat(
            [
                state[b_name].to(device) * weights[index]
                for index, state in enumerate(states)
            ],
            dim=1,
        )
        a4, b4 = low_rank_balanced_svd(a, b, output_rank=4)
        output[a_name] = a4
        output[b_name] = b4
    validate_lora_state(output, contract)
    return output
