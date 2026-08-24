"""Frozen data, adapter, and scale authorities for the G1 held5 oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.g1_objective import initial_success_ids, low_rank_distance_squared
from ember.ecp.native_materialization import (
    extract_rank12_carrier,
    extract_rank4_residual,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import WriterTaskAuthority


G1_CONFIG_SCHEMA = "ember_ecp_native_factor_g1_v1"
G1_MEMBER_NAMES = ("latest", "independent", "earliest")
G1_REFERENCE_CANDIDATES = ("carrier", *G1_MEMBER_NAMES)


@dataclass(frozen=True)
class G1TaskAssets:
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    language: str
    video_authority: WriterTaskAuthority
    effect_manifest: Path
    reference_adapters: Mapping[str, Path]
    initial_success: Mapping[str, set[int]]
    carrier_success: set[int]


@dataclass(frozen=True)
class G1RankAssets:
    contract: LoRAContract
    carrier_complete: Mapping[str, torch.Tensor]
    carrier_rank12: Mapping[str, torch.Tensor]
    reference_rank4: Mapping[int, tuple[Mapping[str, torch.Tensor], ...]]
    s_ref: torch.Tensor
    fit_expert_count: int


def authority_path(config: Mapping[str, Any], name: str, *, asset_root: Path) -> Path:
    value = Path(str(config["authorities"][name]))
    return value.resolve() if value.is_absolute() else (asset_root / value).resolve()


def load_g1_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    output_partition = config.get("native_factor", {}).get(
        "output_value_partition", {}
    )
    initialization = config.get("optimization", {}).get("initialization", {})
    if (
        config.get("schema_version") != G1_CONFIG_SCHEMA
        or config.get("status") != "active_native_factor_capacity_oracle"
        or config.get("information_wall", {}).get("action_meta_installed") is not False
        or config.get("video", {}).get("videos_per_task") != 1
        or config.get("video", {}).get("cross_video_weight") != "identity_k1"
        or output_partition.get("q")
        != "eight_real_attention_heads_independent_signed_measures_then_concatenate"
        or any(
            output_partition.get(name) != "whole_native_vector_one_signed_measure"
            for name in ("v", "action_in", "action_out")
        )
        or initialization.get("kind")
        != "best_verified_robust_projection_or_carrier"
        or tuple(initialization.get("selection_candidates", ()))
        != G1_REFERENCE_CANDIDATES
        or initialization.get("selection_metric")
        != "paired_fixed50_success_count"
        or initialization.get("signed_measure_solve_precision") != "float64"
        or not 0
        < float(initialization.get("relative_singular_threshold", 0))
        < 1
        or not 0 <= float(initialization.get("probability_floor_mass", -1)) < 1
        or initialization.get("retain_initialization_checkpoint") is not True
    ):
        raise ValueError("G1 Native-Factor config changed its capacity contract")
    return config


def select_g1_initialization_reference(
    *,
    carrier_success: set[int],
    member_success: Mapping[str, set[int]],
) -> tuple[str, dict[str, int]]:
    """Choose the strongest verified task-local member, preferring carrier ties."""

    if set(member_success) != set(G1_MEMBER_NAMES):
        raise ValueError("G1 initialization member panel changed")
    success_sets = {"carrier": carrier_success, **dict(member_success)}
    if any(not values <= set(range(50)) for values in success_sets.values()):
        raise ValueError("G1 initialization success IDs changed")
    counts = {name: len(success_sets[name]) for name in G1_REFERENCE_CANDIDATES}
    selected = max(
        G1_REFERENCE_CANDIDATES,
        key=lambda name: (counts[name], -G1_REFERENCE_CANDIDATES.index(name)),
    )
    return selected, counts


def _adapter_file(path: Path) -> Path:
    resolved = path.resolve()
    return resolved / "adapter.safetensors" if resolved.is_dir() else resolved


def _checked_file(path: Path, expected_bytes: int | None = None) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or (
        expected_bytes is not None and resolved.stat().st_size != expected_bytes
    ):
        raise ValueError(f"G1 authority is missing or changed size: {resolved}")
    return resolved


def load_g1_task_assets(
    config: Mapping[str, Any], *, asset_root: Path, data_root: Path
) -> tuple[G1TaskAssets, ...]:
    requested = tuple(map(int, config["tasks"]["held_ordinals"]))
    global_ids = tuple(map(int, config["tasks"]["global_task_ids"]))
    if requested != (90, 91, 92, 93, 94) or len(set(global_ids)) != 5:
        raise ValueError("G1 held5 task panel changed")

    target_manifest = read_json(
        authority_path(config, "target_manifest", asset_root=asset_root)
    )
    target_rows = {int(row["global_task_id"]): row for row in target_manifest["tasks"]}
    projection_path = authority_path(config, "mobile_projection", asset_root=asset_root)
    projection = read_json(projection_path)
    records = projection.get("records", ())
    references: dict[int, dict[str, Path]] = {ordinal: {} for ordinal in requested}
    for record in records:
        task = record.get("task", {})
        ordinal = int(task.get("ordinal", -1))
        member = str(record.get("member", ""))
        if ordinal in references and member in G1_MEMBER_NAMES:
            references[ordinal][member] = _checked_file(
                Path(str(record["projected_adapter"])),
                int(record["projected_adapter_bytes"]),
            )
    if any(set(value) != set(G1_MEMBER_NAMES) for value in references.values()):
        raise ValueError("G1 mobile rank4 reference panel is incomplete")

    success_paths = {
        "latest": authority_path(config, "latest_fixed50", asset_root=asset_root),
        "independent": authority_path(
            config, "independent_fixed50", asset_root=asset_root
        ),
        "earliest": authority_path(config, "earliest_fixed50", asset_root=asset_root),
    }
    carrier_success_path = authority_path(
        config, "carrier_strict250", asset_root=asset_root
    )
    effect_root = authority_path(config, "effect_bank_root", asset_root=asset_root)
    result = []
    for ordinal, global_task_id in zip(requested, global_ids, strict=True):
        row = target_rows.get(global_task_id)
        if row is None or row.get("split_role") != "train":
            raise ValueError("G1 held task escaped the development-train authority")
        hdf5 = row["hdf5"]
        authority = WriterTaskAuthority(
            task_id=global_task_id,
            language=str(row["language"]),
            path=(data_root / str(hdf5["relative_path"])).resolve(),
            expected_bytes=int(hdf5["bytes"]),
        )
        _checked_file(authority.path, authority.expected_bytes)
        effect_manifest = _checked_file(
            effect_root / f"task_{ordinal}" / "manifest.json"
        )
        result.append(
            G1TaskAssets(
                ordinal=ordinal,
                global_task_id=global_task_id,
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                language=str(row["language"]),
                video_authority=authority,
                effect_manifest=effect_manifest,
                reference_adapters={
                    name: references[ordinal][name] for name in G1_MEMBER_NAMES
                },
                initial_success={
                    name: initial_success_ids(
                        _checked_file(success_paths[name]),
                        suite=str(row["suite"]),
                        task_id=int(row["task_id"]),
                    )
                    for name in G1_MEMBER_NAMES
                },
                carrier_success=initial_success_ids(
                    _checked_file(carrier_success_path),
                    suite=str(row["suite"]),
                    task_id=int(row["task_id"]),
                ),
            )
        )
    return tuple(result)


def effective_matrix_rms_distance(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    values = []
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        squared = low_rank_distance_squared(
            left[a_name].float(),
            left[b_name].float(),
            right[a_name].float(),
            right[b_name].float(),
        )
        values.append(
            (squared / float(target.in_features * target.out_features)).sqrt()
        )
    return torch.stack(values)


def calibrate_s_ref(
    *,
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    projection_manifest: Path,
    held_global_ids: set[int],
) -> tuple[torch.Tensor, int]:
    projection = read_json(projection_manifest.resolve())
    expert_paths = [
        _adapter_file(Path(str(row["expert_checkpoint"])))
        for row in projection.get("tasks", ())
        if int(row["global_task_id"]) not in held_global_ids
    ]
    if len(expert_paths) != 19:
        raise ValueError("G1 s_ref requires the fixed fit19 expert panel")
    rows = []
    for path in expert_paths:
        expert = load_file(str(_checked_file(path)), device="cpu")
        validate_lora_state(expert, contract)
        rows.append(effective_matrix_rms_distance(expert, carrier, contract))
    s_ref = torch.stack(rows).median(0).values
    if s_ref.shape != (38,) or not torch.isfinite(s_ref).all() or torch.any(s_ref <= 0):
        raise ValueError("G1 fit19 target scales are invalid")
    return s_ref, len(rows)


def load_g1_rank_assets(
    config: Mapping[str, Any],
    tasks: tuple[G1TaskAssets, ...],
    *,
    asset_root: Path,
    device: torch.device | str,
) -> G1RankAssets:
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", asset_root=asset_root)
    )
    carrier_path = _checked_file(
        authority_path(config, "stable_carrier", asset_root=asset_root)
    )
    carrier_cpu = load_file(str(carrier_path), device="cpu")
    validate_lora_state(carrier_cpu, contract)
    rank12_cpu = extract_rank12_carrier(carrier_cpu, contract)
    references: dict[int, tuple[Mapping[str, torch.Tensor], ...]] = {}
    for task in tasks:
        values = []
        for member in G1_MEMBER_NAMES:
            complete = load_file(str(task.reference_adapters[member]), device="cpu")
            values.append(
                extract_rank4_residual(complete, contract, carrier_state=carrier_cpu)
            )
        references[task.ordinal] = tuple(
            {
                name: value.to(device=device, dtype=torch.float32)
                for name, value in state.items()
            }
            for state in values
        )
    s_ref, fit_count = calibrate_s_ref(
        carrier=carrier_cpu,
        contract=contract,
        projection_manifest=authority_path(
            config, "carrier_projection_manifest", asset_root=asset_root
        ),
        held_global_ids={task.global_task_id for task in tasks},
    )
    return G1RankAssets(
        contract=contract,
        carrier_complete={
            name: value.to(device=device) for name, value in carrier_cpu.items()
        },
        carrier_rank12={
            name: value.to(device=device, dtype=torch.float32)
            for name, value in rank12_cpu.items()
        },
        reference_rank4=references,
        s_ref=s_ref.to(device=device),
        fit_expert_count=fit_count,
    )
