"""Audited carrier and 95-task/118-member authorities for G3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.g1_assets import calibrate_s_ref
from ember.ecp.native_materialization import (
    extract_rank12_carrier,
    low_rank_balanced_svd,
)
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_data import NaturalProgramTask
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.stage0_training import load_stage0_config
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json


G3_CONFIG_SCHEMA_V1 = "ember_ecp_shared_compiler_g3_v1"
G3_CONFIG_SCHEMA = "ember_ecp_shared_compiler_g3_v2"


@dataclass(frozen=True)
class SharedCompilerMember:
    name: str
    step: int
    adapter: Path
    adapter_bytes: int
    successes: int


@dataclass(frozen=True)
class SharedTaskMembers:
    task: NaturalProgramTask
    members: tuple[SharedCompilerMember, ...]


@dataclass(frozen=True)
class SharedCompilerRankAssets:
    contract: LoRAContract
    carrier_complete: Mapping[str, torch.Tensor]
    carrier_rank12: Mapping[str, torch.Tensor]
    s_ref: torch.Tensor


def authority_path(
    config: Mapping[str, Any], name: str, *, asset_root: Path
) -> Path:
    value = Path(str(config["authorities"][name]))
    return value.resolve() if value.is_absolute() else (asset_root / value).resolve()


def _checked(path: Path, expected_bytes: int | None = None) -> Path:
    value = path.resolve()
    if not value.is_file() or (
        expected_bytes is not None and value.stat().st_size != expected_bytes
    ):
        raise ValueError(f"G3 authority is missing or changed size: {value}")
    return value


def load_shared_compiler_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    schema = config.get("schema_version")
    model = config.get("model", {})
    data = config.get("data", {})
    wall = config.get("information_wall", {})
    losses = config.get("optimization", {}).get("loss_weights", {})
    optimization = config.get("optimization", {})
    profile = config.get("profile_defaults", {})
    formal = config.get("formal_run", {})
    common_invalid = (
        model.get("target_owners") != 38
        or model.get("event_slots") != 8
        or model.get("program_width") != 128
        or model.get("residual_rank") != 4
        or model.get("materialization") != "unique_rank12_plus_rank4_rank16"
        or model.get("selection")
        != "program_query_native_content_key_two_softmax_difference"
        or model.get("input_candidate_index")
        != ["video", "frame", "probe", "horizon"]
        or model.get("output_candidate_index")
        != ["video", "frame", "probe", "horizon", "type"]
        or data.get("K_values") != [1, 2, 4]
        or data.get("video_action_cross_episode") is not True
        or data.get("action_chunk_size") != 50
        or data.get("task_role_weighting") != "meta50_target50"
        or optimization.get("functional_query_count") != 4
        or optimization.get("functional_policy_microbatch_size") != 2
        or optimization.get("same_task_consistency_gradient")
        != "rotating_primary_stop_gradient_other_update"
        or wall.get("action_meta_installed") is not False
        or wall.get("held_gradient_tasks") != 0
        or wall.get("shuffled_or_reversed_use") is not False
        or profile.get("allowed_world_sizes") != [1, 2]
        or profile.get("task_pairs") != [[23, 72], [27, 73], [1, 93]]
        or formal.get("allowed_world_sizes") != [1, 2]
    )
    v1_invalid = schema == G3_CONFIG_SCHEMA_V1 and (
        config.get("status") != "historical_frozen_program_shared_compiler_v1"
        or set(losses)
        != {
            "global_member_effect",
            "family_functional",
            "cross_episode_flow",
            "effective_update",
            "carrier_preservation",
            "same_task_consistency",
        }
    )
    teacher = optimization.get("native_teacher", {})
    optimizer = optimization.get("optimizer", {})
    v2_invalid = schema == G3_CONFIG_SCHEMA and (
        config.get("status") != "active_native_teacher_shared_compiler"
        or "native_teacher_manifest" not in config.get("authorities", {})
        or teacher.get("K_values") != [1]
        or teacher.get("member_reduction")
        != "detached_set_valued_functional_responsibilities"
        or teacher.get("target_reduction") != "four_families_equal"
        or teacher.get("selection")
        != "equal_input_output_subspace_and_update_direction"
        or teacher.get("scale") != "small_core_singular_spectrum"
        or teacher.get("scale_video_shared_context_gradient") != "stopped"
        or teacher.get("K2_K4_tensor_reads") != 0
        or teacher.get("confidence_gate") is not False
        or set(losses)
        != {
            "global_member_effect",
            "family_functional",
            "cross_episode_flow",
            "native_teacher_selection",
            "native_teacher_scale",
            "carrier_preservation",
            "same_task_consistency",
        }
        or not {
            "selection_gradient_clip_norm",
            "scale_video_gradient_clip_norm",
        }
        <= set(optimizer)
        or "gradient_clip_norm" in optimizer
        or wall.get("native_teacher_training_only") is not True
        or wall.get("native_teacher_deployment_reads") != 0
        or wall.get("task_video_member_lookup_parameters") is not False
    )
    if common_invalid or schema not in {G3_CONFIG_SCHEMA_V1, G3_CONFIG_SCHEMA}:
        raise ValueError("unsupported G3 shared compiler config")
    if v1_invalid or v2_invalid:
        raise ValueError("unsupported G3 shared compiler config")
    return config


def _success_counts(results: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    panel = results.get("per_task", {})
    rows = panel.values() if isinstance(panel, Mapping) else panel
    return {
        (str(row["suite"]), int(row["task_id"])): int(row["successes"])
        for row in rows
    }


def _member_rows(
    path: Path, *, name: str, keep_success_only: bool
) -> list[dict[str, Any]]:
    results = read_json(_checked(path))
    counts = _success_counts(results)
    rows = []
    for record in results.get("adapter", {}).get("tasks", ()):
        key = (str(record["suite"]), int(record["task_id"]))
        successes = counts.get(key, -1)
        if successes < 0:
            raise ValueError("G3 expert result and adapter panels differ")
        if keep_success_only and successes == 0:
            continue
        adapter = _checked(
            Path(str(record["checkpoint"])) / "adapter.safetensors",
            int(record["adapter_bytes"]),
        )
        rows.append(
            {
                **record,
                "member_name": name,
                "successes": successes,
                "adapter": adapter,
            }
        )
    return rows


def load_shared_task_members(
    config: Mapping[str, Any],
    tasks: Sequence[NaturalProgramTask],
    *,
    asset_root: Path,
) -> tuple[SharedTaskMembers, ...]:
    rows = []
    rows.extend(
        _member_rows(
            authority_path(config, name, asset_root=asset_root),
            name="meta_step1000",
            keep_success_only=True,
        )
        for name in ("meta_train_expert_results", "meta_held_expert_results")
    )
    flat = [record for group in rows for record in group]
    flat.extend(
        _member_rows(
            authority_path(config, "target_step1000_results", asset_root=asset_root),
            name="target_step1000",
            keep_success_only=True,
        )
    )
    flat.extend(
        _member_rows(
            authority_path(config, "target_step2000_results", asset_root=asset_root),
            name="target_step2000",
            keep_success_only=True,
        )
    )
    if len(flat) != 118:
        raise ValueError(f"G3 expert member count changed: {len(flat)}")

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in flat:
        domain = "meta" if str(record["suite"]) == "libero_90" else "target"
        task_id = (
            int(record["task_id"])
            if domain == "meta"
            else int(record["global_task_id"])
        )
        by_key.setdefault((domain, task_id), []).append(record)
    output = []
    for task in tasks:
        domain = "meta" if task.domain == "libero90_nonheld" else "target"
        records = by_key.get((domain, task.domain_task_id), ())
        expected = 1 if domain == "meta" else None
        if not records or (expected is not None and len(records) != expected):
            raise ValueError(f"G3 member authority missing task {task.authority_id}")
        output.append(
            SharedTaskMembers(
                task=task,
                members=tuple(
                    SharedCompilerMember(
                        name=str(record["member_name"]),
                        step=int(record["step"]),
                        adapter=Path(record["adapter"]),
                        adapter_bytes=int(record["adapter_bytes"]),
                        successes=int(record["successes"]),
                    )
                    for record in sorted(records, key=lambda row: int(row["step"]))
                ),
            )
        )
    if len(output) != 95 or sum(len(row.members) for row in output) != 118:
        raise ValueError("G3 95-task/118-member panel changed")
    return tuple(output)


def load_shared_rank_assets(
    config: Mapping[str, Any],
    *,
    asset_root: Path,
    held_global_ids: set[int],
    device: torch.device | str,
) -> SharedCompilerRankAssets:
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", asset_root=asset_root)
    )
    carrier_path = _checked(
        authority_path(config, "stable_carrier", asset_root=asset_root)
    )
    carrier = load_file(str(carrier_path), device="cpu")
    validate_lora_state(carrier, contract)
    carrier_rank12 = extract_rank12_carrier(carrier, contract)
    s_ref, fit_count = calibrate_s_ref(
        carrier=carrier,
        contract=contract,
        projection_manifest=authority_path(
            config, "carrier_projection_manifest", asset_root=asset_root
        ),
        held_global_ids=held_global_ids,
    )
    if fit_count != 19:
        raise ValueError("G3 scale authority lost target-fit19")
    return SharedCompilerRankAssets(
        contract=contract,
        carrier_complete={
            name: value.to(device=device) for name, value in carrier.items()
        },
        carrier_rank12={
            name: value.to(device=device, dtype=torch.float32)
            for name, value in carrier_rank12.items()
        },
        s_ref=s_ref.to(device=device),
    )


def load_frozen_g2_program(
    model: NaturalProgramModel,
    checkpoint: Path,
    *,
    device: torch.device | str,
) -> None:
    """Load the qualified macro20 Program without restoring G2 optimizer state."""

    checkpoint = checkpoint.resolve()
    manifest = read_json(_checked(checkpoint / "checkpoint_manifest.json"))
    tensor_record = manifest.get("files", {}).get("ecp.safetensors", {})
    tensor_path = _checked(
        checkpoint / "ecp.safetensors", int(tensor_record.get("bytes", -1))
    )
    if (
        checkpoint_macro(checkpoint) != 20
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != "g2_natural_program"
        or int(manifest.get("next_macro", -1)) != 20
        or manifest.get("run_contract_schema")
        != "ember_ecp_natural_program_g2_run_v2"
    ):
        raise ValueError("G3 frozen Program authority changed")
    model.load_state_dict(load_file(str(tensor_path), device=str(device)), strict=True)
    model.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("G3 Program was not frozen")


def build_frozen_g2_program(
    config: Mapping[str, Any],
    *,
    asset_root: Path,
    owners: Sequence[Any],
    device: torch.device | str,
) -> NaturalProgramModel:
    """Rebuild the qualified G2 topology over the pure Native observer."""

    g2 = read_json(authority_path(config, "g2_config", asset_root=asset_root))
    model_cell = g2.get("model", {})
    if (
        g2.get("schema_version") != "ember_ecp_natural_program_g2_v1"
        or model_cell.get("canonical_alignment")
        != "boundary_anchored_forward_only_dp_v2"
        or model_cell.get("native_observer_training") != "frozen_stage0_v3"
    ):
        raise ValueError("G3 frozen Program topology changed")
    native = load_frozen_native_observer(
        stage0_config=load_stage0_config(
            authority_path(config, "stage0_config", asset_root=asset_root)
        ),
        owners=tuple(owners),
        native_checkpoint=authority_path(
            config, "native_observer_checkpoint", asset_root=asset_root
        ),
        device=device,
        max_frames_per_call=int(model_cell["max_frames_per_call"]),
    )
    model = NaturalProgramModel(
        native.encoder,
        prefix_width=int(model_cell["prefix_width"]),
        width=int(model_cell["program_width"]),
        owners=int(model_cell["target_owners"]),
        event_slots=int(model_cell["event_slots"]),
        action_phases=int(model_cell["action_phases"]),
        predicate_slots=int(model_cell["predicate_slots"]),
    ).to(device)
    load_frozen_g2_program(
        model,
        authority_path(config, "g2_program_checkpoint", asset_root=asset_root),
        device=device,
    )
    return model


def project_member_to_mobile_rank4(
    *,
    member: Mapping[str, torch.Tensor],
    carrier: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    """Best rank-four residual of one expert relative to the frozen carrier."""

    validate_lora_state(member, contract)
    validate_lora_state(carrier, contract)
    result: dict[str, torch.Tensor] = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        carrier_a = carrier[a_name][:12]
        carrier_b = carrier[b_name][:, :12]
        difference_a = torch.cat((member[a_name], carrier_a), dim=0)
        difference_b = torch.cat((member[b_name], -carrier_b), dim=1)
        projected_a, projected_b = low_rank_balanced_svd(
            difference_a, difference_b, output_rank=4
        )
        result[a_name] = projected_a
        result[b_name] = projected_b
    return result
