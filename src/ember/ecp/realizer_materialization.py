"""Materialize held fixed-effect codes as one complete carrier12+residual4 LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.realizer_code import EFFECT_CODE_AUTHORITY_SCHEMA
from ember.ecp.realizer_model import FixedEffectRealizer
from ember.ecp.realizer_training import REALIZER_CHECKPOINT_SCHEMA
from ember.ecp.realizer_training_data import load_held_effect_code_batch
from ember.expert_manifold.projection import ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


MATERIALIZATION_SCHEMA = "ember_ecp_fixed_effect_realizer_materialization_v1"
ALLOWED_STEPS = (800, 1000)


def _asset(asset_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def merge_carrier_residual(
    *,
    carrier: Mapping[str, torch.Tensor],
    residual: Sequence[tuple[torch.Tensor, torch.Tensor]],
    contract: LoRAContract,
    carrier_rank: int = 12,
) -> dict[str, torch.Tensor]:
    residual_rank = int(contract.rank) - int(carrier_rank)
    if len(residual) != len(contract.targets) or residual_rank != 4:
        raise ValueError("fixed effect materialization rank contract changed")
    validate_lora_state(carrier, contract)
    result: dict[str, torch.Tensor] = {}
    for target, (predicted_a, predicted_b) in zip(
        contract.targets, residual, strict=True
    ):
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        if (
            predicted_a.shape != (residual_rank, target.in_features)
            or predicted_b.shape != (target.out_features, residual_rank)
            or torch.count_nonzero(carrier[b_name][:, carrier_rank:]) != 0
        ):
            raise ValueError("fixed effect carrier/residual topology changed")
        result[a_name] = torch.cat(
            (
                carrier[a_name][:carrier_rank],
                predicted_a.to(carrier[a_name]),
            ),
            dim=0,
        ).contiguous()
        result[b_name] = torch.cat(
            (
                carrier[b_name][:, :carrier_rank],
                predicted_b.to(carrier[b_name]),
            ),
            dim=1,
        ).contiguous()
    validate_lora_state(result, contract)
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("fixed effect materialization produced non-finite LoRA")
    return result


def _projection_manifest(
    *,
    repository: Mapping[str, Any],
    purpose: str,
    base_projection_path: Path,
    base_projection: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_rows = {
        int(row["global_task_id"]): dict(row)
        for row in base_projection.get("tasks", ())
    }
    projection_tasks = []
    for row in tasks:
        base = base_rows.get(int(row["global_task_id"]))
        if base is None:
            raise ValueError("fixed effect base projection is missing a held task")
        projection_tasks.append(
            {
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "ordinal": int(base["ordinal"]),
                "global_task_id": int(row["global_task_id"]),
                "expert_checkpoint": str(base["expert_checkpoint"]),
                "projected_adapter": str(row["adapter_path"]),
                "projected_adapter_bytes": int(row["adapter_bytes"]),
            }
        )
    return {
        "schema_version": ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA,
        "projection_kind": "ecp_stage1_privileged_static_lora",
        "purpose": purpose,
        "task_panel": "held5",
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "base_projection_manifest": {
            "path": str(base_projection_path),
            "bytes": base_projection_path.stat().st_size,
        },
        "optimization": {
            "held_shared_gradient_steps": 0,
            "single_complete_lora": True,
            "final_lora_averaging": False,
            "rank": 16,
            "second_adapter_deployed": False,
            "parameterization": "one complete rank16 static LoRA",
        },
        "information_wall": {
            "role": "development_train_leave_task_out_oracle_only",
            "deployment_carrier": False,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "second_adapter_deployed": False,
        },
        "tasks": projection_tasks,
    }


def materialize_fixed_effect_realizer(args: Any) -> Path:
    repository = git_state(Path(__file__).resolve().parents[3])
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal fixed effect materialization requires clean authority")
    config_path = args.config.resolve()
    code_manifest_path = args.effect_code_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    base_projection_path = args.base_projection_manifest.resolve()
    config = read_json(config_path)
    code_manifest = read_json(code_manifest_path)
    training_root = checkpoint_path.parents[2]
    training_contract = read_json(training_root / "run_contract.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = int(checkpoint.get("step", -1))
    if (
        config.get("schema_version") != "ember_ecp_fixed_effect_realizer_v1"
        or code_manifest.get("schema_version") != EFFECT_CODE_AUTHORITY_SCHEMA
        or checkpoint.get("schema_version") != REALIZER_CHECKPOINT_SCHEMA
        or step not in ALLOWED_STEPS
        or Path(str(checkpoint.get("effect_code_authority", ""))).resolve()
        != code_manifest_path
        or training_contract.get("held_members_loaded_for_training") != 0
        or training_contract.get("fit_tasks") != 90
    ):
        raise ValueError("fixed effect checkpoint authority changed")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    contract_path = _asset(
        args.asset_root.resolve(), config["authorities"]["lora_contract"]
    )
    contract = load_pi05_lora_contract(contract_path)
    transform = load_file(str(Path(code_manifest["coordinate"]["transform_path"])))
    model = FixedEffectRealizer(
        contract=contract,
        owners=build_target_owners(contract),
        a_scales=transform["target_a_scales"],
        b_scales=transform["target_b_scales"],
        token_width=int(config["model"]["token_width"]),
        state_width=int(config["model"]["owner_state_width"]),
        bottleneck=int(config["model"]["output_bottleneck_width"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    held = load_held_effect_code_batch(
        manifest_path=code_manifest_path, device=device, member="latest"
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        prediction = model(held.code, held.particle_mask, held.reliability)

    carrier_path = _asset(
        args.asset_root.resolve(), config["authorities"]["stable_carrier"]
    )
    carrier = load_file(str(carrier_path), device="cpu")
    base_projection = read_json(base_projection_path)
    base_keys = {
        int(row["global_task_id"]): (str(row["suite"]), int(row["task_id"]))
        for row in base_projection.get("tasks", ())
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapters = output_dir / "adapters"
    adapters.mkdir()
    task_rows = []
    for batch_index, code_row in enumerate(held.rows):
        global_id = int(code_row["global_task_id"])
        suite, task_id = base_keys[global_id]
        residual = tuple(
            (a[batch_index].cpu(), b[batch_index].cpu()) for a, b in prediction
        )
        state = merge_carrier_residual(
            carrier=carrier, residual=residual, contract=contract
        )
        adapter_path = adapters / f"task_global_{global_id:02d}.safetensors"
        save_file(state, str(adapter_path))
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "global_task_id": global_id,
                "code_member": str(code_row["member"]),
                "code_reliability": float(code_row["reliability"]),
                "adapter_path": str(adapter_path),
                "adapter_bytes": adapter_path.stat().st_size,
            }
        )
    purpose = f"stage1b_fixed_effect_realizer_step{step}"
    projection = _projection_manifest(
        repository=repository,
        purpose=purpose,
        base_projection_path=base_projection_path,
        base_projection=base_projection,
        tasks=task_rows,
    )
    projection_path = output_dir / "projection_manifest.json"
    write_json_atomic(projection_path, projection)
    manifest = output_dir / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": MATERIALIZATION_SCHEMA,
            "status": "complete",
            "repository": repository,
            "step": step,
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "effect_code_authority": str(code_manifest_path),
            "carrier": {
                "path": str(carrier_path),
                "bytes": carrier_path.stat().st_size,
            },
            "projection_manifest": {
                "path": str(projection_path),
                "bytes": projection_path.stat().st_size,
            },
            "tasks": task_rows,
            "information_wall": {
                "held_target_lora_reads": 0,
                "held_optimizer_steps": 0,
                "task_id_model_input": False,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "single_complete_lora": True,
                "second_adapter_deployed": False,
            },
        },
    )
    return manifest
