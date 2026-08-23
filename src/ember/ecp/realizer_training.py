"""Train the held-free fixed ECP effect realizer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.realizer_code import EFFECT_CODE_AUTHORITY_SCHEMA
from ember.ecp.realizer_model import (
    FixedEffectRealizer,
    fixed_effect_realizer_loss,
)
from ember.ecp.realizer_training_data import PackedEffectCodeDataset
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REALIZER_CHECKPOINT_SCHEMA = "ember_ecp_fixed_effect_realizer_checkpoint_v1"


def _save_checkpoint(
    *,
    root: Path,
    step: int,
    model: FixedEffectRealizer,
    optimizer: torch.optim.Optimizer,
    config_path: Path,
    code_manifest: Path,
) -> Path:
    path = root / "checkpoints" / f"step_{step:08d}" / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": REALIZER_CHECKPOINT_SCHEMA,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": str(config_path),
            "effect_code_authority": str(code_manifest),
            "world_size": 1,
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def run_fixed_effect_realizer(args: Any) -> Path:
    config_path = args.config.resolve()
    code_manifest = args.effect_code_manifest.resolve()
    config = read_json(config_path)
    code = read_json(code_manifest)
    repository = git_state(Path(__file__).resolve().parents[3])
    if (
        config.get("schema_version") != "ember_ecp_fixed_effect_realizer_v1"
        or code.get("schema_version") != EFFECT_CODE_AUTHORITY_SCHEMA
        or code.get("status") != "complete_fit_only_effect_code_coordinate"
        or (
            args.mode == "formal"
            and not git_state_is_clean_pushed_or_frozen_authority(repository)
        )
    ):
        raise ValueError("fixed effect realizer training authority changed")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(config["training"]["seed"]))
    torch.set_float32_matmul_precision("high")
    contract = load_pi05_lora_contract(args.lora_contract.resolve())
    transform = load_file(str(Path(code["coordinate"]["transform_path"]).resolve()))
    a_scales = transform["target_a_scales"].float().to(device)
    b_scales = transform["target_b_scales"].float().to(device)
    dataset = PackedEffectCodeDataset(
        manifest_path=code_manifest,
        contract=contract,
        device=device,
        include_held=False,
    )
    cell = config["model"]
    model = FixedEffectRealizer(
        contract=contract,
        owners=build_target_owners(contract),
        a_scales=a_scales,
        b_scales=b_scales,
        token_width=int(cell["token_width"]),
        state_width=int(cell["owner_state_width"]),
        bottleneck=int(cell["output_bottleneck_width"]),
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(training["total_steps"]) if args.mode == "formal" else 3
    checkpoints = (
        set(int(value) for value in training["checkpoint_steps"])
        if args.mode == "formal"
        else set()
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output_dir / "run_contract.json",
        {
            "schema_version": "ember_ecp_fixed_effect_realizer_run_v1",
            "mode": str(args.mode),
            "repository": repository,
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size},
            "effect_code_authority": str(code_manifest),
            "fold": int(code["fold"]),
            "fit_tasks": len(dataset.task_groups),
            "fit_members": len(dataset.fit_indices),
            "held_members_loaded_for_training": 0,
            "task_batching": "all fit tasks once per optimizer step; one alternating member per task",
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "total_steps": total_steps,
            "checkpoint_steps": sorted(checkpoints),
            "device": str(device),
            "information_wall": {
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "held_optimizer_steps": 0,
                "task_id_model_input": False,
            },
        },
    )
    metrics_path = output_dir / "metrics.jsonl"
    null_code = torch.zeros(1, 4, 8, 38, 128, device=device)
    null_mask = torch.tensor([[True, False, False, False]], device=device)
    null_reliability = torch.zeros(1, device=device)
    started = time.monotonic()
    final = None
    for step in range(1, total_steps + 1):
        batch = dataset.training_batch(step - 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            prediction = model(batch.code, batch.particle_mask, batch.reliability)
            null_prediction = model(null_code, null_mask, null_reliability)
            loss = fixed_effect_realizer_loss(
                prediction=prediction,
                target=batch.targets,
                null_prediction=null_prediction,
                a_scales=a_scales,
                b_scales=b_scales,
                null_weight=float(training["null_code_weight"]),
            )
        if not torch.isfinite(loss.total):
            raise FloatingPointError("fixed effect realizer loss became non-finite")
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        final = {
            "step": step,
            "total": float(loss.total.detach()),
            "factor": float(loss.factor.detach()),
            "effective": float(loss.effective.detach()),
            "null": float(loss.null.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "elapsed_seconds": time.monotonic() - started,
        }
        if step == 1 or step % 10 == 0 or step == total_steps:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(final, sort_keys=True) + "\n")
        if step in checkpoints:
            _save_checkpoint(
                root=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                config_path=config_path,
                code_manifest=code_manifest,
            )
    if final is None:
        raise RuntimeError("fixed effect realizer did not run")
    write_json_atomic(
        output_dir / "completion.json",
        {
            "status": "complete",
            "final": final,
            "checkpoints": sorted(checkpoints),
            "elapsed_seconds": time.monotonic() - started,
            "max_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
    )
    return output_dir / "completion.json"
