"""Optimization, checkpointing, and CLI for the G1 Native-Factor oracle."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.g1_assets import load_g1_config
from ember.ecp.g1_objective import (
    carrier_preservation_loss,
    global_member_effect_loss,
    sensitivity_normalized_update_losses,
    verified_member_effects,
)
from ember.ecp.g1_queries import functional_batch, functional_gradient, gradient_bridge
from ember.ecp.g1_runtime import (
    G1_CHECKPOINT_SCHEMA,
    G1_RUN_SCHEMA,
    REPO_ROOT,
    G1Runtime,
    candidate_states,
    capture_effect_response,
    prepare_runtime,
)
from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics


def _save_checkpoint(runtime: G1Runtime, step: int) -> Path:
    checkpoints = runtime.args.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"step_{step:08d}"
    temporary = checkpoints / f".step_{step:08d}.tmp-{os.getpid()}"
    if final.exists() or temporary.exists():
        raise ValueError("G1 checkpoint already exists")
    temporary.mkdir()
    with torch.no_grad():
        _residual, complete = candidate_states(runtime, canonicalize=True)
    adapter = {
        name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for name, value in complete.items()
    }
    validate_lora_state(adapter, runtime.ranks.contract)
    save_file(adapter, str(temporary / "adapter.safetensors"))
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in runtime.oracle.state_dict().items()
        },
        str(temporary / "oracle.safetensors"),
    )
    torch.save(
        {
            "schema_version": G1_CHECKPOINT_SCHEMA,
            "step": step,
            "task_ordinal": runtime.task.ordinal,
            "global_task_id": runtime.task.global_task_id,
            "metrics_rows": runtime.metrics_rows,
            "optimizer": runtime.optimizer.state_dict(),
            "responsibilities": runtime.responsibilities.detach().cpu(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state(runtime.args.torch_device),
        },
        temporary / "trainer.pt",
    )
    files = {
        path.name: path.stat().st_size for path in temporary.iterdir() if path.is_file()
    }
    write_json_atomic(
        temporary / "manifest.json",
        {
            "schema_version": G1_CHECKPOINT_SCHEMA,
            "step": step,
            "task_ordinal": runtime.task.ordinal,
            "global_task_id": runtime.task.global_task_id,
            "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
            "single_complete_rank16": True,
            "state_tensor_count": len(adapter),
            "files": files,
            "content_hash_policy": "disabled_by_owner",
        },
    )
    os.replace(temporary, final)
    write_json_atomic(
        runtime.args.output_dir / "latest_checkpoint.json",
        {"path": str(final), "step": step},
    )
    return final


def _load_checkpoint(runtime: G1Runtime, checkpoint: Path) -> None:
    manifest = read_json(checkpoint / "manifest.json")
    step = int(manifest.get("step", -1))
    if (
        manifest.get("schema_version") != G1_CHECKPOINT_SCHEMA
        or int(manifest.get("task_ordinal", -1)) != runtime.task.ordinal
        or int(manifest.get("global_task_id", -1)) != runtime.task.global_task_id
        or manifest.get("single_complete_rank16") is not True
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ValueError("G1 resume checkpoint authority changed")
    for name, expected in manifest.get("files", {}).items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected):
            raise ValueError("G1 resume checkpoint file changed")
    runtime.oracle.load_state_dict(
        load_file(
            str(checkpoint / "oracle.safetensors"),
            device=str(runtime.args.torch_device),
        ),
        strict=True,
    )
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=False
    )
    if (
        trainer.get("schema_version") != G1_CHECKPOINT_SCHEMA
        or int(trainer.get("step", -1)) != step
        or int(trainer.get("task_ordinal", -1)) != runtime.task.ordinal
    ):
        raise ValueError("G1 trainer checkpoint changed")
    runtime.optimizer.load_state_dict(trainer["optimizer"])
    runtime.responsibilities = trainer["responsibilities"].to(runtime.args.torch_device)
    torch.set_rng_state(trainer["torch_cpu_rng"])
    torch.cuda.set_rng_state(trainer["torch_cuda_rng"], runtime.args.torch_device)
    runtime.start_step = step
    runtime.metrics_rows = int(trainer["metrics_rows"])


def run_step(runtime: G1Runtime, step: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    residual_state, complete_state = candidate_states(runtime, canonicalize=False)
    member_updates = sensitivity_normalized_update_losses(
        candidate_state=residual_state,
        reference_states=runtime.ranks.reference_rank4[runtime.task.ordinal],
        contract=runtime.ranks.contract,
        s_ref=runtime.ranks.s_ref,
        sensitivity_weights=runtime.sensitivity_weights,
    )
    effect_due = step % int(runtime.config["optimization"]["effect_interval"]) == 0
    global_effect = None
    member_effects = None
    carrier_loss = None
    if effect_due:
        response = capture_effect_response(runtime=runtime, state=complete_state)
        member_effects = verified_member_effects(response, runtime.effect_objective)
        global_effect, responsibilities = global_member_effect_loss(
            member_effects, runtime.effect_objective
        )
        runtime.responsibilities = responsibilities.detach()
        carrier_loss = carrier_preservation_loss(response, runtime.effect_objective)
    effective_update = (runtime.responsibilities * member_updates).sum()

    batch = functional_batch(
        dataset=runtime.query_dataset,
        processor=runtime.query_processor,
        task=runtime.task,
        config=runtime.config,
        step=step,
    )
    functional_loss, functional_gradients = functional_gradient(
        policy=runtime.policy,
        state=complete_state,
        contract=runtime.ranks.contract,
        batch=batch,
        config=runtime.config,
        task=runtime.task,
        step=step,
    )
    functional_bridge = gradient_bridge(
        functional_loss, complete_state, functional_gradients
    )
    weights = runtime.config["optimization"]["loss_weights"]
    total = (
        float(weights["effective_update"]) * effective_update
        + float(weights["independent_functional"]) * functional_bridge
    )
    if global_effect is not None and carrier_loss is not None:
        total = (
            total
            + float(weights["global_member_effect"]) * global_effect
            + float(weights["carrier_preservation"]) * carrier_loss
        )
    total.backward()
    gradients = {name: value.grad for name, value in runtime.oracle.named_parameters()}
    if any(
        value is None or not torch.isfinite(value).all() for value in gradients.values()
    ):
        raise ValueError("G1 oracle gradient is missing or non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        runtime.oracle.parameters(),
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    runtime.optimizer.step()
    if any(parameter.grad is not None for parameter in runtime.policy.parameters()):
        raise ValueError("G1 frozen source policy accumulated gradients")
    return {
        "step": step + 1,
        "task_ordinal": runtime.task.ordinal,
        "global_task_id": runtime.task.global_task_id,
        "effect_due": effect_due,
        "loss": {
            "total": float(total.detach()),
            "global_member_effect": (
                float(global_effect.detach()) if global_effect is not None else None
            ),
            "member_effects": (
                member_effects.detach().cpu().tolist()
                if member_effects is not None
                else None
            ),
            "effective_update": float(effective_update.detach()),
            "member_updates": member_updates.detach().cpu().tolist(),
            "independent_functional": float(functional_loss),
            "carrier_preservation": (
                float(carrier_loss.detach()) if carrier_loss is not None else None
            ),
        },
        "responsibilities": runtime.responsibilities.detach().cpu().tolist(),
        "gradient_norm_before_clip": float(gradient_norm),
        "gradient_nonzero": {
            name: int(torch.count_nonzero(value)) for name, value in gradients.items()
        },
        "scale": {
            "minimum": float(
                (runtime.ranks.s_ref[:, None] * runtime.oracle.scale_logits.tanh())
                .min()
                .detach()
            ),
            "maximum": float(
                (runtime.ranks.s_ref[:, None] * runtime.oracle.scale_logits.tanh())
                .max()
                .detach()
            ),
        },
        "wall_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.args.torch_device
        ),
    }


def train(args: argparse.Namespace) -> None:
    runtime: G1Runtime | None = None
    try:
        runtime = prepare_runtime(args)
        if args.resume is not None:
            _load_checkpoint(runtime, args.resume)
        elif runtime.config["optimization"]["initialization"][
            "retain_initialization_checkpoint"
        ]:
            _save_checkpoint(runtime, 0)
        runtime.metrics_rows = reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            runtime.start_step,
            runtime.metrics_rows,
            cursor_key="step",
        )
        if not runtime.start_step < args.stop_after_step:
            raise ValueError("G1 resume cursor is not before requested stop")
        torch.cuda.reset_peak_memory_stats(runtime.args.torch_device)
        checkpoint_steps = set(
            map(int, runtime.config["optimization"]["checkpoint_steps"])
        ) | {int(args.stop_after_step)}
        for step in range(runtime.start_step, args.stop_after_step):
            row = run_step(runtime, step)
            append_jsonl(args.output_dir / "metrics.jsonl", row)
            runtime.metrics_rows += 1
            if (step + 1) % args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step + 1 in checkpoint_steps:
                _save_checkpoint(runtime, step + 1)
        write_json_atomic(
            args.output_dir / "segment_completion.json",
            {
                "schema_version": G1_RUN_SCHEMA,
                "task_ordinal": runtime.task.ordinal,
                "completed_steps": args.stop_after_step,
                "initial_segment_steps": int(
                    runtime.config["optimization"]["initial_segment_steps"]
                ),
                "status": "segment_complete",
            },
        )
    finally:
        if runtime is not None:
            runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-ordinal", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "asset_root", "data_root", "output_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    config = load_g1_config(args.config)
    if args.stop_after_step is None:
        args.stop_after_step = int(config["optimization"]["initial_segment_steps"])
    if args.stop_after_step <= 0 or args.log_every <= 0:
        raise ValueError("G1 stop and log intervals must be positive")
    return args
