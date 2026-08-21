"""Independent shared Action Meta-LoRA calibration for frozen ECP Stage 0A."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import checkpoint_macro, load_ecp_checkpoint
from ember.ecp.contracts import build_target_owners
from ember.ecp.stage0 import ECPStage0Model
from ember.ecp.stage0_data import ECPStage0Task
from ember.ecp.stage0_training import (
    ECPStage0Runtime,
    build_stage0_optimizer,
    build_stage0_model,
    build_stage0_scheduler,
    build_stage0_tasks_and_schedule,
    build_stage0_training_stores,
    load_stage0_config,
    publish_stage0_contract,
    run_stage0_training,
    stage0_authority_path,
    stage0_source_authority,
    tokenize_stage0_languages,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_config,
    load_policy,
    seed_everything,
)
from ember.writer.meta_lora import MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_ecp_stage0_action_meta_run_v3"
STAGE = "stage0_action_meta"


def _resolve_runtime(
    args: argparse.Namespace,
    config: dict[str, Any],
    world_size: int,
) -> tuple[int, int, tuple[int, ...], int]:
    meta = config["action_meta_lora"]
    cell = meta["formal_run" if args.mode == "formal" else "profile_defaults"]
    if world_size not in cell["allowed_world_sizes"]:
        raise ValueError("ECP Action Meta-LoRA world size is outside its contract")
    total = int(cell["total_macros"])
    stop = int(args.stop_after_macro or cell.get("stop_after_macro", total))
    checkpoints = tuple(map(int, cell["checkpoint_macros"]))
    tasks_per_rank = 90 // world_size if args.mode == "formal" else 1
    if args.mode == "formal":
        if stop not in set(map(int, cell["stage_stop_macros"])):
            raise ValueError("Action Meta-LoRA stop is not pre-registered")
        if not git_state_is_clean_pushed_or_frozen_authority(git_state(REPO_ROOT)):
            raise ValueError("formal Action Meta-LoRA requires a clean pushed commit")
    return total, stop, checkpoints, tasks_per_rank


def _load_native_observer(
    model: ECPStage0Model,
    checkpoint: Path,
    required_macro: int,
    device: torch.device,
) -> dict[str, Any]:
    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    weights = checkpoint / "ecp.safetensors"
    if (
        checkpoint_macro(checkpoint) != required_macro
        or manifest.get("stage") != "stage0_native"
        or int(manifest.get("next_macro", -1)) != required_macro
        or not weights.is_file()
        or weights.stat().st_size != int(manifest["files"][weights.name]["bytes"])
    ):
        raise ValueError("native ECP observer checkpoint authority changed")
    model.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    model.requires_grad_(False).eval()
    return {
        "path": str(checkpoint),
        "macro": required_macro,
        "weights_bytes": weights.stat().st_size,
    }


def _build_action_meta(
    policy: torch.nn.Module, rank: int, device: torch.device
) -> MetaLoRAStack:
    bridge = policy.model.paligemma_with_expert
    expert = bridge.gemma_expert.model
    adapter = MetaLoRAStack(expert.layers, rank).to(device)
    # Adapter gradients traverse all 18 frozen expert layers. The bridge already
    # owns the canonical per-layer checkpoint path; activate it without putting
    # the frozen source policy into train mode.
    expert.gradient_checkpointing = True
    bridge.training = True
    return adapter


def _build_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    context: DistributedContext,
    tasks: tuple[ECPStage0Task, ...],
    source: dict[str, Any],
    native: dict[str, Any],
    adapter: MetaLoRAStack,
    total_macros: int,
    checkpoint_macros: tuple[int, ...],
    tasks_per_rank: int,
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    state = git_state(REPO_ROOT)
    meta = config["action_meta_lora"]
    return {
        "schema_version": RUN_SCHEMA,
        "stage": STAGE,
        "mode": args.mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config_path": str(args.config),
        "source": source,
        "native_observer": native,
        "data_root": str(args.data_root),
        "tokenizer": {
            "path": str(args.tokenizer_path),
            "bytes": args.tokenizer_path.stat().st_size,
        },
        "information_wall": dict(config["information_wall"]),
        "task_roles": dict(config["task_roles"]),
        "tasks": [
            {
                "authority_id": task.authority_id,
                "domain": task.domain,
                "domain_task_id": task.domain_task_id,
                "language": task.language,
                "path": str(task.path),
                "bytes": task.expected_bytes,
            }
            for task in tasks
        ],
        "action_meta_lora": {
            "rank": int(meta["rank"]),
            "targets": list(meta["targets"]),
            "parameters": sum(value.numel() for value in adapter.parameters()),
            "parameter_tensors": sum(1 for _ in adapter.parameters()),
            "scope": "observer_only_never_rollout_second_adapter",
        },
        "model": dict(config["model"]),
        "data": dict(config["data"]),
        "objective": dict(config["objective"]),
        "optimization": dict(meta["optimization"]),
        "runtime": {
            "world_size": context.world_size,
            "topology": topology,
            "total_macros": total_macros,
            "checkpoint_macros": list(checkpoint_macros),
            "tasks_per_rank_per_macro": tasks_per_rank,
            "task_assignment": "dynamic_cost_balanced_long_first",
        },
    }


def prepare_meta_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> ECPStage0Runtime:
    config = load_stage0_config(args.config)
    resolved = _resolve_runtime(args, config, context.world_size)
    total, stop, checkpoints, tasks_per_rank = resolved
    meta = config["action_meta_lora"]
    runtime_config = {**config, "optimization": meta["optimization"]}
    seed_everything(int(meta["optimization"]["seed"]), context)
    tasks, schedule = build_stage0_tasks_and_schedule(config, args.data_root)
    source = stage0_source_authority(args)
    source_config = load_config(stage0_authority_path(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    owners = build_target_owners(
        load_pi05_lora_contract(stage0_authority_path(config, "lora_contract"))
    )
    model = build_stage0_model(
        config, owners, max_frames_per_call=int(meta["max_frames_per_call"])
    ).to(context.device)
    native = _load_native_observer(
        model,
        args.native_checkpoint,
        int(meta["native_checkpoint_macro"]),
        context.device,
    )
    adapter = _build_action_meta(policy, int(meta["rank"]), context.device)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in adapter.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer = build_stage0_optimizer(
        adapter.parameters(), meta["optimization"]
    )
    scheduler = build_stage0_scheduler(optimizer, runtime_config, total)
    video_store, action_store = build_stage0_training_stores(
        tasks, config=config, source_config=source_config
    )
    language = tokenize_stage0_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    contract = _build_contract(
        args=args,
        config=config,
        context=context,
        tasks=tasks,
        source=source,
        native=native,
        adapter=adapter,
        total_macros=total,
        checkpoint_macros=checkpoints,
        tasks_per_rank=tasks_per_rank,
    )
    publish_stage0_contract(args, context, contract)
    start_macro = 0
    expected_metrics = 0
    if args.resume is not None:
        start_macro, expected_metrics = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=STAGE,
            context=context,
            model=adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=RUN_SCHEMA,
        )
    if not 0 <= start_macro < stop:
        raise ValueError("Action Meta-LoRA resume cursor is outside this segment")
    metrics_rows = (
        reconcile_metrics(
            args.output_dir / "metrics.jsonl",
            start_macro,
            expected_metrics,
            cursor_key="macro",
        )
        if context.is_main
        else 0
    )
    adapter.train()
    torch.cuda.reset_peak_memory_stats(context.device)
    return ECPStage0Runtime(
        args=args,
        config=runtime_config,
        context=context,
        tasks=tasks,
        task_by_id={task.authority_id: task for task in tasks},
        schedule=schedule,
        video_store=video_store,
        action_store=action_store,
        language_tokens=language,
        policy=policy,
        model=model,
        action_meta_lora=adapter,
        trainable_parameters=tuple(adapter.parameters()),
        frozen_parameters=(*tuple(policy.parameters()), *tuple(model.parameters())),
        checkpoint_module=adapter,
        checkpoint_stage=STAGE,
        run_schema=RUN_SCHEMA,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks_per_rank=tasks_per_rank,
        total_macros=total,
        stop_after_macro=stop,
        checkpoint_macros=checkpoints,
        start_macro=start_macro,
        metrics_rows=metrics_rows,
        run_contract=contract,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_stage0_native_v3.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "native_checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("Action Meta-LoRA log interval must be positive")
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    run_stage0_training(args, prepare_meta_runtime)


if __name__ == "__main__":
    main()
