"""Fresh end-to-end layered Writer + reading Meta training and cost profiling."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import checkpoint_macro, load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.learning_data import JointTrainingData
from ember.writer.native import joint_functional_backward
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import FrozenVideoPrefixCache, build_joint_runtime
from ember.writer.task_execution import cost_balanced_task_assignment


RUN_SCHEMA = "ember_layered_relation_writer_joint_run_v1"
STAGE = "layered_relation_writer_fresh_joint"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if (
        config.get("schema_version") != "ember_layered_relation_writer_config_v1"
        or config["optimization"].get("fresh_joint_writer_and_meta") is not True
        or float(config["optimization"]["normalizer"]) != 1.0
        or config["data"].get("extra_meta_tasks") != []
        or int(config["data"]["frame_stride"]) != 5
        or config["data"].get("include_last_frame") is not True
        or int(config["observer"]["flow_time"]) != 1
    ):
        raise ValueError("joint Writer scientific contract changed")
    return config


def _optimization(state, config):
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(
        state.parameters(), lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    warmup, total = int(opt["warmup_steps"]), int(config["data"]["total_steps"])
    floor = float(opt["min_lr"]) / float(opt["lr"])

    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * (1 + math.cos(math.pi * progress)) / 2

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _run_contract(args, context, config, runtime, state):
    from ember.writer.layered import LayeredWriterConfig

    properties = torch.cuda.get_device_properties(context.local_rank)
    local = {
        "rank": context.rank, "local_rank": context.local_rank,
        "gpu_uuid": str(properties.uuid), "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rank_topology = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rank_topology, local)
    else:
        rank_topology[0] = local
    return {
        "schema_version": RUN_SCHEMA, "stage": STAGE, "mode": args.mode,
        "git": state, "source": runtime.source, "config": config,
        "model_config": asdict(LayeredWriterConfig(**config["model"])),
        "topology": {
            "host": socket.gethostname(), "world_size": context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "ranks": rank_topology,
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "optimizer": "fresh AdamW", "scaler": None, "resume_contract": "same config, topology, sampler and complete state",
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": "disjoint fixed roles", "gradient_normalizer": 1.0,
        },
    }


def _publish_contract(path, contract, *, resume):
    if resume:
        old = read_json(path)
        for key in ("schema_version", "stage", "mode", "config", "model_config", "topology", "source"):
            if old.get(key) != contract[key]:
                raise ValueError(f"exact-resume contract differs: {key}")
    else:
        if path.exists():
            raise ValueError("fresh run refuses an existing contract")
        write_json_atomic(path, contract)


def _grad_norm(parameters) -> float:
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def _execute_step(runtime, data, cache, context, config, step):
    tasks = data.groups[step]
    costs = data.step_costs(step)
    assignment = cost_balanced_task_assignment(
        tasks, costs, {task: tuple(range(context.world_size)) for task in tasks}, world_size=context.world_size,
    )
    rows = []
    for task in assignment[context.rank]:
        start = time.perf_counter()
        occurrence = data.occurrences[step][task]
        demos = data.video_demos(task, occurrence)
        condition = cache.condition(task, demos)
        prefix_seconds = time.perf_counter() - start
        raw_batch, query_trace = data.action_batch(task, occurrence, demos)
        batch = runtime.processor.training_batch(raw_batch)
        metric = joint_functional_backward(
            runtime.state.writer, runtime.observer, condition, policy=runtime.policy,
            contract=runtime.lora, batch=batch, task_weight=1.0 / len(tasks),
            policy_rng_seed=query_trace["policy_rng_seed"],
            policy_microbatch_size=int(config["runtime"]["policy_microbatch"]),
        )
        torch.cuda.synchronize(context.device)
        rows.append({
            "step": step + 1, "task": task, "suite": data.tasks[task].suite,
            "occurrence": occurrence, "K": len(demos), "video_demos": list(demos),
            "frames": costs[task], "queries": len(query_trace["action_demos"]),
            "prefix_seconds": prefix_seconds, "seconds": time.perf_counter() - start,
            **query_trace, **metric,
        })
    return rows


def run(args: argparse.Namespace) -> None:
    config = _config(args.config)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal joint training requires a clean pushed detached worktree")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 6:
        raise ValueError("EMBER joint training needs one node and at most six GPUs")
    torch.set_num_threads(int(args.cpu_threads))
    # Identical trainable initialization on all ranks; per-rank RNG is restored
    # separately on resume. Sample selection/policy noise is keyed by task visit.
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    start = time.perf_counter()
    data = JointTrainingData(args.asset_root, config["data"])
    runtime = build_joint_runtime(args.asset_root, config, context.device)
    runtime.state.train()
    optimizer, scheduler = _optimization(runtime.state, config)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    contract = _run_contract(args, context, config, runtime, state)
    if context.is_main:
        _publish_contract(args.output / "run_contract.json", contract, resume=args.resume is not None)
    barrier(context)
    initial, metrics_rows = 0, 0
    if args.resume:
        if args.resume.resolve().parent.parent != args.output.resolve():
            raise ValueError("exact-resume checkpoint must belong to its original run root")
        initial, metrics_rows = load_ecp_checkpoint(
            checkpoint=args.resume, stage=STAGE, context=context, model=runtime.state,
            optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA,
            expected_sampler_state=data.sampler_state(checkpoint_macro(args.resume)),
        )
        if context.is_main:
            reconcile_metrics(args.output / "exposures.jsonl", initial, metrics_rows, cursor_key="step", packet_label="exposures")
            reconcile_metrics(args.output / "metrics.jsonl", initial, initial, cursor_key="step", packet_label="metrics")
    stop = int(config["data"]["total_steps"])
    if args.stop_after_step is not None:
        stop = min(stop, args.stop_after_step)
    if not initial < stop:
        raise ValueError("joint run has no remaining registered updates")
    cache = FrozenVideoPrefixCache(runtime.observer, data, int(config["runtime"]["prefix_cache_bytes"]))
    parameters = tuple(runtime.state.parameters())
    barrier(context)
    for step in range(initial, stop):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rows = _execute_step(runtime, data, cache, context, config, step)
        sum_writer_gradients(parameters, world_size=context.world_size)
        meta_norm = _grad_norm(runtime.state.meta.parameters())
        writer_norm = _grad_norm(runtime.state.writer.parameters())
        total_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["optimization"]["grad_clip"]), error_if_nonfinite=True))
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(context.device)
        gathered = [None] * context.world_size
        if context.world_size > 1:
            dist.all_gather_object(gathered, rows)
        else:
            gathered[0] = rows
        metrics_rows += sum(len(rank_rows) for rank_rows in gathered)
        if context.is_main:
            for rank_rows in gathered:
                for row in rank_rows:
                    append_jsonl(args.output / "exposures.jsonl", row)
            metric = {
                "step": step + 1, "seconds": time.perf_counter() - tick,
                "mean_flow_loss": sum(r["flow_loss"] for group in gathered for r in group) / len(data.groups[step]),
                "writer_grad_norm": writer_norm, "meta_grad_norm": meta_norm, "total_grad_norm": total_norm,
                "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30,
            }
            append_jsonl(args.output / "metrics.jsonl", metric)
            print(json.dumps(metric), flush=True)
        if step + 1 in config["evidence"]["checkpoint_steps"] or step + 1 == stop:
            save_ecp_checkpoint(
                output_dir=args.output, macro=step + 1, stage=STAGE, context=context,
                model=runtime.state, optimizer=optimizer, scheduler=scheduler,
                run_contract_schema=RUN_SCHEMA, metrics_rows=metrics_rows,
                sampler_state=data.sampler_state(step + 1),
            )
    barrier(context)
    if context.is_main:
        write_json_atomic(args.output / "completion.json", {
            "schema_version": RUN_SCHEMA, "status": "complete", "mode": args.mode,
            "completed_steps": stop, "exposures": metrics_rows, "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered closed-loop evidence",
        })
    data.close()
    if context.world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pi05_layered_writer_v1.json")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
