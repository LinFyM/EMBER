"""Fresh end-to-end supervised FM learning of the complete Horizon Writer."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.horizon import HorizonWriterConfig
from ember.writer.learning_data import WriterTrainingData
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import FrozenVideoPrefixCache, build_runtime
from ember.writer.task_execution import cost_balanced_task_assignment


RUN_SCHEMA = "ember_horizon_causal_learning_run_v1"
STAGE = "horizon_causal_learning_exploratory"
TRAINING_SCHEMA = "ember_horizon_supervised_training_state_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_model = asdict(HorizonWriterConfig(backend_conditioning=config["model"].get("backend_conditioning", "")))
    expected_data = {"extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
                     "queries_per_task": 64, "tasks_per_update": 4, "cardinalities": [1]}
    # Chunk sizes are execution choices; the complete scientific graph is fixed.
    actual = {**config["model"], **{key: expected_model[key] for key in ("edge_chunk", "activation_checkpoint")}}
    if (
        config.get("schema_version") != "ember_horizon_relation_writer_config_v1"
        or actual != expected_model
        or config["optimization"].get("joint_train_all_writer_modules") is not True
        or float(config["optimization"]["normalizer"]) != 1.0
        or {key: config["data"].get(key) for key in expected_data} != expected_data
        or int(config["observer"]["flow_time"]) != 1
        or int(config["observer"]["meta_rank"]) != 4
        or int(config["observer"]["probe_seed"]) != 1729
        or config["optimization"]["loss"] != "supervised_fm"
        or config.get("update_version") != "supervised_fm_writer_meta_v1"
        or "rl" in config or "trust_scales" in config["optimization"]
        or config.get("execution_precision") != "native_mixed_without_outer_autocast"
    ):
        raise ValueError("horizon supervised Writer scientific contract changed")
    for key, expected in (("video_demos", range(16)), ("action_demos", range(16, 42)),
                          ("diagnostic_action_demos", range(42, 46)), ("held_video_demos", range(46, 50))):
        if config["data"][key] != list(expected):
            raise ValueError(f"registered episode roles changed: {key}")
    if len(config["data"]["task_ids"]) != 24:
        raise ValueError("first-run gradients require all fixed train24 tasks")
    if any(int(value) <= 0 for value in config["runtime"].values()):
        raise ValueError("runtime batches and cache budget must be positive")
    _validate_checkpoint_nodes(config["evidence"]["checkpoint_updates"])
    HorizonWriterConfig(**config["model"])
    return config


def _optimization(state, config):
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(
        state.parameters(), lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    warmup = int(opt["warmup_updates"])
    if warmup < 1:
        raise ValueError("supervised-update warmup must be positive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda updates: min(1., (updates + 1) / warmup))
    return optimizer, scheduler


def _execution_config(args, config, context):
    """Physical query chunks do not change the complete logical FM batch."""
    supplied = getattr(args, "policy_microbatches", None)
    batches = ([int(config["runtime"]["policy_microbatch"])] * context.world_size
               if supplied is None else list(map(int, supplied.split(","))))
    if len(batches) != context.world_size or any(value <= 0 for value in batches):
        raise ValueError("physical microbatches need one positive value per rank")
    local = {**config, "runtime": {**config["runtime"], "policy_microbatch": batches[context.rank]}}
    return local, batches


def _gather(value, context):
    if context.world_size == 1:
        return [value]
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _run_contract(args, context, config, runtime, state):
    properties = torch.cuda.get_device_properties(context.local_rank)
    local = {"rank": context.rank, "local_rank": context.local_rank, "gpu_uuid": str(properties.uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    return {
        "schema_version": RUN_SCHEMA, "stage": STAGE, "mode": args.mode, "command": sys.argv,
        "git": state, "source": runtime.source, "config": config,
        "execution": {"policy_microbatches": _execution_config(args, config, context)[1]},
        "model_config": asdict(HorizonWriterConfig(**config["model"])),
        "topology": {
            "host": socket.gethostname(), "world_size": context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"), "ranks": _gather(local, context),
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "optimizer": "fresh AdamW; one FM update per four equally weighted tasks", "scaler": None,
            "resume_contract": "same config, topology, sampler streams, optimizer updates and complete state",
            "logical_batch": {"suite_count": 4, "conditions": 4, "queries_per_condition": 64,
                              "queries_per_update": 256, "task_weight": 0.25, "gradient_reduction": "SUM"},
            "update_version": config["update_version"], "data_version": config["data"]["version"],
            "checkpoint_updates": list(_checkpoint_nodes(args, config)),
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": "disjoint fixed roles", "gradient_normalizer": 1.0,
            "objective": "supervised_fm", "rl_rollouts": False, "rl_loss": False, "trust_rollback": False,
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


def _execute_step(engine, data, context, config, draws, step):
    if not 1 <= context.world_size <= 4:
        raise ValueError("four-condition task parallelism currently supports one to four useful ranks")
    tasks = tuple(int(draw["task"]) for draw in draws)
    if len(tasks) != 4 or len({data.tasks[task].suite for task in tasks}) != 4:
        raise ValueError("each supervised update must contain one task from each suite")
    by_task = {int(draw["task"]): draw for draw in draws}
    costs = {task: int(draw["frames"]) for task, draw in by_task.items()}
    assignment = cost_balanced_task_assignment(
        tasks, costs, {task: tuple(range(context.world_size)) for task in tasks}, world_size=context.world_size,
    )
    rows = []
    for task in assignment[context.rank]:
        draw = by_task[task]
        tick = time.perf_counter()
        if len(draw["video_demos"]) != 1:
            raise ValueError("the current supervised condition must contain exactly one teaching video")
        metric = engine.backward(draw)
        if int(metric["queries"]) != int(config["data"]["queries_per_task"]):
            raise RuntimeError("supervised engine did not execute the registered FM exposure")
        rows.append({**metric, "step": step, "task": task, "suite": data.tasks[task].suite,
                     "occurrence": draw["occurrence"], "K": len(draw["video_demos"]),
                     "video_demos": list(draw["video_demos"]), "frames": costs[task],
                     "query_seed": draw["query_seed"],
                     "queries": int(config["data"]["queries_per_task"]), "seconds": time.perf_counter() - tick})
    return rows


def _update(engine, runtime, data, context, config, optimizer, scheduler, step):
    draws = data.next_iteration()
    optimizer.zero_grad(set_to_none=True)
    rows, error = [], None
    try:
        rows = _execute_step(engine, data, context, config, draws, step)
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, context) if value]
    if failures:
        raise RuntimeError(f"supervised backward failed on a rank: {failures}")
    parameters = tuple(runtime.state.parameters())
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    tick = time.perf_counter()
    sum_writer_gradients(parameters, world_size=context.world_size)
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    sync_seconds = time.perf_counter() - tick
    norms = {"writer_grad_norm": _grad_norm(runtime.state.writer.parameters()),
             "meta_grad_norm": _grad_norm(runtime.state.meta.parameters())}
    norms["total_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
        parameters, float(config["optimization"]["grad_clip"]), error_if_nonfinite=True))
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    tick = time.perf_counter()
    optimizer.step()
    scheduler.step()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
        norms.update(gradient_sync_seconds=sync_seconds, optimizer_seconds=time.perf_counter() - tick)
    return rows, norms


def _training_state(config, updates):
    return {"schema_version": TRAINING_SCHEMA, "updates": updates,
            "update_version": config["update_version"], "data_version": config["data"]["version"]}


def _restore(args, context, runtime, data, optimizer, scheduler, config):
    if not args.resume:
        return 0, 0
    if args.resume.resolve().parent.parent != args.output.resolve():
        raise ValueError("exact-resume checkpoint must belong to its original run root")
    restored = {}
    updates, metrics_rows = load_ecp_checkpoint(
        checkpoint=args.resume, stage=STAGE, context=context, model=runtime.state,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA, restored_state=restored,
    )
    if restored["training_state"] != _training_state(config, updates):
        raise ValueError("supervised checkpoint stage/update/data contract changed")
    data.restore_sampler(restored["sampler_state"])
    if data.sampler_state()["next_step"] != updates:
        raise ValueError("sampler and optimizer-update cursors differ")
    if context.is_main:
        reconcile_metrics(args.output / "exposures.jsonl", updates, metrics_rows, cursor_key="step", packet_label="exposures")
        reconcile_metrics(args.output / "metrics.jsonl", updates, updates, cursor_key="step", packet_label="metrics")
        if args.mode == "exploratory":
            nodes = config["evidence"]["supervised_validation"]["optimizer_updates"]
            count = sum(node <= updates for node in nodes) * len(data.tasks)
            if count or (args.output / "diagnostics.jsonl").exists():
                reconcile_metrics(args.output / "diagnostics.jsonl", updates, count, cursor_key="step", packet_label="diagnostics")
    return updates, metrics_rows


def _record_iteration(args, context, config, rows, norms, updates, metrics_rows, seconds, scheduler):
    packet = {"rank": context.rank, "rows": rows,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30}
    packets = _gather(packet, context)
    gathered = [row for packet in packets for row in packet["rows"]]
    metrics_rows += len(gathered)
    if context.is_main:
        for row in gathered:
            append_jsonl(args.output / "exposures.jsonl", row)
        metric = {
            "step": updates, "optimizer_updates": updates,
            "seconds": seconds, "mean_flow_loss": sum(r["flow_loss"] for r in gathered) / 4,
            **norms, "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
            "supervised_queries": metrics_rows * int(config["data"]["queries_per_task"]),
            "rank_memory": [{key: value for key, value in packet.items() if key != "rows"} for packet in packets],
            "peak_allocated_gib": max(packet["peak_allocated_gib"] for packet in packets),
            "peak_reserved_gib": max(packet["peak_reserved_gib"] for packet in packets),
        }
        append_jsonl(args.output / "metrics.jsonl", metric)
        print(json.dumps(metric), flush=True)
    return metrics_rows


def _validate_checkpoint_nodes(nodes):
    if not nodes or list(nodes) != sorted(set(nodes)) or any(node <= 0 or node % 50 for node in nodes):
        raise ValueError("checkpoint updates must be registered increasing multiples of 50")


def _checkpoint_nodes(args, config):
    supplied = getattr(args, "checkpoint_updates", None)
    nodes = tuple(config["evidence"]["checkpoint_updates"]) if supplied is None else tuple(map(int, supplied.split(",")))
    _validate_checkpoint_nodes(nodes)
    return nodes


def _segment_limit(args, config):
    nodes = _checkpoint_nodes(args, config)
    stop = nodes[-1] if args.stop_after_step is None else args.stop_after_step
    if stop <= 0 or (args.mode == "exploratory" and (len(nodes) != 2 or stop != nodes[-1])):
        raise ValueError("exploratory segment needs two registered checkpoint nodes and must stop at the last")
    return stop


def _validate_actions(args, engine, data, context, config, step):
    spec = config["evidence"]["supervised_validation"]
    tasks = tuple(spec["task_ids"])
    demos = {task: spec["teacher_video_pool"][task % len(spec["teacher_video_pool"])] for task in tasks}
    costs = {task: data.videos.frame_counts(task, demos[task])[1] for task in tasks}
    assignment = cost_balanced_task_assignment(
        tasks, costs, {task: tuple(range(context.world_size)) for task in tasks}, world_size=context.world_size,
    )
    rows, error = [], None
    try:
        for task in assignment[context.rank]:
            rows.append({"step": step, **engine.validate(task, demos[task], seed=spec["seed"] + task,
                                                       queries=spec["queries_per_task"])})
    except Exception:
        error = traceback.format_exc()
    packets = _gather((rows, error), context)
    if any(failure for _, failure in packets):
        raise RuntimeError(f"held-action validation failed: {[e for _, e in packets if e]}")
    if context.is_main:
        gathered = [row for rows, _ in packets for row in rows]
        for row in gathered:
            append_jsonl(args.output / "diagnostics.jsonl", row)
        print(json.dumps({"diagnostic_step": step, "tasks": len(gathered),
                          "held_action_fm": sum(row["flow_loss"] for row in gathered) / len(gathered)}), flush=True)


def _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop, start):
    updates, metrics_rows = cursors
    nodes = _checkpoint_nodes(args, config)
    if args.mode == "exploratory" and any(node <= updates for node in nodes):
        raise ValueError("segment checkpoint nodes must follow the restored update cursor")
    if context.is_main:
        print(json.dumps({"segment_start": updates, "segment_stop": stop, "checkpoint_updates": nodes,
                          "resume": str(args.resume) if getattr(args, "resume", None) else None}), flush=True)
    if args.mode == "exploratory" and updates == 0 and 0 in config["evidence"]["supervised_validation"]["optimizer_updates"]:
        _validate_actions(args, engine, data, context, config, 0)
    while updates < stop:
        tick = time.perf_counter()
        rows, norms = _update(engine, runtime, data, context, config, optimizer, scheduler, updates + 1)
        updates += 1
        torch.cuda.synchronize(context.device)
        metrics_rows = _record_iteration(args, context, config, rows, norms, updates,
                                         metrics_rows, time.perf_counter() - tick, scheduler)
        if updates == stop or updates in nodes:
            if args.mode == "exploratory" and updates in config["evidence"]["supervised_validation"]["optimizer_updates"]:
                _validate_actions(args, engine, data, context, config, updates)
            save_ecp_checkpoint(
                output_dir=args.output, macro=updates, stage=STAGE, context=context,
                model=runtime.state, optimizer=optimizer, scheduler=scheduler,
                run_contract_schema=RUN_SCHEMA, metrics_rows=metrics_rows,
                sampler_state=data.sampler_state(), training_state=_training_state(config, updates),
            )
    barrier(context)
    if context.is_main:
        write_json_atomic(args.output / "completion.json", {
            "schema_version": RUN_SCHEMA, "status": "segment_complete", "mode": args.mode,
            "optimizer_updates": updates, "exposures": metrics_rows, "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered held-action and paired closed-loop evidence",
        })


def run(args: argparse.Namespace) -> None:
    from ember.writer.supervised import SupervisedEngine

    config = _config(args.config)
    state = git_state(REPO_ROOT)
    if args.mode == "exploratory" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("exploratory learning requires a clean pushed detached worktree")
    stop = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 4:
        raise ValueError("four-condition task parallelism currently supports one node and one to four useful GPUs")
    execution_config, microbatches = _execution_config(args, config, context)
    if context.is_main:
        print(json.dumps({"physical_policy_microbatches": microbatches, "logical_queries_per_update": 256}), flush=True)
    torch.set_num_threads(int(args.cpu_threads))
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    start = time.perf_counter()
    data = WriterTrainingData(args.asset_root, config["data"])
    runtime = build_runtime(args.asset_root, config, context.device)
    runtime.state.train()
    optimizer, scheduler = _optimization(runtime.state, config)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    contract = _run_contract(args, context, config, runtime, state)
    if context.is_main:
        _publish_contract(args.output / "run_contract.json", contract, resume=args.resume is not None)
    barrier(context)
    cursors = _restore(args, context, runtime, data, optimizer, scheduler, config)
    updates, _ = cursors
    if updates >= stop:
        raise ValueError("supervised segment has no remaining registered updates")
    cache = FrozenVideoPrefixCache(runtime.observer, data, int(config["runtime"]["prefix_cache_bytes"]))
    engine = SupervisedEngine(runtime, data, cache, context, execution_config)
    barrier(context)
    try:
        _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop, start)
    finally:
        data.close()
        if context.world_size > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pi05_horizon_writer_v1.json")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "exploratory"), required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-updates", help="this segment's two global update nodes, e.g. 300,400")
    parser.add_argument("--policy-microbatches", help="physical FM query chunks by rank, e.g. 8,4,8,8")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
