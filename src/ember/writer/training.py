"""Fresh video-conditioned functional learning with explicit gradient ownership."""
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
from ember.writer.data import teacher_camera_names
from ember.writer.video import VideoWriterConfig
from ember.writer.learning_data import WriterTrainingData
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import FrozenVideoPrefixCache, build_runtime
from ember.writer.task_execution import cost_balanced_task_assignment


RUN_SCHEMA = "ember_local_action_writer_run_v1"
STAGE = "local_action_grounded_writer_fresh"
TRAINING_SCHEMA = "ember_local_action_training_state_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    teacher_camera_names(config["observer"].get("camera_view", "agentview"))
    expected_model = asdict(VideoWriterConfig())
    selected_model = VideoWriterConfig(**config["model"])
    for key in ("process_mode",):
        expected_model[key] = getattr(selected_model, key)
    expected_data = {"extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
                     "queries_per_task": 64, "tasks_per_update": 4, "cardinalities": [1]}
    expected_observer = {"flow_time": 1, "meta_rank": 4, "vl_meta_rank": 4, "probe_seed": 1729}
    # Chunk sizes are execution choices; the complete scientific graph is fixed.
    actual = {**config["model"], **{key: expected_model[key] for key in ("edge_chunk", "activation_checkpoint")}}
    if (
        config.get("schema_version") != "ember_local_action_writer_config_v1"
        or actual != expected_model
        or config["optimization"].get("joint_train_all_writer_modules") is not True
        or float(config["optimization"]["normalizer"]) != 1.0
        or {key: config["data"].get(key) for key in expected_data} != expected_data
        or type(config["data"].get("conditions_per_task")) is not int
        or config["data"].get("conditions_per_task") not in (1, 2)
        or {key: config["observer"].get(key) for key in expected_observer} != expected_observer
        or config["optimization"]["loss"] != "main_fm_plus_local_action_fm"
        or config.get("update_version") != "local_action_grounded_credit_v1"
        or "rl" in config or "trust_scales" in config["optimization"]
        or config.get("execution_precision") != "native_mixed_without_outer_autocast"
    ):
        raise ValueError("video functional Writer scientific contract changed")
    for key, expected in (("video_demos", range(16)), ("action_demos", range(16, 42)),
                          ("diagnostic_action_demos", range(42, 46)), ("held_video_demos", range(46, 50))):
        if config["data"][key] != list(expected):
            raise ValueError(f"registered episode roles changed: {key}")
    if len(config["data"]["task_ids"]) != 24:
        raise ValueError("first-run gradients require all fixed train24 tasks")
    if any(int(value) <= 0 for value in config["runtime"].values()):
        raise ValueError("runtime batches and cache budget must be positive")
    local = config["local_action"]
    if (type(local.get("enabled")) is not bool or
            {key: value for key, value in local.items() if key != "enabled"} !=
            {"weight": 1.0, "frames": 4, "frame_stride": 5, "action_steps": 15, "noise_draws": 8,
             "diagnostic_clips_per_task": 16, "diagnostic_seed": 20260913}):
        raise ValueError("registered local action objective changed")
    if config["evidence"]["local_action_validation"] != {
            "optimizer_updates": config["evidence"]["supervised_validation"]["optimizer_updates"],
            "action_demos": list(range(42, 46)), "clips_per_task": 16, "noise_draws": 8,
            "seed": 20260913, "gradients": False, "checkpoint_selection": False}:
        raise ValueError("local action diagnostic registration changed")
    _validate_checkpoint_nodes(config["evidence"]["checkpoint_updates"])
    VideoWriterConfig(**config["model"])
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
        "model_config": asdict(VideoWriterConfig(**config["model"])),
        "topology": {
            "host": socket.gethostname(), "world_size": context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"), "ranks": _gather(local, context),
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "vl_meta_parameters": sum(p.numel() for p in runtime.state.vl_meta.parameters()),
            "reader_parameters": sum(p.numel() for p in runtime.state.reader.parameters()) if runtime.state.reader else 0,
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "optimizer": "fresh AdamW; one grouped functional update per four equally weighted tasks", "scaler": None,
            "resume_contract": "same config, topology, sampler streams, optimizer updates and complete state",
            "logical_batch": _logical_batch(config),
            "update_version": config["update_version"], "data_version": config["data"]["version"],
            "checkpoint_updates": list(_checkpoint_nodes(args, config)),
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": "main LoRA cross-episode; local RGB/action pairing only within action pool", "gradient_normalizer": 1.0,
            "objective": "main_fm_plus_local_action_fm", "rl_rollouts": False, "rl_loss": False, "trust_rollback": False,
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


def _logical_batch(config):
    conditions = config["data"]["conditions_per_task"]
    return {"suite_count": 4, "tasks": 4, "conditions_per_task": conditions,
            "conditions": 4 * conditions, "K": 1, "queries_per_task": 64,
            "queries_per_condition": 64 // conditions, "queries_per_update": 256,
            "task_weight": 0.25, "condition_weight": 1.0 / (4 * conditions),
            "local_clips_per_update": 4 if config["local_action"]["enabled"] else 0,
            "local_noise_draws_per_update": 32 if config["local_action"]["enabled"] else 0,
            "gradient_reduction": "SUM"}


def _condition_jobs(data, config, draws):
    logical = _logical_batch(config)
    by_job = {draw["job_id"]: draw for draw in draws}
    tasks = {draw["task"] for draw in draws}
    if (len(draws) != logical["conditions"] or len(by_job) != len(draws)
            or len(tasks) != 4 or len({data.tasks[task].suite for task in tasks}) != 4):
        raise ValueError("each supervised update requires distinct condition jobs from four suite tasks")
    for task in tasks:
        group = [draw for draw in draws if draw["task"] == task]
        indices = {draw["condition_index"] for draw in group}
        if (indices != set(range(logical["conditions_per_task"]))
                or len({(draw["occurrence"], draw["query_seed"]) for draw in group}) != 1
                or any(len(draw["video_demos"]) != 1 for draw in group)
                or len({draw["video_demos"][0] for draw in group}) != len(group)):
            raise ValueError("task conditions require distinct K1 videos and one shared occurrence/query seed")
        for draw in group:
            if (draw["query_count"] != logical["queries_per_condition"]
                    or draw["query_offset"] != draw["condition_index"] * draw["query_count"]):
                raise ValueError("condition query slices must partition the full task batch")
    return by_job


def _execute_step(engine, data, context, config, draws, step):
    logical = _logical_batch(config)
    if not 1 <= context.world_size <= min(6, logical["conditions"]):
        raise ValueError("condition parallelism requires 1 to min(6, condition count) useful ranks")
    by_job = _condition_jobs(data, config, draws)
    engine.step = step
    jobs = tuple(by_job)
    costs = {job: int(draw["frames"]) + (4 if config["local_action"]["enabled"] and draw["condition_index"] == 0 else 0)
             for job, draw in by_job.items()}
    assignment = cost_balanced_task_assignment(
        jobs, costs, {job: tuple(range(context.world_size)) for job in jobs}, world_size=context.world_size,
    )
    rows = []
    for job in assignment[context.rank]:
        draw = by_job[job]
        task = draw["task"]
        tick = time.perf_counter()
        metric = engine.backward(draw)
        if int(metric["queries"]) != draw["query_count"]:
            raise RuntimeError("supervised engine did not execute the registered FM exposure")
        rows.append({**metric, "step": step, "job_id": job, "task": task,
                     "suite": data.tasks[task].suite, "condition_index": draw["condition_index"],
                     "occurrence": draw["occurrence"], "K": 1,
                     "condition_weight": logical["condition_weight"], "task_weight": logical["task_weight"],
                     "video_demos": list(draw["video_demos"]), "frames": draw["frames"], "scheduling_frames": costs[job],
                     "query_seed": draw["query_seed"], "query_offset": draw["query_offset"],
                     "queries": draw["query_count"], "seconds": time.perf_counter() - tick})
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
             "meta_grad_norm": _grad_norm(runtime.state.meta.parameters()),
             "vl_meta_grad_norm": _grad_norm(runtime.state.vl_meta.parameters())}
    norms["reader_grad_norm"] = _grad_norm(runtime.state.reader.parameters()) if runtime.state.reader else 0.0
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
        if args.mode == "formal":
            nodes = config["evidence"]["supervised_validation"]["optimizer_updates"]
            count = sum(node <= updates for node in nodes) * len(data.tasks)
            if count or (args.output / "diagnostics.jsonl").exists():
                reconcile_metrics(args.output / "diagnostics.jsonl", updates, count, cursor_key="step", packet_label="diagnostics")
            if config["local_action"]["enabled"]:
                local_count = count * config["local_action"]["diagnostic_clips_per_task"]
                reconcile_metrics(args.output / "local_diagnostics.jsonl", updates, local_count,
                                  cursor_key="step", packet_label="local_diagnostics")
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
            "seconds": seconds,
            "mean_flow_loss": sum(r["flow_loss"] * r["condition_weight"] for r in gathered),
            "mean_local_flow_loss": sum(r["local_flow_loss"] * r["local_weight"] for r in gathered),
            "local_clips": updates * _logical_batch(config)["local_clips_per_update"],
            "local_noise_draws": updates * _logical_batch(config)["local_noise_draws_per_update"],
            **norms, "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
            "condition_exposures": metrics_rows, "task_exposures": updates * 4,
            "supervised_queries": updates * _logical_batch(config)["queries_per_update"],
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
    if stop <= 0 or (args.mode == "formal" and (len(nodes) != 2 or stop != nodes[-1])):
        raise ValueError("formal segment needs two registered checkpoint nodes and must stop at the last")
    return stop


def _validate_actions(args, engine, data, context, config, step):
    spec = config["evidence"]["supervised_validation"]
    tasks = tuple(spec["task_ids"])
    demos = {task: spec["teacher_video_pool"][task % len(spec["teacher_video_pool"])] for task in tasks}
    costs = {task: data.videos.frame_counts(task, demos[task])[1] for task in tasks}
    assignment = cost_balanced_task_assignment(
        tasks, costs, {task: tuple(range(context.world_size)) for task in tasks}, world_size=context.world_size,
    )
    rows, local_rows, error = [], [], None
    try:
        for task in assignment[context.rank]:
            rows.append({"step": step, **engine.validate(task, demos[task], seed=spec["seed"] + task,
                                                       queries=spec["queries_per_task"])})
            if config["local_action"]["enabled"]:
                for clip in range(config["local_action"]["diagnostic_clips_per_task"]):
                    local_rows.append({"step": step, "task": task, "suite": data.tasks[task].suite,
                        "clip": clip, "gradients": False, **engine.local_action(task, clip, diagnostic=True)})
    except Exception:
        error = traceback.format_exc()
    packets = _gather((rows, local_rows, error), context)
    if any(failure for _, _, failure in packets):
        raise RuntimeError(f"held-action validation failed: {[e for _, _, e in packets if e]}")
    if context.is_main:
        gathered = [row for rows, _, _ in packets for row in rows]
        local_gathered = [row for _, rows, _ in packets for row in rows]
        for row in local_gathered:
            append_jsonl(args.output / "local_diagnostics.jsonl", row)
        for row in gathered:
            append_jsonl(args.output / "diagnostics.jsonl", row)
        print(json.dumps({"diagnostic_step": step, "tasks": len(gathered),
                          "held_action_fm": sum(row["flow_loss"] for row in gathered) / len(gathered),
                          "held_local_action_fm": (sum(row["local_flow_loss"] for row in local_gathered) / len(local_gathered)
                                                   if local_gathered else None)}), flush=True)


def _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop, start):
    updates, metrics_rows = cursors
    nodes = _checkpoint_nodes(args, config)
    if args.mode == "formal" and any(node <= updates for node in nodes):
        raise ValueError("segment checkpoint nodes must follow the restored update cursor")
    if context.is_main:
        print(json.dumps({"segment_start": updates, "segment_stop": stop, "checkpoint_updates": nodes,
                          "resume": str(args.resume) if getattr(args, "resume", None) else None}), flush=True)
    if args.mode == "formal" and updates == 0 and 0 in config["evidence"]["supervised_validation"]["optimizer_updates"]:
        _validate_actions(args, engine, data, context, config, 0)
    while updates < stop:
        tick = time.perf_counter()
        rows, norms = _update(engine, runtime, data, context, config, optimizer, scheduler, updates + 1)
        updates += 1
        torch.cuda.synchronize(context.device)
        metrics_rows = _record_iteration(args, context, config, rows, norms, updates,
                                         metrics_rows, time.perf_counter() - tick, scheduler)
        if updates == stop or updates in nodes:
            if args.mode == "formal" and updates in config["evidence"]["supervised_validation"]["optimizer_updates"]:
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
            "optimizer_updates": updates, "exposures": metrics_rows,
            "condition_exposures": metrics_rows, "task_exposures": updates * 4,
            "supervised_queries": updates * _logical_batch(config)["queries_per_update"],
            "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered held-action and paired closed-loop evidence",
        })


def run(args: argparse.Namespace) -> None:
    from ember.writer.supervised import SupervisedEngine

    config = _config(args.config)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal supervised training requires a clean pushed detached worktree")
    stop = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= min(6, _logical_batch(config)["conditions"]):
        raise ValueError("condition parallelism requires one node and 1 to min(6, condition count) useful GPUs")
    execution_config, microbatches = _execution_config(args, config, context)
    if context.is_main:
        print(json.dumps({"physical_policy_microbatches": microbatches, "logical_queries_per_update": 256}), flush=True)
    torch.set_num_threads(int(args.cpu_threads))
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    start = time.perf_counter()
    data = WriterTrainingData(args.asset_root, config["data"],
                              camera_view=config["observer"].get("camera_view", "agentview"))
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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pi05_video_functional.json")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-updates", help="this segment's two global update nodes, e.g. 300,400")
    parser.add_argument("--policy-microbatches", help="physical FM query chunks by rank, e.g. 8,4,8,8")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
