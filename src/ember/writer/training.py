"""Fresh horizon Writer/Meta, same-version FM+RL and accepted-step trust updates."""
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
from ember.writer.learning_data import JointTrainingData
from ember.writer.replay import sum_writer_gradients
from ember.writer.rl_math import TRUST_SCALES
from ember.writer.runtime import FrozenVideoPrefixCache, build_joint_runtime
from ember.writer.task_execution import cost_balanced_task_assignment


RUN_SCHEMA = "ember_horizon_relation_writer_joint_run_v1"
STAGE = "horizon_relation_writer_fresh_fm_rl_joint"
TRAINING_SCHEMA = "ember_horizon_joint_training_state_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_model = asdict(HorizonWriterConfig())
    # Chunk sizes are execution choices; the complete scientific graph is fixed.
    actual = {**config["model"], "edge_chunk": expected_model["edge_chunk"]}
    if (
        config.get("schema_version") != "ember_horizon_relation_writer_config_v1"
        or actual != expected_model
        or config["optimization"].get("fresh_joint_writer_and_meta") is not True
        or float(config["optimization"]["normalizer"]) != 1.0
        or config["data"].get("extra_meta_tasks") != []
        or int(config["data"]["frame_stride"]) != 5
        or config["data"].get("include_last_frame") is not True
        or int(config["data"]["queries_per_task"]) != 64
        or int(config["observer"]["flow_time"]) != 1
        or int(config["observer"]["meta_rank"]) != 4
        or int(config["observer"]["probe_seed"]) != 1729
        or config["optimization"]["loss"] != "fm_plus_extended_action_writer_rl_same_version"
        or tuple(config["optimization"].get("trust_scales", ())) != TRUST_SCALES
        or config.get("execution_precision") != "native_mixed_without_outer_autocast"
    ):
        raise ValueError("horizon joint Writer scientific contract changed")
    expected_rl = {
        "episodes_per_condition": 4, "initial_state_ids": list(range(32)),
        "decisions_per_episode": 16, "trust_decisions_per_episode": 4,
        "coefficient": 0.1, "flow_steps": 10, "executed_actions": 5,
        "noise": {"rho": 0.8, "std": [0.05] * 6 + [0.1]}, "max_task_kl": 0.02,
    }
    if config["rl"] != expected_rl:
        raise ValueError("configured RL protocol differs from the implemented first-run contract")
    for key, expected in (("video_demos", range(16)), ("action_demos", range(16, 42)),
                          ("diagnostic_action_demos", range(42, 46)), ("held_video_demos", range(46, 50))):
        if config["data"][key] != list(expected):
            raise ValueError(f"registered episode roles changed: {key}")
    if len(config["data"]["task_ids"]) != 24:
        raise ValueError("first-run gradients require all fixed train24 tasks")
    if any(int(value) <= 0 for value in config["runtime"].values()):
        raise ValueError("runtime batches and cache budget must be positive")
    collected_batch = min(4, int(config["runtime"]["rollout_microbatch"]))
    if any(int(config["runtime"][key]) < collected_batch for key in ("rl_microbatch", "trust_microbatch")):
        raise ValueError("replay capacity must support the collected numerical batch shape")
    HorizonWriterConfig(**config["model"])
    return config


def _optimization(state, config):
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(
        state.parameters(), lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    warmup = int(opt["warmup_accepted"])
    if warmup < 1:
        raise ValueError("accepted-update warmup must be positive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda accepted: min(1., (accepted + 1) / warmup))
    return optimizer, scheduler


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
            "optimizer": "fresh AdamW; one joint direction; finite scaled candidates", "scaler": None,
            "resume_contract": "same config, topology, sampler streams, attempted/accepted and complete state",
            "update_version": config["update_version"], "data_version": config["data"]["version"],
            "checkpoint_accepted": config["evidence"]["checkpoint_accepted"],
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": "disjoint fixed roles", "gradient_normalizer": 1.0,
            "sampling_and_gradient_version": "same; one combined FM+RL update per attempted iteration",
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
    tasks = tuple(int(draw["task"]) for draw in draws)
    if len(tasks) != 4 or len({data.tasks[task].suite for task in tasks}) != 4:
        raise ValueError("each attempted update must contain one task from each suite")
    by_task = {int(draw["task"]): draw for draw in draws}
    costs = {task: int(draw["frames"]) for task, draw in by_task.items()}
    assignment = cost_balanced_task_assignment(
        tasks, costs, {task: tuple(range(context.world_size)) for task in tasks}, world_size=context.world_size,
    )
    rows, evidence = [], []
    for task in assignment[context.rank]:
        draw = by_task[task]
        tick = time.perf_counter()
        metric, saved = engine.backward(draw)
        if int(metric["queries"]) != int(config["data"]["queries_per_task"]) or int(metric["rl_episodes"]) != 4:
            raise RuntimeError("joint engine did not execute the registered FM/RL exposure")
        rows.append({**metric, "step": step, "task": task, "suite": data.tasks[task].suite,
                     "occurrence": draw["occurrence"], "K": len(draw["video_demos"]),
                     "video_demos": list(draw["video_demos"]), "frames": costs[task],
                     "query_seed": draw["query_seed"], "episodes": draw["episodes"],
                     "queries": int(config["data"]["queries_per_task"]), "seconds": time.perf_counter() - tick})
        evidence.append((task, saved))
    return rows, evidence


def _trust_scores(engine, evidence, context, tasks):
    local, error = {}, None
    try:
        local = {task: float(engine.trust_score(saved)) for task, saved in evidence}
    except Exception:
        error = traceback.format_exc()
    packets = _gather((local, error), context)
    if any(failure for _, failure in packets):
        raise RuntimeError(f"candidate evaluation failed on a rank: {[failure for _, failure in packets if failure]}")
    scores = {}
    for values, _ in packets:
        if scores.keys() & values.keys():
            raise ValueError("candidate trust evaluation duplicated a task")
        scores.update(values)
    if scores.keys() != set(tasks):
        raise ValueError("candidate trust evaluation must cover the complete four-task update")
    return scores


def _attempt_update(engine, runtime, data, context, config, optimizer, scheduler, attempted, *, profile=False):
    from ember.writer.rl_math import adamw_trust_step

    draws = data.next_iteration()  # Consumed even if every candidate is rejected.
    optimizer.zero_grad(set_to_none=True)
    rows, evidence, error = [], [], None
    try:
        rows, evidence = _execute_step(engine, data, context, config, draws, attempted)
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, context) if value]
    if failures:
        raise RuntimeError(f"joint backward failed on a rank: {failures}")
    parameters = tuple(runtime.state.parameters())
    sum_writer_gradients(parameters, world_size=context.world_size)
    norms = {"writer_grad_norm": _grad_norm(runtime.state.writer.parameters()),
             "meta_grad_norm": _grad_norm(runtime.state.meta.parameters())}
    norms["total_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
        parameters, float(config["optimization"]["grad_clip"]), error_if_nonfinite=True))
    if profile:
        tick = time.perf_counter()
        norms["current_version_task_kl"] = _trust_scores(engine, evidence, context, [d["task"] for d in draws])
        norms["current_version_trust_seconds"] = time.perf_counter() - tick
    result = adamw_trust_step(optimizer, lambda: _trust_scores(engine, evidence, context, [d["task"] for d in draws]),
                              max_task_kl=float(config["rl"]["max_task_kl"]))
    if result.accepted:
        scheduler.step()
    return rows, result, norms  # No rollout evidence survives into the next version.


def _training_state(config, attempted, accepted):
    return {"schema_version": TRAINING_SCHEMA, "attempted": attempted, "accepted": accepted,
            "rejected": attempted - accepted, "Sigma": config["rl"]["noise"],
            "update_version": config["update_version"], "data_version": config["data"]["version"]}


def _restore(args, context, runtime, data, optimizer, scheduler, config):
    if not args.resume:
        return 0, 0, 0
    if args.resume.resolve().parent.parent != args.output.resolve():
        raise ValueError("exact-resume checkpoint must belong to its original run root")
    restored = {}
    attempted, metrics_rows = load_ecp_checkpoint(
        checkpoint=args.resume, stage=STAGE, context=context, model=runtime.state,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA, restored_state=restored,
    )
    training = restored["training_state"]
    if not isinstance(training, dict) or not 0 <= training.get("accepted", -1) <= attempted:
        raise ValueError("joint checkpoint has no valid accepted-update cursor")
    accepted = int(training["accepted"])
    if training != _training_state(config, attempted, accepted):
        raise ValueError("joint checkpoint training/noise/update/data contract changed")
    data.restore_sampler(restored["sampler_state"])
    if data.sampler_state()["next_step"] != attempted:
        raise ValueError("sampler and attempted-update cursors differ")
    if context.is_main:
        reconcile_metrics(args.output / "exposures.jsonl", attempted, metrics_rows, cursor_key="step", packet_label="exposures")
        reconcile_metrics(args.output / "metrics.jsonl", attempted, attempted, cursor_key="step", packet_label="metrics")
    return attempted, accepted, metrics_rows


def _record_iteration(args, context, config, rows, result, norms, attempted, accepted, metrics_rows, seconds, scheduler):
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
            "step": attempted, "attempted": attempted, "accepted": accepted, "rejected": attempted - accepted,
            "update_accepted": result.accepted, "alpha": result.alpha,
            "trust_candidates": [asdict(candidate) for candidate in result.attempts],
            "seconds": seconds, "mean_flow_loss": sum(r["flow_loss"] for r in gathered) / 4,
            **norms, "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
            "supervised_queries": metrics_rows * int(config["data"]["queries_per_task"]),
            "rollout_episodes": metrics_rows * int(config["rl"]["episodes_per_condition"]),
            "rl_mixed_groups": sum(r["rl_mixed_group"] for r in gathered),
            "rl_mixed_group_fraction": sum(r["rl_mixed_group"] for r in gathered) / 4,
            "rl_successes": sum(r["rl_successes"] for r in gathered),
            "rank_memory": [{key: value for key, value in packet.items() if key != "rows"} for packet in packets],
            "peak_allocated_gib": max(packet["peak_allocated_gib"] for packet in packets),
            "peak_reserved_gib": max(packet["peak_reserved_gib"] for packet in packets),
        }
        append_jsonl(args.output / "metrics.jsonl", metric)
        print(json.dumps(metric), flush=True)
    return metrics_rows


def _segment_limit(args, config):
    if args.mode == "formal" and args.stop_after_step is not None:
        raise ValueError("an attempted-step cap is a profile control; formal segments stop at accepted-update nodes")
    stop = max(config["evidence"]["checkpoint_accepted"]) if args.stop_after_accepted is None else args.stop_after_accepted
    if stop <= 0 or (args.stop_after_step is not None and args.stop_after_step <= 0):
        raise ValueError("segment stop cursors must be positive")
    return stop


def _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop_accepted, start):
    attempted, accepted, metrics_rows = cursors
    while accepted < stop_accepted and (args.stop_after_step is None or attempted < args.stop_after_step):
        tick = time.perf_counter()
        rows, result, norms = _attempt_update(engine, runtime, data, context, config, optimizer, scheduler,
                                             attempted + 1, profile=args.mode == "profile")
        attempted += 1
        accepted += int(result.accepted)
        torch.cuda.synchronize(context.device)
        metrics_rows = _record_iteration(args, context, config, rows, result, norms, attempted, accepted,
                                         metrics_rows, time.perf_counter() - tick, scheduler)
        segment_end = accepted >= stop_accepted or (args.stop_after_step is not None and attempted >= args.stop_after_step)
        if segment_end or (result.accepted and accepted in config["evidence"]["checkpoint_accepted"]):
            save_ecp_checkpoint(
                output_dir=args.output, macro=attempted, stage=STAGE, context=context,
                model=runtime.state, optimizer=optimizer, scheduler=scheduler,
                run_contract_schema=RUN_SCHEMA, metrics_rows=metrics_rows,
                sampler_state=data.sampler_state(), training_state=_training_state(config, attempted, accepted),
            )
    barrier(context)
    if context.is_main:
        write_json_atomic(args.output / "completion.json", {
            "schema_version": RUN_SCHEMA, "status": "segment_complete", "mode": args.mode,
            "attempted": attempted, "accepted": accepted, "rejected": attempted - accepted,
            "exposures": metrics_rows, "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered paired closed-loop evidence and continued iteration",
        })


def run(args: argparse.Namespace) -> None:
    from ember.writer.joint import JointUpdateEngine

    config = _config(args.config)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal joint training requires a clean pushed detached worktree")
    stop_accepted = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 6:
        raise ValueError("EMBER joint training needs one node and at most six GPUs")
    torch.set_num_threads(int(args.cpu_threads))
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
    cursors = _restore(args, context, runtime, data, optimizer, scheduler, config)
    attempted, accepted, _ = cursors
    if accepted >= stop_accepted or (args.stop_after_step is not None and attempted >= args.stop_after_step):
        raise ValueError("joint segment has no remaining registered updates")
    cache = FrozenVideoPrefixCache(runtime.observer, data, int(config["runtime"]["prefix_cache_bytes"]))
    engine = JointUpdateEngine(runtime, data, cache, context, config, args.asset_root, args.output)
    barrier(context)
    try:
        _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop_accepted, start)
    finally:
        engine.close()
        data.close()
        if context.world_size > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pi05_horizon_writer_v1.json")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--stop-after-accepted", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
