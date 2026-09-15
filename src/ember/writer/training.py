"""Fresh video-conditioned functional learning with explicit gradient ownership."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.learning_data import WriterTrainingData
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import VideoConditionCache, build_runtime, require_architecture_identity
from ember.writer.task_execution import cost_balanced_task_assignment


RUN_SCHEMA = "ember_language_axial_writer_run_v1"
STAGE = "language_axial_writer_fresh"
TRAINING_SCHEMA = "ember_language_axial_training_state_v1"
UPDATE_VERSION = "full_ab_pure_fm_joint_text_vl_action_meta_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def observer_mode_contract(model: dict[str, Any]) -> dict[str, str]:
    """Bind declared RGB and full-horizon reads to the registered model pair."""
    require_architecture_identity(model)
    camera, read = model["camera_view"], model["horizon_read"]
    patches = 512 if camera == "dual" else 256
    return {"camera_view": camera,
            "native_inputs": f"full{patches}_patch_content_and_full50_{read}_horizon_read",
            "horizon_read": "softmax_shared_query_nonaffine_rms_H_plus_relative_bias_then_raw_H_values_uniform_init"
                if read == "learned" else "uniform_fixed_zero_query_and_bias_over_all_50_raw_H_values"}


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_data = {
        "extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
        "queries_per_task": 21, "tasks_per_update": 4, "conditions_per_task": 1, "cardinalities": [1],
        "action_start_offset": 1, "query_alignment": "post_action_observation_future_control_v1",
        "version": "v52_full_video_cross_episode_events_v1",
        "event_schema_version": "v52_full_video_cross_episode_events_v1",
        "seed": 7, "sampler_seed": 20260721, "teacher_video_seed": 20260722, "maximum_updates": 1200,
    }
    expected_observer = {
        "flow_time": 1, "meta_rank": 4, "vl_meta_rank": 4, "text_meta_rank": 4,
        "probe_seed": 7 + 0x5A17, **observer_mode_contract(config["model"]),
    }
    if (config.get("schema_version") != "ember_language_axial_writer_config_v1"
            or config["optimization"].get("joint_train_all_writer_modules") is not True
            or float(config["optimization"]["normalizer"]) != 1.0
            or config["optimization"]["loss"] != "main_fm"
            or any(config["data"].get(key) != value for key, value in expected_data.items())
            or type(config["data"].get("action_start_offset")) is not int
            or any(config["observer"].get(key) != value for key, value in expected_observer.items())
            or config.get("update_version") != UPDATE_VERSION
            or {"rl", "video_prior", "spatial_supervision", "correction_supervision",
                "native_output_calibration", "local_field_supervision"} & config.keys()
            or "trust_scales" in config["optimization"]
            or config.get("execution_precision") != "native_bf16_writer_fm_fp32_lora"):
        raise ValueError("canonical Core/Procedure Writer scientific contract changed")
    for key, expected in (("video_demos", range(46)), ("action_demos", range(46)),
                          ("diagnostic_action_demos", range(46, 50)), ("held_video_demos", range(46, 50))):
        if config["data"][key] != list(expected):
            raise ValueError(f"registered episode roles changed: {key}")
    if len(config["data"]["task_ids"]) != 24:
        raise ValueError("development gradients require all fixed train24 tasks")
    if any(type(value) is not int or value <= 0 for value in config["runtime"].values()):
        raise ValueError("runtime batches and cache budget must be positive integers")
    if type(config["observer"]["frame_chunk"]) is not int or config["observer"]["frame_chunk"] <= 0:
        raise ValueError("native frame chunk must be a positive integer")
    _validate_checkpoint_nodes(config["evidence"]["checkpoint_updates"], allow_empty=True)
    return config


def _optimization(state, config):
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    opt = config["optimization"]
    optimizer = torch.optim.AdamW(
        state.parameters(), lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=int(opt["warmup_updates"]), num_decay_steps=int(opt["decay_updates"]),
        peak_lr=float(opt["lr"]), decay_lr=float(opt["decay_lr"]),
    # This bounded run covers only the beginning of the original 12k clock.
    # The upstream builder otherwise rescales warmup/decay to the 1200 budget.
    ).build(optimizer, int(opt["decay_updates"]))
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
        "model_config": dict(config["model"]),
        "topology": {
            "host": socket.gethostname(), "world_size": context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"), "ranks": _gather(local, context),
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "vl_meta_parameters": sum(p.numel() for p in runtime.state.vl_meta.parameters()),
            "text_meta_parameters": sum(p.numel() for p in runtime.state.text_meta.parameters()),
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "optimizer": "fresh AdamW; one grouped functional update per four equally weighted tasks", "scaler": None,
            "resume_contract": "same config, topology, sampler streams, optimizer updates and complete state",
            "logical_batch": _logical_batch(config),
            "event_plan": str((args.output / "training_events.json").resolve()),
            "update_version": config["update_version"], "data_version": config["data"]["version"],
            "checkpoint_updates": list(_checkpoint_nodes(args, config)),
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": "main LoRA cross-episode", "gradient_normalizer": 1.0,
            "objective": config["optimization"]["loss"],
            "training_only_actions": "same-task cross-episode main FM execution queries only",
            "native_read": f"same-version final {config['model']['camera_view']} Z and full50 H to "
                           f"{config['model']['horizon_read']}; joint three-Meta checkpoint replay",
            "complete_lora": "shared eight-family full A/B heads from video Core and Procedure modulation",
            "deployment_frozen_source_vjp": False, "deployment_loss_or_optimizer": False,
            "rl_rollouts": False, "rl_loss": False, "trust_rollback": False,
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
    return {"tasks": 4, "conditions_per_task": 1, "conditions": 4, "K": 1,
            "queries_per_task": 21, "queries_per_condition": 21, "queries_per_update": 84,
            "task_weight": .25, "condition_weight": .25, "gradient_reduction": "SUM"}


def _condition_jobs(data, config, draws):
    by_job = {draw["job_id"]: draw for draw in draws}
    tasks = {draw["task"] for draw in draws}
    if (len(draws) != 4 or len(by_job) != 4 or len(tasks) != 4 or not tasks <= set(data.tasks)
            or any(draw["condition_index"] != 0 or len(draw["video_demos"]) != 1
                   or draw["query_count"] != 21 or draw["query_offset"] != 0 for draw in draws)):
        raise ValueError("each update requires four distinct equal-weight K1 task events and 21 queries each")
    return by_job


def _execute_step(engine, data, context, config, draws, step):
    logical = _logical_batch(config)
    if not 1 <= context.world_size <= min(6, logical["conditions"]):
        raise ValueError("condition parallelism requires 1 to min(6, condition count) useful ranks")
    by_job = _condition_jobs(data, config, draws)
    engine.step = step
    jobs = tuple(by_job)
    costs = {job: int(draw["frames"]) for job, draw in by_job.items()}
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
             "vl_meta_grad_norm": _grad_norm(runtime.state.vl_meta.parameters()),
             "text_meta_grad_norm": _grad_norm(runtime.state.text_meta.parameters())}
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


def _validate_checkpoint_nodes(nodes, *, allow_empty=False):
    if (not nodes and not allow_empty) or any(type(node) is not int or node <= 0 for node in nodes):
        raise ValueError("checkpoint updates must be registered increasing positive integers")
    if list(nodes) != sorted(set(nodes)):
        raise ValueError("checkpoint updates must be registered increasing positive integers")


def _checkpoint_nodes(args, config):
    supplied = getattr(args, "checkpoint_updates", None)
    nodes = tuple(config["evidence"]["checkpoint_updates"]) if supplied is None else tuple(map(int, supplied.split(",")))
    _validate_checkpoint_nodes(nodes, allow_empty=args.mode != "formal")
    return nodes


def _segment_limit(args, config):
    nodes = _checkpoint_nodes(args, config)
    stop = args.stop_after_step if args.stop_after_step is not None else (nodes[-1] if nodes else None)
    if type(stop) is not int or stop <= 0:
        raise ValueError("smoke/profile without registered nodes needs an explicit positive --stop-after-step")
    if args.mode == "formal" and stop != nodes[-1]:
        raise ValueError("formal segment must stop at the last registered checkpoint node")
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
    # A resumed segment keeps its original registered nodes. The restored
    # cursor skips completed nodes while the loop retains the registered stop.
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
    if args.mode == "formal" and (config["status"] != "registered_source_aligned_v52_learning"
                                  or config["evidence"]["profile_registration"]["status"] != "complete"):
        raise ValueError("formal learning needs the post-profile checkpoint and exposure registration")
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal supervised training requires a clean pushed detached worktree")
    stop = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= min(6, _logical_batch(config)["conditions"]):
        raise ValueError("condition parallelism requires one node and 1 to min(6, condition count) useful GPUs")
    execution_config, microbatches = _execution_config(args, config, context)
    if context.is_main:
        print(json.dumps({"physical_policy_microbatches": microbatches, "logical_queries_per_update": 84}), flush=True)
    torch.set_num_threads(int(args.cpu_threads))
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    start = time.perf_counter()
    data = WriterTrainingData(args.asset_root, config["data"],
                              camera_view=config["observer"]["camera_view"])
    if context.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        events = data.event_plan()
        event_path = args.output / "training_events.json"
        if args.resume:
            if read_json(event_path) != events:
                raise ValueError("exact-resume training events or grouping changed")
        elif event_path.exists():
            raise ValueError("fresh training refuses an existing event plan")
        else:
            write_json_atomic(event_path, events)
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
    cache = VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"]))
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
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pi05_writer.json")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "profile", "formal"), required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--checkpoint-updates", help="this segment's registered global optimizer-update nodes")
    parser.add_argument("--policy-microbatches", help="physical FM query chunks by rank, e.g. 8,4,8,8")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
