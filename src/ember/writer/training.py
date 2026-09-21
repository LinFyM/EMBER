"""Fresh video-conditioned functional learning with explicit gradient ownership."""
from __future__ import annotations

import argparse
import json
import math
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
from ember.writer.learning_data import EVENT_SCHEMA, WriterTrainingData
from ember.writer.auxiliary_pairing import declared_dynamic_episode
from ember.writer.continuation import (
    LOW_LR_REPAIR, inherit_history, prepare_continuation, prepare_phase_continuation,
    require_continuation_config, require_continuation_start, require_extended_prefix,
)
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import VideoConditionCache, build_runtime, require_architecture_identity
from ember.writer.task_execution import (
    condition_rank_groups, cost_balanced_task_assignment, initialize_condition_group,
    merge_condition_rows, query_shard,
)


CONFIG_SCHEMA = "ember_video_teaching_writer_config_v1"
RUN_SCHEMA = "ember_video_teaching_writer_run_v1"
STAGE = "video_teaching_writer_fresh"
TRAINING_SCHEMA = "ember_video_teaching_training_state_v1"
UPDATE_VERSION = "video_teaching_full_ab_joint_meta_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_TRANSITION_SCHEMA = "ember_writer_topology_transition_v1"


def observer_mode_contract(model: dict[str, Any]) -> dict[str, str]:
    """Bind the ordered full-horizon and adjacent-content native reads."""
    require_architecture_identity(model)
    patches = 512 if model["camera_view"] == "dual" else 256
    return {"camera_view": model["camera_view"],
            "native_inputs": f"full{patches}_patch_content_and_repeated_full50_H_adjacent_E_reads",
            "horizon_read": "repeated_content_position_attention_over_all_50_raw_H_values",
            "video_order": "causal_RoPE_with_real_frame_positions_and_ordered_adjacent_roles"}


def _validate_dynamic_schedule(config):
    expected_episode = declared_dynamic_episode(config, REPO_ROOT)
    opt = config["optimization"]
    if (any(type(opt.get(key)) is not int for key in
            ("warmup_updates", "tail_start_update", "tail_end_update", "decay_updates"))
            or not 0 <= opt["warmup_updates"] <= opt["tail_start_update"] < opt["tail_end_update"]
            or opt["decay_updates"] <= opt["tail_end_update"]
            or not 0 < opt["tail_final_ratio"] <= 1
            or config["model"]["camera_view"] != "agentview"
            or config["data"].get("teaching_episode") != expected_episode
            or not config["data"].get("protocol")):
        raise ValueError("dynamic Writer schedule or single-camera teaching contract changed")


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    require_continuation_config(config)
    expected_data = {
        "extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
        "queries_per_task": 21, "tasks_per_update": 4, "conditions_per_task": 1, "cardinalities": [1],
        "action_start_offset": 1, "query_alignment": "post_action_observation_future_control_v1",
        "version": EVENT_SCHEMA, "event_schema_version": EVENT_SCHEMA,
        "seed": 7, "sampler_seed": 20260721, "teacher_video_seed": 20260722,
        "teaching_queries_per_task": 7, "teaching_seed": 20260919,
    }
    expected_observer = {
        "flow_time": 1, "meta_rank": 4, "vl_meta_rank": 4, "text_meta_rank": 4,
        "probe_seed": 7 + 0x5A17, **observer_mode_contract(config["model"]),
    }
    expected_optimization = {
        "loss": "main_fm_plus_video_teaching", "joint_train_all_writer_modules": True,
        "normalizer": 1., "teaching_weight": 1 / 3, "teaching_prefix_steps": 5, "teaching_flow_time": 1,
        "tail_start_update": 900, "tail_end_update": 1500, "tail_final_ratio": .1,
        "seed": 7, "lr": 3e-4, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 1e-4,
        "grad_clip": 1., "warmup_updates": 100, "decay_updates": 12000, "decay_lr": 1e-5,
    }
    dynamic = config.get("training_control") is not None
    if dynamic:
        for key in ("warmup_updates", "tail_start_update", "tail_end_update", "tail_final_ratio", "decay_updates"):
            expected_optimization.pop(key)
        _validate_dynamic_schedule(config)
    if (config.get("schema_version") != CONFIG_SCHEMA
            or any(config["optimization"].get(key) != value for key, value in expected_optimization.items())
            or config["data"].get("teaching_episode") not in {"same_video", "cross_episode"}
            or any(config["data"].get(key) != value for key, value in expected_data.items())
            or type(config["data"].get("action_start_offset")) is not int
            or any(config["observer"].get(key) != value for key, value in expected_observer.items())
            or config.get("update_version") != UPDATE_VERSION
            or {"rl", "video_prior", "spatial_supervision", "correction_supervision",
                "native_output_calibration", "local_field_supervision"} & config.keys()
            or "trust_scales" in config["optimization"]
            or config.get("execution_precision") != "native_bf16_writer_fm_fp32_lora"):
        raise ValueError("canonical video-teaching Writer scientific contract changed")
    for key, expected in (("video_demos", range(46)), ("action_demos", range(46)),
                          ("diagnostic_action_demos", range(46, 50)), ("held_video_demos", range(46, 50))):
        if config["data"][key] != list(expected):
            raise ValueError(f"registered episode roles changed: {key}")
    if len(set(config["data"]["task_ids"])) != (36 if dynamic else 24):
        raise ValueError("development gradients require the complete registered training task set")
    if any(type(value) is not int or value <= 0 for value in config["runtime"].values()):
        raise ValueError("runtime batches and cache budget must be positive integers")
    if type(config["observer"]["frame_chunk"]) is not int or config["observer"]["frame_chunk"] <= 0:
        raise ValueError("native frame chunk must be a positive integer")
    _validate_checkpoint_nodes(config["evidence"]["checkpoint_updates"], allow_empty=True)
    return config


def _learning_rate_multiplier(step, opt):
    def original(index):
        if index < opt["warmup_updates"]:
            return (index + 1) / (opt["warmup_updates"] + 1)
        floor = opt["decay_lr"] / opt["lr"]
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * index / opt["decay_updates"]))
    start, end = opt["tail_start_update"], opt["tail_end_update"]
    if step <= start:
        return original(step)
    progress = min(1., (step - start) / (end - start))
    ratio = opt["tail_final_ratio"]
    return original(start) * (ratio + (1 - ratio) * .5 * (1 + math.cos(math.pi * progress)))


def _optimization(state, config):

    opt = config["optimization"]
    optimizer = torch.optim.AdamW(
        state.parameters(), lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    phase = config.get("phase_continuation")
    multiplier = ((lambda _step: float(phase["fixed_lr"]) / float(opt["lr"]))
                  if phase else (lambda step: _learning_rate_multiplier(step, opt)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
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


def _event_plan_path(args):
    return args.output / "training_events.json"


def _publish_event_plan(args, events):
    path = _event_plan_path(args)
    parent_checkpoint = getattr(args, "phase_from", None) or getattr(args, "extend_from", None)
    if parent_checkpoint:
        parent = parent_checkpoint.resolve().parent.parent / "training_events.json"
        phase = getattr(args, "phase_from", None)
        require_extended_prefix(read_json(parent), events,
                                parent_updates=(LOW_LR_REPAIR["parent_updates"] if phase else 1500),
                                child_updates=(events.get("maximum_updates") or 2100))
    if path.exists():
        if not args.resume:
            raise ValueError("fresh training refuses an existing event plan")
        if read_json(path) != events:
            raise ValueError("exact-resume training events or grouping changed")
    elif args.resume:
        raise ValueError("exact-resume requires its original registered event plan")
    else:
        write_json_atomic(path, events)


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
            "condition_rank_groups": [list(group) for group in condition_rank_groups(context.world_size)],
            "within_condition": "disjoint_native_frames_and_queries_compact_autograd_gather_v1",
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "vl_meta_parameters": sum(p.numel() for p in runtime.state.vl_meta.parameters()),
            "text_meta_parameters": sum(p.numel() for p in runtime.state.text_meta.parameters()),
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "optimizer": ("parent AdamW state preserved" if
                          (config.get("continuation") or config.get("phase_continuation")) else "fresh AdamW")
                         + "; one grouped functional update per four equally weighted tasks", "scaler": None,
            "resume_contract": {
                "default": "same config, physical topology, sampler streams, optimizer updates and complete state",
                "topology_transition": "ordinary dynamic --resume with explicit --allow-topology-change only",
                "preserved_on_transition": ["logical_batch", "event_plan", "optimizer", "scheduler", "sampler"],
            },
            "logical_batch": _logical_batch(config),
            "event_plan": str(_event_plan_path(args).resolve()),
            "maximum_updates": config["data"]["maximum_updates"],
            "update_version": config["update_version"], "data_version": config["data"]["version"],
            "checkpoint_updates": list(_checkpoint_nodes(args, config)),
        },
        "information_wall": {
            "deployment_inputs": ["exact language", "complete internally ordered RGB video"],
            "frame_indices": "real sampled positions for ordered Procedure",
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": {"main": "cross_episode", "teaching": config["data"]["teaching_episode"]},
            "gradient_normalizer": 1.0,
            "objective": config["optimization"]["loss"],
            "training_only_actions": "execution-query inputs and targets, loaded after RGB-language-only compilation",
            "native_read": "repeated full50 H and ordered adjacent E content; joint three-Meta replay",
            "complete_lora": "shared eight-family full A/B heads from Core-conditioned centered Procedure AdaLN",
            "deployment_frozen_source_vjp": False, "deployment_loss_or_optimizer": False,
            "rl_rollouts": False, "rl_loss": False, "trust_rollback": False,
        },
    }


def require_resume_identity(old, contract, *, allow_topology_change=False):
    for key in ("schema_version", "stage", "mode", "config", "model_config", "source"):
        if old.get(key) != contract[key]:
            raise ValueError(f"exact-resume contract differs: {key}")
    if old.get("topology") != contract["topology"] and not allow_topology_change:
        raise ValueError("exact-resume contract differs: topology")
    if allow_topology_change and old.get("training", {}).get("logical_batch") != contract["training"]["logical_batch"]:
        raise ValueError("topology transition changed the logical Writer update")


def _publish_contract(path, contract, *, resume, allow_topology_change=False):
    if resume:
        require_resume_identity(read_json(path), contract, allow_topology_change=allow_topology_change)
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
            "teaching_queries_per_task": 7, "teaching_queries_per_update": 28,
            "total_queries_per_update": 112, "teaching_weight": config["optimization"]["teaching_weight"],
            "task_weight": .25, "condition_weight": .25, "gradient_reduction": "SUM"}


def _condition_jobs(data, config, draws):
    by_job = {draw["job_id"]: draw for draw in draws}
    tasks = {draw["task"] for draw in draws}
    if (len(draws) != 4 or len(by_job) != 4 or len(tasks) != 4 or not tasks <= set(data.tasks)
            or any(draw["condition_index"] != 0 or len(draw["video_demos"]) != 1
                   or draw["query_count"] != 21 or draw["query_offset"] != 0
                   or draw["teaching_count"] != 7 or draw["teaching_offset"] != 0 for draw in draws)):
        raise ValueError("each update requires four distinct equal-weight K1 tasks with 21 main and seven teaching queries")
    return by_job


def _execute_step(engine, data, context, config, draws, step):
    logical = _logical_batch(config)
    groups = condition_rank_groups(context.world_size)
    group_index = next(index for index, members in enumerate(groups) if context.rank in members)
    members = groups[group_index]
    by_job = _condition_jobs(data, config, draws)
    engine.step = step
    jobs = tuple(by_job)
    costs = {job: int(draw["frames"]) for job, draw in by_job.items()}
    assignment = cost_balanced_task_assignment(
        jobs, costs, {job: tuple(range(len(groups))) for job in jobs}, world_size=len(groups),
    )
    rows = []
    for job in assignment[group_index]:
        offset, count = query_shard(by_job[job]["query_count"], members, context.rank)
        teaching_offset, teaching_count = query_shard(by_job[job]["teaching_count"], members, context.rank)
        draw = {**by_job[job], "query_offset": offset, "query_count": count,
                "teaching_offset": teaching_offset, "teaching_count": teaching_count}
        task = draw["task"]
        tick = time.perf_counter()
        metric = engine.backward(draw)
        if int(metric["queries"]) != draw["query_count"] or int(metric["teaching_queries"]) != teaching_count:
            raise RuntimeError("supervised engine did not execute the registered FM exposure")
        rows.append({**metric, "step": step, "job_id": job, "task": task,
                     "suite": data.tasks[task].suite, "condition_index": draw["condition_index"],
                     "occurrence": draw["occurrence"], "K": 1,
                     "condition_weight": logical["condition_weight"] * count / logical["queries_per_condition"],
                     "task_weight": logical["task_weight"], "execution_rank": context.rank,
                     "condition_ranks": list(members),
                     "video_demos": list(draw["video_demos"]), "frames": draw["frames"], "scheduling_frames": costs[job],
                     "query_seed": draw["query_seed"], "query_offset": draw["query_offset"],
                     "queries": draw["query_count"], "seconds": time.perf_counter() - tick})
    return rows


def _update(engine, runtime, data, context, config, optimizer, scheduler, step):
    applied_lrs = {float(group["lr"]) for group in optimizer.param_groups}
    if len(applied_lrs) != 1:
        raise ValueError("Writer requires one applied learning rate across optimizer groups")
    applied_lr = next(iter(applied_lrs))
    phase = config.get("phase_continuation")
    if phase and not math.isclose(applied_lr, float(phase["fixed_lr"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("low-LR repair update would use a non-registered learning rate")
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
    norms["lr_applied"] = applied_lr
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


def _activate_phase_schedule(optimizer, scheduler, runtime, config, updates, *, initial_transition):
    phase = config.get("phase_continuation")
    if not phase:
        return
    parent = int(phase["parent_updates"])
    if updates < parent or (initial_transition and updates != parent):
        raise ValueError("low-LR repair restored an invalid global cursor")
    parameters = list(runtime.state.parameters())
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}:
        raise ValueError("low-LR repair optimizer parameter ownership changed")
    if set(optimizer.state) != set(owned):
        raise ValueError("low-LR repair requires complete AdamW state for every trainable parameter")
    steps = {int(state["step"].item()) for state in optimizer.state.values()}
    if steps != {updates}:
        raise ValueError("low-LR repair AdamW step cursor differs from the global cursor")
    if scheduler.last_epoch != updates:
        raise ValueError("low-LR repair scheduler cursor differs from the global cursor")
    fixed = float(phase["fixed_lr"])
    current = {float(group["lr"]) for group in optimizer.param_groups}
    expected = float(phase["parent_applied_lr"] if initial_transition else fixed)
    if len(current) != 1 or not math.isclose(next(iter(current)), expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("low-LR repair restored an unexpected applied learning rate")
    if initial_transition:
        for group in optimizer.param_groups:
            group["lr"] = fixed
        scheduler._last_lr = [fixed for _ in optimizer.param_groups]
    elif any(not math.isclose(value, fixed, rel_tol=1e-10, abs_tol=1e-12) for value in current):
        raise ValueError("low-LR repair exact-resume did not preserve the fixed learning rate")


def _restore(args, context, runtime, data, optimizer, scheduler, config):
    parent = getattr(args, "phase_from", None) or getattr(args, "extend_from", None)
    checkpoint = args.resume or parent
    if not checkpoint:
        return 0, 0
    if not parent and checkpoint.resolve().parent.parent != args.output.resolve():
        raise ValueError("exact-resume checkpoint must belong to its original run root")
    restored = {}
    updates, metrics_rows = load_ecp_checkpoint(
        checkpoint=checkpoint, stage=STAGE, context=context, model=runtime.state,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA, restored_state=restored,
        allow_world_size_change=bool(getattr(args, "allow_topology_change", False)),
    )
    if restored["training_state"] != _training_state(config, updates):
        raise ValueError("supervised checkpoint stage/update/data contract changed")
    phase_parent = getattr(args, "phase_from", None)
    data.restore_sampler(restored["sampler_state"], extend_completed=(parent is not None and phase_parent is None),
                         extend_from_step=(LOW_LR_REPAIR["parent_updates"] if phase_parent else None))
    if data.sampler_state()["next_step"] != updates or scheduler.last_epoch != updates:
        raise ValueError("sampler, scheduler and optimizer-update cursors differ")
    _activate_phase_schedule(optimizer, scheduler, runtime, config, updates,
                             initial_transition=phase_parent is not None)
    if context.is_main:
        if getattr(args, "allow_topology_change", False):
            transition = restored.get("topology_resume", {
                "checkpoint_world_size": context.world_size,
                "current_world_size": context.world_size,
                "checkpoint_rng_ranks": list(range(context.world_size)),
                "fresh_seeded_ranks": [],
            })
            append_jsonl(args.output / "topology_transitions.jsonl", {
                "schema_version": TOPOLOGY_TRANSITION_SCHEMA,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_macro": updates,
                **transition,
                "preserved": {
                    "logical_batch": _logical_batch(config),
                    "event_plan": "registered immutable dynamic events",
                    "optimizer_scheduler": "restored checkpoint trainer state",
                    "task_weighting": "one grouped update over four equally weighted tasks",
                },
            })
        if parent:
            inherit_history(checkpoint, args.output)
        reconcile_metrics(args.output / "exposures.jsonl", updates, metrics_rows, cursor_key="step", packet_label="exposures")
        reconcile_metrics(args.output / "metrics.jsonl", updates, updates, cursor_key="step", packet_label="metrics")
        if args.mode == "formal":
            nodes = config["evidence"]["supervised_validation"]["optimizer_updates"]
            count = sum(node <= updates for node in nodes) * len(data.tasks)
            if count or (args.output / "diagnostics.jsonl").exists():
                reconcile_metrics(args.output / "diagnostics.jsonl", updates, count, cursor_key="step", packet_label="diagnostics")
    return updates, metrics_rows


def _require_topology_resume(args, config):
    requested = bool(getattr(args, "allow_topology_change", False))
    if requested and (
        not getattr(args, "resume", None)
        or getattr(args, "extend_from", None)
        or getattr(args, "phase_from", None)
        or not config.get("training_control")
        or config.get("continuation")
        or config.get("phase_continuation")
    ):
        raise ValueError("physical topology transition requires an ordinary dynamic --resume")
    return requested


def _record_iteration(args, context, config, rows, norms, updates, metrics_rows, seconds, scheduler):
    packet = {"rank": context.rank, "rows": rows,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30}
    packets = _gather(packet, context)
    gathered = merge_condition_rows([row for packet in packets for row in packet["rows"]])
    metrics_rows += len(gathered)
    if context.is_main:
        for row in gathered:
            append_jsonl(args.output / "exposures.jsonl", row)
        metric = {
            "step": updates, "global_step": updates, "optimizer_updates": updates,
            "seconds": seconds,
            "mean_flow_loss": sum(r["flow_loss"] * r["condition_weight"] for r in gathered),
            "mean_teaching_loss": sum(r["teaching_loss"] * r["task_weight"] for r in gathered),
            "mean_total_loss": sum(r["flow_loss"] * r["condition_weight"]
                                   + r["teaching_loss"] * r["teaching_weight"] for r in gathered),
            **norms, "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
            "condition_exposures": metrics_rows, "task_exposures": updates * 4,
            "supervised_queries": updates * _logical_batch(config)["queries_per_update"],
            "teaching_queries": updates * _logical_batch(config)["teaching_queries_per_update"],
            "total_queries": updates * _logical_batch(config)["total_queries_per_update"],
            "rank_memory": [{key: value for key, value in packet.items() if key != "rows"} for packet in packets],
            "peak_allocated_gib": max(packet["peak_allocated_gib"] for packet in packets),
            "peak_reserved_gib": max(packet["peak_reserved_gib"] for packet in packets),
        }
        if config.get("phase_continuation"):
            metric["phase_step"] = updates - int(config["phase_continuation"]["parent_updates"])
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
    if config.get("training_control"):
        stop = args.stop_after_step
        if type(stop) is not int or stop <= 0:
            raise ValueError("dynamic training requires an explicit positive --stop-after-step")
        interval = config["training_control"]["checkpoint_interval"]
        nodes = tuple(range(interval, stop + 1, interval))
        if supplied is not None and tuple(map(int, supplied.split(","))) != nodes:
            raise ValueError("dynamic checkpoint nodes must follow the registered interval")
        return nodes
    nodes = tuple(config["evidence"]["checkpoint_updates"]) if supplied is None else tuple(map(int, supplied.split(",")))
    _validate_checkpoint_nodes(nodes, allow_empty=args.mode != "formal")
    return nodes


def _segment_limit(args, config):
    nodes = _checkpoint_nodes(args, config)
    stop = args.stop_after_step if args.stop_after_step is not None else (nodes[-1] if nodes else None)
    if type(stop) is not int or stop <= 0:
        raise ValueError("smoke/profile without registered nodes needs an explicit positive --stop-after-step")
    if config.get("training_control"):
        if args.mode == "formal" and stop % config["training_control"]["validation_interval"]:
            raise ValueError("formal dynamic segments must end at a complete validation boundary")
        return stop
    if args.mode == "formal" and stop != nodes[-1]:
        raise ValueError("formal segment must stop at the last registered checkpoint node")
    if stop > config["data"]["maximum_updates"]:
        raise ValueError(f"segment exceeds the registered {config['data']['maximum_updates']}-update event budget")
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
        checkpoint = (getattr(args, "resume", None) or getattr(args, "phase_from", None)
                      or getattr(args, "extend_from", None))
        print(json.dumps({"segment_start": updates, "segment_stop": stop, "checkpoint_updates": nodes,
                          "global_step": updates,
                          "phase_step": (updates - LOW_LR_REPAIR["parent_updates"]
                                         if config.get("phase_continuation") else None),
                          "resume": str(checkpoint) if checkpoint else None}), flush=True)
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
            "teaching_queries": updates * _logical_batch(config)["teaching_queries_per_update"],
            "total_queries": updates * _logical_batch(config)["total_queries_per_update"],
            "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered held-action and paired closed-loop evidence",
        })


def run(args: argparse.Namespace) -> None:
    from ember.writer.supervised import SupervisedEngine

    config = _config(args.config)
    require_continuation_start(args, config)
    allow_topology_change = _require_topology_resume(args, config)
    if args.mode == "formal" and (config["status"] != "registered_video_teaching_learning"
                                  or config["evidence"]["profile_registration"]["status"] != "complete"):
        raise ValueError("formal learning needs the post-profile checkpoint and exposure registration")
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal supervised training requires a clean pushed detached worktree")
    stop = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    condition_rank_groups(context.world_size)
    if (args.mode == "formal" and not allow_topology_change
            and context.world_size != config["evidence"]["profile_registration"]["world_size"]):
        raise ValueError("formal video-teaching training requires its registered profiled topology")
    execution_config, microbatches = _execution_config(args, config, context)
    if context.is_main:
        print(json.dumps({"physical_policy_microbatches": microbatches,
                          "logical_batch": _logical_batch(config)}), flush=True)
    torch.set_num_threads(int(args.cpu_threads))
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    start = time.perf_counter()
    data = WriterTrainingData(args.asset_root, config["data"],
                              camera_view=config["observer"]["camera_view"], planned_updates=stop)
    if context.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        _publish_event_plan(args, data.event_plan())
    runtime = build_runtime(args.asset_root, config, context.device)
    runtime.state.train()
    optimizer, scheduler = _optimization(runtime.state, config)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    frame_parallel_group = initialize_condition_group(context)
    contract = _run_contract(args, context, config, runtime, state)
    if context.is_main:
        if getattr(args, "phase_from", None):
            prepare_phase_continuation(args, contract)
        elif getattr(args, "extend_from", None):
            prepare_continuation(args, contract)
        _publish_contract(args.output / "run_contract.json", contract, resume=args.resume is not None,
                          allow_topology_change=allow_topology_change)
    barrier(context)
    cursors = _restore(args, context, runtime, data, optimizer, scheduler, config)
    updates, _ = cursors
    if updates >= stop:
        raise ValueError("supervised segment has no remaining registered updates")
    cache = VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"]))
    engine = SupervisedEngine(runtime, data, cache, context, execution_config,
                              frame_parallel_group=frame_parallel_group)
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
    parser.add_argument("--allow-topology-change", action="store_true",
                        help="allow an ordinary dynamic resume to use a changed physical topology")
    parser.add_argument("--extend-from", type=Path, help="complete parent1500 state for the registered 2100 continuation")
    parser.add_argument("--phase-from", type=Path,
                        help="complete formal N1800 state for the registered low-LR repair phase")
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
