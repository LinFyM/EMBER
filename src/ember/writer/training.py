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
from ember.writer.conditional_contract import (
    CONFIG_SCHEMA as CONDITIONAL_CONFIG_SCHEMA,
    EXPERIMENT as CONDITIONAL_EXPERIMENT,
    UPDATE_VERSION as CONDITIONAL_UPDATE_VERSION,
    validate_config as _conditional_config,
)
from ember.writer.relational_contract import (
    EXPERIMENT as RELATIONAL_EXPERIMENT,
    validate_config as _relational_config,
)
from ember.writer.learning_data import (
    CONDITIONAL_EVENT_SCHEMA, RELATIONAL_EVENT_SCHEMA, EVENT_SCHEMA, MAIN_EVENT_QUERIES, TEACHING_EVENT_QUERIES,
    TASKS_PER_UPDATE, WriterTrainingData, query_allocation,
)
from ember.writer.continuation import (
    inherit_history, prepare_continuation,
    require_continuation_config, require_continuation_start, require_extended_prefix,
)
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import VideoConditionCache, build_runtime, require_architecture_identity
from ember.writer.task_execution import (
    condition_assignment, condition_rank_groups, cost_balanced_task_assignment, merge_condition_rows,
)
from ember.writer.support_slot_credit import (
    ForkTrainingData, attach_branch_completion, attach_branch_contract, execute_registered_event,
    inspect_fork, restore_fork,
)


CONFIG_SCHEMA = "ember_video_teaching_writer_config_v1"
RUN_SCHEMA = "ember_video_teaching_writer_run_v1"
STAGE = "video_teaching_writer_fresh"
TRAINING_SCHEMA = "ember_video_teaching_training_state_v1"
UPDATE_VERSION = "video_teaching_twelve_condition_joint_meta_v2"
TASK_MIXING_DECLARATION = {
    "kind": "coverage_task_mixing_fresh_v1", "conditions": TASKS_PER_UPDATE,
    "initialization": "fresh", "event_query_pools": [MAIN_EVENT_QUERIES, TEACHING_EVENT_QUERIES],
}
REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_TRANSITION_SCHEMA = "ember_writer_topology_transition_v1"


def _bounded_conditional(config):
    return config.get("experiment", {}).get("kind") in {CONDITIONAL_EXPERIMENT, RELATIONAL_EXPERIMENT}


def observer_mode_contract(model: dict[str, Any]) -> dict[str, str]:
    """Bind the ordered full-horizon and adjacent-content native reads."""
    require_architecture_identity(model)
    patches = 512 if model["camera_view"] == "dual" else 256
    return {"camera_view": model["camera_view"],
            "native_inputs": f"full{patches}_patch_content_and_repeated_full50_H_adjacent_E_reads",
            "horizon_read": "repeated_content_position_attention_over_all_50_raw_H_values",
            "video_order": "causal_RoPE_with_real_frame_positions_and_ordered_adjacent_roles"}


def _validate_dynamic_schedule(config):
    reference = read_json(REPO_ROOT / "configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json")
    if (config["source"] != reference["source"]
            or any(config["data"].get(key) != reference["data"].get(key) for key in ("task_ids", "protocol"))):
        raise ValueError("task-mixing source or audited train-task authority changed")
    opt = config["optimization"]
    if opt != reference["optimization"]:
        raise ValueError("task-mixing scientific contract preserves the original loss, Adam and LR schedule")
    if (any(type(opt.get(key)) is not int for key in
            ("warmup_updates", "tail_start_update", "tail_end_update", "decay_updates"))
            or not 0 <= opt["warmup_updates"] <= opt["tail_start_update"] < opt["tail_end_update"]
            or opt["decay_updates"] <= opt["tail_end_update"]
            or not 0 < opt["tail_final_ratio"] <= 1
            or config["model"]["camera_view"] != "agentview"
            or config["data"].get("teaching_episode") != "cross_episode"
            or not config["data"].get("protocol")):
        raise ValueError("dynamic Writer schedule or single-camera teaching contract changed")


def _query_contract(config):
    if config.get("experiment", {}).get("kind") == RELATIONAL_EXPERIMENT:
        if config["data"].get("event_schema_version") != RELATIONAL_EVENT_SCHEMA:
            raise ValueError("relation-support event schema changed")
        return query_allocation(config["data"], 0)
    if config.get("experiment", {}).get("kind") == CONDITIONAL_EXPERIMENT:
        if config["experiment"].get("arm_id") not in {"A_direct16", "B_language", "C_video_fm", "D_video_aux"}:
            raise ValueError("conditional compilation requires one of its four registered arms")
        if config["data"].get("event_schema_version") != CONDITIONAL_EVENT_SCHEMA:
            raise ValueError("conditional compilation event schema changed")
        return query_allocation(config["data"], 0)
    if config.get("experiment") != TASK_MIXING_DECLARATION:
        raise ValueError("canonical task-mixing scientific contract requires its explicit declaration")
    return query_allocation(config["data"], 0)


def _config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    validator = {RELATIONAL_EXPERIMENT: _relational_config,
                 CONDITIONAL_EXPERIMENT: _conditional_config}.get(config.get("experiment", {}).get("kind"))
    if validator is not None:
        return validator(config)
    require_continuation_config(config)
    expected_data = {
        "extra_meta_tasks": [], "frame_stride": 5, "include_last_frame": True,
        "tasks_per_update": TASKS_PER_UPDATE, "conditions_per_task": 1, "cardinalities": [1],
        "action_start_offset": 1, "query_alignment": "post_action_observation_future_control_v1",
        "version": EVENT_SCHEMA, "event_schema_version": EVENT_SCHEMA,
        "seed": 7, "sampler_seed": 20260721, "teacher_video_seed": 20260722,
        "teaching_seed": 20260919,
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
    _query_contract(config)
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
    conditional = _bounded_conditional(config)
    parameters = (tuple(parameter for parameter in state.parameters() if parameter.requires_grad)
                  if conditional else tuple(state.parameters()))
    if not parameters:
        raise ValueError("Writer parameterization has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(opt["lr"]), betas=tuple(opt["betas"]),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_multiplier(step, opt),
    )
    return optimizer, scheduler


def _execution_config(args, config, context):
    """Physical query chunks do not change the complete logical FM batch."""
    supplied = getattr(args, "policy_microbatches", None)
    conditional = _bounded_conditional(config)
    profile = config.get("evidence", {}).get("profile_registration", {})
    registered = profile.get("policy_microbatches") if conditional and profile.get("status") == "complete" else None
    batches = (list(map(int, registered)) if supplied is None and registered is not None else
               [int(config["runtime"]["policy_microbatch"])] * context.world_size
               if supplied is None else list(map(int, supplied.split(","))))
    if len(batches) != context.world_size or any(value <= 0 for value in batches):
        raise ValueError("physical microbatches need one positive value per rank")
    if registered is not None and batches != list(map(int, registered)):
        raise ValueError("formal physical policy microbatches must match the registered full-video profile")
    local = {**config, "runtime": {**config["runtime"], "policy_microbatch": batches[context.rank]}}
    return local, batches


def _event_plan_path(args):
    return args.output / "training_events.json"


def _publish_event_plan(args, events):
    path = _event_plan_path(args)
    parent_checkpoint = getattr(args, "extend_from", None)
    if parent_checkpoint:
        parent = parent_checkpoint.resolve().parent.parent / "training_events.json"
        require_extended_prefix(read_json(parent), events,
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
    conditional = _bounded_conditional(config)
    tasks_per_update = int(config["data"].get("tasks_per_update", TASKS_PER_UPDATE))
    parameterization = getattr(runtime, "parameterization", "video_writer")
    uses_video = getattr(runtime, "uses_video", True)
    deployment_inputs = {
        "direct_lora": [],
        "language_writer": ["exact task language"],
        "video_writer": ["exact task language", "complete internally ordered RGB video"],
    }[parameterization]
    trainable = ([(name, parameter) for name, parameter in runtime.state.named_parameters()
                  if parameter.requires_grad] if conditional else [])
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
            "within_condition": ("whole_video_and_query_prefixes_on_one_rank_no_frame_subgroup"
                                 if parameterization == "video_writer" else
                                 "one_task_parameterization_and_query_prefixes_on_one_rank"),
        },
        "training": {
            "writer_parameters": sum(p.numel() for p in runtime.state.writer.parameters()),
            "meta_parameters": sum(p.numel() for p in runtime.state.meta.parameters()),
            "vl_meta_parameters": sum(p.numel() for p in runtime.state.vl_meta.parameters()),
            "text_meta_parameters": sum(p.numel() for p in runtime.state.text_meta.parameters()),
            "source_trainable_parameters": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            **({"trainable_parameter_names": [name for name, _ in trainable],
                "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable)}
               if conditional else {}),
            "optimizer": ("parent AdamW state preserved" if
                          config.get("continuation") else "fresh AdamW")
                         + f"; one grouped functional update per {tasks_per_update} equally weighted tasks", "scaler": None,
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
            "deployment_inputs": deployment_inputs,
            "parameterization": parameterization,
            "frame_indices": ("real sampled positions for ordered Procedure" if uses_video else None),
            "execution_adapters": 1, "reading_meta_in_execution": False,
            "validation_test_gradients": False, "shuffled_reversed": False,
            "video_action_episodes": {"main": "cross_episode", "teaching": config["data"]["teaching_episode"]},
            "gradient_normalizer": 1.0,
            "objective": config["optimization"]["loss"],
            "training_only_actions": ("execution-query inputs and targets, loaded after task compilation"
                                      if conditional else
                                      "execution-query inputs and targets, loaded after RGB-language-only compilation"),
            "native_read": ("repeated full50 H and ordered adjacent E content; joint three-Meta replay"
                            if uses_video else
                            "exact native text-only language path" if parameterization == "language_writer" else
                            "none; shared direct identity-initialized full A/B parameters"),
            "complete_lora": ("shared eight-family full A/B heads from Core-conditioned centered Procedure AdaLN"
                              if uses_video else
                              "shared eight-family full A/B heads from language-only Core slots" if parameterization == "language_writer" else
                              "shared trainable full A/B parameters across all tasks"),
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
    main, teaching = _query_contract(config)
    tasks_per_update = int(config["data"].get("tasks_per_update", TASKS_PER_UPDATE))
    return {"tasks": tasks_per_update, "conditions_per_task": 1, "conditions": tasks_per_update, "K": 1,
            "queries_per_task": main, "queries_per_condition": main, "queries_per_update": tasks_per_update * main,
            "teaching_query_counts": list(teaching), "teaching_queries_per_update": sum(teaching),
            "total_queries_per_update": tasks_per_update * main + sum(teaching),
            "policy_random_batch_sizes": {"main": MAIN_EVENT_QUERIES, "teaching": TEACHING_EVENT_QUERIES},
            "teaching_weight": config["optimization"]["teaching_weight"],
            "task_weight": 1 / tasks_per_update, "condition_weight": 1 / tasks_per_update,
            "gradient_reduction": "SUM"}


def _condition_jobs(data, config, draws, step):
    by_job = {draw["job_id"]: draw for draw in draws}
    tasks = {draw["task"] for draw in draws}
    main, teaching = query_allocation(config["data"], step - 1)
    tasks_per_update = int(config["data"].get("tasks_per_update", TASKS_PER_UPDATE))
    if (len(draws) != tasks_per_update or set(by_job) != set(range(tasks_per_update))
            or len(tasks) != tasks_per_update or not tasks <= set(data.tasks)
            or any(draw["condition_index"] != 0 or len(draw["video_demos"]) != 1
                   or draw["query_count"] != main or draw["query_offset"] != 0
                   or draw["teaching_count"] != teaching[draw["job_id"]]
                   or draw["teaching_offset"] != 0 for draw in draws)):
        count = "four" if tasks_per_update == 4 else "twelve"
        raise ValueError(f"each update requires {count} distinct K1 conditions with the registered query allocation")
    return by_job


def _execute_step(engine, data, context, config, draws, step):
    logical = _logical_batch(config)
    condition_rank_groups(context.world_size)
    by_job = _condition_jobs(data, config, draws, step)
    engine.step = step
    jobs = tuple(by_job)
    uses_video = getattr(getattr(engine, "runtime", None), "uses_video", True)
    costs = {job: int(draw["frames"]) if uses_video else 1
             for job, draw in by_job.items()}
    assignment = condition_assignment(jobs, costs, world_size=context.world_size)
    rows = []
    for job in assignment[context.rank]:
        draw = by_job[job]
        task = draw["task"]
        tick = time.perf_counter()
        gate = draw.get("credit_gate", 1)
        metric = execute_registered_event(engine, data, draw, step)
        if int(metric["queries"]) != draw["query_count"] or int(metric["teaching_queries"]) != draw["teaching_count"]:
            raise RuntimeError("supervised engine did not execute the registered FM exposure")
        rows.append({**metric, "step": step, "job_id": job, "task": task,
                     "suite": data.tasks[task].suite, "condition_index": draw["condition_index"],
                     "occurrence": draw["occurrence"], "K": 1,
                     "condition_weight": logical["condition_weight"],
                     "task_weight": logical["task_weight"], "execution_rank": context.rank,
                     "condition_ranks": [context.rank],
                     "video_demos": list(draw["video_demos"]), "frames": draw["frames"], "scheduling_frames": costs[job],
                     "query_seed": draw["query_seed"], "query_offset": draw["query_offset"],
                     "queries": draw["query_count"], "seconds": time.perf_counter() - tick,
                     "credit_gate": gate,
                     "effective_main_weight": gate * logical["condition_weight"],
                     "effective_teaching_weight": gate * metric["teaching_weight"],
                     **({"source_event_plan": draw["source_event_plan"],
                         "source_event_index": draw["source_event_index"],
                         "branch_cursor": draw["branch_cursor"]}
                        if "source_event_plan" in draw else {})})
    return rows


def _update(engine, runtime, data, context, config, optimizer, scheduler, step):
    applied_lrs = {float(group["lr"]) for group in optimizer.param_groups}
    if len(applied_lrs) != 1:
        raise ValueError("Writer requires one applied learning rate across optimizer groups")
    applied_lr = next(iter(applied_lrs))
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
    conditional = _bounded_conditional(config)
    parameters = (tuple(parameter for parameter in runtime.state.parameters() if parameter.requires_grad)
                  if conditional else tuple(runtime.state.parameters()))
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


def _restore(args, context, runtime, data, optimizer, scheduler, config):
    parent = getattr(args, "extend_from", None)
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
    data.restore_sampler(restored["sampler_state"], extend_completed=parent is not None)
    if data.sampler_state()["next_step"] != updates or scheduler.last_epoch != updates:
        raise ValueError("sampler, scheduler and optimizer-update cursors differ")
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
                    "task_weighting": f"one grouped update over {config['data'].get('tasks_per_update', TASKS_PER_UPDATE)} equally weighted tasks",
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
        or not config.get("training_control")
        or config.get("continuation")
    ):
        raise ValueError("physical topology transition requires an ordinary dynamic --resume")
    return requested


def _record_iteration(args, context, config, rows, norms, updates, metrics_rows, seconds, scheduler):
    packet = {"rank": context.rank, "rows": rows,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30}
    packets = _gather(packet, context)
    main, teaching = query_allocation(config["data"], updates - 1)
    gathered = merge_condition_rows([row for packet in packets for row in packet["rows"]],
                                   main_queries=main, teaching_queries=teaching,
                                   tasks_per_update=int(config["data"].get("tasks_per_update", TASKS_PER_UPDATE)))
    metrics_rows += len(gathered)
    if context.is_main:
        for row in gathered:
            append_jsonl(args.output / "exposures.jsonl", row)
        metric = {
            "step": updates, "global_step": updates, "optimizer_updates": updates,
            "seconds": seconds,
            "mean_flow_loss": sum(r["flow_loss"] * r["effective_main_weight"] for r in gathered),
            "mean_teaching_loss": sum(r["teaching_loss"] * r["task_weight"] * r["credit_gate"]
                                      for r in gathered),
            "mean_total_loss": sum(r["flow_loss"] * r["effective_main_weight"]
                                   + r["teaching_loss"] * r["effective_teaching_weight"] for r in gathered),
            **norms, "lr_next": scheduler.get_last_lr()[0], "exposures": metrics_rows,
            "condition_exposures": metrics_rows,
            "task_exposures": updates * config["data"].get("tasks_per_update", TASKS_PER_UPDATE),
            "supervised_queries": updates * _logical_batch(config)["queries_per_update"],
            "teaching_queries": updates * _logical_batch(config)["teaching_queries_per_update"],
            "total_queries": updates * _logical_batch(config)["total_queries_per_update"],
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
    if getattr(args, "support_slot_arm", None) is not None:
        nodes = (1160, 1183)
        if supplied is not None and tuple(map(int, supplied.split(","))) != nodes:
            raise ValueError("support-slot fork saves only its first-slot and terminal full checkpoints")
        return nodes
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
    if getattr(args, "support_slot_arm", None) is not None:
        if stop not in nodes:
            raise ValueError("support-slot fork ends only at macro1160 or macro1183")
        return stop
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


def _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop, start,
                 *, support_slot_scope=None):
    updates, metrics_rows = cursors
    nodes = _checkpoint_nodes(args, config)
    # A resumed segment keeps its original registered nodes. The restored
    # cursor skips completed nodes while the loop retains the registered stop.
    if context.is_main:
        checkpoint = getattr(args, "resume", None) or getattr(args, "extend_from", None)
        print(json.dumps({"segment_start": updates, "segment_stop": stop, "checkpoint_updates": nodes,
                          "global_step": updates,
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
                sampler_state=data.sampler_state(),
                training_state=(support_slot_scope.training_state(config, updates)
                                if support_slot_scope is not None else _training_state(config, updates)),
            )
    barrier(context)
    if context.is_main:
        completed = {
            "schema_version": RUN_SCHEMA, "status": "segment_complete", "mode": args.mode,
            "optimizer_updates": updates, "exposures": metrics_rows,
            "condition_exposures": metrics_rows,
            "task_exposures": updates * config["data"].get("tasks_per_update", TASKS_PER_UPDATE),
            "supervised_queries": updates * _logical_batch(config)["queries_per_update"],
            "teaching_queries": updates * _logical_batch(config)["teaching_queries_per_update"],
            "total_queries": updates * _logical_batch(config)["total_queries_per_update"],
            "seconds": time.perf_counter() - start,
            "scientific_qualification": False, "next": "registered held-action and paired closed-loop evidence",
        }
        if support_slot_scope is not None:
            attach_branch_completion(completed, data, support_slot_scope)
        write_json_atomic(args.output / "completion.json", completed)


def run(args: argparse.Namespace) -> None:
    from ember.writer.supervised import SupervisedEngine

    config = _config(args.config)
    support_slot_scope = None
    if getattr(args, "support_slot_arm", None) is not None:
        support_slot_scope = inspect_fork(args.support_slot_arm, config, args.output, mode=args.mode)
        if (getattr(args, "extend_from", None) or getattr(args, "phase_from", None)
                or getattr(args, "allow_topology_change", False)):
            raise ValueError("registered support-slot fork preserves world2 and uses only its own branch resume")
    conditional = _bounded_conditional(config)
    if conditional and support_slot_scope is None:
        if getattr(args, "extend_from", None) or getattr(args, "phase_from", None):
            raise ValueError("conditional compilation arms are fresh and cannot inherit a checkpoint")
        if getattr(args, "allow_topology_change", False):
            raise ValueError("conditional compilation exact-resume preserves its registered topology")
        allow_topology_change = False
        if args.mode == "formal" and config["evidence"]["profile_registration"].get("status") != "complete":
            raise ValueError("formal conditional training needs completed smoke/profile registration")
        if args.mode == "smoke" and (args.stop_after_step is None or args.stop_after_step > 4):
            raise ValueError("conditional arm smoke is limited to four consecutive macro updates")
        if args.mode == "profile" and (args.stop_after_step is None or args.stop_after_step > 1):
            raise ValueError("conditional full-video differentiable profile is limited to one disposable macro update")
    elif support_slot_scope is None:
        require_continuation_start(args, config)
        allow_topology_change = _require_topology_resume(args, config)
    else:
        allow_topology_change = False
    if args.mode == "formal" and not conditional and (
            config["status"] != "registered_video_teaching_learning"
            or config["evidence"]["profile_registration"]["status"] != "complete"):
        raise ValueError("formal learning needs the post-profile checkpoint and exposure registration")
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal supervised training requires a clean pushed detached worktree")
    stop = _segment_limit(args, config)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    condition_rank_groups(context.world_size)
    if support_slot_scope is not None and context.world_size != 2:
        raise ValueError("support-slot fork must keep the parent's two physical ranks")
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
    if support_slot_scope is not None:
        data = ForkTrainingData(args.asset_root, config, support_slot_scope)
    else:
        data = WriterTrainingData(args.asset_root, config["data"],
                                  camera_view=config["observer"]["camera_view"], planned_updates=stop,
                                  use_videos=config.get("experiment", {}).get("parameterization", "video_writer") == "video_writer")
    if context.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        _publish_event_plan(args, data.event_plan())
    runtime = build_runtime(args.asset_root, config, context.device)
    runtime.state.train()
    optimizer, scheduler = _optimization(runtime.state, config)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    contract = _run_contract(args, context, config, runtime, state)
    if support_slot_scope is not None:
        attach_branch_contract(contract, support_slot_scope)
    if context.is_main:
        if getattr(args, "extend_from", None):
            prepare_continuation(args, contract)
        _publish_contract(args.output / "run_contract.json", contract, resume=args.resume is not None,
                          allow_topology_change=allow_topology_change)
    barrier(context)
    if support_slot_scope is not None:
        cursors = restore_fork(args, context, runtime, data, optimizer, scheduler, config, support_slot_scope)
    else:
        cursors = _restore(args, context, runtime, data, optimizer, scheduler, config)
    updates, _ = cursors
    if updates >= stop:
        raise ValueError("supervised segment has no remaining registered updates")
    cache = VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"]))
    engine = SupervisedEngine(runtime, data, cache, context, execution_config)
    barrier(context)
    try:
        _run_segment(args, context, config, runtime, data, engine, optimizer, scheduler, cursors, stop, start,
                     support_slot_scope=support_slot_scope)
    finally:
        data.close()
        if context.world_size > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/libero_24_8_8_coverage_v1/writer_task_diversity.json")
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
    parser.add_argument("--support-slot-arm", choices=("KEEP77", "SWAP76", "DROP77"),
                        help="registered controlled fork from C_S00@1155")
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())
