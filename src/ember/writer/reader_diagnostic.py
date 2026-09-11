"""Bounded train-task readout fitting on frozen native queries and video memory.

This probe cannot update or export a Writer. Its auxiliary-only artifact is not
a deployment checkpoint, and its FM scores cannot qualify video causality.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.learning_data import WriterTrainingData
from ember.writer.materialization import frozen_authority, inspect_writer_checkpoint
from ember.writer.native import autocast
from ember.writer.runtime import FrozenVideoPrefixCache, build_runtime

SEED = 20260912
REPO_ROOT = Path(__file__).resolve().parents[3]


@torch.no_grad()
def _native_queries(runtime, state, raw, trace, device, microbatch):
    batch = runtime.processor.training_batch(raw)
    count = len(trace["action_demos"])
    tensors = {name: [] for name in ("query", "source", "target")}
    student_loss, owner = 0.0, NativeFlowPrediction(runtime.policy)
    for start in range(0, count, microbatch):
        stop = min(count, start + microbatch)
        part = {k: v[start:stop] if isinstance(v, torch.Tensor) and v.ndim and len(v) == count else v
                for k, v in batch.items()}
        sample = flow_sample(runtime.policy, part, seed=trace["policy_rng_seed"], device=device,
                             random_batch=trace["policy_random_batch_size"], offset=start)
        with autocast(device):
            source, query = owner(sample)
            student, _ = torch.func.functional_call(
                owner, {"policy." + k: v for k, v in state.items()}, (sample,), strict=False)
        for key, value in (("query", query), ("source", source), ("target", sample.target)):
            tensors[key].append(value.detach().cpu())
        student_loss += float(mean_velocity_loss(student, sample.target, sample.action_width)) * (stop - start) / count
    return {**{k: torch.cat(v) for k, v in tensors.items()}, "action_width": sample.action_width,
            "student_loss": student_loss}


@torch.no_grad()
def _cache_condition(runtime, data, cache, task, role, config, device, smoke):
    held = config["evidence"]["supervised_validation"]
    pool = data.video_pool if role == "support" else tuple(held["teacher_video_pool"])
    demo = pool[task % len(pool)]
    condition = cache.condition(task, (demo,))
    responses = runtime.observer.responses(condition)
    inputs = runtime.observer.writer_arguments(condition)
    with autocast(device):
        videos = runtime.state.writer.encode(responses, *inputs)
        state = runtime.state.writer.decode(videos, inputs[0])
        memory, _, prior = runtime.state.writer.memory(videos, inputs[0])
    if role == "support":
        raw, trace = data.action_batch(task, 0, (demo,), query_seed=SEED + task,
                                       query_count=8 if smoke else 64)
    else:
        raw, trace = data.diagnostic_batch(task, seed=held["seed"] + task, count=8 if smoke else 32)
    result = _native_queries(runtime, state, raw, trace, device, config["runtime"]["policy_microbatch"])
    return {**result, "memory": memory.cpu(), "prior": prior.cpu(), "task": task, "role": role,
            "suite": data.tasks[task].suite, "video_demo": demo, "trace": trace}


def _predict(reader, record, device, memory_record=None):
    memory_record = record if memory_record is None else memory_record
    with autocast(device):
        prediction = reader(record["query"].to(device), record["source"].to(device),
                            memory_record["memory"].to(device), memory_record["prior"].to(device))
    return mean_velocity_loss(prediction, record["target"].to(device), record["action_width"])


@torch.no_grad()
def _report(reader, records, device, epoch):
    rows = []
    for record in records:
        peers = sorted((r for r in records if r["role"] == record["role"] and r["suite"] != record["suite"]),
                       key=lambda r: r["task"])
        donor = peers[record["task"] % len(peers)]
        loss = float(_predict(reader, record, device))
        swapped = float(_predict(reader, record, device, donor))
        source = float(mean_velocity_loss(record["source"], record["target"], record["action_width"]))
        rows.append({"task": record["task"], "suite": record["suite"], "role": record["role"],
                     "queries": len(record["query"]), "reader_loss": loss, "source_loss": source,
                     "student_loss": record["student_loss"], "cross_task_memory_loss": swapped,
                     "cross_task_memory_donor": donor["task"]})
    means = {role: {key: statistics.mean(r[key] for r in rows if r["role"] == role)
                    for key in ("reader_loss", "source_loss", "student_loss", "cross_task_memory_loss")}
             for role in ("support", "held")}
    if not all(math.isfinite(value) for values in means.values() for value in values.values()):
        raise ValueError("reader diagnostic produced a non-finite FM score")
    return {"epoch": epoch, "means": means, "per_task": rows, "held_gradients": False}


def _fit_epoch(reader, records, optimizer, scheduler, config, device, epoch):
    rng = random.Random(SEED + epoch)
    groups = [[r for r in records if r["role"] == "support" and r["suite"] == suite] for suite in SUITE_ORDER]
    if len({len(group) for group in groups}) != 1 or not all(groups):
        raise ValueError("probe requires equally represented train suites")
    for group in groups:
        rng.shuffle(group)
    norms = []
    for conditions in zip(*groups, strict=True):
        optimizer.zero_grad(set_to_none=True)
        for record in conditions:
            (_predict(reader, record, device) / 4).backward()
        norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), config["optimization"]["grad_clip"],
                                             error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        scheduler.step()
    return {"epoch": epoch, "updates": len(groups[0]), "reader_grad_norm_mean": statistics.mean(norms)}


def _contract(args, run, checkpoint, repository, tasks, epochs):
    return {"schema_version": "ember_frozen_reader_fit_probe_v1", "mode": args.mode,
            "checkpoint": checkpoint, "git": repository, "source": run["source"],
            "task_ids": tasks, "seed": SEED, "epochs": epochs,
            "report_epochs": [0, 1] if args.mode == "smoke" else [0, 4, 16, 64],
            "support": {"video": "train pool indexed by task modulo pool length", "queries_per_task": 64,
                        "action_demos": run["config"]["data"]["action_demos"], "query_seed": "20260912 + task"},
            "held": {k: run["config"]["evidence"]["supervised_validation"][k]
                     for k in ("task_ids", "action_demos", "teacher_video_pool", "queries_per_task", "seed")},
            "optimization": {**run["config"]["optimization"], "joint_train_all_writer_modules": False,
                             "trainable": "existing execution reader only", "optimizer_state": "fresh",
                             "loss": "reader FM only", "reader_initialization": "loaded checkpoint reader",
                             "update_batch": "four equally weighted suites; each task once per epoch",
                             "fixed_cache_reuse": True},
            "frozen": ["source", "Meta", "video encoder", "Compiler", "LoRA decoder", "cached queries/memory"],
            "cross_task_memory": "post-hoc interface substitution; not raw-video intervention or causal qualification",
            "validation_test_used": False, "deployment_checkpoint": False, "checkpoint_selection": False,
            "smoke_override": "four train tasks, eight support/held queries each, one epoch" if args.mode == "smoke" else None}


def _prepare(args, device):
    run, checkpoint = inspect_writer_checkpoint(args.checkpoint)
    repository = git_state(REPO_ROOT)
    if args.mode == "diagnostic" and not frozen_authority(repository):
        raise ValueError("retained diagnostic requires clean pushed detached code")
    if (not run["config"]["auxiliary"]["enabled"] or checkpoint["macro"] != 200
            or run["model_config"]["process_mode"] != "ordered"):
        raise ValueError("registered probe requires an ordered auxiliary-enabled step200 checkpoint")
    config = copy.deepcopy(run["config"])
    data = WriterTrainingData(args.asset_root, config["data"],
                              camera_view=config["observer"].get("camera_view", "agentview"))
    tasks = sorted(data.tasks)
    if args.mode == "smoke":
        tasks = [next(t for t in tasks if data.tasks[t].suite == suite) for suite in SUITE_ORDER]
    epochs = 1 if args.mode == "smoke" else 64
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "run_contract.json", _contract(args, run, checkpoint, repository, tasks, epochs))
    runtime = build_runtime(args.asset_root, config, device)
    runtime.state.load_state_dict(load_file(str(args.checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.requires_grad_(False).eval()
    if any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError("the probe requires a frozen source")
    cache = FrozenVideoPrefixCache(runtime.observer, data, 0)
    return runtime, data, cache, tasks, config, epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-root", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("smoke", "diagnostic"), default="diagnostic")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime, data, cache, tasks, config, epochs = _prepare(args, device)
    records = []
    for role in ("support", "held"):
        for task in tasks:
            records.append(_cache_condition(runtime, data, cache, task, role, config, device, args.mode == "smoke"))
            print(json.dumps({"cached": len(records), "task": task, "role": role}), flush=True)
    write_json_atomic(args.output / "cache_provenance.json", [{k: r[k] for k in
        ("task", "suite", "role", "video_demo", "trace")} for r in records])
    reader = runtime.state.reader
    data.close()
    del runtime, data, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reader.requires_grad_(True).train()
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(reader.parameters(), lr=opt["lr"], betas=tuple(opt["betas"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(1., (n + 1) / opt["warmup_updates"]))
    reports = [_report(reader, records, device, 0)]
    for epoch in range(1, epochs + 1):
        row = _fit_epoch(reader, records, optimizer, scheduler, config, device, epoch)
        if epoch in (4, 16, 64) or epoch == epochs:
            reports.append(_report(reader, records, device, epoch))
            print(json.dumps({**row, "means": reports[-1]["means"]}), flush=True)
    write_json_atomic(args.output / "results.json", {"reports": reports, "goal_complete": False,
        "interpretation": "fixed-representation reader fitting only; no deployment or video causal qualification"})
    torch.save({"schema_version": "ember_reader_probe_state_v1", "reader": reader.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epochs": epochs,
                "optimizer_updates": epochs * len(tasks) // 4,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "deployment_checkpoint": False}, args.output / "reader_probe_state.pt")
    print(json.dumps({"status": "probe_complete", "mode": args.mode, "epochs": epochs,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                      if device.type == "cuda" else None}), flush=True)


if __name__ == "__main__":
    main()
