"""Bounded native intermediate readout fitting; never a deployable Writer.

The source and main200 video representation are frozen. Only a fresh hidden
residual reader learns; fixed execution prefix KV stays in CPU memory.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache

from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.function_reader import ExecutionVideoReader
from ember.writer.learning_data import WriterTrainingData
from ember.writer.materialization import frozen_authority, inspect_writer_checkpoint
from ember.writer.native import autocast
from ember.writer.runtime import FrozenVideoPrefixCache, build_runtime

SEED, SITE = 20260912, 9
ROOT = Path(__file__).resolve().parents[3]


class CachedPrediction(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, noisy_action, flow_time):
        captured = []
        handle = self.policy.model.action_out_proj.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            velocity = self.policy.model.denoise_step(padding, cache, noisy_action, flow_time)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("native prediction must use one real action output")
        return velocity, captured[0]


@contextmanager
def native_reader_scope(policy, reader, memory, prior):
    layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[SITE].input_layernorm
    calls = []

    def inject(module, args, output):
        hidden, gate = output
        if hidden.ndim != 3 or hidden.shape[1:] != (50, 1024):
            raise ValueError("intermediate read lost the complete native execution horizon")
        calls.append(True)
        return reader(hidden, hidden, memory, prior).to(hidden.dtype), gate

    handle = layer.register_forward_hook(inject)
    try:
        yield
        if len(calls) != 1:
            raise RuntimeError("registered intermediate site must execute exactly once")
    finally:
        handle.remove()


def _arguments(record, start, stop, device):
    cache = DynamicCache(tuple((key[start:stop].to(device), value[start:stop].to(device), window)
                               for key, value, window in record["layers"]))
    return (record["padding"][start:stop].to(device), cache,
            record["noisy_action"][start:stop].to(device), record["flow_time"][start:stop].to(device))


@torch.no_grad()
def _cache_queries(runtime, state, reference, memory, prior, raw, trace, device):
    batch, chunks = runtime.processor.training_batch(raw), []
    count, owner = len(trace["action_demos"]), CachedPrediction(runtime.policy)
    names = ("source_loss", "student_loss", "legacy_reader_loss", "full_source_loss")
    losses = dict.fromkeys(names, 0.)
    for start in range(0, count, 8):
        stop = min(count, start + 8)
        part = {k: v[start:stop] if isinstance(v, torch.Tensor) and v.ndim and len(v) == count else v
                for k, v in batch.items()}
        sample = flow_sample(runtime.policy, part, seed=trace["policy_rng_seed"], device=device,
                             random_batch=trace["policy_random_batch_size"], offset=start)
        prefix = prepare_execution_policy_prefix(runtime.policy, part)
        cache = prepare_prefix_kv_cache(runtime.policy, prefix)
        actions, noise, flow_time = sample.arguments[-3:]
        noisy = flow_time[:, None, None] * noise + (1 - flow_time[:, None, None]) * actions
        args = (prefix.padding, cache, noisy, flow_time)
        with autocast(device):
            source, query = owner(*args)
            student, _ = torch.func.functional_call(
                owner, {"policy." + k: v for k, v in state.items()}, args, strict=False)
            legacy = reference(query, source, memory, prior)
            full, _ = NativeFlowPrediction(runtime.policy)(sample)
        for name, prediction in zip(names, (source, student, legacy, full), strict=True):
            losses[name] += float(mean_velocity_loss(prediction, sample.target, sample.action_width)) * (stop-start)/count
        chunks.append({"padding": prefix.padding.cpu(), "noisy_action": noisy.cpu(),
                       "flow_time": flow_time.cpu(), "target": sample.target.cpu(),
                       "layers": tuple((k.cpu(), v.cpu(), window) for k, v, window in cache)})
    if (not all(math.isfinite(v) for v in losses.values())
            or abs(losses["source_loss"] - losses["full_source_loss"]) > 1e-3):
        raise ValueError("cached native source FM disagrees materially with official full forward")
    arrays = {k: torch.cat([c[k] for c in chunks]) for k in ("padding", "noisy_action", "flow_time", "target")}
    layers = tuple((torch.cat([c["layers"][i][0] for c in chunks]),
                    torch.cat([c["layers"][i][1] for c in chunks]), chunks[0]["layers"][i][2])
                   for i in range(len(chunks[0]["layers"])))
    return {**arrays, "layers": layers, "action_width": sample.action_width, **losses}


@torch.no_grad()
def _cache_condition(runtime, reference, data, cache, task, role, config, device, smoke):
    held = config["evidence"]["supervised_validation"]
    pool = data.video_pool if role == "support" else tuple(held["teacher_video_pool"])
    demo = pool[task % len(pool)]
    condition = cache.condition(task, (demo,))
    responses, inputs = runtime.observer.responses(condition), runtime.observer.writer_arguments(condition)
    with autocast(device):
        videos = runtime.state.writer.encode(responses, *inputs)
        state = runtime.state.writer.decode(videos, inputs[0])
        memory, _, prior = runtime.state.writer.memory(videos, inputs[0])
    if role == "support":
        raw, trace = data.action_batch(task, 0, (demo,), query_seed=SEED + task, query_count=8 if smoke else 64)
    else:
        raw, trace = data.diagnostic_batch(task, seed=held["seed"] + task, count=8 if smoke else 32)
    record = _cache_queries(runtime, state, reference, memory, prior, raw, trace, device)
    return {**record, "memory": memory.cpu(), "prior": prior.cpu(), "task": task, "role": role,
            "suite": data.tasks[task].suite, "video_demo": demo, "trace": trace}


def _predict(owner, reader, record, device, microbatch, *, backward_weight=None, donor=None):
    condition, count, total = record if donor is None else donor, len(record["target"]), 0.
    memory, prior = condition["memory"].to(device), condition["prior"].to(device)
    for start in range(0, count, microbatch):
        stop = min(count, start + microbatch)
        args = _arguments(record, start, stop, device)
        with autocast(device), native_reader_scope(owner.policy, reader, memory, prior):
            prediction, _ = owner(*args)
            loss = mean_velocity_loss(prediction, record["target"][start:stop].to(device), record["action_width"])
            if backward_weight is not None:
                (loss * ((stop - start) / count * backward_weight)).backward()
        total += float(loss.detach()) * (stop - start) / count
    if not math.isfinite(total):
        raise ValueError("native readout produced non-finite FM")
    return total


@torch.no_grad()
def _report(owner, reader, records, device, microbatch, epoch, *, final=False):
    rows = []
    for record in records:
        row = {k: record[k] for k in ("task", "suite", "role", "source_loss", "student_loss", "legacy_reader_loss")}
        row.update(queries=len(record["target"]), reader_loss=_predict(owner, reader, record, device, microbatch))
        if epoch == 0 and abs(row["reader_loss"] - row["source_loss"]) > 1e-3:
            raise ValueError("zero hidden residual did not preserve the source function")
        if final and record["role"] == "held":
            peers = sorted((r for r in records if r["role"] == "held" and r["suite"] != record["suite"]),
                           key=lambda r: r["task"])
            donor = peers[record["task"] % len(peers)]
            row.update(cross_task_memory_loss=_predict(owner, reader, record, device, microbatch, donor=donor),
                       cross_task_memory_donor=donor["task"])
        rows.append(row)
    means = {role: {k: statistics.mean(r[k] for r in rows if r["role"] == role)
                    for k in ("reader_loss", "source_loss", "student_loss", "legacy_reader_loss")}
             for role in ("support", "held")}
    return {"epoch": epoch, "means": means, "per_task": rows, "held_gradients": False}


def _profile(owner, reader, records, device):
    record = max((r for r in records if r["role"] == "support"), key=lambda r: len(r["memory"]))
    rows = []
    for microbatch in (8, 16, 32, 64):
        if microbatch > len(record["target"]):
            continue
        reader.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            loss = _predict(owner, reader, record, device, microbatch, backward_weight=.25)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            rows.append({"microbatch": microbatch, "seconds": seconds, "queries_per_second": len(record["target"])/seconds,
                         "loss": loss, "peak_gib": torch.cuda.max_memory_allocated(device)/2**30})
        except torch.cuda.OutOfMemoryError:
            rows.append({"microbatch": microbatch, "oom": True})
        reader.zero_grad(set_to_none=True)
    usable = [r for r in rows if not r.get("oom")]
    if not usable:
        raise RuntimeError("no profiled native readout microbatch fits")
    return {"task": record["task"], "memory_tokens": len(record["memory"]), "rows": rows,
            "selected_microbatch": max(usable, key=lambda r: r["queries_per_second"])["microbatch"]}


def _fit_epoch(owner, reader, records, optimizer, scheduler, config, device, microbatch, epoch):
    rng = random.Random(SEED + epoch)
    groups = [[r for r in records if r["role"] == "support" and r["suite"] == suite] for suite in SUITE_ORDER]
    if len({len(group) for group in groups}) != 1 or not all(groups):
        raise ValueError("native readout requires equal train suite weights")
    for group in groups:
        rng.shuffle(group)
    norms, start = [], time.perf_counter()
    for conditions in zip(*groups, strict=True):
        optimizer.zero_grad(set_to_none=True)
        for record in conditions:
            _predict(owner, reader, record, device, microbatch, backward_weight=.25)
        norms.append(float(torch.nn.utils.clip_grad_norm_(reader.parameters(), config["optimization"]["grad_clip"],
                                                         error_if_nonfinite=True)))
        optimizer.step()
        scheduler.step()
    return {"epoch": epoch, "updates": len(groups[0]), "seconds": time.perf_counter()-start,
            "reader_grad_norm_mean": statistics.mean(norms)}


def _contract(args, run, checkpoint, repository, tasks, config, epochs):
    counts = (8, 8) if epochs == 1 else (64, 32)
    return {"schema_version": "ember_native_intermediate_readout_probe_v1", "mode": args.mode,
                "checkpoint": checkpoint, "source": run["source"], "git": repository, "task_ids": tasks,
                "seed": SEED, "epochs": epochs, "reports": [0, 1] if epochs == 1 else [0, 4, 16, 32],
                "site": SITE, "reader_initialization": "fresh zero-output hidden residual",
                "legacy_reader": str(args.legacy_reader), "config": config,
                "support_queries_per_task": counts[0], "held_queries_per_task": counts[1],
                "cache": "frozen real execution prefix KV and fixed E in CPU RAM only",
                "physical_microbatch": "profile 8/16/32/64 on longest fixed support memory; no optimizer updates",
                "trainable": "new intermediate reader only", "held_gradients": False,
                "validation_test_used": False, "deployment_checkpoint": False, "checkpoint_selection": False}


def _prepare(args, device):
    run, checkpoint = inspect_writer_checkpoint(args.checkpoint)
    repository = git_state(ROOT)
    if args.mode == "diagnostic" and not frozen_authority(repository):
        raise ValueError("retained diagnostic requires clean pushed detached code")
    if (checkpoint["macro"] != 200 or run["model_config"]["process_mode"] != "ordered"
            or run["config"]["auxiliary"]["distill_max"] != .25):
        raise ValueError("registered native probe requires original main200")
    config = copy.deepcopy(run["config"])
    data = WriterTrainingData(args.asset_root, config["data"], camera_view=config["observer"].get("camera_view", "agentview"))
    tasks = sorted(data.tasks)
    if args.mode == "smoke":
        tasks = [next(t for t in tasks if data.tasks[t].suite == suite) for suite in SUITE_ORDER]
    runtime = build_runtime(args.asset_root, config, device)
    runtime.state.load_state_dict(load_file(str(args.checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.requires_grad_(False).eval()
    reference = copy.deepcopy(runtime.state.reader)
    old = torch.load(args.legacy_reader, map_location=device, weights_only=False)
    reference.load_state_dict(old["reader"], strict=True)
    if old["epochs"] != 64 or any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError("registered source/reference freeze contract failed")
    epochs = 1 if args.mode == "smoke" else 32
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "run_contract.json", _contract(args, run, checkpoint, repository, tasks, config, epochs))
    return runtime, reference, data, tasks, config, epochs


def _smoke_gradients(owner, reader, record, device, microbatch):
    reader.zero_grad(set_to_none=True)
    _predict(owner, reader, record, device, microbatch, backward_weight=.25)
    names = ("output.weight", "query_projection.weight", "read.value.weight")
    parameters = dict(reader.named_parameters())
    norms = {name: float(parameters[name].grad.norm()) if parameters[name].grad is not None else 0. for name in names}
    if not all(math.isfinite(value) and value > 0 for value in norms.values()):
        raise RuntimeError("native hidden read did not receive its real control gradient")
    if any(p.grad is not None for p in owner.policy.parameters()):
        raise RuntimeError("native probe accumulated a source parameter gradient")
    reader.zero_grad(set_to_none=True)
    return {"reader_gradient_norms": norms, "source_gradients": False, "optimizer_updates_added": 0,
            "scope_checks": "one native site call per forward; handles removed in finally"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-root", "checkpoint", "legacy-reader", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("smoke", "diagnostic"), default="diagnostic")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime, reference, data, tasks, config, epochs = _prepare(args, device)
    cache, records = FrozenVideoPrefixCache(runtime.observer, data, 0), []
    for role in ("support", "held"):
        for task in tasks:
            records.append(_cache_condition(runtime, reference, data, cache, task, role, config, device, epochs == 1))
            print(json.dumps({"cached": len(records), "task": task, "role": role}), flush=True)
    provenance = [{k: r[k] for k in ("task", "suite", "role", "video_demo", "trace", "full_source_loss", "source_loss")}
                  for r in records]
    write_json_atomic(args.output / "cache_provenance.json", provenance)
    owner = CachedPrediction(runtime.policy).eval()
    data.close()
    del runtime, reference, data, cache
    torch.manual_seed(SEED)
    reader = ExecutionVideoReader(action_width=1024).to(device).train()
    profile = _profile(owner, reader, records, device)
    write_json_atomic(args.output / "profile.json", profile)
    microbatch = profile["selected_microbatch"]
    opt = config["optimization"]
    optimizer = torch.optim.AdamW(reader.parameters(), lr=opt["lr"], betas=tuple(opt["betas"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(1., (n+1)/opt["warmup_updates"]))
    reports = [_report(owner, reader, records, device, microbatch, 0)]
    for epoch in range(1, epochs+1):
        row = _fit_epoch(owner, reader, records, optimizer, scheduler, config, device, microbatch, epoch)
        if epoch in (4, 16, 32) or epoch == epochs:
            reports.append(_report(owner, reader, records, device, microbatch, epoch, final=epoch == epochs))
            write_json_atomic(args.output / "results.json", {"reports": reports, "goal_complete": False,
                "complete": epoch == epochs, "interpretation": "frozen-E native control readout, not video qualification"})
            row["means"] = reports[-1]["means"]
        print(json.dumps(row), flush=True)
    if args.mode == "smoke":
        write_json_atomic(args.output / "smoke_checks.json",
                          _smoke_gradients(owner, reader, records[0], device, microbatch))
    torch.save({"schema_version": "ember_native_readout_probe_state_v1", "reader": reader.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epochs": epochs,
                "optimizer_updates": epochs*len(tasks)//4, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "deployment_checkpoint": False},
               args.output / "reader_probe_state.pt")
    print(json.dumps({"status": "probe_complete", "epochs": epochs,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30}), flush=True)


if __name__ == "__main__":
    main()
