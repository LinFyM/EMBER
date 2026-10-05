"""The fixed 270-update, full-FM frozen-T transition-read comparison."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import LORA_B_SUFFIX, copy_task_lora_state_
from ember.pi05_source_checkpoint import capture_rng, read_json, restore_rng, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss, paired_functional_credit
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast
from ember.writer.task_execution import condition_assignment

from .data import FormalData
from .run import build_runtime, frozen_git, gather
from .specification import SPEC_PATH, specification
from .transition_read import (PARENT, ROOT, STUDY, LinearEffectHooks, ReaderHooks,
                              TransitionReadout, read_frozen_memory, summarize_stats)

SCHEMA = "ember_query_conditioned_transition_read_run_v1"


def group_norms(model):
    return {name: float(torch.stack([getattr(t, name).grad.float().norm()
                                   for t in model.targets if getattr(t, name).grad is not None]).norm())
            for name in ("j", "q", "e")}


def one_condition(runtime, model, data, event, arm, *, microbatch, frame_chunk, hooks):
    started = time.perf_counter()
    memory = read_frozen_memory(runtime, data, event["task"], event["teacher_demo"], frame_chunk=frame_chunk)
    torch.cuda.synchronize(runtime.device)
    native_seconds = time.perf_counter() - started
    batch = runtime.processor.training_batch(data.batch(event))
    statistics = {}
    initial = all(torch.count_nonzero(t.j).item() == 0 for t in model.targets)
    if arm == "L":
        state = model.linear(memory)
        initial_difference = (max(float((state[name] - value).abs().max())
                                  for name, value in memory.state.items()) if initial else None)
        # This is the actual complete-FM chain derivative, not a label or auxiliary target.
        with autocast(runtime.device), hooks.activate([state], [memory.state], stats=[statistics]):
            credit = paired_functional_credit(runtime.policy, state, runtime.lora, batch,
                seed=event["flow_seed"], device=runtime.device, random_batch=28, offset=0,
                microbatch=microbatch, condition_weight=.25)
        names = tuple(name for name in state if name.endswith(LORA_B_SUFFIX))
        torch.autograd.backward(tuple(state[name] for name in names),
                                tuple(credit["lora_cotangent"][name] for name in names))
        loss = credit["flow_loss"]
        calls = credit["compiled_forward_calls"]
    else:
        initial_difference = None
        copy_task_lora_state_(runtime.policy, memory.state, runtime.lora)
        owner = NativeFlowPrediction(runtime.policy)
        loss, calls = 0., 0
        for start in range(0, 28, microbatch):
            stop = min(28, start + microbatch)
            sliced = {name: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim and len(value) == 28 else value
                      for name, value in batch.items()}
            with autocast(runtime.device):
                sample = flow_sample(runtime.policy, sliced, seed=event["flow_seed"], device=runtime.device,
                                     random_batch=28, offset=start)
                prepared = owner.prepare(sample)
                # A single teacher memory broadcasts to all same-condition query samples.
                with hooks.activate([memory], parent_installed=True, stats=[statistics]):
                    prediction = owner(sample, prepared)
                    value = mean_velocity_loss(prediction, sample.target, sample.action_width)
                    (value * ((stop - start) / 28) * .25).backward()
            loss += float(value.detach()) * (stop - start) / 28
            calls += 1
    if any(p.grad is not None for p in runtime.writer.parameters()) or any(p.grad is not None for p in runtime.policy.parameters()):
        raise ValueError("frozen father/source received a parameter gradient")
    effect = summarize_stats(statistics)
    if initial and (initial_difference is not None and initial_difference > 1e-5
                    or any(row["delta_rms"] != 0 for row in effect["rows"])):
        raise ValueError("J=0 actual consumer differs from parent")
    if len({row["target"] for row in effect["rows"]}) != 38:
        raise ValueError("actual FM consumer lost a target")
    torch.cuda.synchronize(runtime.device)
    return {**event, "queries": 28, "query_rows": event["queries"],
            "flow_loss": loss, "native_seconds": native_seconds,
            "seconds": time.perf_counter() - started, "sampled_frames": memory.sampled_frames,
            "raw_frames": memory.raw_frames, "compiled_forward_calls": calls,
            "initial_parent_max_difference": initial_difference, "effect": effect}


def update(runtime, model, optimizer, data, context, arm, output, step, *, microbatch, frame_chunk, hooks, profile=False):
    started = time.perf_counter()
    jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
    costs = {i: data.videos.frame_counts(job["task"], job["teacher_demo"])[1] for i, job in enumerate(jobs)}
    assigned = condition_assignment(tuple(costs), costs, world_size=context.world_size)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(context.device)
    records, error = [], None
    try:
        for index in assigned[context.rank]:
            records.append(one_condition(runtime, model, data, jobs[index], arm, microbatch=microbatch,
                                         frame_chunk=frame_chunk, hooks=hooks))
    except Exception:
        error = traceback.format_exc()
    failures = [message for message in gather(error, context.world_size) if message]
    if failures:
        raise RuntimeError(f"fixed macro failed: {failures}")
    sum_writer_gradients(tuple(model.parameters()), world_size=context.world_size)
    groups = group_norms(model)
    norm = float(torch.nn.utils.clip_grad_norm_(tuple(model.parameters()), 1., error_if_nonfinite=True))
    optimizer.step()
    data.next_step = step + 1
    torch.cuda.synchronize(context.device)
    packets = gather(records, context.world_size)
    memory = gather({"rank": context.rank, "peak_allocated_GiB": torch.cuda.max_memory_allocated(context.device) / 2**30,
                     "peak_reserved_GiB": torch.cuda.max_memory_reserved(context.device) / 2**30}, context.world_size)
    row = {"macro": step + 1, "arm": arm, "queries": 112, "lr": 1e-4, "microbatch": microbatch,
           "frame_chunk": frame_chunk, "seconds": time.perf_counter() - started,
           "group_grad_norms": groups, "grad_norm_before_clip": norm, "rank_memory": memory,
           "jobs": [record for packet in packets for record in packet]}
    if context.is_main:
        append_jsonl(output / ("profile_metrics.jsonl" if profile else "metrics.jsonl"), row)
        if step == 0:
            write_json_atomic(output / ("profile_first_consumer.json" if profile else "first_consumer.json"), row)
            print(json.dumps({"event": "first_actual_consumer", "arm": arm,
                              "seconds": row["seconds"], "memory": memory}), flush=True)
    return row


def run_updates(args, runtime, model, optimizer, data, context, output, hooks, git, topology, cursor, rows):
    if args.phase == "profile":
        if cursor:
            raise ValueError("profile may not resume")
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        rng = capture_rng(context)
        measurements = []
        for step, micro, chunk in ((0, 14, 8), (1, 28, 16)):
            measurements.append(update(runtime, model, optimizer, data, context, args.arm, output, step,
                microbatch=micro, frame_chunk=chunk, hooks=hooks, profile=True))
        model.load_state_dict(initial, strict=True)
        optimizer.state.clear()
        data.next_step = 0
        restore_rng(rng, context)
        if context.is_main:
            write_json_atomic(output / "profile_complete.json", {"discarded_updates": 2,
                "restored_initial_parameters_optimizer_rng_cursor": True,
                "rows": [{k: value for k, value in row.items() if k != "jobs"} for row in measurements]})
        return
    for step in range(cursor, 270):
        update(runtime, model, optimizer, data, context, args.arm, output, step,
               microbatch=args.microbatch, frame_chunk=args.frame_chunk, hooks=hooks)
        rows += 1
        if step + 1 not in (90, 180, 270):
            continue
        checkpoint = save_ecp_checkpoint(output_dir=output, macro=step + 1, stage=STUDY, context=context,
            model=model, optimizer=optimizer, scheduler=None, run_contract_schema=SCHEMA, metrics_rows=rows,
            sampler_state=data.sampler_state(), training_state={"study": STUDY, "arm": args.arm,
                "parent": str(PARENT), "parent_source": runtime.source, "git": git, "topology": topology})
        if step + 1 == 90:
            restored = {}
            macro, restored_rows = load_ecp_checkpoint(checkpoint=checkpoint, stage=STUDY, context=context,
                model=model, optimizer=optimizer, scheduler=None, run_contract_schema=SCHEMA,
                expected_sampler_state=data.sampler_state(), restored_state=restored)
            if macro != 90 or restored_rows != rows:
                raise ValueError("actual complete checkpoint restore lost cursor")
            if context.is_main:
                write_json_atomic(output / "restore_consumer90.json", {"checkpoint": str(checkpoint), "macro": macro,
                    "sampler": restored["sampler_state"], "optimizer_parameter_states": len(optimizer.state),
                    "scheduler": None, "rank_rng_restored": True, "topology": topology})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile", "train"))
    parser.add_argument("--arm", required=True, choices=("L", "R"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--attempt", default="first")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.microbatch <= 28 or args.frame_chunk <= 0 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("physical packing/NCCL contract changed")
    started = time.time()
    git = frozen_git(continuation=True)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size not in range(1, 5):
        raise ValueError("four-condition macro admits only useful ranks1..4")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    spec = specification(SPEC_PATH)
    data = FormalData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device, "T")
    runtime.writer.load_state_dict(load_file(str(PARENT / "ecp.safetensors"), device=str(context.device)), strict=True)
    runtime.writer.requires_grad_(False).eval()
    runtime.policy.requires_grad_(False).eval()
    model = TransitionReadout(runtime.lora).to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
    seed_everything(7, context)
    output = ROOT / args.arm / args.phase / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ranks": gather({"rank": context.rank, "uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
                                 "numa": context.numa_node, "affinity": context.cpu_affinity}, context.world_size)}
    contract = {"schema_version": SCHEMA, "study": STUDY, "arm": args.arm, "parent": str(PARENT),
                "parent_manifest": read_json(PARENT / "checkpoint_manifest.json"), "source": runtime.source,
                "git": git, "topology": topology, "phase": args.phase,
                "sampling_spec": str(SPEC_PATH), "events": spec["events"], "updates": 270,
                "queries": 30240, "trainable_parameters": 2490368, "loss": "full50x7 FM only",
                "optimizer": {"lr": 1e-4, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 1e-4, "clip": 1., "scheduler": None},
                "microbatch": args.microbatch, "frame_chunk": args.frame_chunk, "resume": str(args.resume) if args.resume else None}
    if context.is_main:
        write_json_atomic(output / "run_contract.json", contract)
    hooks = ReaderHooks(runtime.policy, runtime.lora, model) if args.arm == "R" else LinearEffectHooks(runtime.policy, runtime.lora)
    cursor, rows = 0, 0
    try:
        if args.resume:
            restored = {}
            cursor, rows = load_ecp_checkpoint(checkpoint=args.resume, stage=STUDY, context=context,
                model=model, optimizer=optimizer, scheduler=None, run_contract_schema=SCHEMA,
                restored_state=restored)
            if restored["training_state"]["arm"] != args.arm or restored["training_state"]["parent"] != str(PARENT):
                raise ValueError("resume arm/frozen parent changed")
            data.restore(restored["sampler_state"])
        run_updates(args, runtime, model, optimizer, data, context, output, hooks, git, topology, cursor, rows)
        if context.is_main:
            write_json_atomic(output / "completion.json", {"status": "complete", "phase": args.phase,
                "macro": 270 if args.phase == "train" else 0, "queries": 30240 if args.phase == "train" else 224,
                "seconds_including_load": time.time() - started, "contract": contract})
    finally:
        hooks.close()
        data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
