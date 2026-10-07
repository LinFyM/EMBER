"""Finite shared-MT training owner; task-weighted native credit and full ECP."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import socket
import time
import traceback

import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_setup import initialize_distributed, initialize_deferred_process_group, seed_everything
from ember.pi05_source_contract import append_jsonl
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_assignment
from . import checkpoint
from .credit import condition_credit
from .data import NativeData
from .runtime import build_runtime, frozen_git
from .specification import ROOT, MT_WEIGHTS, SCHEMA, specification


def gather(value, world):
    if world == 1:
        return [value]
    rows = [None] * world
    dist.all_gather_object(rows, value)
    return rows


def gradient_groups(model):
    groups = {"beta": [], "encoder": [], "reader_Q": [], "reader_K": [], "reader_V": [], "reader_O": []}
    for name, value in model.named_parameters():
        if name.startswith("common."):
            key = "beta"
        elif name.startswith("readers."):
            projection = name.split(".")[2].lower()
            key = "reader_" + projection.upper()
            if key not in groups:
                key = "encoder"
        else:
            key = "encoder"
        if value.grad is not None:
            groups[key].append(value.grad.detach().float().square().sum())
    return {name: float(torch.stack(values).sum().sqrt()) if values else 0.
            for name, values in groups.items()}


@dataclass
class Session:
    spec: dict
    context: object
    data: NativeData
    runtime: object
    optimizer: torch.optim.Optimizer
    parameters: tuple
    output: Path
    origin: dict
    contract: dict
    microbatch: int
    frame_chunk: int


def prepare(args) -> Session:
    git = frozen_git()
    spec = specification()
    if args.arm not in ("M", "V") or args.kind not in ("formal", "smoke", "profile"):
        raise ValueError("unknown finite native-video branch")
    if args.kind != "formal" and args.arm != "V":
        raise ValueError("only the registered V engineering branch exists")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("explicit BCI NCCL contract is required")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    data = NativeData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device, args.arm)
    parameters = tuple(runtime.controller.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
    output = ROOT / ("training" if args.kind == "formal" else "engineering") / args.arm / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    local = {"rank": context.rank, "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "nccl_p2p_disable": "1",
                "ranks": gather(local, context.world_size), "condition_assignment": "whole_condition_cost_balanced",
                "microbatch": args.microbatch, "frame_chunk": args.frame_chunk}
    origin = {"kind": "smoke" if args.kind != "formal" else "formal", "source": runtime.source,
              "mt_checkpoint": str(MT_WEIGHTS), "training_git": git,
              "fresh_optimizer": True, "lora": runtime.lora.to_dict()}
    contract = {"schema_version": SCHEMA, "study_id": spec["task"], "arm": args.arm, "kind": args.kind,
                "git": git, "spec": spec, "origin": origin, "topology": topology,
                "scheduler": None, "source_trainable": 0, "learning_dtype": "float32",
                "source_parameter_bytes": sum(p.numel() * p.element_size() for p in runtime.policy.parameters()),
                "learning_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                "allocated_gib_after_setup": torch.cuda.memory_allocated(context.device) / 2**30,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "condition_weight": .25, "query_weight": "actual_micro_count/28",
                "information_wall": "teacher: dual RGB/exact L only; independent cross-episode query FM"}
    if context.is_main:
        path = output / "run_contract.json"
        if path.exists():
            raise ValueError("attempt path already has a launch contract; choose a new retained attempt")
        write_json_atomic(path, contract)
    return Session(spec, context, data, runtime, optimizer, parameters, output, origin, contract,
                   args.microbatch, args.frame_chunk)


def one_update(session, step, *, optimize=True):
    started = time.monotonic()
    context = session.context
    jobs = [session.data.event(step, task) for task in session.data.tasks_for_step(step)]
    costs = {i: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             if session.runtime.controller.arm == "V" else 28 for i, job in enumerate(jobs)}
    assignment = condition_assignment(tuple(costs), costs, world_size=context.world_size)
    session.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(context.device)
    local, error = [], None
    try:
        for i in assignment[context.rank]:
            local.append(condition_credit(session.runtime, session.data, jobs[i],
                microbatch=session.microbatch, frame_chunk=session.frame_chunk,
                check_initial=(step == 0 and session.runtime.controller.arm == "V")))
    except Exception:
        error = traceback.format_exc()
        write_json_atomic(session.output / f"failure_rank_{context.rank:02d}_step_{step:04d}.json",
                          {"step": step, "error": error, "completed_conditions": local,
                           "assigned_jobs": [jobs[i] for i in assignment[context.rank]]})
    failures = [x for x in gather(error, context.world_size) if x]
    if failures:
        raise RuntimeError(f"native-video actual condition failed: {failures}")
    sum_writer_gradients(session.parameters, world_size=context.world_size, bucket_bytes=64 * 2**20)
    groups = gradient_groups(session.runtime.controller)
    norm = float(torch.nn.utils.clip_grad_norm_(session.parameters, 1., error_if_nonfinite=True))
    if optimize:
        session.optimizer.step()
        session.data.next_step = step + 1
    torch.cuda.synchronize(context.device)
    packets = gather(local, context.world_size)
    memory = gather({"rank": context.rank, "allocated_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
                     "reserved_gib": torch.cuda.max_memory_reserved(context.device) / 2**30}, context.world_size)
    rows = [row for packet in packets for row in packet]
    if len(rows) != 4 or len({row["event"]["task"] for row in rows}) != 4:
        raise ValueError("an update lost a whole equally weighted task condition")
    record = {"update": step + 1, "arm": session.runtime.controller.arm, "queries": 112,
              "flow_loss": sum(row["flow_loss"] for row in rows) / 4,
              "conditions": rows, "gradients": groups, "gradient_norm_before_clip": norm,
              "memory": memory, "seconds": time.monotonic() - started, "optimizer_applied": optimize}
    if context.is_main:
        append_jsonl(session.output / "metrics.jsonl", record)
        print(json.dumps({k: record[k] for k in ("update", "arm", "flow_loss", "seconds", "gradients", "memory")}), flush=True)
    return record


def _restore_or_origin(session, args):
    if args.resume:
        restored = {}
        macro, rows = checkpoint.restore(checkpoint=args.resume, stage=args.arm, context=session.context,
            model=session.runtime.controller, optimizer=session.optimizer, data=session.data,
            origin=session.origin, topology=session.contract["topology"],
            allow_topology_change=True, restored_state=restored)
        if session.context.is_main:
            prefix = (args.resume.parent.parent / "metrics.jsonl").read_text().splitlines()[:rows]
            if len(prefix) != rows:
                raise ValueError("resume lost its actual metrics prefix")
            (session.output / "metrics.jsonl").write_text("\n".join(prefix) + ("\n" if prefix else ""))
            write_json_atomic(session.output / "resume_provenance.json", restored)
        return macro
    if args.kind == "formal":
        checkpoint.save(output_dir=session.output, macro=0, stage=args.arm, context=session.context,
            model=session.runtime.controller, optimizer=session.optimizer, data=session.data,
            origin=session.origin, metrics_rows=0, topology=session.contract["topology"])
    return 0


def _validate_end(args, macro):
    if args.kind == "formal":
        if args.stop_after != 128:
            raise ValueError("registered native-video stopping cursor changed")
    elif args.kind == "smoke":
        expected = 4 if args.resume else 2
        if args.arm != "V" or args.stop_after != expected or args.resume and macro != 2:
            raise ValueError("smoke must be fresh two updates then full ECP continuation to four")
    else:
        raise ValueError("profile never applies a training update")


def _complete(session, args, macro, start):
    states = {str(value.dtype) for state in session.optimizer.state.values()
              for name, value in state.items() if name != "step" and isinstance(value, torch.Tensor)}
    if states != {"torch.float32"}:
        raise ValueError(f"learning optimizer dtype changed: {states}")
    write_json_atomic(session.output / "completion.json", {"status": "complete", "arm": args.arm,
        "kind": args.kind, "completed_updates": macro, "queries": macro * 112,
        "seconds_after_setup": time.monotonic() - start, "scheduler": None,
        "optimizer_tensor_dtypes": sorted(states), "source_trainable": 0,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "git": session.contract["git"]})


def run_training(args):
    session = prepare(args)
    start = time.monotonic()
    try:
        macro = _restore_or_origin(session, args)
        _validate_end(args, macro)
        end = args.stop_after
        for step in range(macro, end):
            one_update(session, step)
            macro = step + 1
            if macro in (64, 128) or args.kind == "smoke" and macro == end:
                checkpoint.save(output_dir=session.output, macro=macro, stage=args.arm, context=session.context,
                    model=session.runtime.controller, optimizer=session.optimizer, data=session.data,
                    origin=session.origin, metrics_rows=macro, topology=session.contract["topology"])
        if session.context.is_main:
            _complete(session, args, macro, start)
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()
