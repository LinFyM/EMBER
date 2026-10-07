"""Matched-distribution fork of the existing full T training consumers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
import traceback

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.operator_writer.credit import gradient_groups, one_job
from ember.operator_writer.run import build_runtime, gather, optimizer_for
from ember.operator_writer.specification import SCHEMA as PARENT_SCHEMA, STAGE as PARENT_STAGE
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_contract import git_state
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed, initialize_deferred_process_group, seed_everything
from ember.writer.function_credit import sample_flow_randomness
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_assignment

SCHEMA = "ember_cross_context_pairing_run_v1"
STAGE = "cross_context_pairing"


def query_randomness(runtime, event):
    """Draw each original 28-query panel once, then index its actual tensors."""
    cache = {}
    rows = event["queries"]
    for row in rows:
        seed = row["origin_flow_seed"]
        if seed not in cache:
            cache[seed] = sample_flow_randomness(runtime.policy, (28, 50, 32),
                seed=seed, device=runtime.device, random_batch=28)
    noise = torch.stack([cache[row["origin_flow_seed"]][0][row["origin_offset"]] for row in rows])
    tau = torch.stack([cache[row["origin_flow_seed"]][1][row["origin_offset"]] for row in rows])
    if len(rows) != 28:
        raise ValueError("actual condition must retain 28 equally weighted queries")
    return noise, tau


def _restore(runtime, optimizer, scheduler, context, checkpoint, data, arm, *, smoke=False):
    restored = {}
    parent = checkpoint.resolve() == Path(data.spec["pairing"]["parent_checkpoint"]).resolve()
    updates, rows = load_ecp_checkpoint(checkpoint=checkpoint,
        stage=PARENT_STAGE if parent else STAGE, context=context,
        model=runtime.writer, optimizer=optimizer, scheduler=scheduler,
        run_contract_schema=PARENT_SCHEMA if parent else SCHEMA,
        restored_state=restored, allow_world_size_change=True)
    if parent:
        expected = {"updates": 2340, "mode": "T", "loss_variant": "full"}
        if (updates != 2340 or rows != 2340 or restored["training_state"] != expected
                or restored["sampler_state"]["next_step"] != 2340):
            raise ValueError("parent T2340 complete ECP identity/cursor changed")
        updates, rows = 0, 0
    else:
        data.restore(restored["sampler_state"], smoke=smoke)
        if restored["training_state"] != _state(arm, updates):
            raise ValueError("pairing arm or cumulative update identity changed")
    if (scheduler.last_epoch != 2340 + updates or rows != updates
            or optimizer.param_groups[0]["lr"] != 1e-5
            or {int(value["step"]) for value in optimizer.state.values()} != {2340 + updates}):
        raise ValueError("inherited optimizer/scheduler clock changed")
    return updates, rows, restored


def _state(arm, updates):
    return {"arm": arm, "updates": updates, "cumulative_updates": 2340 + updates,
            "loss_variant": "full", "scientific_distribution_fork": True}


def _checkpoint(output, context, runtime, optimizer, scheduler, data, arm, updates):
    return save_ecp_checkpoint(output_dir=output, macro=updates, stage=STAGE,
        context=context, model=runtime.writer, optimizer=optimizer, scheduler=scheduler,
        run_contract_schema=SCHEMA, metrics_rows=updates,
        sampler_state=data.sampler_state(), training_state=_state(arm, updates))


def _job(runtime, data, event, args, output, *, save_randomness):
    random = query_randomness(runtime, event)
    record = one_job(runtime, data, event, args.microbatch, args.frame_chunk,
                     "full", flow_randomness=random)
    record.update(query_records=event["queries"],
        actual_tau=random[1].detach().cpu().tolist(),
        noise_mean=float(random[0].mean()), noise_rms=float(random[0].square().mean().sqrt()),
        noise_origin="canonical original-query 28-panel then actual tensor indexing")
    if save_randomness:
        path = output / "flow_randomness" / f"u{event['update']:03d}_t{event['task']:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"query_records": event["queries"], "noise": random[0].detach().cpu(),
                    "tau": random[1].detach().cpu()}, path)
        record["actual_flow_randomness"] = str(path)
    return record


def execute(spec, args):
    from .data import PairingData

    from ember.writer.materialization import frozen_authority

    git = git_state(Path(__file__).resolve().parents[3])
    if not frozen_authority(git):
        raise ValueError("pairing execution requires clean pushed detached source")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 6 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("pairing requires one-node NUMA/NCCL physical topology")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    data = PairingData(args.asset_root, spec, arm=args.arm)
    runtime = build_runtime(args.asset_root, spec, context.device, "T")
    runtime.writer.train()
    seed_everything(7, context)
    optimizer, scheduler, parameters = optimizer_for(runtime.writer, spec)
    output = Path(spec["run_root"]) / args.phase / args.arm / args.attempt if args.phase != "formal" else (
        Path(spec["run_root"]) / args.arm / "train/attempts" / args.attempt)
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ranks": gather({"rank": context.rank, "numa": context.numa_node,
            "cpu_affinity": list(context.cpu_affinity or ()),
            "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid)}, context.world_size)}
    contract = {"schema_version": SCHEMA, "git": git, "phase": args.phase, "arm": args.arm,
        "spec": spec, "source": runtime.source, "lora": runtime.lora.to_dict(),
        "parent_checkpoint": spec["pairing"]["parent_checkpoint"], "resumed_from": str(args.resume),
        "parent_training_git": spec["continuation"]["parent_training_git"],
        "topology": topology, "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
        "trainable_names": [name for name,p in runtime.writer.named_parameters() if p.requires_grad],
        "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
        "information_wall": "only dual RGB/exact L teach; query RGB/state/actions supervise FM; metadata routes data only"}
    try:
        parent_contract = read_json(Path(spec["pairing"]["parent_checkpoint"]).parent.parent / "run_contract.json")
        if (parent_contract["trainable_names"] != contract["trainable_names"]
                or parent_contract["lora"] != contract["lora"]
                or parent_contract["optimizer"] != spec["optimization"]
                or contract["source_trainable"] != 0):
            raise ValueError("parent complete parameter/optimizer topology changed")
        updates, rows, restored = _restore(runtime, optimizer, scheduler, context, args.resume, data, args.arm,
                                           smoke=args.phase == "smoke")
        if context.is_main:
            if (output / "run_contract.json").exists():
                raise ValueError("attempt already registered; use a new attempt for an ECP resume")
            write_json_atomic(output / "run_contract.json", contract)
            write_json_atomic(output / "resume_provenance.json", {"old_world_size":
                read_json(args.resume / "checkpoint_manifest.json")["world_size"],
                "new_topology": topology, "restored_state": restored,
                "scheduler_epoch": scheduler.last_epoch,
                "optimizer_steps": sorted({int(value["step"]) for value in optimizer.state.values()}),
                "sampler_state": data.sampler_state(), "new_distribution_not_original_exact_resume": True})
        if args.phase == "profile":
            _profile(runtime, data, args, output, context, optimizer)
            return
        if updates == 0:
            _checkpoint(output, context, runtime, optimizer, scheduler, data, args.arm, 0)
        target = args.stop_after if args.stop_after is not None else 216
        if args.phase == "smoke" and target not in (2,4):
            raise ValueError("independent smoke has only its 2/4-update ECP boundaries")
        if args.phase == "formal" and target not in (108,216):
            raise ValueError("only registered 108 recovery /216 scientific endpoint")
        started = time.perf_counter()
        while updates < target:
            begun = time.perf_counter()
            jobs = [data.event(updates, task) for task in data.tasks_for_step(updates)]
            costs = {i: data.videos.frame_counts(job["task"], job["teacher_demo"])[1] for i,job in enumerate(jobs)}
            assigned = condition_assignment(tuple(costs), costs, world_size=context.world_size)
            optimizer.zero_grad(set_to_none=True)
            local, error = [], None
            try:
                for i in assigned[context.rank]:
                    local.append(_job(runtime, data, jobs[i], args, output, save_randomness=True))
            except Exception:
                error = traceback.format_exc()
            failures = [value for value in gather(error, context.world_size) if value]
            if failures:
                raise RuntimeError(f"pairing actual consumer failed: {failures}")
            sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 2**20)
            groups = gradient_groups(runtime.writer)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, spec["optimization"]["grad_clip"], error_if_nonfinite=True))
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            updates += 1
            rows += 1
            data.next_step = updates
            packets = gather(local, context.world_size)
            memory = gather({"rank": context.rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(context.device)/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(context.device)/2**30}, context.world_size)
            if context.is_main:
                record = {"update": updates, "cumulative_update":2340+updates, "arm":args.arm,
                    "queries":112, "lr_applied":lr, "scheduler_epoch":scheduler.last_epoch,
                    "jobs":[row for packet in packets for row in packet], "rank_memory":memory,
                    "grad_norms_before_clip":groups, "total_grad_norm":norm,
                    "seconds":time.perf_counter()-begun}
                append_jsonl(output / "metrics.jsonl", record)
                if updates in (1,2,4,108,216):
                    print(json.dumps({"event":"update", "update":updates, "seconds":record["seconds"],
                        "rank_memory":memory, "lr":lr}),flush=True)
            if updates in (108,216) or args.phase == "smoke" and updates in (2,4):
                _checkpoint(output, context, runtime, optimizer, scheduler, data, args.arm, updates)
        if context.is_main:
            write_json_atomic(output / "completion.json", {"phase":args.phase,"arm":args.arm,
                "updates":updates,"queries_in_this_attempt":(updates - (0 if args.resume.name == 'macro_00002340' else int(args.resume.name[-8:]))) *112,
                "checkpoint":str(output / "checkpoints" / f"macro_{updates:08d}"),
                "seconds":time.perf_counter()-started,"scientific_endpoint":args.phase=='formal' and updates==216})
    except BaseException:
        if context.is_main:
            write_json_atomic(output / "failure.json", {"error":traceback.format_exc(),"git":git,"phase":args.phase})
        raise
    finally:
        data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _profile(runtime, data, args, output, context, optimizer):
    if context.world_size != 1:
        raise ValueError("physical single-condition profile uses one measured rank")
    events = [data.event(u,t) for u in range(216) for t in data.tasks_for_step(u)]
    event = max(events, key=lambda row:data.videos.frame_counts(row["task"], row["teacher_demo"])[1])
    records = []
    for microbatch, frame_chunk in ((14,16),(28,32),(28,64))[args.profile_skip:]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
        args.microbatch, args.frame_chunk = microbatch, frame_chunk
        began = time.perf_counter()
        try:
            record = _job(runtime,data,event,args,output,save_randomness=False)
            record.update(status="ok", microbatch=microbatch,frame_chunk=frame_chunk,
                peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(runtime.device)/2**30)
        except torch.cuda.OutOfMemoryError:
            record = {"status":"oom","microbatch":microbatch,"frame_chunk":frame_chunk,
                "seconds":time.perf_counter()-began,"error":traceback.format_exc()}
        records.append(record)
        write_json_atomic(output / "profile.json", {"event":event,"trials":records,"optimizer_updates":0})
        print(json.dumps({key:record.get(key) for key in ('status','microbatch','frame_chunk','total_seconds','peak_reserved_gib')}),flush=True)
        if record["status"] == "oom":
            break
