"""Bounded shared-Writer score stage; source and logical event stream stay fixed.

This owns collection/update/checkpoint lifecycle, reusing the existing native
compiler, full-LoRA functional interface, environment pool and ECP checkpoint.
Readout scheduling and resource allocation are launch-time operations.
"""
from __future__ import annotations

import argparse
import copy
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
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_source_checkpoint import barrier, capture_rng, restore_rng, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed, initialize_deferred_process_group, seed_everything
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_assignment

from .credit import gradient_groups
from .data import FormalData, TASKS
from .denoising_collection import collect_group
from .denoising_credit import score_group
from .run import build_runtime, frozen_git, gather


STAGE = "denoising_return_writer_20261006"
ROOT = Path("/data1/user/ymdai/ember_runs") / STAGE
SCHEMA = "denoising_return_writer_v1"


def optimizer_for(writer):
    optimizer = torch.optim.AdamW(tuple(writer.parameters()), lr=1e-5, betas=(.9, .95),
                                 eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return optimizer, scheduler


def reduce_update(runtime, optimizer, scheduler, context) -> dict:
    parameters = tuple(runtime.writer.parameters())
    sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=16 * 1024**2)
    norms = gradient_groups(runtime.writer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"preclip_norm": float(norm), "gradient_groups": norms, "lr": optimizer.param_groups[0]["lr"]}


def coordinated_call(function, context):
    value, error = None, None
    try:
        value = function()
    except Exception:
        error = traceback.format_exc()
    failures = [item for item in gather(error, context.world_size) if item]
    if failures:
        raise RuntimeError("a bounded stage consumer failed: " + "\n".join(failures))
    return value


def execute(args) -> None:
    wall_started = time.monotonic()
    root = args.root.resolve()
    if root != ROOT or (root / "completion.json").exists() or (root / "RETIRED").exists():
        raise ValueError("run is outside its only root or already sealed")
    contract = read_json(root / "contract.json")
    git = frozen_git(continuation=True)
    if contract["git"] != git or contract["schema_version"] != SCHEMA:
        raise ValueError("training requires its clean pushed detached root identity")
    gate = read_json(root / "readouts/parent/seen/SDE/evaluation/results.json")
    parent_rows = gate.get("episodes", gate.get("rows", []))
    if len(parent_rows) != 144 or not any(row["success"] for row in parent_rows):
        raise ValueError("positive complete parent SDE144 is required before training")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 6:
        raise ValueError("invalid registered physical rank count")
    seed_everything(20261006, context)
    torch.set_num_threads(args.cpu_threads)
    output = root / "train" if args.attempt == "fresh" else root / "train/attempts" / args.attempt
    if context.is_main:
        if (output / "run_contract.json").exists():
            raise ValueError("completed/started attempt cannot be overwritten")
        output.mkdir(parents=True, exist_ok=True)
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    physical = int(visible[context.local_rank])
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(physical),
                      LIBERO_CONFIG_PATH=str(output / f"libero_config_rank{context.rank}"))
    spec = read_json(Path(contract["source_spec"]["path"]))
    runtime = build_runtime(args.asset_root, spec, context.device, "T")
    runtime.writer.load_state_dict(load_file(str(Path(contract["parent_checkpoint"]) / "ecp.safetensors"),
                                            device=str(context.device)), strict=True)
    data = FormalData(args.asset_root, spec, query_labels=False)
    if tuple(data.tasks) != tuple(TASKS) or data.seed != 20260928:
        raise ValueError("original 36-task FormalData event stream changed")
    environment = read_json(args.environment_contract)
    environment = {key: environment[key] for key in ("libero_paths", "environment", "parallel")}
    environment["parallel"] = dict(environment["parallel"], envs_per_replica=4)
    metadata = read_json(args.environment_contract)["tasks"]
    tasks = {(row["suite"], int(row["task_id"])): row for row in metadata}
    if len(tasks) != 36:
        raise ValueError("training environment metadata must cover original36")
    initialize_deferred_process_group(context, rendezvous_root=output)
    pool = PersistentTaskEnvironmentPool(environment, physical_gpu_id=physical)
    topology = gather({"rank": context.rank, "host": socket.gethostname(), "physical_GPU": physical,
                       "device": str(context.device), "numa": context.numa_node,
                       "cpu_affinity": context.cpu_affinity}, context.world_size)
    optimizer, scheduler = optimizer_for(runtime.writer)
    source_trainable = sum(parameter.numel() for parameter in runtime.policy.parameters() if parameter.requires_grad)
    if source_trainable or any(not p.requires_grad for p in runtime.writer.parameters()):
        raise ValueError("source must freeze and all original shared Writer parameters must learn")
    training_contract = {**contract, "stage": STAGE, "topology": topology, "world_size": context.world_size,
                         "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
                         "source_trainable": source_trainable,
                         "trainable_names": [name for name, _ in runtime.writer.named_parameters()],
                         "profile": args.profile, "environment_contract": str(args.environment_contract),
                         "seed_schedule": "SeedSequence([20261006,macro1based,task,replica,purpose,replan])",
                         "purposes": ["init", "env", "initial_noise", "SDE", "reservoir", "steps"]}
    if context.is_main:
        write_json_atomic(output / "run_contract.json", training_contract)
    macro, rows, nonzero_groups = 0, 0, 0
    if args.resume:
        restored = {}
        macro, rows = coordinated_call(lambda: load_ecp_checkpoint(checkpoint=args.resume, stage=STAGE,
            context=context, model=runtime.writer, optimizer=optimizer, scheduler=scheduler,
            run_contract_schema=SCHEMA, restored_state=restored, allow_world_size_change=True), context)
        if macro not in range(9, 72, 9) or rows != macro:
            raise ValueError("resume only at an existing complete 9-macro boundary")
        nonzero_groups = restored["training_state"]["nonzero_groups"]
        data.next_step = macro
        if context.is_main:
            prefix = (args.resume.parent.parent / "metrics.jsonl").read_text().splitlines()[:rows]
            if len(prefix) != rows:
                raise ValueError("checkpoint metric prefix missing")
            (output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
            write_json_atomic(output / "resume_provenance.json", {"checkpoint": str(args.resume),
                "restored": restored.get("topology_resume"), "topology": topology})
    microbatch, frame_chunk = args.microbatch, args.frame_chunk
    try:
        while macro < 72:
            update_started = time.perf_counter()
            jobs = [data.event(macro, task) for task in data.tasks_for_step(macro)]
            costs = {index: tasks[(data.tasks[event["task"]].suite,
                                  data.tasks[event["task"]].suite_task_id)]["horizon"]
                     for index, event in enumerate(jobs)}
            assigned = condition_assignment(tuple(costs), costs, world_size=context.world_size)
            local = []
            for index in assigned[context.rank]:
                event = jobs[index]
                condition, raw, sampled = data.condition(runtime, event["task"], event["teacher_demo"])
                with torch.no_grad():
                    state, _ = runtime.compile(condition, frame_chunk=frame_chunk)
                teaching = data.tasks[event["task"]]
                suite, local_task = teaching.suite, teaching.suite_task_id
                group = collect_group(runtime, pool, tasks[suite, int(local_task)], environment, event, state,
                                      output / "raw" / f"macro_{macro+1:03d}" / f"task_{event['task']:03d}", macro+1)
                group.update(raw_teacher_frames=raw, sampled_teacher_frames=sampled,
                             complete_lora_targets=len(state) // 2, source_git=git,
                             parent_checkpoint=contract["parent_checkpoint"])
                local.append((condition, state, group))
            # Each rank owns whole conditions; all four logical conditions still
            # use the identical unmodified omega before the single score update.
            if macro == 0 and args.profile:
                original = copy.deepcopy(runtime.writer.state_dict())
                initial_rng = capture_rng(context)
                profiles = []
                for physical_micro, physical_frames in ((32, 16), (64, 32)):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats(context.device)
                    tick = time.perf_counter()
                    failure = None
                    try:
                        credits = coordinated_call(lambda: [score_group(runtime, condition, state, group,
                            microbatch=physical_micro, frame_chunk=physical_frames, force_zero=True)
                            for condition, state, group in local], context)
                        reduce_update(runtime, optimizer, scheduler, context)
                    except RuntimeError as error:
                        if "out of memory" not in str(error).lower():
                            raise
                        failure, credits = str(error), []
                    seconds = time.perf_counter() - tick
                    profile = {"microbatch": physical_micro, "frame_chunk": physical_frames,
                               "seconds": seconds, "transitions": sum(c["scored_transitions"] for c in credits),
                               "failure": failure,
                               "peak_memory_bytes": torch.cuda.max_memory_allocated(context.device)}
                    profiles.append(gather(profile, context.world_size))
                    runtime.writer.load_state_dict(original, strict=True)
                    optimizer, scheduler = optimizer_for(runtime.writer)
                    restore_rng(initial_rng, context)
                    torch.cuda.empty_cache()
                means = [float("inf") if any(row["failure"] for row in profile) else
                         max(row["seconds"] for row in profile) for profile in profiles]
                if all(value == float("inf") for value in means):
                    raise ValueError("both bounded profiles exceed actual physical capacity")
                chosen = min(range(2), key=means.__getitem__)
                microbatch, frame_chunk = ((32, 16), (64, 32))[chosen]
                if context.is_main:
                    write_json_atomic(output / "profile.json", {"discarded_updates": 2, "profiles": profiles,
                        "chosen": {"microbatch": microbatch, "frame_chunk": frame_chunk},
                        "stop_reason": "two authorised discard updates exhausted; true logical batch unchanged",
                        "parent_optimizer_RNG_restored": True})
                del original
            optimizer.zero_grad(set_to_none=True)
            credits = coordinated_call(lambda: [score_group(runtime, condition, state, group,
                microbatch=microbatch, frame_chunk=frame_chunk) for condition, state, group in local], context)
            update = reduce_update(runtime, optimizer, scheduler, context)
            all_groups = [group for rank_groups in gather([item[2] for item in local], context.world_size)
                          for group in rank_groups]
            all_credits = [item for rank_credits in gather(credits, context.world_size) for item in rank_credits]
            macro += 1
            rows += 1
            data.next_step = macro
            nonzero_groups += sum(bool(group["nonzero_LOO"]) for group in all_groups)
            metric = {"macro": macro, "groups": all_groups, "score_credit": all_credits, **update,
                      "physical_microbatch": microbatch, "native_frame_chunk": frame_chunk,
                      "seconds": time.perf_counter() - update_started, "nonzero_groups_to_date": nonzero_groups,
                      "source_trainable": source_trainable, "finite": True}
            if context.is_main:
                append_jsonl(output / "metrics.jsonl", metric)
            if macro % 9 == 0:
                checkpoint = save_ecp_checkpoint(output_dir=output, macro=macro, stage=STAGE,
                    context=context, model=runtime.writer, optimizer=optimizer, scheduler=scheduler,
                    run_contract_schema=SCHEMA, metrics_rows=rows, sampler_state=data.sampler_state(),
                    training_state={"updates": macro, "nonzero_groups": nonzero_groups,
                                    "noise_seed_root": 20261006, "topology": topology,
                                    "physical_microbatch": microbatch, "frame_chunk": frame_chunk})
                if context.is_main:
                    manifest = read_json(checkpoint / "checkpoint_manifest.json")
                    manifest.update(source=contract["source"], parent_checkpoint=contract["parent_checkpoint"],
                                    training_git=git["commit"])
                    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
                barrier(context)
                if context.is_main and macro in (63, 72):
                    print(json.dumps({"event": "readout_checkpoint", "macro": macro,
                                      "checkpoint": str(checkpoint)}), flush=True)
            if macro == 9 and nonzero_groups == 0:
                break
            del local
        if context.is_main:
            write_json_atomic(output / "completion.json", {"status": "no_credit_first9" if macro == 9 else "complete72",
                "updates": macro, "episodes": macro * 16, "nonzero_groups": nonzero_groups,
                "seconds": time.monotonic() - wall_started, "exit": 0, "git": git})
    except Exception:
        write_json_atomic(output / f"failure_rank{context.rank}.json", {"macro_completed": macro,
            "error": traceback.format_exc(), "seconds": time.monotonic() - wall_started})
        raise
    finally:
        pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--environment-contract", type=Path, required=True)
    parser.add_argument("--attempt", default="fresh")
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--frame-chunk", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", type=Path)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
