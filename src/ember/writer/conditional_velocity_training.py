"""Canonical fresh V/L conditional-velocity training on the fixed coverage36 stream."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import (initialize_deferred_process_group, initialize_distributed,
                                      load_policy, seed_everything)
from ember.writer.conditional_velocity import ConditionalVelocityOperator
from ember.writer.conditional_velocity_data import COVERAGE_TASKS, VelocityTrainingData
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast, require_architecture_identity
from ember.writer.task_execution import condition_assignment


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "configs/conditional_velocity_operator_v1/learning_spec.json"
SCHEMA = "ember_conditional_velocity_learning_run_v1"
STAGE = "conditional_velocity_learning"


def contract() -> tuple[dict, dict]:
    spec = read_json(SPEC)
    expected = {
        "schema_version": "ember_conditional_velocity_learning_v1",
        "run_root": "/data1/user/ymdai/ember_runs/conditional_velocity_operator_learning_20260927",
        "reference_model_config": "configs/language_content_path_causality_v1/train_C0.json",
        "protocol": "configs/libero_24_8_8_coverage_v1/protocol.json",
        "train_tasks": list(COVERAGE_TASKS),
        "validation_tasks": [3, 6, 11, 16, 23, 26, 31, 39],
        "teacher_pool": [0, 49], "query_count_per_task": 28, "tasks_per_update": 4,
        "sampler_seed": 20260927, "optimization_seed": 7, "module_seed": 7,
        "world_size": 2, "first_stop_update": 270, "checkpoint_updates": [90, 180, 270],
        "initialization": {"common_seed": 7, "text_core_seed": 7,
            "construction_order": ["identity_common_beta", "seven_row_readout", "free_U",
                                   "mode_specific_teaching"], "checkpoint_source": "fresh_only"},
        "lora": {"common_rank": 128, "conditional_rank": 7, "deploy_rank": 135,
                 "targets": 38, "alpha": 135, "conditional_target": "model.action_out_proj"},
        "optimizer": {"name": "AdamW", "lr": 0.0003, "betas": [0.9, 0.95],
                      "eps": 1e-8, "weight_decay": 0.0001, "clip": 1.0,
                      "warmup_updates": 150, "cosine_updates": 1200, "floor_lr": 0.00001},
        "evaluation": {"role": "validation", "arm": "correct", "state_ids": [0, 49],
                       "video_schedule_seed": 20260911, "episodes_per_mode": 400,
                       "full_state_id": 0},
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("conditional-velocity canonical specification changed")
    reference = read_json(ROOT / spec["reference_model_config"])
    require_architecture_identity(reference["model"])
    if (reference["model"]["camera_view"] != "agentview"
            or reference["model"]["procedure_blocks"] != 2
            or reference["experiment"]["language_content_path"] is not False):
        raise ValueError("conditional-velocity canonical contract changed")
    return spec, reference


def require_frozen(output: Path, mode: str) -> dict:
    spec, _ = contract()
    if mode not in ("V", "L") or output.resolve() != (Path(spec["run_root"]) / mode).resolve():
        raise ValueError("formal V/L output must use the registered data1 mode root")
    state = git_state(ROOT)
    pushed = subprocess.run(["git", "merge-base", "--is-ancestor", state["commit"],
                             "origin/main"], cwd=ROOT, check=False)
    if state["branch"] or state["dirty_paths"] or pushed.returncode != 0:
        raise ValueError("formal GPU work requires a clean pushed detached main ancestor")
    return state


@dataclass
class VelocityRuntime:
    policy: torch.nn.Module
    state: ConditionalVelocityOperator
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device
    mode: str

    def condition(self, videos, task: int, demo: int, language: str):
        tokens, mask, span = self.tokenizer([language])
        if self.mode == "L":
            return (tokens, mask, span), None, 0
        video = videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(self.device, non_blocking=True)
        positions = torch.from_numpy(video.frame_indices).to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], device=self.device, dtype=torch.long)
        return (pixels, positions, offsets, tokens, mask, span), video.raw_frame_count, len(pixels)

    def compile(self, condition: tuple):
        with autocast(self.device):
            return self.state(self.policy, condition)


def build_runtime(asset_root: Path, reference: dict, device: torch.device, *, mode: str,
                  evaluation_policy: dict | None = None) -> VelocityRuntime:
    if mode not in ("V", "L"):
        raise ValueError("unknown conditional-velocity arm")
    source_config = reference["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    tokenizer_path = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    if evaluation_policy is None:
        policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
        processor = Pi05LiberoProcessor(stats, tokenizer_path, 200, str(device))
    else:
        if device != torch.device("cuda:0"):
            raise ValueError("canonical evaluator requires one visible cuda:0")
        from ember.pi05_eval.worker_setup import load_policy as load_eval_policy
        policy, processor, _ = load_eval_policy(
            Path(source["model_path"]), stats, tokenizer_path, evaluation_policy,
        )
    base_lora = load_pi05_lora_contract(asset_root / reuse["lora_contract"])
    lora = derive_pi05_lora_rank(base_lora, rank=135)
    prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    # Shared identity, row readout and U are drawn before mode-specific teaching.
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    common_template = identity_lora_state(derive_pi05_lora_rank(base_lora, rank=128))
    state = ConditionalVelocityOperator(policy, reference["model"], common_template, mode=mode).to(device)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("source physical policy retained trainable parameters")
    return VelocityRuntime(policy, state, Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device)),
                           processor, lora, source, device, mode)


def _gather(value, world_size):
    packets = [None] * world_size
    dist.all_gather_object(packets, value)
    return packets


def _norm(parameters) -> float:
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _gradient_groups(state):
    encoder = state.teaching.semantic_encoder
    return {"common": _norm(state.common.parameters()), "U": _norm(state.U.parameters()),
            "readout": _norm(state.readout.parameters()),
            "text_meta": _norm(encoder.text_meta_lora.parameters()),
            "vl_meta": _norm(encoder.vl_meta_lora.parameters()),
            "action_meta": _norm(encoder.action_meta_lora.parameters()),
            "core": _norm(state.teaching.semantic_core.parameters()),
            "procedure": _norm(state.teaching.procedure.parameters()) if state.teaching.procedure else 0.0}


def _job(runtime: VelocityRuntime, data: VelocityTrainingData, event: dict,
         *, microbatch: int) -> dict:
    tick = time.perf_counter()
    task = event["task"]
    condition, raw_frames, sampled_frames = runtime.condition(
        data.videos, task, event["teacher_demo"], data.tasks[task].authority.language,
    )
    with torch.no_grad():
        generated, coefficient = runtime.compile(condition)
    compile_seconds = time.perf_counter() - tick
    batch = runtime.processor.training_batch(data.batch(event))
    with autocast(runtime.device):
        credit = paired_functional_credit(
            runtime.policy, generated, runtime.lora, batch,
            seed=event["policy_rng_seed"], device=runtime.device, random_batch=28,
            offset=0, microbatch=microbatch, condition_weight=0.25,
        )
    credit_seconds = time.perf_counter() - tick - compile_seconds
    with autocast(runtime.device):
        replay, _ = runtime.compile(condition)
    cotangent = credit["lora_cotangent"]
    if set(cotangent) != set(replay) or any(not torch.isfinite(value).all() for value in cotangent.values()):
        raise ValueError("complete LoRA cotangent is missing or nonfinite")
    torch.autograd.backward(tuple(replay.values()), tuple(cotangent[name] for name in replay))
    torch.cuda.synchronize(runtime.device)
    return {"task": task, "visit": event["visit"], "teacher_demo": event["teacher_demo"],
            "raw_frames": raw_frames, "sampled_frames": sampled_frames,
            "queries": 28, "query_trace": event, "flow_loss": credit["flow_loss"],
            "compiled_forward_calls": credit["compiled_forward_calls"],
            "compile_seconds": compile_seconds, "credit_seconds": credit_seconds,
            "replay_seconds": time.perf_counter() - tick - compile_seconds - credit_seconds,
            "total_seconds": time.perf_counter() - tick,
            "coefficient_norm": float(coefficient.detach().float().norm())}


def _lr_multiplier(step: int, setting: dict) -> float:
    peak, floor = setting["lr"], setting["floor_lr"]
    warmup, cosine = setting["warmup_updates"], setting["cosine_updates"]
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup + 1) / cosine)
    return (floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))) / peak


def _optimizer(state, spec):
    setting = spec["optimizer"]
    params = tuple(parameter for parameter in state.parameters() if parameter.requires_grad)
    if not params:
        raise ValueError("velocity model has no active parameters")
    optimizer = torch.optim.AdamW(params, lr=setting["lr"], betas=tuple(setting["betas"]),
                                  eps=setting["eps"], weight_decay=setting["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_multiplier(step, setting),
    )
    return optimizer, scheduler, params


@dataclass
class TrainSession:
    args: argparse.Namespace
    spec: dict
    context: object
    data: VelocityTrainingData
    runtime: VelocityRuntime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    run_contract: dict


def _prepare_train(args):
    spec, reference = contract()
    git = require_frozen(args.output, args.mode)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("formal velocity training requires same-node world2 with NCCL_P2P_DISABLE=1")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7 + context.rank, context)
    data = VelocityTrainingData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, reference, context.device, mode=args.mode)
    seed_everything(7 + context.rank, context)
    runtime.state.train()
    optimizer, scheduler, parameters = _optimizer(runtime.state, spec)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    topology = {"host": socket.gethostname(), "world_size": 2,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_uuids": _gather(str(torch.cuda.get_device_properties(context.local_rank).uuid), 2),
                "numa_nodes": _gather(context.numa_node, 2)}
    frozen_git = {"commit": git["commit"], "branch": git["branch"],
                  "dirty_paths": git["dirty_paths"], "pushed_ref": "origin/main"}
    run_contract = {"schema_version": SCHEMA, "stage": STAGE, "git": frozen_git,
                    "spec": str(SPEC), "mode": args.mode, "source": runtime.source,
                    "topology": topology, "model": reference["model"],
                    "physical_microbatch": args.microbatch,
                    "lora": runtime.lora.to_dict(), "optimizer": spec["optimizer"],
                    "initialization": spec["initialization"],
                    "sampler": {k: v for k, v in data.events.sampler_state().items() if k != "next_step"},
                    "trainable_parameters": [name for name, value in runtime.state.named_parameters()
                                             if value.requires_grad],
                    "source_trainable": sum(p.numel() for p in runtime.policy.parameters()
                                            if p.requires_grad),
                    "scientific_qualification": True}
    error = None
    try:
        if context.is_main:
            path = args.output / "run_contract.json"
            if path.exists():
                if read_json(path) != run_contract:
                    raise ValueError("existing velocity run contract changed")
            else:
                write_json_atomic(path, run_contract)
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, 2) if value]
    if failures:
        raise RuntimeError(f"run contract failure: {failures}")
    return TrainSession(args, spec, context, data, runtime, optimizer, scheduler,
                        parameters, run_contract)


def _resume_prefix(root: Path, rows: int) -> list[str]:
    prefix = (root / "metrics.jsonl").read_text().splitlines()[:rows]
    if len(prefix) != rows or [json.loads(row)["update"] for row in prefix] != list(range(1, rows + 1)):
        raise ValueError("checkpoint metrics history prefix changed")
    return prefix


def _restore(session):
    if session.args.resume is None:
        return 0, 0
    result, error = None, None
    try:
        parent = session.args.resume.parent.parent
        if read_json(parent / "run_contract.json") != session.run_contract:
            raise ValueError("resume source, mode, git or physical topology changed")
        restored = {}
        updates, rows = load_ecp_checkpoint(
            checkpoint=session.args.resume, stage=STAGE, context=session.context,
            model=session.runtime.state, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA, restored_state=restored,
        )
        session.data.events.restore(restored["sampler_state"])
        if (restored["training_state"] != {"updates": updates}
                or session.scheduler.last_epoch != updates or session.data.events.next_step != updates):
            raise ValueError("optimizer, schedule, sampler or training cursor changed")
        prefix = _resume_prefix(parent, rows)
        if session.context.is_main:
            (session.args.output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
        result = updates, rows
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, 2) if value]
    if failures:
        raise RuntimeError(f"exact world2 restore failed: {failures}")
    return result


def _one_update(session, updates, rows):
    tick = time.perf_counter()
    jobs = session.data.events.event(updates)
    costs = {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             if session.args.mode == "V" else 1 for index, job in enumerate(jobs)}
    assignment = condition_assignment(tuple(costs), costs, world_size=2)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        for index in assignment[session.context.rank]:
            local.append(_job(session.runtime, session.data, jobs[index],
                              microbatch=session.args.microbatch))
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, 2) if value]
    if failures:
        raise RuntimeError(f"conditional-velocity macro failed: {failures}")
    sum_writer_gradients(session.parameters, world_size=2)
    gradients = _gradient_groups(session.runtime.state)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.spec["optimizer"]["clip"], error_if_nonfinite=True,
    ))
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    updates, rows = updates + 1, rows + 1
    packets = _gather(local, 2)
    if session.context.is_main:
        with (session.args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"update": updates, "round": (updates - 1) // 9,
                                     "seconds": time.perf_counter() - tick,
                                     "jobs": [item for packet in packets for item in packet],
                                     "grad_norms_before_clip": gradients,
                                     "total_grad_norm": norm,
                                     "lr_after_step": session.scheduler.get_last_lr()[0]}) + "\n")
    session.data.events.next_step = updates
    if updates in session.spec["checkpoint_updates"]:
        save_ecp_checkpoint(output_dir=session.args.output, macro=updates, stage=STAGE,
                            context=session.context, model=session.runtime.state,
                            optimizer=session.optimizer, scheduler=session.scheduler,
                            run_contract_schema=SCHEMA, metrics_rows=rows,
                            sampler_state=session.data.events.sampler_state(),
                            training_state={"updates": updates})
    return updates, rows


def train(args):
    if args.stop_after not in (90, 180, 270) or not 0 < args.microbatch <= 28:
        raise ValueError("registered first batch stops only at 90/180/270 with microbatch1..28")
    session = _prepare_train(args)
    try:
        updates, rows = _restore(session)
        if updates >= args.stop_after:
            raise ValueError("velocity segment has no registered updates remaining")
        first, started = updates, time.perf_counter()
        while updates < args.stop_after:
            updates, rows = _one_update(session, updates, rows)
        if session.context.is_main:
            write_json_atomic(args.output / "completion.json", {
                "schema_version": SCHEMA, "status": "segment_complete", "mode": args.mode,
                "updates": updates, "segment_updates": updates - first,
                "segment_queries": (updates - first) * 112,
                "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "latest_checkpoint": str(args.output / "checkpoints" / f"macro_{updates:08d}"),
                "scientific_qualification": updates == 270,
            })
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("V", "L"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=270)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--cpu-threads", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
