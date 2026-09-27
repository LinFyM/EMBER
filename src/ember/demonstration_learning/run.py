"""Bounded P/I complete-LoRA engineering: audit, 4+2-to-4, profile and two cases."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (initialize_deferred_process_group, initialize_distributed,
                                      load_policy, seed_everything)
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast, require_architecture_identity
from ember.writer.task_execution import condition_assignment

from .data import TASKS, SUPPORTED, TransferData, audit_full_cycle
from .model import CorrespondenceLoRA


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "configs/demonstration_transfer_v1/learning_engineering_spec.json"
RUN_SCHEMA = "ember_demonstration_transfer_learning_engineering_run_v1"
STAGE = "demonstration_transfer_learning_engineering"


def specification() -> dict:
    spec = read_json(SPEC_PATH)
    if (spec.get("schema_version") != "ember_demonstration_transfer_learning_engineering_spec_v1"
            or spec.get("task") != "demonstration_transfer_learning_engineering_20260927"
            or tuple(spec["data"]["task_ids"]) != TASKS
            or tuple(spec["data"]["new_query_task_ids"]) != SUPPORTED
            or spec["data"]["new_crossing_rows"] != 296
            or spec["data"]["original_teacher_demos"] != list(range(46))
            or spec["data"]["original_action_demos"] != list(range(46))
            or spec["data"]["new_teacher_demos"] != [0, 1, 2, 3]
            or spec["data"]["frame_stride"] != 5
            or spec["execution"]["actual_macro_updates_total"] != 12
            or spec["execution"]["actual_training_queries_total"] != 1344
            or spec["operator"]["complete_rank"] != 144
            or spec["operator"]["target_count"] != 38
            or spec["operator"]["alpha"] != 144
            or spec["operator"]["template_identity_seed"] != 20260721
            or spec["optimization"]["extra_auxiliary_loss"] is not False
            or spec["runtime"]["run_root"] != "/data1/user/ymdai/ember_runs/demonstration_transfer_learning_engineering_20260927"):
        raise ValueError("bounded complete-LoRA engineering specification changed")
    require_architecture_identity(spec["model"])
    if spec["model"]["camera_view"] != "agentview" or spec["model"]["max_frames_per_encoder_call"] != 8:
        raise ValueError("legal native teaching encoder topology changed")
    return spec


def _frozen_git() -> dict:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"]:
        raise ValueError("GPU engineering requires a clean detached frozen checkout")
    remote = subprocess.run(["git", "branch", "-r", "--contains", state["commit"]],
                            cwd=REPO, text=True, capture_output=True, check=True).stdout.splitlines()
    if not any(line.strip() == "origin/codex/demonstration-transfer" for line in remote):
        raise ValueError("frozen engineering commit is not pushed to its owner branch")
    return {"commit": state["commit"], "branch": "", "dirty_paths": [],
            "pushed_ref": "origin/codex/demonstration-transfer"}


def _gather(value, world_size: int):
    if world_size == 1:
        return [value]
    packets = [None] * world_size
    dist.all_gather_object(packets, value)
    return packets


@dataclass
class Runtime:
    policy: torch.nn.Module
    state: CorrespondenceLoRA
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device

    def condition(self, data: TransferData, task: int, demo: int) -> tuple[tuple, int, int]:
        video = data.videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(self.device, non_blocking=True)
        positions = torch.from_numpy(video.frame_indices).to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], dtype=torch.long, device=self.device)
        tokens, mask, span = self.tokenizer([data.tasks[task].authority.language])
        return (pixels, positions, offsets, tokens, mask, span), video.raw_frame_count, len(pixels)

    def compile(self, condition: tuple) -> dict[str, torch.Tensor]:
        with autocast(self.device):
            return self.state(self.policy, condition)


def build_runtime(asset_root: Path, spec: dict, device: torch.device, *, evaluation: bool = False) -> Runtime:
    source_config = spec["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    tokenizer_path = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    if evaluation:
        from ember.pi05_eval.worker_setup import load_policy as load_eval_policy

        recipe = read_json(asset_root / "configs/pi05_target_evaluation_v1.json")
        policy, processor, _ = load_eval_policy(
            Path(source["model_path"]), stats, tokenizer_path, recipe["policy"])
    else:
        policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
        processor = Pi05LiberoProcessor(stats, tokenizer_path, 200, str(device))
    base = load_pi05_lora_contract(asset_root / reuse["lora_contract"])
    lora = derive_pi05_lora_rank(base, rank=144)
    full_template = prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    torch.manual_seed(int(spec["model"]["initialization_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["model"]["initialization_seed"]))
    state = CorrespondenceLoRA(policy, spec["model"], full_template).to(device)
    if any(p.requires_grad for p in policy.parameters()):
        raise ValueError("physical source retained a trainable parameter")
    return Runtime(policy, state, Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device)),
                   processor, lora, source, device)


def _optimizer(state: CorrespondenceLoRA, spec: dict):
    opt = spec["optimization"]
    parameters = tuple(p for p in state.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])

    def warmup(step: int) -> float:
        if step >= opt["warmup_updates"]:
            raise ValueError("post-warmup learning is outside this engineering batch")
        return (step + 1) / (opt["warmup_updates"] + 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)
    return optimizer, scheduler, parameters


def _norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _gradient_groups(state: CorrespondenceLoRA) -> dict[str, float]:
    encoder = state.writer.semantic_encoder
    return {"common": _norm(state.common), "factor_heads": _norm(state.writer.factor_heads),
            "text_meta": _norm(encoder.text_meta_lora), "vl_meta": _norm(encoder.vl_meta_lora),
            "action_meta": _norm(encoder.action_meta_lora),
            "core": _norm(state.writer.semantic_core), "procedure": _norm(state.writer.procedure),
            "compiler": _norm(state.writer.compiler)}


def _one_job(runtime: Runtime, data: TransferData, event: dict, microbatch: int) -> dict:
    tick = time.perf_counter()
    condition, raw_frames, sampled_frames = runtime.condition(data, event["task"], event["teacher_demo"])
    with torch.no_grad():
        state = runtime.compile(condition)
    torch.cuda.synchronize(runtime.device)
    compile_seconds = time.perf_counter() - tick
    raw = data.batch(event)
    batch = runtime.processor.training_batch(raw)
    with autocast(runtime.device):
        credit = paired_functional_credit(
            runtime.policy, state, runtime.lora, batch,
            seed=event["flow_seed"], device=runtime.device, random_batch=28,
            offset=0, microbatch=microbatch, condition_weight=0.25)
    torch.cuda.synchronize(runtime.device)
    credit_seconds = time.perf_counter() - tick - compile_seconds
    cotangent = credit["lora_cotangent"]
    with autocast(runtime.device):
        replay = runtime.compile(condition)
    if set(cotangent) != set(replay) or any(not torch.isfinite(v).all() for v in cotangent.values()):
        raise ValueError("final rank144 FM cotangent is missing or nonfinite")
    torch.autograd.backward(tuple(replay.values()),
                            tuple(cotangent[name].to(replay[name]) for name in replay))
    torch.cuda.synchronize(runtime.device)
    return {"task": event["task"], "visit": event["visit"], "kind": event["kind"],
            "source": event.get("source"), "independent_reference": event.get("independent_reference"),
            "teacher_demo": event["teacher_demo"], "raw_frames": raw_frames,
            "sampled_frames": sampled_frames, "queries": len(event["queries"]),
            "flow_seed": event["flow_seed"], "flow_loss": credit["flow_loss"],
            "compiled_forward_calls": credit["compiled_forward_calls"],
            "compile_seconds": compile_seconds, "fm_seconds": credit_seconds,
            "replay_seconds": time.perf_counter() - tick - compile_seconds - credit_seconds,
            "total_seconds": time.perf_counter() - tick,
            "fm_cotangent_norm": float(torch.stack([v.norm() for v in cotangent.values()]).norm())}


@dataclass
class Session:
    spec: dict
    context: object
    data: TransferData
    runtime: Runtime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    output: Path
    contract: dict
    arm: str
    microbatch: int


def _prepare_train(spec: dict, args) -> Session:
    git = _frozen_git()
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size not in (1, 2) or (context.world_size == 2 and os.environ.get("NCCL_P2P_DISABLE") != "1"):
        raise ValueError("engineering training requires one or two same-node A40 ranks")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(int(spec["optimization"]["seed"]), context)
    data = TransferData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device)
    runtime.state.train()
    seed_everything(int(spec["optimization"]["seed"]), context)
    optimizer, scheduler, parameters = _optimizer(runtime.state, spec)
    output = Path(spec["runtime"]["run_root"]) / args.arm / ("resume" if args.resume else "fresh")
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    local = {"rank": context.rank, "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "ranks": _gather(local, context.world_size)}
    contract = {"schema_version": RUN_SCHEMA, "stage": STAGE, "git": git,
                "spec": str(SPEC_PATH), "arm": args.arm, "source": runtime.source,
                "model": spec["model"], "lora": runtime.lora.to_dict(),
                "optimizer": spec["optimization"], "topology": topology,
                "microbatch": args.microbatch, "frame_chunk": spec["execution"]["initial_frame_chunk"],
                "sampler": {k: v for k, v in data.events.sampler_state().items() if k != "next_step"},
                "trainable_names": [name for name, p in runtime.state.named_parameters() if p.requires_grad],
                "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
                "information_wall": "teacher exact language+agentview RGB/positions only; own query RGB/state/actions only to FM",
                "qualification": False}
    error = None
    try:
        if context.is_main:
            if args.resume:
                parent = args.resume.parent.parent
                if args.resume != (Path(spec["runtime"]["run_root"]) / args.arm /
                                   "fresh" / "checkpoints" / "macro_00000002"):
                    raise ValueError("resume is not this arm's registered fresh macro2")
                if read_json(parent / "run_contract.json") != contract:
                    raise ValueError("same-arm source, parameter, sampler or physical topology changed")
            write_json_atomic(output / "run_contract.json", contract)
    except Exception:
        error = traceback.format_exc()
    failures = [item for item in _gather(error, context.world_size) if item]
    if failures:
        raise RuntimeError(f"run contract failure: {failures}")
    return Session(spec, context, data, runtime, optimizer, scheduler, parameters,
                   output, contract, args.arm, args.microbatch)


def _restore(session: Session, checkpoint: Path) -> tuple[int, int]:
    restored, result, error = {}, None, None
    try:
        updates, rows = load_ecp_checkpoint(
            checkpoint=checkpoint, stage=STAGE, context=session.context,
            model=session.runtime.state, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=RUN_SCHEMA,
            restored_state=restored)
        session.data.events.restore(restored["sampler_state"])
        if (updates != 2 or rows != 2 or restored["training_state"] != {"updates": 2, "arm": session.arm}
                or session.scheduler.last_epoch != 2 or session.data.events.next_step != 2):
            raise ValueError("ECP optimizer/scheduler/sampler/RNG cursor changed")
        parent_rows = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        prefix = parent_rows[:rows]
        if len(prefix) != 2 or [json.loads(row)["update"] for row in prefix] != [1, 2]:
            raise ValueError("checkpoint metrics prefix is not complete and continuous")
        if session.context.is_main:
            (session.output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
        result = updates, rows
    except Exception:
        error = traceback.format_exc()
    failures = [item for item in _gather(error, session.context.world_size) if item]
    if failures:
        raise RuntimeError(f"ECP restore failed on a rank: {failures}")
    return result


def _step(session: Session, update: int, rows: int) -> tuple[int, int]:
    tick = time.perf_counter()
    jobs = session.data.events.event(update, session.arm)
    costs = {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             for index, job in enumerate(jobs)}
    assignment = condition_assignment(tuple(costs), costs, world_size=session.context.world_size)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        for index in assignment[session.context.rank]:
            local.append(_one_job(session.runtime, session.data, jobs[index], session.microbatch))
    except Exception:
        error = traceback.format_exc()
    failures = [item for item in _gather(error, session.context.world_size) if item]
    if failures:
        raise RuntimeError(f"paired engineering macro failed on a rank: {failures}")
    sum_writer_gradients(session.parameters, world_size=session.context.world_size)
    gradients = _gradient_groups(session.runtime.state)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.spec["optimization"]["grad_clip"], error_if_nonfinite=True))
    lr = session.optimizer.param_groups[0]["lr"]
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    update += 1
    session.data.events.next_step = update
    packets = _gather(local, session.context.world_size)
    memory = _gather({"rank": session.context.rank,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(session.context.device) / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(session.context.device) / 2**30},
                     session.context.world_size)
    if session.context.is_main:
        append_jsonl(session.output / "metrics.jsonl", {
            "update": update, "arm": session.arm, "queries": 112, "jobs": [r for p in packets for r in p],
            "lr_applied": lr, "lr_next": session.scheduler.get_last_lr()[0],
            "grad_norms_before_clip": gradients, "total_grad_norm": norm,
            "rank_memory": memory, "seconds": time.perf_counter() - tick})
    rows += 1
    if update in (2, 4):
        save_ecp_checkpoint(
            output_dir=session.output, macro=update, stage=STAGE, context=session.context,
            model=session.runtime.state, optimizer=session.optimizer, scheduler=session.scheduler,
            run_contract_schema=RUN_SCHEMA, metrics_rows=rows,
            sampler_state=session.data.events.sampler_state(),
            training_state={"updates": update, "arm": session.arm})
    return update, rows


def train(spec: dict, args) -> None:
    if (args.arm not in ("P", "I") or args.stop_after != 4
            or args.microbatch not in spec["execution"]["oom_only_policy_microbatches"]
            + [spec["execution"]["initial_policy_microbatch"]]):
        raise ValueError("training requires a fixed arm, four updates and registered physical batch")
    session = _prepare_train(spec, args)
    try:
        updates, rows = _restore(session, args.resume) if args.resume else (0, 0)
        started = time.perf_counter()
        while updates < 4:
            updates, rows = _step(session, updates, rows)
        if session.context.is_main:
            write_json_atomic(session.output / "completion.json", {
                "schema_version": RUN_SCHEMA, "status": "engineering_complete", "arm": args.arm,
                "updates": updates, "actual_segment_updates": updates - (2 if args.resume else 0),
                "actual_segment_queries": (updates - (2 if args.resume else 0)) * 112,
                "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "checkpoint": str(session.output / "checkpoints" / "macro_00000004"),
                "scientific_qualification": False})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def audit(spec: dict, args) -> None:
    data = TransferData(args.asset_root, spec)
    try:
        result = audit_full_cycle(data.events)
        output = Path(spec["runtime"]["run_root"])
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "event_audit.json", result)
        plan = output / "events.jsonl"
        if plan.exists():
            raise ValueError("bounded CPU event plan already exists")
        with plan.open("x", encoding="utf-8") as handle:
            for update in range(288):
                p = data.events.event(update, "P")
                i = data.events.event(update, "I")
                handle.write(json.dumps({"update": update + 1, "P": p, "I_teacher_demos":
                                         [row["teacher_demo"] for row in i]}, sort_keys=True) + "\n")
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train", "profile-cases"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("P", "I"))
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    spec = specification()
    if args.phase == "audit":
        audit(spec, args)
    elif args.phase == "train":
        train(spec, args)
    else:
        from .episode import profile_and_cases
        profile_and_cases(spec, args, _frozen_git(), build_runtime)


if __name__ == "__main__":
    main()
