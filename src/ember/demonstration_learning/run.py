"""One bounded complete-LoRA FM/ECP loop for paired Writer and direct MT."""

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
from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import copy_task_lora_state_, task_lora_state_dict
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
from ember.source_sft.control import clamped_lr_multiplier
from ember.writer.model import DirectLoRAParameters

from .data import TASKS, SUPPORTED, TransferData, audit_two_cycles
from .model import CorrespondenceLoRA


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "configs/demonstration_transfer_v1/learning_spec.json"
RUN_SCHEMA = "ember_demonstration_transfer_formal_stage1_run_v1"
STAGE = "demonstration_transfer_learning_stage1"


def specification() -> dict:
    spec = read_json(SPEC_PATH)
    expected = (
        (spec.get("schema_version"), "ember_demonstration_transfer_learning_spec_v1"),
        (spec.get("task"), "demonstration_transfer_learning_stage1_20260927"),
        (tuple(spec["data"]["task_ids"]), TASKS),
        (tuple(spec["data"]["new_query_task_ids"]), SUPPORTED),
        (spec["data"]["new_crossing_rows"], 296),
        (spec["data"]["original_teacher_demos"], list(range(46))),
        (spec["data"]["original_action_demos"], list(range(46))),
        (spec["data"]["new_teacher_demos"], [0, 1, 2, 3]),
        (spec["data"]["frame_stride"], 5),
        (spec["execution"]["actual_macro_updates_total"], 576),
        (spec["execution"]["actual_training_queries_total"], 64512),
        (spec["execution"]["training_arms"], ["P", "I"]),
        (spec["execution"]["fresh_macro_updates_per_arm"], 288),
        (spec["execution"]["resume_arms"], ["P", "I"]),
        (spec["execution"]["checkpoint_macros"], [72, 144, 216, 288]),
        (spec["execution"]["allowed_world_sizes"], [2]),
        (spec["execution"]["initial_policy_microbatch"], 28),
        (spec["execution"]["oom_only_policy_microbatches"], [14, 7]),
        (spec["bank"]["arms"], ["P", "I"]),
        (spec["bank"]["source_macro"], 288),
        (spec["bank"]["task_ids"], [3, 6, 11, 16, 23, 26, 31, 39]),
        (spec["bank"]["init_state_ids"], list(range(50))),
        (spec["bank"]["conditions_each"], 400),
        (spec["bank"]["video_schedule"]["mode"], "correct"),
        (spec["bank"]["video_schedule"]["seed"], 7),
        (spec["bank"]["video_schedule"]["cardinality"], 1),
        (spec["bank"]["video_schedule"]["demos"], list(range(50))),
        (spec["scene"]["count"], 400),
        (spec["scene"]["task_ids"], [3, 6, 11, 16, 23, 26, 31, 39]),
        (spec["scene"]["init_state_ids"], list(range(50))),
        (spec["official_interface"]["arms"], ["P288", "I288"]),
        (spec["official_interface"]["episodes_per_arm"], 400),
        (spec["official_interface"]["full_episodes_per_arm"], 8),
        (spec["official_interface"]["compact_episodes_per_arm"], 392),
        (spec["operator"]["complete_rank"], 144),
        (spec["operator"]["target_count"], 38),
        (spec["operator"]["alpha"], 144),
        (spec["operator"]["template_identity_seed"], 20260721),
        (spec["mt"]["rank"], 128), (spec["mt"]["queries_per_update"], 576),
        (spec["runtime"]["run_root"],
         "/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/stage1"),
    )
    if (not all(actual == wanted for actual, wanted in expected)
            or spec["optimization"]["extra_auxiliary_loss"] is not False):
        raise ValueError("formal complete-LoRA learning specification changed")
    require_architecture_identity(spec["model"])
    if spec["model"]["camera_view"] != "agentview" or spec["model"]["max_frames_per_encoder_call"] != 8:
        raise ValueError("legal native teaching encoder topology changed")
    return spec


def _frozen_git() -> dict:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"]:
        raise ValueError("formal execution requires a clean detached frozen checkout")
    remote = subprocess.run(["git", "branch", "-r", "--contains", state["commit"]],
                            cwd=REPO, text=True, capture_output=True, check=True).stdout.splitlines()
    if not any(line.strip() == "origin/main" for line in remote):
        raise ValueError("frozen formal commit is not contained in origin/main")
    return {"commit": state["commit"], "branch": "", "dirty_paths": [],
            "pushed_ref": "origin/main"}


def _gather(value, world_size: int):
    if world_size == 1:
        return [value]
    packets = [None] * world_size
    dist.all_gather_object(packets, value)
    return packets


@dataclass
class Runtime:
    policy: torch.nn.Module
    state: torch.nn.Module
    tokenizer: Pi05TeacherPrefixTokenizer | None
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device
    source_identity: dict[str, torch.Tensor]
    source_identity_restores: int = 0
    arm: str = "P"

    def condition(self, data, task: int, demo: int) -> tuple[tuple, int, int]:
        video = data.videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(self.device, non_blocking=True)
        positions = torch.from_numpy(video.frame_indices).to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], dtype=torch.long, device=self.device)
        tokens, mask, span = self.tokenizer([data.tasks[task].authority.language])
        return (pixels, positions, offsets, tokens, mask, span), video.raw_frame_count, len(pixels)

    def compile(self, condition: tuple | None) -> dict[str, torch.Tensor]:
        self.restore_source_identity()
        with autocast(self.device):
            return self.state() if self.arm == "M" else self.state(self.policy, condition)

    def restore_source_identity(self) -> None:
        copy_task_lora_state_(self.policy, self.source_identity, self.lora)
        self.source_identity_restores += 1

    def source_delta_norm(self) -> float:
        """One bounded interface check of physical source identity between compiles."""
        physical = task_lora_state_dict(self.policy)
        values = [(physical[name].detach().float() - reference.float()).norm()
                  for name, reference in self.source_identity.items()]
        return float(torch.stack(values).norm())


def build_runtime(asset_root: Path, spec: dict, device: torch.device, *, arm: str = "P",
                  evaluation: bool = False, frame_chunk: int = 8) -> Runtime:
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
    lora = derive_pi05_lora_rank(base, rank=128 if arm == "M" else 144)
    full_template = prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    torch.manual_seed(int(spec["model"]["initialization_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(spec["model"]["initialization_seed"]))
    state = (DirectLoRAParameters(full_template) if arm == "M" else
             CorrespondenceLoRA(policy, {**spec["model"], "max_frames_per_encoder_call": frame_chunk},
                                full_template)).to(device)
    if any(p.requires_grad for p in policy.parameters()):
        raise ValueError("physical source retained a trainable parameter")
    tokenizer = None if arm == "M" else Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device))
    return Runtime(policy, state, tokenizer, processor, lora, source, device, full_template, arm=arm)


def _optimizer(state: torch.nn.Module, spec: dict):
    opt = spec["optimization"]
    parameters = tuple(p for p in state.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: clamped_lr_multiplier(
            step, warmup=opt["warmup_updates"], decay=1200,
            peak=opt["lr"], floor=1e-5))
    return optimizer, scheduler, parameters


def _norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _gradient_groups(state: torch.nn.Module) -> dict[str, float]:
    if isinstance(state, DirectLoRAParameters):
        return {"shared_complete_lora": _norm(state)}
    encoder = state.writer.semantic_encoder
    return {"common": _norm(state.common), "factor_heads": _norm(state.writer.factor_heads),
            "text_meta": _norm(encoder.text_meta_lora), "vl_meta": _norm(encoder.vl_meta_lora),
            "action_meta": _norm(encoder.action_meta_lora),
            "core": _norm(state.writer.semantic_core), "procedure": _norm(state.writer.procedure),
            "compiler": _norm(state.writer.compiler)}


def _one_job(runtime: Runtime, data: TransferData, event: dict, microbatch: int) -> dict:
    tick = time.perf_counter()
    if runtime.arm == "M":
        condition, raw_frames, sampled_frames = None, 0, 0
    else:
        condition, raw_frames, sampled_frames = runtime.condition(data, event["task"], event["teacher_demo"])
    with torch.no_grad():
        state = runtime.compile(condition)
    torch.cuda.synchronize(runtime.device)
    compile_seconds = time.perf_counter() - tick
    segments = data.physical_segments(event) if runtime.arm == "M" else ((0, data.batch(event)),)
    cotangent, loss, calls = {}, 0.0, 0
    for offset, raw in segments:
        batch = runtime.processor.training_batch(raw)
        count = int(batch["action"].shape[0])
        with autocast(runtime.device):
            credit = paired_functional_credit(
                runtime.policy, state, runtime.lora, batch,
                seed=event["flow_seed"], device=runtime.device,
                random_batch=16 if runtime.arm == "M" else 28,
                offset=offset, microbatch=min(microbatch, count),
                condition_weight=(count / 16 / 36 if runtime.arm == "M" else 0.25))
        for name, value in credit["lora_cotangent"].items():
            cotangent[name] = cotangent.get(name, 0) + value
        loss += credit["flow_loss"] * (count / 16 if runtime.arm == "M" else 1)
        calls += credit["compiled_forward_calls"]
    torch.cuda.synchronize(runtime.device)
    credit_seconds = time.perf_counter() - tick - compile_seconds
    with autocast(runtime.device):
        replay = runtime.compile(condition)
    if set(cotangent) != set(replay) or any(not torch.isfinite(v).all() for v in cotangent.values()):
        raise ValueError("final complete-LoRA FM cotangent is missing or nonfinite")
    torch.autograd.backward(tuple(replay.values()),
                            tuple(cotangent[name].to(replay[name]) for name in replay))
    torch.cuda.synchronize(runtime.device)
    return {"task": event["task"], "visit": event.get("visit"), "kind": event["kind"],
            "source": event.get("source"), "independent_reference": event.get("independent_reference"),
            "teacher_demo": event.get("teacher_demo"), "raw_frames": raw_frames,
            "sampled_frames": sampled_frames, "queries": len(event["queries"]),
            "flow_seed": event["flow_seed"], "flow_loss": loss,
            "compiled_forward_calls": calls,
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


def _registered_resume(checkpoint: Path, contract: dict, spec: dict) -> int:
    """Admit only the latest complete same-arm ECP under this formal frozen run."""
    attempts = Path(spec["runtime"]["run_root"]) / contract["arm"] / "train" / "attempts"
    if (attempts.parent / "final_checkpoint.json").exists():
        raise ValueError("completed formal arm cannot be resumed again")
    checkpoint = checkpoint.resolve()
    if (checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.parent != attempts.resolve()
            or not (checkpoint / "checkpoint_manifest.json").is_file()):
        raise ValueError("resume source is outside this arm's formal attempt ECPs")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    cursor = manifest.get("next_macro")
    if (cursor not in spec["execution"]["checkpoint_macros"]
            or checkpoint.name != f"macro_{cursor:08d}"
            or manifest.get("stage") != STAGE or manifest.get("run_contract_schema") != RUN_SCHEMA):
        raise ValueError("resume ECP stage or registered cursor changed")
    latest = max((int(row.name.removeprefix("macro_"))
                  for row in attempts.glob("*/checkpoints/macro_*")
                  if row.name.removeprefix("macro_").isdigit()
                  and (row / "checkpoint_manifest.json").is_file()), default=0)
    if latest != cursor:
        raise ValueError("resume must use this arm's latest registered complete ECP")
    parent = read_json(checkpoint.parent.parent / "run_contract.json")
    for field in contract:
        if field not in ("microbatch", "frame_chunk") and parent.get(field) != contract[field]:
            raise ValueError(f"formal resume changed {field}")
    allowed_batch = [28, 14, 7]
    if (parent["microbatch"] not in allowed_batch or contract["microbatch"] not in allowed_batch
            or allowed_batch.index(contract["microbatch"]) < allowed_batch.index(parent["microbatch"])
            or allowed_batch.index(contract["microbatch"]) - allowed_batch.index(parent["microbatch"]) > 1
            or parent["frame_chunk"] not in (8, 4) or contract["frame_chunk"] not in (8, 4)
            or contract["frame_chunk"] > parent["frame_chunk"]):
        raise ValueError("physical FM chunks can only shrink after OOM")
    return cursor


def _prepare_train(spec: dict, args) -> Session:
    git = _frozen_git()
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("formal training requires same-node world2 with NCCL P2P disabled")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(int(spec["optimization"]["seed"]), context)
    data = TransferData(args.asset_root, spec, arm=args.arm)
    runtime = build_runtime(args.asset_root, spec, context.device, arm=args.arm,
                            frame_chunk=args.frame_chunk)
    runtime.state.train()
    seed_everything(int(spec["optimization"]["seed"]), context)
    optimizer, scheduler, parameters = _optimizer(runtime.state, spec)
    output = Path(spec["runtime"]["run_root"]) / args.arm / "train" / "attempts" / args.attempt
    if (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("formal attempt already has retained evidence")
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
                "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
                "sampler": {k: v for k, v in data.events.sampler_state().items() if k != "next_step"},
                "trainable_names": [name for name, p in runtime.state.named_parameters() if p.requires_grad],
                "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
                "source_identity_before_every_compile": True,
                "information_wall": "teacher exact language+agentview RGB/positions only; own query RGB/state/actions only to FM",
                "qualification": "formal_stage1_macro288"}
    error = None
    try:
        if context.is_main:
            if (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
                raise ValueError("formal attempt already has retained evidence")
            if (output.parent.parent / "final_checkpoint.json").exists():
                raise ValueError("formal arm already completed macro288")
            if args.resume:
                _registered_resume(args.resume, contract, spec)
            elif args.attempt != "fresh":
                raise ValueError("fresh formal training must use the fresh attempt")
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
        if (updates not in session.spec["execution"]["checkpoint_macros"] or rows != updates
                or restored["training_state"] != {"updates": updates, "arm": session.arm}
                or session.scheduler.last_epoch != updates or session.data.events.next_step != updates):
            raise ValueError("ECP optimizer/scheduler/sampler/RNG cursor changed")
        parent_rows = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        prefix = parent_rows[:rows]
        if len(prefix) != rows or [json.loads(row)["update"] for row in prefix] != list(range(1, rows + 1)):
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
    jobs = session.data.events.event(update) if session.arm == "M" else session.data.events.event(update, session.arm)
    costs = ({index: 16 for index, job in enumerate(jobs)} if session.arm == "M" else
             {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
              for index, job in enumerate(jobs)})
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
        raise RuntimeError(f"formal paired macro failed on a rank: {failures}")
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
            "update": update, "arm": session.arm, "queries": 576 if session.arm == "M" else 112,
            "jobs": [r for p in packets for r in p],
            "lr_applied": lr, "lr_next": session.scheduler.get_last_lr()[0],
            "grad_norms_before_clip": gradients, "total_grad_norm": norm,
            "rank_memory": memory, "seconds": time.perf_counter() - tick})
    rows += 1
    if update in session.spec["execution"]["checkpoint_macros"]:
        save_ecp_checkpoint(
            output_dir=session.output, macro=update, stage=STAGE, context=session.context,
            model=session.runtime.state, optimizer=session.optimizer, scheduler=session.scheduler,
            run_contract_schema=RUN_SCHEMA, metrics_rows=rows,
            sampler_state=session.data.events.sampler_state(),
            training_state={"updates": update, "arm": session.arm})
    return update, rows


def train(spec: dict, args) -> None:
    if (args.arm not in ("P", "I") or args.stop_after != 288
            or args.frame_chunk not in (8, 4)
            or args.microbatch not in spec["execution"]["oom_only_policy_microbatches"]
            + [spec["execution"]["initial_policy_microbatch"]]
            or (args.resume is None and (args.microbatch, args.frame_chunk) != (28, 8))
            or (args.resume is not None and args.attempt == "fresh")
            or not args.attempt or "/" in args.attempt or args.attempt in (".", "..")):
        raise ValueError("formal P/I requires macro288 and registered physical chunks")
    session = _prepare_train(spec, args)
    try:
        updates, rows = _restore(session, args.resume) if args.resume else (0, 0)
        restored_cursor = updates
        started = time.perf_counter()
        while updates < 288:
            updates, rows = _step(session, updates, rows)
        if session.context.is_main:
            checkpoint = session.output / "checkpoints" / "macro_00000288"
            write_json_atomic(session.output / "completion.json", {
                "schema_version": RUN_SCHEMA, "status": "formal_training_complete", "arm": args.arm,
                "updates": updates, "actual_segment_updates": updates - restored_cursor,
                "actual_segment_queries": (updates - restored_cursor) * 112,
                "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "checkpoint": str(checkpoint), "scientific_qualification": "pending_correct400"})
            write_json_atomic(Path(spec["runtime"]["run_root"]) / args.arm / "train" /
                              "final_checkpoint.json", {"arm": args.arm, "checkpoint": str(checkpoint),
                                                        "run_contract": str(session.output / "run_contract.json")})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def audit(spec: dict, args) -> None:
    data = TransferData(args.asset_root, spec)
    try:
        result = audit_two_cycles(data.events)
        print(json.dumps(result, sort_keys=True))
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("P", "I"))
    parser.add_argument("--stop-after", type=int, default=288)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--attempt", default="fresh")
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    spec = specification()
    if args.phase == "audit":
        audit(spec, args)
    elif args.phase == "train":
        train(spec, args)


if __name__ == "__main__":
    main()
