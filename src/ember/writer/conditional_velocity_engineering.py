"""Bounded conditional-velocity graph, world2 resume and longest-video profile."""

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

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import default_collate

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import (initialize_deferred_process_group, initialize_distributed,
                                      load_policy, seed_everything)
from ember.writer.conditional_velocity import ConditionalVelocityOperator
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import prepare_frozen_writer_policy, task_logical_batch_policy_rng_seed
from ember.writer.learning_data import load_learning_tasks
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast, require_architecture_identity
from ember.writer.task_execution import condition_assignment


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "configs/conditional_velocity_operator_v1/engineering_spec.json"
SCHEMA = "ember_conditional_velocity_operator_engineering_run_v1"
STAGE = "conditional_velocity_operator_engineering"


def contract() -> tuple[dict, dict]:
    spec = read_json(SPEC)
    reference = read_json(ROOT / spec["reference_model_config"])
    require_architecture_identity(reference["model"])
    if (spec["schema_version"] != "ember_conditional_velocity_operator_engineering_v1"
            or spec["train_tasks"] != [2, 12, 22, 32]
            or spec["teacher_demo"] != 0 or spec["query_demos"] != [1, 49]
            or spec["query_count_per_task"] != 28 or spec["sampler_seed"] != 20260927
            or spec["world_size"] != 2 or spec["fresh_updates"] != 4
            or (spec["resume_from"], spec["resume_to"]) != (2, 4)
            or spec["lora"] != {"common_rank": 128, "conditional_rank": 7,
                                "deploy_rank": 135, "targets": 38, "alpha": 135,
                                "conditional_target": "model.action_out_proj"}
            or reference["model"]["camera_view"] != "agentview"
            or reference["model"]["procedure_blocks"] != 2
            or reference["experiment"]["language_content_path"] is not False):
        raise ValueError("conditional-velocity engineering contract changed")
    return spec, reference


def require_frozen(output: Path) -> dict:
    spec, _ = contract()
    if (not Path(spec["run_root"]).is_relative_to("/data1/user/ymdai")
            or not output.resolve().is_relative_to(Path(spec["run_root"]).resolve())):
        raise ValueError("new evidence must remain under the registered data1 root")
    state = git_state(ROOT)
    remote = subprocess.run(
        ["git", "rev-parse", "origin/codex/conditional-velocity-operator"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if state["branch"] or state["dirty_paths"] or state["commit"] != remote:
        raise ValueError("GPU work requires the clean pushed detached implementation commit")
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

    def condition(self, videos: RawTeacherVideoStore, task, demo: int, language: str):
        video = videos.load(task, demo)
        tokens, mask, span = self.tokenizer([language])
        pixels = torch.from_numpy(video.frames).to(self.device, non_blocking=True)
        positions = torch.from_numpy(video.frame_indices).to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], device=self.device, dtype=torch.long)
        return (pixels, positions, offsets, tokens, mask, span), video.raw_frame_count

    def compile(self, condition: tuple):
        with autocast(self.device):
            return self.state(self.policy, condition)


def build_runtime(asset_root: Path, reference: dict, device: torch.device,
                  *, evaluation_policy: dict | None = None) -> VelocityRuntime:
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
    common_template = identity_lora_state(derive_pi05_lora_rank(base_lora, rank=128))
    state = ConditionalVelocityOperator(policy, reference["model"], common_template).to(device)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("source physical policy retained trainable parameters")
    return VelocityRuntime(policy, state, Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device)),
                           processor, lora, source, device)


class EngineeringData:
    """Four fixed train tasks; demo0 video and 28 unique cross-episode queries per macro."""

    def __init__(self, asset_root: Path, spec: dict, task_ids: tuple[int, ...]) -> None:
        self.spec = spec
        self.tasks = load_learning_tasks(asset_root, task_ids, role="train",
                                         protocol_path=spec["protocol"])
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="agentview")
        self.queries = FunctionalQueryDataset(authorities, demo_indices=tuple(range(1, 50)),
                                              action_chunk_size=50, action_start_offset=1)
        self.next_step = 0

    def draw(self, task: int, update: int) -> tuple[dict, dict]:
        if task not in self.tasks or update < 0:
            raise ValueError("unregistered conditional-velocity query")
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.spec["sampler_seed"], update, task],
        ))
        demos = [int(value) for value in rng.choice(np.arange(1, 50), size=28, replace=False)]
        lengths = self.tasks[task].episode_lengths
        frames = [int(rng.integers(lengths[demo] - 1)) for demo in demos]
        rows = self.queries.task_episode_rows
        raw = default_collate([self.queries[rows[task][demo][frame]]
                               for demo, frame in zip(demos, frames, strict=True)])
        trace = {"task": task, "update_index": update, "teacher_demo": 0,
                 "action_demos": demos, "action_frames": frames,
                 "action_start_indices": [frame + 1 for frame in frames],
                 "policy_rng_seed": task_logical_batch_policy_rng_seed(
                     optimization_seed=self.spec["flow_seed"], task_id=task,
                     task_visit=update, demo_indices=demos, frame_indices=frames,
                 )}
        return raw, trace

    def sampler_state(self):
        return {"next_step": self.next_step, "tasks": self.spec["train_tasks"],
                "teacher_demo": 0, "query_demos": [1, 49], "queries_per_task": 28,
                "sampler_seed": self.spec["sampler_seed"], "offset": 1}

    def restore(self, state):
        expected = self.sampler_state()
        if ({key: value for key, value in state.items() if key != "next_step"}
                != {key: value for key, value in expected.items() if key != "next_step"}
                or type(state.get("next_step")) is not int or not 0 <= state["next_step"] <= 4):
            raise ValueError("conditional-velocity sampler contract changed")
        self.next_step = state["next_step"]

    def close(self):
        self.videos.close()
        self.queries.close()


def _gather(value, world_size):
    if world_size == 1:
        return [value]
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
            "readout_output": _norm(state.readout.output.parameters()),
            "text_meta": _norm(encoder.text_meta_lora.parameters()),
            "vl_meta": _norm(encoder.vl_meta_lora.parameters()),
            "action_meta": _norm(encoder.action_meta_lora.parameters()),
            "core": _norm(state.teaching.semantic_core.parameters()),
            "procedure": _norm(state.teaching.procedure.parameters())}


def _job(runtime: VelocityRuntime, data: EngineeringData, task: int, update: int,
         *, microbatch: int, task_weight: float) -> dict:
    tick = time.perf_counter()
    condition, raw_frames = runtime.condition(data.videos, task, 0,
                                               data.tasks[task].authority.language)
    with torch.no_grad():
        generated, coefficient = runtime.compile(condition)
    compile_seconds = time.perf_counter() - tick
    raw, trace = data.draw(task, update)
    batch = runtime.processor.training_batch(raw)
    with autocast(runtime.device):
        credit = paired_functional_credit(
            runtime.policy, generated, runtime.lora, batch,
            seed=trace["policy_rng_seed"], device=runtime.device, random_batch=28,
            offset=0, microbatch=microbatch, condition_weight=task_weight,
        )
    credit_seconds = time.perf_counter() - tick - compile_seconds
    with autocast(runtime.device):
        replay, _ = runtime.compile(condition)
    cotangent = credit["lora_cotangent"]
    if set(cotangent) != set(replay) or any(not torch.isfinite(value).all() for value in cotangent.values()):
        raise ValueError("complete LoRA cotangent is missing or nonfinite")
    torch.autograd.backward(tuple(replay.values()), tuple(cotangent[name] for name in replay))
    torch.cuda.synchronize(runtime.device)
    return {"task": task, "teacher_demo": 0, "raw_frames": raw_frames,
            "sampled_frames": len(condition[0]), "queries": 28, "query_trace": trace,
            "flow_loss": credit["flow_loss"], "compiled_forward_calls": credit["compiled_forward_calls"],
            "compile_seconds": compile_seconds, "credit_seconds": credit_seconds,
            "replay_seconds": time.perf_counter() - tick - compile_seconds - credit_seconds,
            "total_seconds": time.perf_counter() - tick,
            "coefficient_norm": float(coefficient.detach().float().norm())}


def _optimizer(state, spec):
    setting = spec["optimizer"]
    params = tuple(state.parameters())
    optimizer = torch.optim.AdamW(params, lr=setting["lr"], betas=tuple(setting["betas"]),
                                  eps=setting["eps"], weight_decay=setting["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / setting["warmup_updates"], 1.0),
    )
    return optimizer, scheduler, params


@dataclass
class TrainSession:
    args: argparse.Namespace
    spec: dict
    context: object
    data: EngineeringData
    runtime: VelocityRuntime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    run_contract: dict


def _prepare_train(args):
    spec, reference = contract()
    git = require_frozen(args.output)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("conditional-velocity training requires same-node world2 with NCCL_P2P_DISABLE=1")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(spec["flow_seed"] - context.rank, context)
    data = EngineeringData(args.asset_root, spec, tuple(spec["train_tasks"]))
    runtime = build_runtime(args.asset_root, reference, context.device)
    seed_everything(spec["flow_seed"], context)
    runtime.state.train()
    optimizer, scheduler, parameters = _optimizer(runtime.state, spec)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    topology = {"host": socket.gethostname(), "world_size": 2,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_uuids": _gather(str(torch.cuda.get_device_properties(context.local_rank).uuid), 2),
                "numa_nodes": _gather(context.numa_node, 2)}
    frozen_git = {"commit": git["commit"], "branch": git["branch"],
                  "dirty_paths": git["dirty_paths"],
                  "pushed_ref": "origin/codex/conditional-velocity-operator"}
    run_contract = {"schema_version": SCHEMA, "stage": STAGE, "git": frozen_git,
                    "spec": str(SPEC), "source": runtime.source,
                    "topology": topology, "model": reference["model"],
                    "physical_microbatch": args.microbatch,
                    "lora": runtime.lora.to_dict(), "optimizer": spec["optimizer"],
                    "sampler": {k: v for k, v in data.sampler_state().items() if k != "next_step"},
                    "trainable_parameters": [name for name, value in runtime.state.named_parameters()
                                             if value.requires_grad],
                    "source_trainable": sum(p.numel() for p in runtime.policy.parameters()
                                            if p.requires_grad),
                    "scientific_qualification": False}
    error = None
    try:
        if context.is_main:
            path = args.output / "run_contract.json"
            if path.exists():
                if read_json(path) != run_contract:
                    raise ValueError("existing conditional-velocity run contract changed")
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
    parent = (root / "metrics.jsonl").read_text().splitlines()
    prefix = parent[:rows]
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
            raise ValueError("resume source, git or physical topology changed")
        restored = {}
        updates, rows = load_ecp_checkpoint(
            checkpoint=session.args.resume, stage=STAGE, context=session.context,
            model=session.runtime.state, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA, restored_state=restored,
        )
        session.data.restore(restored["sampler_state"])
        if (restored["training_state"] != {"updates": updates}
                or session.scheduler.last_epoch != updates or session.data.next_step != updates):
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
    tasks = session.spec["train_tasks"]
    costs = {index: session.data.videos.frame_counts(task, 0)[1]
             for index, task in enumerate(tasks)}
    assignment = condition_assignment(tuple(costs), costs, world_size=2)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        for job in assignment[session.context.rank]:
            local.append(_job(session.runtime, session.data, tasks[job], updates,
                              microbatch=session.args.microbatch, task_weight=1 / 4))
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
            handle.write(json.dumps({"update": updates, "seconds": time.perf_counter() - tick,
                                     "jobs": [item for packet in packets for item in packet],
                                     "grad_norms_before_clip": gradients,
                                     "total_grad_norm": norm,
                                     "lr_after_step": session.scheduler.get_last_lr()[0]}) + "\n")
    session.data.next_step = updates
    if updates in (2, 4):
        save_ecp_checkpoint(output_dir=session.args.output, macro=updates, stage=STAGE,
                            context=session.context, model=session.runtime.state,
                            optimizer=session.optimizer, scheduler=session.scheduler,
                            run_contract_schema=SCHEMA, metrics_rows=rows,
                            sampler_state=session.data.sampler_state(),
                            training_state={"updates": updates})
    return updates, rows


def train(args):
    if args.stop_after != 4 or not 0 < args.microbatch <= 28:
        raise ValueError("registered four-step run requires a valid 1..28 physical microbatch")
    session = _prepare_train(args)
    try:
        updates, rows = _restore(session)
        if updates >= args.stop_after:
            raise ValueError("engineering segment has no registered updates remaining")
        first, started = updates, time.perf_counter()
        while updates < args.stop_after:
            updates, rows = _one_update(session, updates, rows)
        if session.context.is_main:
            write_json_atomic(args.output / "completion.json", {
                "schema_version": SCHEMA, "status": "segment_complete",
                "updates": updates, "segment_updates": updates - first,
                "segment_queries": (updates - first) * 112,
                "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "latest_checkpoint": str(args.output / "checkpoints" / f"macro_{updates:08d}"),
                "scientific_qualification": False,
            })
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def profile(args):
    spec, reference = contract()
    require_frozen(args.output)
    if args.checkpoint is None or args.output != Path(spec["run_root"]):
        raise ValueError("profile requires the registered root and fresh4 checkpoint")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(spec["flow_seed"])
    runtime = build_runtime(args.asset_root, reference, device)
    runtime.state.load_state_dict(load_file(str(args.checkpoint / "ecp.safetensors"), device="cuda:0"),
                                  strict=True)
    runtime.state.train()
    data = EngineeringData(args.asset_root, spec, (29,))
    try:
        profile_spec = spec["profile"]
        raw, sampled = data.videos.frame_counts(29, 0)
        if (raw, sampled) != (profile_spec["raw_frames"], profile_spec["sampled_frames"]):
            raise ValueError("registered longest video asset identity changed")
        runtime.state.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = _job(runtime, data, 29, 4, microbatch=args.microbatch, task_weight=1.0)
        torch.cuda.synchronize(device)
        write_json_atomic(args.output / "engineering/profile.json", {
            "schema_version": "ember_conditional_velocity_profile_v1", "status": "complete",
            "result": result, "wall_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "gradient_norms": _gradient_groups(runtime.state),
            "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
            "checkpoint": str(args.checkpoint), "optimizer_updates": 0,
            "scientific_qualification": False,
        })
    finally:
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "profile"))
    parser.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    (train if args.phase == "train" else profile)(args)


if __name__ == "__main__":
    main()
