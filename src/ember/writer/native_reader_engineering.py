"""Bounded world2 Reader engineering, profile and canonical train-only episodes."""

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
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
                                      inspect_source_checkpoint, load_evaluation_authorities)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import (initialize_deferred_process_group,
                                      initialize_distributed, load_policy, seed_everything)
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.learning_data import WriterTrainingData
from ember.writer.native_conditional_reader import (NativeConditionalReader,
                                                     NativeTeachingEncoder, TeachingMemory)
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast, require_architecture_identity
from ember.writer.task_execution import condition_assignment
from ember.writer.training import _learning_rate_multiplier


SCHEMA = "ember_native_conditional_reader_engineering_run_v1"
STAGE = "native_conditional_reader_engineering"
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "configs/native_conditional_reader_v1/engineering_spec.json"
REFERENCE = ROOT / "configs/language_content_path_causality_v1/train_C0.json"


class ReaderState(torch.nn.Module):
    def __init__(self, teaching: NativeTeachingEncoder, reader: NativeConditionalReader) -> None:
        super().__init__()
        self.teaching = teaching
        self.reader = reader


@dataclass
class ReaderRuntime:
    policy: torch.nn.Module
    state: ReaderState
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    source: dict
    device: torch.device
    mode: str

    def condition(self, data: WriterTrainingData, task: int, demo: int) -> tuple:
        language = data.tasks[task].authority.language
        tokens, mask, span = self.tokenizer([language])
        if self.mode == "R_L":
            return tokens, mask, span
        frames, positions = data.load_videos(task, (demo,))
        pixels = frames[0].to(self.device, non_blocking=True)
        indices = positions[0].to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], dtype=torch.long, device=self.device)
        return pixels, indices, offsets, tokens, mask, span

    def memory(self, condition: tuple) -> TeachingMemory:
        with autocast(self.device):
            return self.state.teaching(self.policy, condition)


def _contract() -> tuple[dict, dict]:
    spec, config = read_json(SPEC), read_json(REFERENCE)
    require_architecture_identity(config["model"])
    if (spec.get("schema_version") != "ember_native_conditional_reader_engineering_v1"
            or spec["fit_task_ids"] != config["data"]["task_ids"]
            or spec["reference_training_config"] != str(REFERENCE.relative_to(ROOT))
            or spec["assets"]["source_config"] != config["source"]
            or spec["teaching_camera"] != "agentview"
            or spec["native_targets"] != [
                "gemma_expert.model.layers.9.self_attn.q_proj",
                "gemma_expert.model.layers.9.self_attn.v_proj",
            ]):
        raise ValueError("Reader engineering specification changed")
    if (config["model"]["camera_view"] != "agentview"
            or config["model"]["action_horizon"] != 50
            or config["data"]["queries_per_task"] != 21
            or config["data"]["teaching_queries_per_task"] != 7
            or config["optimization"]["teaching_weight"] != 1 / 3
            or config["experiment"]["extra_endpoint_prefix"] is not False):
        raise ValueError("C0 reference events, input or objective changed")
    return spec, config


def require_frozen_data1(spec: dict, *, output: Path, exact_root: bool = False) -> None:
    root = Path(spec["run_root"]).resolve()
    target = output.resolve()
    if (not root.is_relative_to("/data1/user/ymdai")
            or (target != root if exact_root else not target.is_relative_to(root))):
        raise ValueError("Reader outputs must stay under the registered data1 study root")
    state = git_state(ROOT)
    task_remote = subprocess.run(
        ["git", "rev-parse", "origin/codex/native-conditional-reader-engineering-20260926"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout.strip()
    pushed_task_commit = state["commit"] == task_remote
    if (state["branch"] or state["dirty_paths"]
            or not (pushed_task_commit or git_state_is_clean_pushed_or_frozen_authority(state))):
        raise ValueError("GPU engineering requires a clean pushed detached frozen tree")


def build_runtime(asset_root: Path, config: dict, device: torch.device, mode: str,
                  *, evaluation_policy: dict | None = None) -> ReaderRuntime:
    source_config = config["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    tokenizer = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    if evaluation_policy is None:
        policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
        processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(device))
    else:
        if device != torch.device("cuda:0"):
            raise ValueError("canonical evaluator requires one visible cuda:0")
        from ember.pi05_eval.worker_setup import load_policy as load_eval_policy

        policy, processor, _ = load_eval_policy(
            Path(source["model_path"]), stats, tokenizer, evaluation_policy)
    policy.requires_grad_(False).eval()
    policy.model.gradient_checkpointing_disable()
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("source policy is trainable")
    teaching = NativeTeachingEncoder(policy, config["model"], mode=mode)
    # R_V has a Procedure and R_L does not. Seed only the shared reader so its
    # initial common components match without perturbing the query RNG stream.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(int(config["model"]["initialization_seed"]) + 0x9A71)
        reader = NativeConditionalReader()
    if mode == "R_L":
        for module in (reader.read_norm, reader.procedure_norm,
                       reader.procedure_read, reader.procedure_mix):
            module.requires_grad_(False)
    state = ReaderState(teaching, reader).to(device)
    return ReaderRuntime(policy, state,
                         Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)),
                         processor,
                         source, device, mode)


def _grad_norm(parameters) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters
              if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _credit_group(runtime: ReaderRuntime, memory: TeachingMemory, batch: dict, trace: dict,
                  *, weight: float, endpoint: bool, prefix_steps: int | None,
                  microbatch: int) -> tuple[float, dict[str, int]]:
    total = len(trace["action_demos"])
    owner = NativeFlowPrediction(runtime.policy)
    loss_total, counts_total = 0.0, {"q": 0, "v": 0}
    for start in range(0, total, microbatch):
        stop = min(total, start + microbatch)
        sliced = {key: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim
                  and len(value) == total else value for key, value in batch.items()}
        sample = flow_sample(
            runtime.policy, sliced, seed=trace["policy_rng_seed"], device=runtime.device,
            random_batch=trace["policy_random_batch_size"], offset=trace["query_offset"] + start,
            noise_endpoint=endpoint,
        )
        with runtime.state.reader.installed(runtime.policy, memory) as counts, autocast(runtime.device):
            prediction = owner(sample)
            loss = mean_velocity_loss(prediction, sample.target, sample.action_width,
                                      prefix_steps=prefix_steps)
            (loss * weight * (stop - start) / total).backward()
        if not counts["q"] or not counts["v"]:
            raise ValueError("reader missed a native q/v projection in real FM")
        for name in counts_total:
            counts_total[name] += counts[name]
        loss_total += float(loss.detach()) * (stop - start) / total
    return loss_total, counts_total


def _job(runtime: ReaderRuntime, data: WriterTrainingData, draw: dict, config: dict,
         *, microbatch: int, condition_demo: int | None = None) -> dict:
    condition_demo = draw["video_demos"][0] if condition_demo is None else condition_demo
    condition = runtime.condition(data, draw["task"], condition_demo)
    started = time.perf_counter()
    with torch.no_grad():
        memory = runtime.memory(condition)
    memory_seconds = time.perf_counter() - started
    leaf = memory.leaves()
    groups = ((False, draw["query_offset"], draw["query_count"], 1 / 4),
              (True, draw["teaching_offset"], draw["teaching_count"],
               config["optimization"]["teaching_weight"] / 4))
    losses, calls = {}, {"q": 0, "v": 0}
    for teaching, offset, count, weight in groups:
        raw, trace = data.action_batch(draw["task"], draw["occurrence"], draw["video_demos"],
                                       query_seed=draw["query_seed"], query_offset=offset,
                                       query_count=count, teaching=teaching)
        batch = runtime.processor.training_batch(raw)
        loss, consumed = _credit_group(runtime, leaf, batch, trace, weight=weight,
                                       endpoint=teaching and config["experiment"]["extra_endpoint_prefix"],
                                       prefix_steps=5 if teaching and config["experiment"]["extra_endpoint_prefix"] else None,
                                       microbatch=microbatch)
        losses["teaching" if teaching else "main"] = loss
        for name in calls:
            calls[name] += consumed[name]
    cotangents = tuple(value.grad for value in leaf.values())
    if any(value is None or not torch.isfinite(value).all() for value in cotangents):
        raise ValueError("real FM did not return finite teaching-memory cotangents")
    with autocast(runtime.device):
        replay = runtime.memory(condition)
    torch.autograd.backward(replay.values(), cotangents)
    return {"task": draw["task"], "teacher_demo": condition_demo,
            "action_event_teacher": draw["video_demos"][0],
            "occurrence": draw["occurrence"], "query_seed": draw["query_seed"],
            "queries": draw["query_count"] + draw["teaching_count"],
            "memory_seconds": memory_seconds, "flow_loss": losses,
            "native_target_calls": calls, "frames": draw["frames"]}


def _optimizer(state: ReaderState, config: dict):
    opt = config["optimization"]
    parameters = tuple(parameter for parameter in state.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_multiplier(step, opt))
    return optimizer, scheduler, parameters


def _gather(value, world_size):
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


@dataclass
class TrainSession:
    args: argparse.Namespace
    config: dict
    context: object
    data: WriterTrainingData
    runtime: ReaderRuntime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    contract: dict


def _prepare_train(args: argparse.Namespace) -> TrainSession:
    spec, config = _contract()
    if args.mode not in spec["modes"] or args.stop_after not in (2, 4):
        raise ValueError("only registered R_V/R_L two or four macro engineering segments are allowed")
    require_frozen_data1(spec, output=args.output)
    state = git_state(ROOT)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("Reader training and resume require one world2 A40 node")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    data = WriterTrainingData(args.asset_root, config["data"], camera_view="agentview",
                              planned_updates=4, use_videos=args.mode == "R_V")
    runtime = build_runtime(args.asset_root, config, context.device, args.mode)
    runtime.state.train()
    optimizer, scheduler, parameters = _optimizer(runtime.state, config)
    args.output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=args.output)
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_uuids": _gather(str(torch.cuda.get_device_properties(context.local_rank).uuid), 2),
                "numa_nodes": _gather(context.numa_node, 2)}
    contract = {"schema_version": SCHEMA, "stage": STAGE, "git": state,
                "mode": args.mode, "reference": str(REFERENCE), "spec": str(SPEC),
                "source": runtime.source, "topology": topology,
                "queries_per_macro": 112, "training_seed": config["optimization"]["seed"],
                "data_event_contract": data.sampler_state()["event_contract"],
                "trainable_parameters": [name for name, value in runtime.state.named_parameters()
                                         if value.requires_grad],
                "source_trainable": sum(value.numel() for value in runtime.policy.parameters()
                                        if value.requires_grad)}
    if context.is_main:
        path = args.output / "run_contract.json"
        if path.exists():
            if read_json(path) != contract:
                raise ValueError("Reader exact-resume run contract changed")
        else:
            write_json_atomic(path, contract)
    barrier(context)
    return TrainSession(args, config, context, data, runtime, optimizer, scheduler, parameters, contract)


def _restore_train(session: TrainSession) -> tuple[int, int]:
    args = session.args
    if args.resume is None:
        return 0, 0
    if read_json(args.resume.parent.parent / "run_contract.json") != session.contract:
        raise ValueError("Reader resume source, mode or physical topology changed")
    restored = {}
    updates, rows = load_ecp_checkpoint(
        checkpoint=args.resume, stage=STAGE, context=session.context,
        model=session.runtime.state, optimizer=session.optimizer,
        scheduler=session.scheduler, run_contract_schema=SCHEMA, restored_state=restored)
    session.data.restore_sampler(restored["sampler_state"])
    if (restored["training_state"] != {"mode": args.mode, "updates": updates}
            or session.scheduler.last_epoch != updates
            or session.data.sampler_state()["next_step"] != updates):
        raise ValueError("Reader optimizer, schedule or sampler cursor changed")
    if session.context.is_main:
        prior = (args.resume.parent.parent / "metrics.jsonl").read_text().splitlines()
        if len(prior) != rows or [json.loads(row)["update"] for row in prior] != list(range(1, rows + 1)):
            raise ValueError("Reader resume exposure history changed")
        (args.output / "metrics.jsonl").write_text("\n".join(prior) + "\n")
    return updates, rows


def _local_jobs(session: TrainSession, draws: tuple) -> list[dict]:
    costs = {draw["job_id"]: draw["frames"] if session.args.mode == "R_V" else 1
             for draw in draws}
    assignment = condition_assignment(tuple(costs), costs, world_size=2)
    local, error = [], None
    try:
        for job_id in assignment[session.context.rank]:
            local.append(_job(session.runtime, session.data, draws[job_id], session.config,
                              microbatch=session.args.microbatch))
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in _gather(error, 2) if value]
    if failures:
        raise RuntimeError(f"Reader real-FM job failed: {failures}")
    return local


def _gradient_metrics(session: TrainSession) -> dict:
    state = session.runtime.state
    return {"reader": _grad_norm(state.reader.parameters()),
            "text": _grad_norm(state.teaching.semantic_encoder.text_meta_lora.parameters()),
            "vl": _grad_norm(state.teaching.semantic_encoder.vl_meta_lora.parameters()),
            "action": _grad_norm(state.teaching.semantic_encoder.action_meta_lora.parameters()),
            "core": _grad_norm(state.teaching.semantic_core.parameters()),
            "procedure": (_grad_norm(state.teaching.procedure.parameters())
                          if state.teaching.procedure is not None else None)}


def _one_update(session: TrainSession, updates: int, rows: int) -> tuple[int, int]:
    tick = time.perf_counter()
    draws = session.data.next_iteration()
    session.optimizer.zero_grad(set_to_none=True)
    local = _local_jobs(session, draws)
    sum_writer_gradients(session.parameters, world_size=2)
    onset = _gradient_metrics(session)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.config["optimization"]["grad_clip"], error_if_nonfinite=True))
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    updates, rows = updates + 1, rows + 1
    packets = _gather(local, 2)
    if session.context.is_main:
        with (session.args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"update": updates, "mode": session.args.mode,
                                     "seconds": time.perf_counter() - tick,
                                     "jobs": [row for packet in packets for row in packet],
                                     "grad_norms_before_clip": onset,
                                     "total_grad_norm": norm,
                                     "lr": session.scheduler.get_last_lr()[0]}) + "\n")
    if updates in (2, 4):
        save_ecp_checkpoint(output_dir=session.args.output, macro=updates, stage=STAGE,
                            context=session.context, model=session.runtime.state,
                            optimizer=session.optimizer, scheduler=session.scheduler,
                            run_contract_schema=SCHEMA, metrics_rows=rows,
                            sampler_state=session.data.sampler_state(),
                            training_state={"mode": session.args.mode, "updates": updates})
    return updates, rows


def train(args: argparse.Namespace) -> None:
    session = _prepare_train(args)
    try:
        updates, rows = _restore_train(session)
        if updates >= args.stop_after:
            raise ValueError("Reader segment has no registered updates remaining")
        started_at, first = time.perf_counter(), updates
        while updates < args.stop_after:
            updates, rows = _one_update(session, updates, rows)
        if session.context.is_main:
            write_json_atomic(args.output / "completion.json", {
                "schema_version": SCHEMA, "status": "segment_complete", "mode": args.mode,
                "updates": updates, "segment_updates": updates - first,
                "queries": updates * 112, "seconds": time.perf_counter() - started_at,
                "resumed_from": str(args.resume) if args.resume else None,
                "latest_checkpoint": str(args.output / "checkpoints" / f"macro_{updates:08d}"),
                "scientific_qualification": False,
            })
    finally:
        session.data.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", nargs="?")
    parser.add_argument("--mode", choices=("R_V", "R_L"), required=True)
    parser.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
