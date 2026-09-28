"""Bounded T/U engineering: true cross-episode FM, ECP and one canonical case."""

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
from torch.utils.data import default_collate

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import copy_task_lora_state_, task_lora_state_dict, validate_lora_state
from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (initialize_deferred_process_group, initialize_distributed,
                                      load_policy, seed_everything)
from ember.source_sft.control import clamped_lr_multiplier
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import prepare_frozen_writer_policy, task_logical_batch_policy_rng_seed
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast
from ember.writer.task_execution import condition_assignment

from .model import OperatorReadWrite
from .native import read_native_video


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "configs/operator_read_write_v1/engineering_spec.json"
SCHEMA = "ember_operator_read_write_run_v1"
STAGE = "operator_read_write_engineering"
TASKS = (2, 29, 38, 97)


def specification() -> dict:
    spec = read_json(SPEC_PATH)
    if (spec.get("schema_version") != "ember_operator_read_write_engineering_v1"
            or spec.get("task") != "operator_read_write_engineering_20260928"
            or tuple(spec["events"]["task_ids"]) != TASKS
            or spec["events"]["teacher_demos_by_update"] != [0, 1, 2, 3]
            or spec["events"]["query_action_offset"] != 1
            or spec["operator"]["rank"] != 128 or spec["operator"]["targets"] != 38
            or spec["operator"]["probe_shape"] != [50, 32]
            or spec["operator"]["teaching_camera"] != "dual"
            or spec["operator"]["frame_stride"] != 5
            or spec["operator"]["frame_chunk"] != 8
            or spec["execution"]["actual_updates_total"] != 10
            or spec["execution"]["actual_queries_total"] != 1120
            or spec["execution"]["checkpoints"] != [2, 4]
            or spec["execution"]["world_size"] != 2):
        raise ValueError("bounded operator read/write contract changed")
    return spec


def frozen_git() -> dict:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"]:
        raise ValueError("GPU calculation requires clean detached frozen source")
    refs = subprocess.run(["git", "branch", "-r", "--contains", state["commit"]], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    if not any(row.strip() == "origin/codex/demonstration-transfer" for row in refs):
        raise ValueError("frozen code commit must be pushed to executor branch")
    return {"commit": state["commit"], "branch": "", "dirty_paths": [],
            "pushed_ref": "origin/codex/demonstration-transfer"}


def gather(value, world: int):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


class EngineeringData:
    """Only action-hidden RGB is teaching; own HDF5 state/action is query label."""

    def __init__(self, asset_root: Path, spec: dict, *, query_labels: bool = True) -> None:
        from ember.writer.learning_data import load_learning_tasks

        self.tasks = load_learning_tasks(asset_root, TASKS, role="train",
                                         protocol_path=spec["source"]["data_protocol"])
        authorities = tuple(row.authority for row in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
        self.queries = (FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                              action_chunk_size=50, action_start_offset=1)
                        if query_labels else None)
        self.rows = self.queries.task_episode_rows if self.queries is not None else None
        self.seed = int(spec["events"]["seed"])
        self.next_step = 0

    def event(self, step: int, task: int, *, teacher: int | None = None) -> dict:
        if task not in TASKS or step not in range(4):
            raise ValueError("event is outside four fixed train tasks/updates")
        teacher = step if teacher is None else teacher
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, task, step]))
        demos = rng.choice(np.asarray([i for i in range(50) if i != teacher]),
                           size=28, replace=False)
        lengths = self.tasks[task].episode_lengths
        frames = [int(rng.integers(lengths[int(demo)] - 1)) for demo in demos]
        demos = [int(demo) for demo in demos]
        flow_seed = task_logical_batch_policy_rng_seed(
            optimization_seed=7, task_id=task, task_visit=step,
            demo_indices=demos, frame_indices=frames)
        return {"update": step + 1, "task": task, "teacher_demo": teacher,
                "queries": [{"demo": demo, "frame": frame}
                            for demo, frame in zip(demos, frames, strict=True)],
                "flow_seed": flow_seed, "query_offset": 1}

    def sampler_state(self) -> dict:
        return {"schema_version": "ember_operator_read_write_events_v1", "next_step": self.next_step,
                "seed": self.seed, "tasks": list(TASKS), "teacher_demos": [0, 1, 2, 3],
                "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28}

    def restore(self, state: dict) -> None:
        expected = self.sampler_state()
        if (state.get("next_step") != 2 or {k: v for k, v in state.items() if k != "next_step"}
                != {k: v for k, v in expected.items() if k != "next_step"}):
            raise ValueError("operator sampler identity or ECP2 cursor changed")
        self.next_step = 2

    def condition(self, runtime, task: int, demo: int) -> tuple[tuple, int, int]:
        video = self.videos.load(task, demo)
        pixels = torch.from_numpy(video.frames).to(runtime.device, non_blocking=True)
        indices = torch.from_numpy(video.frame_indices).to(runtime.device, non_blocking=True)
        tokens, mask, _ = runtime.tokenizer([self.tasks[task].authority.language])
        return (pixels, indices, tokens, mask), video.raw_frame_count, len(pixels)

    def batch(self, event: dict) -> dict:
        if self.queries is None or self.rows is None:
            raise ValueError("deployment/video-only path has no action-query dataset")
        rows = [self.queries[self.rows[event["task"]][row["demo"]][row["frame"]]]
                for row in event["queries"]]
        if len(rows) != 28 or any(row["demo_index"] == event["teacher_demo"] for row in rows):
            raise ValueError("FM query touched its teacher episode")
        return default_collate(rows)

    def close(self) -> None:
        self.videos.close()
        if self.queries is not None:
            self.queries.close()


@dataclass
class Runtime:
    policy: torch.nn.Module
    writer: OperatorReadWrite
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device
    identity: dict[str, torch.Tensor]
    identity_resets: int = 0

    def restore_identity(self) -> None:
        copy_task_lora_state_(self.policy, self.identity, self.lora)
        self.identity_resets += 1

    def physical_delta(self) -> float:
        current = task_lora_state_dict(self.policy)
        return float(torch.stack([(current[name].float() - expected.float()).norm()
                                  for name, expected in self.identity.items()]).norm())

    def compile(self, condition: tuple, *, frame_chunk: int = 8,
                retain_native: bool = False) -> tuple[dict, dict | None]:
        self.restore_identity()
        with autocast(self.device):
            x, h = read_native_video(self.policy, self.writer.public_state(), self.writer.probe,
                                     condition, self.writer.names, frame_chunk=frame_chunk)
            if retain_native:
                if h.requires_grad:
                    h.retain_grad()
                for value in x.values():
                    if value.requires_grad:
                        value.retain_grad()
            state = self.writer(x, h)
        validate_lora_state(state, self.lora)
        return state, ({"x": x, "h": h} if retain_native else None)


def build_runtime(asset_root: Path, spec: dict, device: torch.device, mode: str, *,
                  evaluation: bool = False) -> Runtime:
    source_config = spec["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    tokenizer_path = asset_root / source_config["tokenizer"]
    stats = read_json(asset_root / source_config["normalization"])["stats"]
    if evaluation:
        from ember.pi05_eval.worker_setup import load_policy as load_eval_policy

        recipe = read_json(asset_root / "configs/pi05_target_evaluation_v1.json")
        policy, processor, _ = load_eval_policy(
            Path(source["model_path"]), stats, tokenizer_path, recipe["policy"])
    else:
        policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
        processor = Pi05LiberoProcessor(stats, tokenizer_path, 200, str(device))
    base = load_pi05_lora_contract(asset_root / source_config["lora_contract"])
    lora = derive_pi05_lora_rank(base, rank=128)
    identity = prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    writer = OperatorReadWrite(lora, identity, mode).to(device)
    if any(p.requires_grad for p in policy.parameters()):
        raise ValueError("physical source parameters must be frozen")
    return Runtime(policy, writer, Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device)),
                   processor, lora, source, device, identity)


def optimizer_for(writer: OperatorReadWrite, spec: dict):
    opt = spec["optimization"]
    parameters = tuple(p for p in writer.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: clamped_lr_multiplier(step, warmup=opt["warmup_updates"],
                                                      decay=opt["decay_updates"], peak=opt["lr"],
                                                      floor=opt["floor_lr"]))
    return optimizer, scheduler, parameters


def gradient_groups(writer: OperatorReadWrite) -> dict[str, float]:
    def norm(parameters):
        values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return float(torch.stack(values).norm()) if values else 0.0

    common = writer.public_state()
    return {"public_A": norm(writer.common.values[i] for i, name in enumerate(writer.common.names)
                             if name.endswith(".lora_A.default.weight")),
            "public_B0": norm(writer.common.values[i] for i, name in enumerate(writer.common.names)
                              if name.endswith(".lora_B.default.weight")),
            "independent_S": norm(writer.separate_keys or ()),
            **{name: norm(parameter for unit in writer.writes
                          for parameter in getattr(unit, name).parameters())
               for name in ("p", "c", "d", "o")}}


def one_job(runtime: Runtime, data: EngineeringData, event: dict, microbatch: int,
            frame_chunk: int) -> dict:
    started = time.perf_counter()
    condition, raw, sampled = data.condition(runtime, event["task"], event["teacher_demo"])
    with torch.no_grad():
        state, _ = runtime.compile(condition, frame_chunk=frame_chunk)
    torch.cuda.synchronize(runtime.device)
    compilation = time.perf_counter() - started
    batch = runtime.processor.training_batch(data.batch(event))
    with autocast(runtime.device):
        credit = paired_functional_credit(
            runtime.policy, state, runtime.lora, batch, seed=event["flow_seed"],
            device=runtime.device, random_batch=28, offset=0, microbatch=microbatch,
            condition_weight=0.25)
    torch.cuda.synchronize(runtime.device)
    fm = time.perf_counter() - started - compilation
    cotangent = credit["lora_cotangent"]
    with torch.enable_grad():
        replay, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
        if set(replay) != set(cotangent) or any(not torch.isfinite(v).all() for v in cotangent.values()):
            raise ValueError("full-rank FM cotangent incomplete or nonfinite")
        torch.autograd.backward(tuple(replay.values()),
                                tuple(cotangent[name].to(replay[name]) for name in replay))
    torch.cuda.synchronize(runtime.device)
    x_norms = [value.grad.float().norm() for value in native["x"].values() if value.grad is not None]
    native_norm = {"h": float(native["h"].grad.float().norm()) if native["h"].grad is not None else 0.0,
                   "x": float(torch.stack(x_norms).norm()) if x_norms else 0.0}
    return {"task": event["task"], "teacher_demo": event["teacher_demo"],
            "queries": len(event["queries"]), "query_demos": [row["demo"] for row in event["queries"]],
            "query_frames": [row["frame"] for row in event["queries"]], "flow_seed": event["flow_seed"],
            "raw_frames": raw, "sampled_frames": sampled, "flow_loss": credit["flow_loss"],
            "fm_cotangent_norm": float(torch.stack([v.norm() for v in cotangent.values()]).norm()),
            "native_cotangent_norm": native_norm,
            "compile_seconds": compilation, "fm_seconds": fm,
            "replay_seconds": time.perf_counter() - started - compilation - fm,
            "total_seconds": time.perf_counter() - started}


@dataclass
class Session:
    spec: dict
    context: object
    data: EngineeringData
    runtime: Runtime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    output: Path
    contract: dict
    mode: str
    microbatch: int
    frame_chunk: int


def prepare_train(spec: dict, args) -> Session:
    git = frozen_git()
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2 or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("operator engineering train/resume requires same-node world2")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    data = EngineeringData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device, args.mode)
    runtime.writer.train()
    seed_everything(7, context)
    optimizer, scheduler, parameters = optimizer_for(runtime.writer, spec)
    root = Path(spec["run_root"])
    output = root / args.mode / ("resume" if args.resume else "fresh")
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    local = {"rank": context.rank, "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "ranks": gather(local, context.world_size)}
    contract = {"schema_version": SCHEMA, "stage": STAGE, "git": git,
                "spec": str(SPEC_PATH), "mode": args.mode, "source": runtime.source,
                "lora": runtime.lora.to_dict(), "operator": spec["operator"],
                "events": spec["events"], "optimizer": spec["optimization"],
                "topology": topology, "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
                "sampler": {key: value for key, value in data.sampler_state().items() if key != "next_step"},
                "trainable_names": [name for name, p in runtime.writer.named_parameters() if p.requires_grad],
                "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
                "information_wall": "teacher exact language + dual RGB only; independent query own RGB/state/action FM"}
    error = None
    try:
        if context.is_main:
            if args.resume:
                parent = args.resume.parent.parent
                expected = root / "T" / "fresh" / "checkpoints" / "macro_00000002"
                if args.mode != "T" or args.resume != expected:
                    raise ValueError("only T's registered own fresh ECP2 may resume")
                if read_json(parent / "run_contract.json") != contract:
                    raise ValueError("source, parameter, sampler, numerical or physical topology changed")
            if (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
                raise ValueError("engineering attempt output already exists")
            write_json_atomic(output / "run_contract.json", contract)
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, context.world_size) if value]
    if failures:
        raise RuntimeError(f"run contract failure: {failures}")
    return Session(spec, context, data, runtime, optimizer, scheduler, parameters,
                   output, contract, args.mode, args.microbatch, args.frame_chunk)


def restore(session: Session, checkpoint: Path) -> tuple[int, int]:
    restored, result, error = {}, None, None
    try:
        updates, rows = load_ecp_checkpoint(
            checkpoint=checkpoint, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA,
            restored_state=restored)
        session.data.restore(restored["sampler_state"])
        if (updates != 2 or rows != 2 or session.scheduler.last_epoch != 2
                or restored["training_state"] != {"updates": 2, "mode": "T"}):
            raise ValueError("T ECP2 optimizer/scheduler/sampler cursor changed")
        rows_from_parent = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        prefix = rows_from_parent[:rows]
        if len(prefix) != rows or [json.loads(row)["update"] for row in prefix] != [1, 2]:
            raise ValueError("ECP2 metrics history lacks a complete consecutive prefix")
        if session.context.is_main:
            (session.output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
        result = updates, rows
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, session.context.world_size) if value]
    if failures:
        raise RuntimeError(f"ECP2 restoration failed on a rank: {failures}")
    return result


def update(session: Session, updates: int, rows: int) -> tuple[int, int]:
    started = time.perf_counter()
    jobs = [session.data.event(updates, task) for task in TASKS]
    costs = {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             for index, job in enumerate(jobs)}
    assigned = condition_assignment(tuple(costs), costs, world_size=2)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        for index in assigned[session.context.rank]:
            local.append(one_job(session.runtime, session.data, jobs[index],
                                 session.microbatch, session.frame_chunk))
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, 2) if value]
    if failures:
        raise RuntimeError(f"operator macro job failed on a rank: {failures}")
    sum_writer_gradients(session.parameters, world_size=2)
    gradients = gradient_groups(session.runtime.writer)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.spec["optimization"]["grad_clip"], error_if_nonfinite=True))
    lr = session.optimizer.param_groups[0]["lr"]
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    updates += 1
    session.data.next_step = updates
    packets = gather(local, 2)
    memory = gather({"rank": session.context.rank,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated(session.context.device) / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(session.context.device) / 2**30}, 2)
    if session.context.is_main:
        append_jsonl(session.output / "metrics.jsonl", {
            "update": updates, "mode": session.mode, "queries": 112,
            "jobs": [row for packet in packets for row in packet],
            "lr_applied": lr, "lr_next": session.scheduler.get_last_lr()[0],
            "grad_norms_before_clip": gradients, "total_grad_norm": norm,
            "rank_memory": memory, "seconds": time.perf_counter() - started})
    rows += 1
    if updates in (2, 4):
        save_ecp_checkpoint(
            output_dir=session.output, macro=updates, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA, metrics_rows=rows,
            sampler_state=session.data.sampler_state(),
            training_state={"updates": updates, "mode": session.mode})
    return updates, rows


def train(spec: dict, args) -> None:
    if (args.mode not in ("T", "U") or args.stop_after != 4
            or (args.resume and args.mode != "T")
            or args.microbatch not in (28, 14, 7) or args.frame_chunk not in (8, 4)):
        raise ValueError("only bounded T/U four-update engineering is active")
    session = prepare_train(spec, args)
    try:
        updates, rows = restore(session, args.resume) if args.resume else (0, 0)
        started = time.perf_counter()
        while updates < 4:
            updates, rows = update(session, updates, rows)
        if session.context.is_main:
            write_json_atomic(session.output / "completion.json", {
                "schema_version": SCHEMA, "mode": session.mode, "updates": updates,
                "actual_segment_updates": updates - (2 if args.resume else 0),
                "actual_segment_queries": (updates - (2 if args.resume else 0)) * 112,
                "metrics_rows": rows, "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "checkpoint": str(session.output / "checkpoints" / "macro_00000004"),
                "scientific_qualification": False})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def load_T4(runtime: Runtime, spec: dict, git: dict) -> Path:
    from safetensors.torch import load_file

    root = Path(spec["run_root"]) / "T" / "fresh"
    checkpoint = root / "checkpoints" / "macro_00000004"
    contract = read_json(root / "run_contract.json")
    completion = read_json(root / "completion.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    if (contract.get("schema_version") != SCHEMA or contract.get("stage") != STAGE
            or contract.get("git") != git or contract.get("mode") != "T"
            or contract.get("source") != runtime.source or contract.get("lora") != runtime.lora.to_dict()
            or contract.get("operator") != spec["operator"]
            or completion.get("updates") != 4 or completion.get("metrics_rows") != 4
            or manifest.get("next_macro") != 4 or manifest.get("stage") != STAGE
            or manifest.get("run_contract_schema") != SCHEMA):
        raise ValueError("T4 engineering source or checkpoint identity changed")
    runtime.writer.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(runtime.device)),
                                   strict=True)
    return checkpoint


def profile(spec: dict, args) -> None:
    if torch.cuda.device_count() != 1 or args.mode is not None or args.resume is not None:
        raise ValueError("one fixed T4 longest-video profile requires one visible GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    runtime = build_runtime(args.asset_root, spec, torch.device("cuda:0"), "T")
    checkpoint = load_T4(runtime, spec, frozen_git())
    data = EngineeringData(args.asset_root, spec)
    output = Path(spec["run_root"]) / "profile"
    output.mkdir(parents=True, exist_ok=False)
    try:
        actual = data.videos.frame_counts(38, 36)
        if actual != (517, 105):
            raise ValueError(f"registered longest legal video identity changed: {actual}")
        event = data.event(3, 38, teacher=36)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        started = time.perf_counter()
        result = one_job(runtime, data, event, 28, 8)
        write_json_atomic(output / "profile.json", {
            "schema_version": SCHEMA, "checkpoint": str(checkpoint), "event": result,
            "updates": 0, "queries": 28, "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 2**30,
            "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad)})
    finally:
        data.close()


def case(spec: dict, args) -> None:
    from dataclasses import asdict

    from ember.pi05_assets import configure_libero_runtime_assets
    from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
    from ember.pi05_eval_contract import inspect_installed_target_tasks
    from ember.pi05_evaluation import rollout_shard
    from ember.writer.materialization import file_record

    if torch.cuda.device_count() != 1 or args.mode is not None or args.resume is not None:
        raise ValueError("one fixed T4 train-only case requires one visible GPU")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not physical.isdigit():
        raise ValueError("canonical EGL case requires a physical GPU index")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    output = Path(spec["run_root"]) / "sequential_case"
    output.mkdir(parents=True, exist_ok=False)
    runtime = build_runtime(args.asset_root, spec, torch.device("cuda:0"), "T", evaluation=True)
    checkpoint = load_T4(runtime, spec, frozen_git())
    data = EngineeringData(args.asset_root, spec, query_labels=False)
    try:
        runtime.writer.eval()

        def compile_one(task: int, demo: int):
            condition, raw, sampled = data.condition(runtime, task, demo)
            started = time.perf_counter()
            with torch.no_grad():
                state, _ = runtime.compile(condition)
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("compiled complete task LoRA is nonfinite")
            return state, {"task": task, "demo": demo, "raw_frames": raw,
                           "sampled_frames": sampled, "seconds": time.perf_counter() - started}

        first, first_info = compile_one(2, 46)
        copy_task_lora_state_(runtime.policy, first, runtime.lora)
        installed_delta = runtime.physical_delta()
        second, second_info = compile_one(38, 46)
        restored_delta = runtime.physical_delta()
        if installed_delta <= 0 or restored_delta != 0 or runtime.identity_resets != 2:
            raise ValueError("second compile retained first condition's task adapter")
        isolation = {"schema_version": SCHEMA, "checkpoint": str(checkpoint),
                     "first": first_info, "second": second_info,
                     "installed_first_delta_norm": installed_delta,
                     "second_compile_source_delta_norm": restored_delta,
                     "identity_resets": runtime.identity_resets}
        write_json_atomic(output / "source_isolation.json", isolation)
        copy_task_lora_state_(runtime.policy, second, runtime.lora)
        authorities = load_evaluation_authorities(args.asset_root / spec["source"]["evaluation_config"],
                                                  args.asset_root)
        tasks, paths = inspect_installed_target_tasks(
            authorities, role="development_train", state_count=1,
            libero_config_dir=output / "libero_config")
        configure_libero_runtime_assets(Path(paths["assets"]))
        task_rows = {(task.suite, task.task_id): asdict(task) for task in tasks}
        manifest = read_json(args.asset_root / "configs/libero_24_8_8_coverage_v1/manifest.json")
        task38 = next(row for row in manifest["tasks"] if row["global_task_id"] == 38)
        selected = task_rows[(task38["suite"], int(task38["task_id"]))]
        if selected["language"] != task38["language"] or selected["horizon"] != 520:
            raise ValueError("global38 official train task identity changed")
        eval_contract = dict(authorities.config)
        eval_contract["libero_paths"] = paths
        eval_contract["parallel"] = {**eval_contract["parallel"], "envs_per_replica": 1}
        eval_contract["diagnostic_stage_predicates"] = {"full_conditions_only": False}
        eval_contract["diagnostic_occupancy_capture"] = {
            "mode": "full", "trajectory_root": str(output / "trajectories"),
            "passive_trace": {"trace_root": str(output / "passive_traces")}}
        pool = PersistentTaskEnvironmentPool(eval_contract, physical_gpu_id=int(physical))
        try:
            envs, states = pool.switch(selected)
            started = time.perf_counter()
            result = rollout_shard(
                envs=envs, init_states=states, task=selected, state_ids=(0,),
                contract=eval_contract, policy=runtime.policy, preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action)
            if len(result) != 1:
                raise ValueError("official train-only interface returned other than one episode")
        finally:
            pool.close()
        row = {**result[0], "global_task_id": 38, "teacher_demo": 46,
               "compiled_lora": {"rank": 128, "alpha": 128, "targets": 38,
                                 "factors": 76, "checkpoint": str(checkpoint)},
               "source_isolation": file_record(output / "source_isolation.json"),
               "interface_seconds": time.perf_counter() - started}
        path = output / "case_global_38_state_00_demo_46.json"
        write_json_atomic(path, row)
        write_json_atomic(output / "completion.json", {
            "schema_version": SCHEMA, "checkpoint": str(checkpoint),
            "case": file_record(path), "source_isolation": file_record(output / "source_isolation.json"),
            "success": row["success"], "steps": row["steps"],
            "full_capture": row.get("occupancy_trajectory"),
            "scientific_qualification": False})
    finally:
        data.close()


def audit(spec: dict, asset_root: Path) -> dict:
    data = EngineeringData(asset_root, spec)
    try:
        plans = [[data.event(step, task) for task in TASKS] for step in range(4)]
        if any(len({row["task"] for row in jobs}) != 4 for jobs in plans):
            raise ValueError("macro lost equal four-task allocation")
        for jobs in plans:
            for row in jobs:
                if (len(row["queries"]) != 28 or any(q["demo"] == row["teacher_demo"]
                                                     for q in row["queries"])):
                    raise ValueError("event lost cross-episode FM")
        return {"schema_version": spec["events"]["schema_version"],
                "events": plans, "modes": ["T", "U"],
                "actual_training_updates": 10, "actual_training_queries": 1120,
                "profile": spec["execution"]["profile"],
                "case": spec["execution"]["sequential_case"]}
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train", "profile", "case"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("T", "U"))
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    spec = specification()
    if args.phase == "audit":
        print(json.dumps(audit(spec, args.asset_root), sort_keys=True))
    elif args.phase == "train":
        train(spec, args)
    elif args.phase == "profile":
        profile(spec, args)
    else:
        case(spec, args)


if __name__ == "__main__":
    main()
