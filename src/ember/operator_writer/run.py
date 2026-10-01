"""Shared operator cross-episode FM training with sealed ECP recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

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
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast
from ember.writer.task_execution import condition_assignment

from .credit import gradient_groups, one_job
from .data import (CHECKPOINTS, CONTINUATION_CHECKPOINTS, CONTINUATION_UPDATES,
                   CONTINUATION1350_CHECKPOINTS, CONTINUATION1350_UPDATES,
                   CONTINUATION1800_CHECKPOINTS, CONTINUATION1800_UPDATES,
                   PILOT_CHECKPOINTS, PILOT_UPDATES,
                   CONTINUATION2340_CHECKPOINTS, CONTINUATION2340_UPDATES,
                   CONTINUATION2790_CHECKPOINTS, CONTINUATION2790_UPDATES,
                   TASKS, UPDATES, FormalData)
from .model import OperatorReadWrite
from .native import read_native_video
from . import change_clock, joint_training


from .specification import (
    REPO, SPEC_PATH, CHANGE_CLOCK_SPEC_PATH, CHANGE_CLOCK_CONTINUATION_SPEC_PATH,
    CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH, CONTINUATION1800_SPEC_PATH, PILOT_SPEC_PATH,
    CONTINUATION2340_SPEC_PATH, CONTINUATION2790_SPEC_PATH, JOINT_SPEC_PATH, CONTEXT_SPEC_PATH, CONTEXT_CONTINUATION_SPEC_PATH, SELF_READ_SPEC_PATH, CONDITIONAL_SPEC_PATH, PILOT_ROOT,
    CONTINUATION2340_ROOT, CONTINUATION2790_ROOT, CONTINUATION900_ROOT, CONTINUATION1350_ROOT,
    SEALED_ROOT, SEALED_SPEC_PATH, SCHEMA, STAGE,
    OPERATOR_CONTRACT, OPTIMIZATION_CONTRACT, EVENT_CONTRACT, EXECUTION_CONTRACT,
    CONTINUATION_EVENTS, CONTINUATION_EXECUTION, CONTINUATION1350_EVENTS, CONTINUATION1350_EXECUTION,
    CONTINUATION1800_EVENTS, CONTINUATION1800_EXECUTION, PILOT_EVENTS, PILOT_EXECUTION,
    CONTINUATION2340_EVENTS, CONTINUATION2340_EXECUTION, CONTINUATION2790_EVENTS, CONTINUATION2790_EXECUTION,
    PILOT_ARMS, PILOT_CONTRACT, specification, specification_path,
)


def frozen_git(*, continuation: bool = False, change_clock_pilot: bool = False) -> dict:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"]:
        raise ValueError("GPU calculation requires clean detached frozen source")
    refs = subprocess.run(["git", "branch", "-r", "--contains", state["commit"]], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    allowed = (("origin/codex/operator-change-clock", "origin/main") if change_clock_pilot else
               ("origin/codex/demonstration-transfer", "origin/main")
               if continuation else ("origin/main",))
    pushed_ref = next((ref for ref in allowed if any(row.strip() == ref for row in refs)), None)
    if pushed_ref is None:
        raise ValueError("formal frozen code commit must be pushed to its registered source ref")
    return {"commit": state["commit"], "branch": "", "dirty_paths": [],
            "pushed_ref": pushed_ref}


def gather(value, world: int):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def resume_contract_compatible(parent: dict, current: dict, *, allow_topology_change: bool = False) -> bool:
    """Only registered packing and explicit physical topology may change."""
    mutable = ("microbatch", "frame_chunk", "topology") if allow_topology_change else ("microbatch", "frame_chunk")
    if current.get("source_resume") is not None:
        if not joint_training.conditional_resume_compatible(parent, current):
            return False
        mutable += ("git", "spec", "parent_checkpoint", "source_resume")
    return (packing_compatible(parent, current)
            and {k: v for k, v in parent.items() if k not in mutable}
            == {k: v for k, v in current.items() if k not in mutable})


def packing_compatible(parent: dict, current: dict) -> bool:
    frames = {(8, 8), (8, 4), (4, 4)}
    if joint_training.conditional_contract(parent) and joint_training.conditional_contract(current):
        frames = {(old, new) for old in (4, 8, 16, 32) for new in (4, 8, 16, 32)}
    return ((parent.get("microbatch"), current.get("microbatch")) in {
            (28, 28), (28, 14), (28, 7), (14, 14), (14, 7), (7, 7)}
            and (parent.get("frame_chunk"), current.get("frame_chunk")) in frames)


def complete_checkpoint(path: Path) -> bool:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    macro = manifest.get("next_macro")
    world = manifest.get("world_size")
    allowed = ((macro in (*CHECKPOINTS, *change_clock.CONTINUATION_CHECKPOINTS) and world in range(1, 7))
               or (macro in (*CONTINUATION_CHECKPOINTS, *CONTINUATION1350_CHECKPOINTS,
                             *CONTINUATION1800_CHECKPOINTS, *PILOT_CHECKPOINTS,
                             *CONTINUATION2340_CHECKPOINTS, *CONTINUATION2790_CHECKPOINTS)
                   and world in range(1, 7)))
    expected_files = ({"ecp.safetensors", "trainer_state.pt"}
                      | {f"rank_{rank:02d}_state.pt" for rank in range(world)}) if allowed else set()
    return (manifest.get("stage") == STAGE and manifest.get("run_contract_schema") == SCHEMA
            and allowed and set(files) == expected_files
            and all((path / name).is_file() and (path / name).stat().st_size == row.get("bytes")
                    for name, row in files.items()))


def validate_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    root = Path(spec["run_root"])
    if joint_training.registered(spec):
        joint_training.validate_attempt(spec, args, contract, output)
        return
    event_schema = spec.get("events", {}).get("schema_version")
    if event_schema in (CONTINUATION1350_EVENTS["schema_version"],
                        CONTINUATION1800_EVENTS["schema_version"],
                        PILOT_EVENTS["schema_version"],
                        CONTINUATION2340_EVENTS["schema_version"],
                        CONTINUATION2790_EVENTS["schema_version"]):
        _validate_late_continuation_attempt(spec, args, contract, output)
        return
    if (spec.get("execution", {}).get("updates_per_mode") == CONTINUATION_UPDATES
            or spec.get("task") == change_clock.CONTINUATION_TASK):
        _validate_continuation_attempt(spec, args, contract, output)
        return
    if args.resume:
        parent = args.resume.resolve().parent.parent
        attempts = (root / args.mode / "train" / "attempts").resolve()
        if (parent.parent != attempts or parent == output.resolve()
                or args.resume.name not in {f"macro_{step:08d}" for step in CHECKPOINTS}):
            raise ValueError("resume requires this arm's registered ECP90/180/270")
        if not resume_contract_compatible(read_json(parent / "run_contract.json"), contract,
                                          allow_topology_change=spec.get("task") == change_clock.TASK):
            raise ValueError("source, parameter, sampler, numerical or physical topology changed")
        if not complete_checkpoint(args.resume):
            raise ValueError("requested same-arm ECP is incomplete")
        latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
            "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
        if int(args.resume.name.split("_")[-1]) != latest:
            raise ValueError("resume must use the latest complete same-arm ECP")
    if (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("formal attempt output already exists")


def _validate_continuation_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    clock = spec.get("task") == change_clock.CONTINUATION_TASK
    checkpoints = tuple(spec["execution"]["checkpoints"])
    if args.resume is None:
        raise ValueError("continuation requires a complete parent ECP")
    checkpoint = args.resume.resolve()
    macro = int(checkpoint.name.removeprefix("macro_")) if re.fullmatch(r"macro_[0-9]{8}", checkpoint.name) else -1
    if macro not in (270, *checkpoints[:-1]) or not complete_checkpoint(checkpoint):
        raise ValueError("continuation requires a registered complete parent/intermediate ECP")
    parent = checkpoint.parent.parent
    if output.resolve() == parent or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("continuation attempt output already exists")
    attempts = Path(spec["run_root"]) / args.mode / "train/attempts"
    latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
        "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
    if (macro == 270 and latest != -1) or (macro != 270 and macro != latest):
        raise ValueError("resume requires the latest complete same-arm continuation ECP")
    if macro == 270:
        expected = (change_clock.PARENT_CHECKPOINT if clock else
                    SEALED_ROOT / args.mode / "train/attempts/fresh/checkpoints/macro_00000270")
        if checkpoint != expected:
            raise ValueError("continuation parent is not the sealed same-arm 270 ECP")
        from .bank import inspect_training_source

        old_spec = change_clock.TRAINING_SPEC_PATH if clock else SEALED_SPEC_PATH
        old = inspect_training_source(read_json(old_spec), checkpoint, args.mode, sealed_evaluation=True)
        if (old.get("events") != EVENT_CONTRACT or old.get("optimizer") != OPTIMIZATION_CONTRACT
                or old.get("source") != contract["source"] or old.get("lora") != contract["lora"]
                or old.get("operator") != contract["operator"]
                or any(old.get(key) != contract.get(key) for key in (
                    "trainable_names", "source_trainable", "information_wall"))):
            raise ValueError("sealed 270 source, parameters or optimization changed")
    else:
        if parent.parent.resolve() != attempts.resolve():
            raise ValueError("continuation ECP is outside this arm's registered attempts")
        old = read_json(parent / "run_contract.json")
        fixed = ("schema_version", "stage", "git", "spec", "mode", "source", "lora", "operator",
                 "events", "optimizer", "sampler", "trainable_names", "source_trainable",
                 "information_wall", "continuation")
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("continuation scientific source or frozen Git changed")
        if not packing_compatible(old, contract):
            raise ValueError("continuation packing changed outside registered OOM transitions")


def _validate_late_continuation_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    """One registered parent migration and same-window ECP resume owner."""
    parent_macro = int(spec["continuation"]["parent_macro"])
    checkpoints = tuple(spec["execution"]["checkpoints"])
    pilot = parent_macro == 1800
    arm = getattr(args, "pilot_arm", None) if pilot else args.mode
    if (parent_macro not in (900, 1350, 1800, 1890, 2340)
            or args.mode not in spec["execution"]["modes"]
            or pilot and (arm not in PILOT_ARMS or contract.get("pilot_arm") != arm
                          or contract.get("loss_variant") != PILOT_ARMS[arm])
            or parent_macro in (1890, 2340) and
            (arm != "T" or contract.get("loss_variant") != "full" or
             spec["continuation"].get("parent_arm") != ("control" if parent_macro == 1890 else "T"))):
        raise ValueError("unregistered continuation parent or arm")
    if args.resume is None:
        raise ValueError("continuation requires a complete same-arm ECP")
    checkpoint = args.resume.resolve()
    macro = int(checkpoint.name.removeprefix("macro_")) if re.fullmatch(r"macro_[0-9]{8}", checkpoint.name) else -1
    if macro not in (parent_macro, *checkpoints[:-1]) or not complete_checkpoint(checkpoint):
        raise ValueError("continuation requires a registered complete parent/intermediate ECP")
    parent = checkpoint.parent.parent
    if output.resolve() == parent or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("continuation attempt output already exists")
    attempts = Path(spec["run_root"]) / arm / "train/attempts"
    if output.parent.resolve() != attempts.resolve():
        raise ValueError("continuation attempt is outside its registered arm")
    latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
        "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
    if (macro == parent_macro and latest != -1) or (macro != parent_macro and macro != latest):
        raise ValueError("resume requires the latest complete same-arm continuation ECP")
    if macro == parent_macro:
        expected = (Path(spec["continuation"]["parent_run_root"])
                    / spec["continuation"].get("parent_arm", args.mode)
                    / "train/attempts/continuation/checkpoints" / f"macro_{parent_macro:08d}")
        if checkpoint != expected:
            raise ValueError("continuation parent is not the original same-arm ECP")
        from .bank import inspect_training_source

        old_spec = (CONTINUATION_SPEC_PATH if parent_macro == 900 else
                    CONTINUATION1350_SPEC_PATH if parent_macro == 1350 else
                    CONTINUATION1800_SPEC_PATH if parent_macro == 1800 else
                    PILOT_SPEC_PATH if parent_macro == 1890 else CONTINUATION2340_SPEC_PATH)
        old = inspect_training_source(specification(old_spec), checkpoint, args.mode,
                                      sealed_evaluation=True)
        fixed = ("schema_version", "stage", "mode", "source", "lora", "operator", "optimizer",
                 "trainable_names", "source_trainable", "information_wall")
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("sealed parent scientific source or optimizer changed")
        if parent_macro in (1890, 2340) and (
                old.get("loss_variant") != "full"
                or parent_macro == 1890 and old.get("pilot_arm") != "control"
                or parent_macro == 2340 and old.get("pilot_arm") is not None
                or old.get("git", {}).get("commit")
                != spec["continuation"]["parent_training_git"]):
            raise ValueError("sealed parent arm, loss or Git changed")
    else:
        if parent.parent.resolve() != attempts.resolve():
            raise ValueError("continuation ECP is outside this arm's registered attempts")
        old = read_json(parent / "run_contract.json")
        fixed = ("schema_version", "stage", "git", "spec", "mode", "source", "lora", "operator",
                 "events", "optimizer", "sampler", "trainable_names", "source_trainable",
                 "information_wall", "continuation")
        if pilot:
            fixed += ("pilot_arm", "loss_variant", "pilot")
        elif parent_macro in (1890, 2340):
            fixed += ("loss_variant",)
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("continuation scientific source or frozen Git changed")
        if not packing_compatible(old, contract):
            raise ValueError("continuation packing changed outside registered OOM transitions")


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
                retain_native: bool = False, capture_mechanism: bool = False,
                target_executor=None) -> tuple[dict, dict | None]:
        """Compose native/Writer reads without replacing the shared public base."""
        self.restore_identity()
        passes = []
        native_state = self.writer.public_state()
        with autocast(self.device):
            for _ in range(2 if self.writer.mode == "self_read" else 1):
                x, h = read_native_video(self.policy, native_state, self.writer.probe,
                                         condition, self.writer.names, frame_chunk=frame_chunk)
                state = self.writer(x, h, frame_indices=condition[1],
                                    **({"capture_mechanism": True} if capture_mechanism else {}),
                                    **({"target_executor": target_executor} if target_executor else {}))
                if retain_native or capture_mechanism:
                    for value in (h, *x.values(), *state.values()):
                        if value.requires_grad:
                            value.retain_grad()
                    passes.append({"x": x, "h": h, "state": state})
                # Only the next native read uses B0+M0; Writer keeps its original β.
                native_state = state
        validate_lora_state(state, self.lora)
        native = ({"x": x, "h": h, "passes": passes} if retain_native or capture_mechanism else None)
        if capture_mechanism:
            native["mechanism"] = self.writer.last_mechanism
        return state, native


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
    writer = OperatorReadWrite(lora, identity, "T" if mode == joint_training.MODE else mode).to(device)
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




@dataclass
class Session:
    spec: dict
    context: object
    data: FormalData
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
    continuation = "continuation" in spec
    clock_pilot = spec["task"] in (change_clock.TASK, change_clock.CONTINUATION_TASK)
    git = frozen_git(continuation=continuation, change_clock_pilot=clock_pilot)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    allowed_worlds = spec["execution"].get("world_sizes", [spec["execution"].get("world_size")])
    if context.world_size not in allowed_worlds or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("operator formal train/resume topology or NCCL contract changed")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    data = FormalData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device, args.mode)
    runtime.writer.train()
    seed_everything(7, context)
    optimizer, scheduler, parameters = optimizer_for(runtime.writer, spec)
    root = Path(spec["run_root"])
    pilot = spec["execution"]["updates_per_mode"] == PILOT_UPDATES
    next_window = spec["execution"]["updates_per_mode"] in (
        CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES)
    output = root / (args.pilot_arm if pilot else args.mode) / "train" / "attempts" / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    local = {"rank": context.rank, "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "ranks": gather(local, context.world_size)}
    contract = {"schema_version": SCHEMA, "stage": STAGE, "git": git,
                "spec": str(specification_path(spec)),
                "mode": args.mode, "source": runtime.source,
                "lora": runtime.lora.to_dict(), "operator": spec["operator"],
                "events": spec["events"], "optimizer": spec["optimization"],
                "topology": topology, "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
                "sampler": {key: value for key, value in data.sampler_state().items() if key != "next_step"},
                "trainable_names": [name for name, p in runtime.writer.named_parameters() if p.requires_grad],
                "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
                "information_wall": "teacher exact language + dual RGB only; independent query own RGB/state/action FM"}
    if joint_training.registered(spec):
        contract.update(loss_variant=joint_training.settings(spec)[2]["loss_variant"], joint=spec["joint"])
    if continuation:
        contract["continuation"] = spec["continuation"]
        contract["parent_checkpoint"] = str(args.resume.resolve())
        if pilot:
            contract.update(pilot_arm=args.pilot_arm, loss_variant=PILOT_ARMS[args.pilot_arm],
                            pilot=spec["pilot"])
        elif next_window:
            contract["loss_variant"] = "full"
        if spec["execution"]["updates_per_mode"] in (CONTINUATION1350_UPDATES,
                                                       CONTINUATION1800_UPDATES,
                                                       CONTINUATION2340_UPDATES,
                                                       CONTINUATION2790_UPDATES):
            contract["stop_after_macro"] = args.stop_after_macro
    joint_training.register_conditional_resume(spec, args, contract)
    error = None
    try:
        if context.is_main:
            validate_attempt(spec, args, contract, output)
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
        updates_target = session.spec["execution"]["updates_per_mode"]
        clock_continuation = session.spec["task"] == change_clock.CONTINUATION_TASK
        continuation = clock_continuation or updates_target in (CONTINUATION_UPDATES, CONTINUATION1350_UPDATES,
                                          CONTINUATION1800_UPDATES, PILOT_UPDATES,
                                          CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES)
        updates, rows = load_ecp_checkpoint(
            checkpoint=checkpoint, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA,
            restored_state=restored, allow_world_size_change=(continuation or session.mode in (change_clock.MODE, *joint_training.MODES)))
        migration = session.data.restore(restored["sampler_state"],
                                         migrate_sealed_270=(clock_continuation or updates_target == CONTINUATION_UPDATES) and updates == 270,
                                         migrate_continuation_900=updates_target == CONTINUATION1350_UPDATES and updates == 900,
                                         migrate_continuation_1350=updates_target == CONTINUATION1800_UPDATES and updates == 1350,
                                         migrate_continuation_1800=updates_target == PILOT_UPDATES and updates == 1800,
                                         migrate_pilot_1890=updates_target == CONTINUATION2340_UPDATES and updates == 1890,
                                         migrate_continuation_2340=updates_target == CONTINUATION2790_UPDATES and updates == 2340)
        valid_checkpoints = set(session.data.checkpoints)
        if "continuation" in session.spec:
            valid_checkpoints.add(session.spec["continuation"]["parent_macro"])
        if (updates not in valid_checkpoints or rows != updates or session.scheduler.last_epoch != updates
                or restored["training_state"] != restored_training_identity(session, updates)):
            raise ValueError("formal ECP optimizer/scheduler/sampler cursor changed")
        rows_from_parent = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        prefix = rows_from_parent[:rows]
        if (len(prefix) != rows or [json.loads(row)["update"] for row in prefix] != list(range(1, rows + 1))
                or any(json.loads(row)["mode"] != session.mode for row in prefix)):
            raise ValueError("ECP metrics history lacks a complete consecutive same-arm prefix")
        if session.context.is_main:
            (session.output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
            if continuation or session.mode in (change_clock.MODE, *joint_training.MODES):
                parent_world = int(read_json(checkpoint / "checkpoint_manifest.json")["world_size"])
                write_json_atomic(session.output / "resume_provenance.json", {
                    "checkpoint": str(checkpoint), "parent_git": read_json(
                        checkpoint.parent.parent / "run_contract.json")["git"],
                    "parent_spec": read_json(checkpoint.parent.parent / "run_contract.json")["spec"],
                    "current_git": session.contract["git"], "current_spec": session.contract["spec"],
                    "source_resume": session.contract.get("source_resume"),
                    "old_world_size": parent_world, "new_world_size": session.context.world_size,
                    "restored_rank_rng": list(range(min(parent_world, session.context.world_size))),
                    "fresh_rank_rng": {str(rank): {"seed_function": "seed_everything(7, context)",
                                                   "rank_seed": 7 + rank}
                                       for rank in range(parent_world, session.context.world_size)},
                    "sampler_migration": migration, "metrics_prefix_rows": rows,
                    "topology": session.contract["topology"]})
        result = updates, rows
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, session.context.world_size) if value]
    if failures:
        raise RuntimeError(f"formal ECP restoration failed on a rank: {failures}")
    return result


def training_state(session: Session, updates: int) -> dict:
    state = {"updates": updates, "mode": session.mode}
    if session.mode in joint_training.MODES:
        state["loss_variant"] = session.contract["loss_variant"]
    elif session.data.updates in (CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES):
        state["loss_variant"] = "full"
    elif "pilot_arm" in session.contract:
        state.update(pilot_arm=session.contract["pilot_arm"],
                     loss_variant=session.contract["loss_variant"])
    return state


def restored_training_identity(session: Session, updates: int) -> dict:
    target = session.data.updates
    if target == CONTINUATION2340_UPDATES and updates == 1890:
        return {"updates": updates, "mode": session.mode,
                "pilot_arm": "control", "loss_variant": "full"}
    if target == PILOT_UPDATES and updates == 1800:
        return {"updates": updates, "mode": session.mode}
    return training_state(session, updates)


def update(session: Session, updates: int, rows: int) -> tuple[int, int]:
    started = time.perf_counter()
    jobs = [session.data.event(updates, task) for task in session.data.tasks_for_step(updates)]
    costs = {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             for index, job in enumerate(jobs)}
    world = session.context.world_size
    owners = min(len(jobs), world)
    shared_targets = session.mode == joint_training.CONDITIONAL_MODE and world > owners
    assigned = condition_assignment(tuple(costs), costs, world_size=owners if shared_targets else world)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        if shared_targets and session.context.rank >= owners:
            from .target_execution import serve_target_shards
            serve_target_shards(session.runtime, owners=owners, world=world)
        for index in (() if shared_targets and session.context.rank >= owners else assigned[session.context.rank]):
            executor = None
            if shared_targets:
                from .target_execution import TargetShardClient
                executor = TargetShardClient(session.runtime.writer, owners=owners, world=world)
            local.append(one_job(session.runtime, session.data, jobs[index],
                                 session.microbatch, session.frame_chunk,
                                 session.contract.get("loss_variant", "full"), target_executor=executor))
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, world) if value]
    if failures:
        raise RuntimeError(f"operator macro job failed on a rank: {failures}")
    sum_writer_gradients(session.parameters, world_size=world,
                         bucket_bytes=64 * 2**20 if session.mode == joint_training.CONDITIONAL_MODE else None)
    gradients = gradient_groups(session.runtime.writer)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.spec["optimization"]["grad_clip"], error_if_nonfinite=True))
    lr = session.optimizer.param_groups[0]["lr"]
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    updates += 1
    session.data.next_step = updates
    packets = gather(local, world)
    memory = gather({"rank": session.context.rank,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated(session.context.device) / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(session.context.device) / 2**30}, world)
    if session.context.is_main:
        record = {
            "update": updates, "mode": session.mode, "queries": 112,
            **{key: value for key, value in training_state(session, updates).items()
               if key not in ("updates", "mode")},
            "jobs": [row for packet in packets for row in packet],
            "lr_applied": lr, "lr_next": session.scheduler.get_last_lr()[0],
            "grad_norms_before_clip": gradients, "total_grad_norm": norm,
            "rank_memory": memory, "seconds": time.perf_counter() - started}
        append_jsonl(session.output / "metrics.jsonl", record)
        if not getattr(session, "first_consumer_recorded", False):
            write_json_atomic(session.output / "first_consumer.json", {
                "git": session.contract["git"], "spec": session.contract["spec"],
                "source_resume": session.contract.get("source_resume"), "row": record})
            print(json.dumps({"event": "first_actual_consumer", "update": updates,
                              "seconds": record["seconds"], "world_size": world,
                              "lr_applied": lr, "rank_memory": memory}), flush=True)
            session.first_consumer_recorded = True
    rows += 1
    if updates in session.data.checkpoints:
        save_ecp_checkpoint(
            output_dir=session.output, macro=updates, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA, metrics_rows=rows,
            sampler_state=session.data.sampler_state(),
            training_state=training_state(session, updates))
    return updates, rows


def validate_train_request(spec: dict, args) -> None:
    if joint_training.registered(spec):
        joint_training.validate_request(spec, args)
        return
    if spec["task"] in (change_clock.TASK, change_clock.CONTINUATION_TASK):
        change_clock.validate_request(spec, args)
        return
    if (spec["execution"]["updates_per_mode"] != CONTINUATION2790_UPDATES
            or spec["events"] != CONTINUATION2790_EVENTS
            or spec.get("continuation", {}).get("parent_arm") != "T"
            or spec.get("pilot") is not None):
        raise ValueError("new formal operator training requires the T2790 continuation spec")
    if (args.mode != "T" or not args.attempt
            or getattr(args, "pilot_arm", None) is not None
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt) is None
            or args.resume is None or args.attempt == "fresh"
            or args.microbatch not in (28, 14, 7) or args.frame_chunk not in (8, 4)):
        raise ValueError("operator train requires registered T attempt and ECP policy")
    if (args.stop_after_macro is not None
            and args.stop_after_macro not in CONTINUATION2790_CHECKPOINTS[:-1]):
        raise ValueError("controlled stop must be a complete intermediate 2790 ECP boundary")


def train(spec: dict, args) -> None:
    validate_train_request(spec, args)
    session = prepare_train(spec, args)
    try:
        updates, rows = restore(session, args.resume) if args.resume else (0, 0)
        start_updates = updates
        started = time.perf_counter()
        target = session.data.updates
        if updates >= target:
            raise ValueError("completed operator checkpoint cannot start a new training attempt")
        if args.stop_after_macro is not None and args.stop_after_macro <= updates:
            raise ValueError("controlled stop is not after the resumed ECP")
        while updates < target:
            updates, rows = update(session, updates, rows)
            if updates in spec["execution"]["checkpoints"][:-1]:
                requested = ((session.output / "stop_at_next_ecp.request").exists()
                             if session.context.is_main else False)
                stop_at_ecp = any(gather(requested or updates == args.stop_after_macro,
                                         session.context.world_size))
                if stop_at_ecp:
                    if session.context.is_main:
                        write_json_atomic(session.output / "stopped_at_ecp.json", {
                            "schema_version": SCHEMA, "mode": session.mode, "updates": updates,
                            "metrics_rows": rows,
                            "checkpoint": str(session.output / "checkpoints"
                                              / f"macro_{updates:08d}"),
                            "reason": ("registered stop-after-macro" if updates == args.stop_after_macro
                                       else "stop_at_next_ecp.request"),
                            "next_resume_from_this_ecp": True})
                    return
        if session.context.is_main:
            write_json_atomic(session.output / "completion.json", {
                "schema_version": SCHEMA, "mode": session.mode, "updates": updates,
                **{key: value for key, value in training_state(session, updates).items()
                   if key not in ("updates", "mode")},
                "actual_segment_updates": updates - start_updates,
                "actual_segment_queries": (updates - start_updates) * 112,
                "metrics_rows": rows, "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "checkpoint": str(session.output / "checkpoints" / f"macro_{target:08d}"),
                "formal_checkpoint_complete": True,
                "scientific_qualification": False})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def audit(spec: dict, asset_root: Path) -> dict:
    data = FormalData(asset_root, spec, query_labels=False)
    try:
        visits = {task: [] for task in TASKS}
        query_total = 0
        for step in range(data.updates):
            jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
            if len(jobs) != 4 or len({row["task"] for row in jobs}) != 4:
                raise ValueError("macro lost equal four-task allocation")
            for row in jobs:
                if (len(row["queries"]) != 28 or any(q["demo"] == row["teacher_demo"]
                                                     for q in row["queries"])
                        or len({q["demo"] for q in row["queries"]}) != 28
                        or any(not 0 <= q["frame"] < data.tasks[row["task"]].episode_lengths[q["demo"]] - 1
                               for q in row["queries"])):
                    raise ValueError("event lost cross-episode FM")
                visits[row["task"]].append(row["teacher_demo"])
                query_total += 28
        count = data.updates // 9
        expected = {task: [] for task in TASKS}
        for task in TASKS:
            for round_index in range((count + 49) // 50):
                seed = ([20260928, 1, task] if round_index == 0 else
                        [20260928, 1, task, round_index])
                length = min(50, count - 50 * round_index)
                expected[task] += list(np.random.default_rng(
                    np.random.SeedSequence(seed)).permutation(50)[:length])
        if visits != expected or query_total != data.updates * 112:
            raise ValueError("36-task teacher rounds or cross-episode queries changed")
        return {"schema_version": spec["events"]["schema_version"],
                "modes": spec["execution"]["modes"], "updates_per_mode": data.updates,
                "conditions_per_mode": sum(len(rows) for rows in visits.values()),
                "queries_per_mode": query_total, "visits_per_task": count,
                "teacher_order": visits, "query_offset": 1}
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("T", "U", change_clock.MODE, *joint_training.MODES))
    parser.add_argument("--pilot-arm", choices=tuple(PILOT_ARMS))
    parser.add_argument("--attempt", type=str)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--spec", type=Path, default=CONDITIONAL_SPEC_PATH)
    args = parser.parse_args()
    spec = specification(args.spec)
    if args.phase == "train":
        train(spec, args)
    else:
        print(json.dumps(audit(spec, args.asset_root), sort_keys=True))


if __name__ == "__main__":
    main()
