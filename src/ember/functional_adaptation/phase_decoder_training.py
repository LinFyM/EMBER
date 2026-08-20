"""Distributed fixed-decoder fitting on successful closed-loop phase panels."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ember.functional_adaptation.decoder import FunctionalAdapterDecoder
from ember.functional_adaptation.functional_response import (
    functional_response_distillation_loss,
)
from ember.functional_adaptation.phase_decoder_codes import (
    PhaseDecoderCodeAuthority,
    load_phase_decoder_code_authority,
)
from ember.functional_adaptation.phase_decoder_panels import (
    CachedPhasePanel,
    PhaseMemberSource,
    cache_phase_member_panels,
    load_phase_member_sources,
)
from ember.functional_adaptation.phase_decoder_projection import (
    materialize_phase_decoder_projections,
    phase_decoder_asset,
    save_phase_decoder,
)
from ember.lora import LoRAContract
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_policy,
    seed_everything,
)
from ember.writer.as_step import (
    accumulate_flat_gradient,
    assign_flat_gradient,
    parameter_layout,
)
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_phase_aligned_functional_decoder_run_v1"
RESULT_SCHEMA = "ember_phase_aligned_functional_decoder_result_v1"
CHECKPOINT_SCHEMA = "ember_phase_aligned_functional_decoder_checkpoint_v1"


@dataclass
class Runtime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    repository: dict[str, Any]
    source: dict[str, Any]
    code_authority: PhaseDecoderCodeAuthority
    contract: LoRAContract
    policy: torch.nn.Module
    identity_state: Mapping[str, torch.Tensor]
    decoder: FunctionalAdapterDecoder
    optimizer: torch.optim.Optimizer
    member_sources: tuple[PhaseMemberSource, ...]
    fit_panels: dict[int, tuple[CachedPhasePanel, ...]]
    schedule: tuple[tuple[int, int], ...]
    topology: tuple[dict[str, Any], ...]
    started: float


def _authority_path(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve())
    if (
        value.get("schema_version")
        != "ember_pi05_train24_phase_aligned_decoder_v1"
        or value.get("status") != "preregistered_before_decoder_optimization"
        or value.get("decoder", {}).get("class") != "FunctionalAdapterDecoder"
        or value.get("decoder", {}).get("fully_fixed_after_fit") is not True
        or value.get("representation", {}).get("held_code_optimization_steps") != 0
    ):
        raise ValueError("unsupported phase-aligned decoder config")
    return value


def _task_schedule(config: Mapping[str, Any], world_size: int) -> tuple[tuple[int, int], ...]:
    fit_count = int(config["roles"]["fit_task_count"])
    visits = int(config["training"]["visits_per_fit_task"])
    generator = torch.Generator(device="cpu")
    rows: list[int] = []
    for epoch in range(visits):
        generator.manual_seed(int(config["training"]["schedule_seed"]) + epoch)
        rows.extend(torch.randperm(fit_count, generator=generator).tolist())
    if (
        len(rows) != int(config["training"]["total_task_visits"])
        or len(rows) % world_size
        or any(rows.count(index) != visits for index in range(fit_count))
    ):
        raise ValueError("phase decoder task-equal schedule changed")
    seen = [0] * fit_count
    schedule = []
    for task_index in rows:
        schedule.append((task_index, seen[task_index]))
        seen[task_index] += 1
    return tuple(schedule)


def _prepare(args: argparse.Namespace) -> Runtime:
    if args.resume is None and args.output_dir.exists():
        raise ValueError("fresh phase decoder output directory already exists")
    if args.resume is not None and not args.output_dir.is_dir():
        raise ValueError("phase decoder resume output directory is missing")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    config_path = args.config.resolve()
    config = load_config(config_path)
    repository = git_state(REPO_ROOT)
    expected_world = int(config["training"]["world_size"])
    if (
        args.mode != "formal"
        or context.world_size != expected_world
        or os.environ.get("NCCL_P2P_DISABLE") != "1"
        or not git_state_is_clean_pushed_or_frozen_authority(repository)
    ):
        raise ValueError("formal phase decoder launch authority changed")
    seed_everything(int(config["decoder"]["initialization_seed"]), context)
    codes = load_phase_decoder_code_authority(
        args.code_artifact,
        config=config,
        config_path=config_path,
        device=context.device,
    )
    authorities = load_evaluation_authorities(
        _authority_path(config, "evaluation"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    policy = load_policy(
        Path(str(source["model_path"])), authorities.source_base_config, context.device
    )
    contract = load_pi05_lora_contract(_authority_path(config, "lora_contract"))
    identity = prepare_frozen_writer_policy(policy, contract)
    policy.requires_grad_(False).eval()
    decoder = FunctionalAdapterDecoder(
        contract,
        identity,
        code_width=int(config["decoder"]["code_width"]),
        address_width=int(config["decoder"]["address_width"]),
        hidden_width=int(config["decoder"]["hidden_width"]),
        initialization_seed=int(config["decoder"]["initialization_seed"]),
    ).to(context.device)
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    member_sources = load_phase_member_sources(
        analysis_path=_authority_path(config, "phase_analysis"),
        codes=codes,
        repo_root=REPO_ROOT,
    )
    runtime = Runtime(
        args=args,
        context=context,
        config=config,
        repository=repository,
        source=source,
        code_authority=codes,
        contract=contract,
        policy=policy,
        identity_state=identity,
        decoder=decoder,
        optimizer=optimizer,
        member_sources=member_sources,
        fit_panels={},
        schedule=_task_schedule(config, context.world_size),
        topology=(),
        started=time.monotonic(),
    )
    fit_member_indices = [
        index
        for index, member in enumerate(codes.members)
        if member.fold_role == "fit"
    ]
    train_seed = int(config["functional_supervision"]["train_policy_seed"])
    runtime.fit_panels = {
        index: cache_phase_member_panels(
            policy=runtime.policy,
            identity_state=runtime.identity_state,
            contract=runtime.contract,
            sources=runtime.member_sources,
            member_index=index,
            device=runtime.context.device,
            policy_seed=train_seed,
        )
        for index in fit_member_indices
    }
    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.resolve().parent
    )
    for value in decoder.state_dict().values():
        dist.broadcast(value, src=0)
    topology: list[Any] = [None] * context.world_size
    dist.all_gather_object(
        topology,
        {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "device": str(context.device),
            "device_name": torch.cuda.get_device_name(context.device),
            "numa_node": context.numa_node,
            "cpu_affinity": list(context.cpu_affinity or ()),
        },
    )
    runtime.topology = tuple(dict(value) for value in topology)
    return runtime


def _run_contract(runtime: Runtime) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "mode": runtime.args.mode,
        "repository": runtime.repository,
        "host": socket.gethostname(),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "deferred_nccl": True,
            "topology": list(runtime.topology),
        },
        "inputs": {
            "config": str(runtime.args.config.resolve()),
            "config_bytes": runtime.args.config.resolve().stat().st_size,
            "code_artifact": str(runtime.code_authority.root),
            "source_run": str(runtime.args.source_run.resolve()),
            "checkpoint": str(runtime.args.checkpoint.resolve()),
            "expert_bank_root": str(runtime.args.expert_bank_root.resolve()),
        },
        "roles": runtime.config["roles"],
        "representation": runtime.config["representation"],
        "functional_supervision": runtime.config["functional_supervision"],
        "decoder": runtime.config["decoder"],
        "training": runtime.config["training"],
        "decision_gates": runtime.config["decision_gates"],
        "information_wall": {
            "privileged_actions": "development_train_successful_experts_only",
            "held_code_gradients": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "deployment_task_id_route": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_contract(runtime: Runtime) -> None:
    root = runtime.args.output_dir.resolve()
    contract = _run_contract(runtime)
    if runtime.context.is_main:
        if runtime.args.resume is None:
            root.mkdir(parents=True, exist_ok=False)
            write_json_atomic(root / "run_contract.json", contract)
        elif read_json(root / "run_contract.json") != contract:
            raise ValueError("phase decoder resume contract changed")
        append_jsonl(
            root / "invocations.jsonl",
            {
                "argv": list(os.sys.argv),
                "started_unix": time.time(),
                "resume": str(runtime.args.resume) if runtime.args.resume else None,
            },
        )
    dist.barrier(device_ids=[runtime.context.local_rank])


def _checkpoint_path(root: Path, task_visits: int) -> Path:
    return root / "checkpoints" / f"task_visits_{task_visits:08d}" / "state.pt"


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state(value["torch_cuda"], context.device)


def _save_checkpoint(runtime: Runtime, *, task_visits: int, metrics_rows: int) -> None:
    states: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(states, _rng_state(runtime.context))
    if runtime.context.is_main:
        path = _checkpoint_path(runtime.args.output_dir, task_visits)
        path.parent.mkdir(parents=True, exist_ok=False)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "task_visits": task_visits,
                "optimizer_updates": task_visits // runtime.context.world_size,
                "metrics_rows": metrics_rows,
                "world_size": runtime.context.world_size,
                "decoder": runtime.decoder.state_dict(),
                "optimizer": runtime.optimizer.state_dict(),
                "rng_by_rank": states,
            },
            temporary,
        )
        os.replace(temporary, path)
    dist.barrier(device_ids=[runtime.context.local_rank])


def _rewind_metrics(path: Path, rows: int) -> None:
    values = path.read_bytes().splitlines(keepends=True) if path.is_file() else []
    if len(values) < rows:
        raise ValueError("phase decoder metrics precede resume checkpoint")
    if len(values) > rows:
        temporary = path.with_name(f".{path.name}.resume.{os.getpid()}")
        temporary.write_bytes(b"".join(values[:rows]))
        os.replace(temporary, path)


def _resume(runtime: Runtime) -> tuple[int, int]:
    if runtime.args.resume is None:
        return 0, 0
    if runtime.args.resume.parents[2] != runtime.args.output_dir:
        raise ValueError("phase decoder resume checkpoint is outside its run root")
    value = torch.load(
        runtime.args.resume.resolve(),
        map_location=runtime.context.device,
        weights_only=False,
    )
    task_visits = int(value.get("task_visits", -1))
    metrics_rows = int(value.get("metrics_rows", -1))
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or int(value.get("world_size", -1)) != runtime.context.world_size
        or task_visits not in runtime.config["training"]["checkpoint_task_visits"]
        or task_visits % runtime.context.world_size
        or len(value.get("rng_by_rank", ())) != runtime.context.world_size
    ):
        raise ValueError("phase decoder exact-resume checkpoint changed")
    runtime.decoder.load_state_dict(value["decoder"], strict=True)
    runtime.optimizer.load_state_dict(value["optimizer"])
    _restore_rng(value["rng_by_rank"][runtime.context.rank], runtime.context)
    if runtime.context.is_main:
        _rewind_metrics(runtime.args.output_dir / "metrics.jsonl", metrics_rows)
    dist.barrier(device_ids=[runtime.context.local_rank])
    return task_visits, metrics_rows


def _fit_code(runtime: Runtime, task_index: int) -> torch.Tensor:
    return runtime.code_authority.fit_task_codes[task_index]


def _member_for_visit(runtime: Runtime, task_index: int, visit: int) -> tuple[int, int]:
    ordinal = runtime.code_authority.fit_ordinals[task_index]
    members = tuple(
        index
        for index, row in enumerate(runtime.code_authority.members)
        if row.ordinal == ordinal
    )
    if not members:
        raise ValueError("phase decoder fit task lost successful members")
    member_index = members[visit % len(members)]
    panel_index = (visit // len(members)) % 4
    return member_index, panel_index


def _fit(runtime: Runtime, *, start_visits: int, metrics_rows: int) -> int:
    context = runtime.context
    layout = parameter_layout(runtime.decoder)
    gradient = torch.zeros(layout[-1].stop, device=context.device, dtype=torch.float32)
    world = context.world_size
    total = len(runtime.schedule)
    checkpoints = set(int(value) for value in runtime.config["training"]["checkpoint_task_visits"])
    clip = float(runtime.config["training"]["optimizer"]["gradient_clip_norm"])
    metrics_path = runtime.args.output_dir / "metrics.jsonl"
    for cursor in range(start_visits, total, world):
        task_index, visit = runtime.schedule[cursor + context.rank]
        member_index, panel_index = _member_for_visit(runtime, task_index, visit)
        panel = runtime.fit_panels[member_index][panel_index]
        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = functional_response_distillation_loss(
                runtime.policy,
                runtime.decoder(_fit_code(runtime, task_index)),
                runtime.contract,
                panel.batch,
                panel.target,
                policy_seed=panel.policy_seed,
            )
        loss.backward()
        gradient.zero_()
        gradients = tuple(item.parameter.grad for item in layout)
        accumulate_flat_gradient(gradient, gradients, layout)
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world)
        gradient_norm = torch.linalg.vector_norm(gradient)
        if float(gradient_norm) > clip:
            gradient.mul_(clip / float(gradient_norm))
        assign_flat_gradient(gradient, layout)
        runtime.optimizer.step()
        local = {
            "rank": context.rank,
            "task_index": task_index,
            "task_ordinal": runtime.code_authority.fit_ordinals[task_index],
            "task_visit": visit,
            "member_index": member_index,
            "member": runtime.code_authority.members[member_index].member,
            "panel_index": panel_index,
            "loss": float(loss.detach()),
        }
        records: list[Any] = [None] * world
        dist.all_gather_object(records, local)
        task_visits = cursor + world
        metrics_rows += 1
        if context.is_main:
            append_jsonl(
                metrics_path,
                {
                    "optimizer_update": task_visits // world,
                    "task_visits": task_visits,
                    "mean_functional_loss": sum(float(row["loss"]) for row in records)
                    / world,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "records": records,
                    "elapsed_seconds": time.monotonic() - runtime.started,
                    "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                },
            )
        if task_visits in checkpoints:
            _save_checkpoint(runtime, task_visits=task_visits, metrics_rows=metrics_rows)
    return metrics_rows


def _member_code(runtime: Runtime, member_index: int) -> torch.Tensor:
    member = runtime.code_authority.members[member_index]
    if member.fold_role == "held_transform_only":
        return runtime.code_authority.member_codes[member_index]
    task_index = runtime.code_authority.fit_ordinals.index(member.ordinal)
    return runtime.code_authority.fit_task_codes[task_index]


def _evaluate_members(runtime: Runtime) -> list[dict[str, Any]]:
    local = []
    seed = int(runtime.config["functional_supervision"]["evaluation_policy_seed"])
    for member_index in range(runtime.context.rank, len(runtime.code_authority.members), runtime.context.world_size):
        panels = cache_phase_member_panels(
            policy=runtime.policy,
            identity_state=runtime.identity_state,
            contract=runtime.contract,
            sources=runtime.member_sources,
            member_index=member_index,
            device=runtime.context.device,
            policy_seed=seed,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = [
                float(
                    functional_response_distillation_loss(
                        runtime.policy,
                        runtime.decoder(_member_code(runtime, member_index)),
                        runtime.contract,
                        panel.batch,
                        panel.target,
                        policy_seed=panel.policy_seed,
                    )
                )
                for panel in panels
            ]
        member = runtime.code_authority.members[member_index]
        local.append(
            {
                "member_index": member_index,
                "suite": member.suite,
                "task_id": member.task_id,
                "global_task_id": member.global_task_id,
                "ordinal": member.ordinal,
                "fold_role": member.fold_role,
                "member": member.member,
                "expert_step": member.expert_step,
                "panel_losses": losses,
                "mean_loss": sum(losses) / len(losses),
            }
        )
    shards: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(shards, local)
    return sorted(
        (row for shard in shards for row in shard),
        key=lambda row: int(row["member_index"]),
    )


def _functional_gate(runtime: Runtime, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held = [row for row in rows if row["fold_role"] == "held_transform_only"]
    families = {}
    threshold = float(
        runtime.config["decision_gates"]["functional"][
            "both_held_member_families_mean_loss_below"
        ]
    )
    minimum_tasks = int(
        runtime.config["decision_gates"]["functional"][
            "minimum_tasks_below_identity_per_member_family"
        ]
    )
    for family in ("earliest", "latest"):
        selected = [row for row in held if row["member"] == family]
        mean = sum(float(row["mean_loss"]) for row in selected) / len(selected)
        below = sum(float(row["mean_loss"]) < 1.0 for row in selected)
        families[family] = {
            "mean_loss": mean,
            "tasks_below_identity": below,
            "passes": mean < threshold and below >= minimum_tasks,
        }
    return {
        "identity_relative_loss": 1.0,
        "families": families,
        "passes": all(row["passes"] for row in families.values()),
        "closed_loop_still_required": True,
    }


def _result(
    runtime: Runtime,
    *,
    rows: Sequence[Mapping[str, Any]],
    functional_gate: Mapping[str, Any],
    metrics_rows: int,
    decoder_path: Path,
) -> dict[str, Any]:
    final_visits = int(runtime.config["training"]["total_task_visits"])
    checkpoint = _checkpoint_path(runtime.args.output_dir, final_visits)
    return {
        "schema_version": RESULT_SCHEMA,
        "formal_authority": True,
        "repository": runtime.repository,
        "config": phase_decoder_asset(runtime.args.config),
        "code_artifact": {
            "root": str(runtime.code_authority.root),
            "result": phase_decoder_asset(runtime.code_authority.root / "result.json"),
            "codes": phase_decoder_asset(runtime.code_authority.root / "phase_codes.safetensors"),
        },
        "decoder": phase_decoder_asset(decoder_path),
        "final_exact_resume_checkpoint": phase_decoder_asset(checkpoint),
        "training": {
            "task_visits": final_visits,
            "optimizer_updates": final_visits // runtime.context.world_size,
            "world_size": runtime.context.world_size,
            "metrics_rows": metrics_rows,
            "elapsed_seconds": time.monotonic() - runtime.started,
        },
        "functional_evaluation": {
            "member_rows": list(rows),
            "fit_mean": sum(
                float(row["mean_loss"]) for row in rows if row["fold_role"] == "fit"
            )
            / sum(row["fold_role"] == "fit" for row in rows),
            "held_mean": sum(
                float(row["mean_loss"])
                for row in rows
                if row["fold_role"] == "held_transform_only"
            )
            / sum(row["fold_role"] == "held_transform_only" for row in rows),
            "gate": dict(functional_gate),
        },
        "information_wall": {
            "fit_task_gradients": 19,
            "held_task_gradients": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "final_lora_averaging": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    args.code_artifact = args.code_artifact.resolve()
    args.source_run = args.source_run.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.expert_bank_root = args.expert_bank_root.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    runtime = _prepare(args)
    _publish_contract(runtime)
    start_visits, metrics_rows = _resume(runtime)
    metrics_rows = _fit(
        runtime, start_visits=start_visits, metrics_rows=metrics_rows
    )
    rows = _evaluate_members(runtime)
    gate = _functional_gate(runtime, rows)
    if runtime.context.is_main:
        decoder_path = save_phase_decoder(runtime.decoder, runtime.args.output_dir)
        result = _result(
            runtime,
            rows=rows,
            functional_gate=gate,
            metrics_rows=metrics_rows,
            decoder_path=decoder_path,
        )
        write_json_atomic(runtime.args.output_dir / "result.json", result)
        materialize_phase_decoder_projections(
            config_path=runtime.args.config,
            task_expert_config_path=_authority_path(runtime.config, "task_experts"),
            codes=runtime.code_authority,
            member_sources=runtime.member_sources,
            decoder=runtime.decoder,
            contract=runtime.contract,
            repository=runtime.repository,
            source=runtime.source,
            expert_bank_root=runtime.args.expert_bank_root,
            output_dir=runtime.args.output_dir,
            functional_rows=rows,
            decoder_path=decoder_path,
        )
        print(
            json.dumps(
                {
                    "event": "complete",
                    "functional_gate": gate,
                    "output_dir": str(runtime.args.output_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(device_ids=[runtime.context.local_rank])
    dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--code-artifact", type=Path, required=True)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("formal",), required=True)
    result.add_argument("--resume", type=Path)
    return result
