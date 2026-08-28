"""Run and aggregate the six-task G3 P1 task-local primal qualification."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.bank_conditioning.consensus import FitConsensusTeacherStore
from ember.ecp.bank_conditioning.f0 import F0Runtime, _prepare_runtime
from ember.ecp.bank_conditioning.mapping import (
    cross_video_consistency_loss,
    paired_mapping_loss,
)
from ember.ecp.bank_conditioning.primal_capacity import (
    TaskLocalPrimalCode,
    optimistic_recovery_record,
    recovery_record,
    subset_teacher,
    task_local_output,
)
from ember.ecp.bank_conditioning.primal_dual_runtime import PrimalDualVideoOperator
from ember.ecp.g1_initialization import cache_native_video_readout
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


P1_SCHEMA = "ember_ecp_primal_capacity_p1_v1"
P1_TASK_SCHEMA = "ember_ecp_primal_capacity_p1_task_v1"
P1_REPORT_SCHEMA = "ember_ecp_primal_capacity_p1_report_v1"


@dataclass
class P1TaskSetup:
    runtime: F0Runtime
    config: dict[str, Any]
    model_config: dict[str, Any]
    model_path: Path
    task: int
    role: str
    target_indices: tuple[int, ...]
    owners: tuple[Any, ...]
    s_ref: torch.Tensor
    fit_rows: tuple[Any, ...]
    fit_videos: tuple[int, int]
    held_video: int
    member_names: tuple[str, ...]
    consensus_rows: dict[int, tuple[Any, ...]]
    native_rows: dict[int, tuple[Any, ...]]
    operator: PrimalDualVideoOperator
    prepared: dict[int, Any]
    capture_metrics: dict[str, object]
    prepare_seconds: float
    code: TaskLocalPrimalCode
    optimizer: torch.optim.Optimizer
    parameters: tuple[torch.nn.Parameter, ...]
    temperature: float

    @property
    def all_videos(self) -> tuple[int, ...]:
        return (*self.fit_videos, self.held_video)

    @property
    def fit_banks(self) -> tuple[Any, Any]:
        return tuple(self.prepared[video] for video in self.fit_videos)

    @property
    def fit_teachers(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return tuple(self.consensus_rows[video] for video in self.fit_videos)


def load_primal_capacity_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    panel = config.get("panel", {})
    optimization = config.get("optimization", {})
    gate = config.get("gate", {})
    wall = config.get("information_wall", {})
    if not all(
        (
            config.get("schema_version") == P1_SCHEMA,
            config.get("status") == "active_preregistered",
            config.get("model_config")
            == "configs/pi05_ecp_shared_compiler_g3_v5.json",
            panel.get("task_ids") == [1, 8, 9, 72, 73, 75],
            panel.get("task_roles")
            == {"meta_fit": [1, 8, 9], "target_fit": [72, 73, 75]},
            panel.get("target_indices") == [0, 16, 34, 1, 17, 35, 36, 37],
            panel.get("fit_video_selection")
            == "lowest_two_mapping_fit_video_ordinals",
            panel.get("held_video_selection")
            == "pre_registered_mapping_video_holdout",
            int(optimization.get("steps", -1)) == 500,
            optimization.get("report_steps") == [0, 1, 50, 100, 200, 350, 500],
            float(optimization.get("primal_learning_rate", 0)) > 0,
            optimization.get("scale_trainable") is False,
            optimization.get("held_video_gradients") == 0,
            optimization.get("held_video_checkpoint_selection") is False,
            optimization.get("early_stop") is False,
            float(gate.get("fit_median_recovery_minimum", 0)) > 0,
            float(gate.get("held_median_recovery_minimum", 0)) > 0,
            gate.get("all_six_tasks_required") is True,
            gate.get("all_four_families_required") is True,
            wall.get("deployment_candidate") is False,
            wall.get("held_video_used_for_gradient") is False,
            wall.get("action_meta_installed") is False,
            wall.get("task_local_scale_trainable_parameters") == 0,
            wall.get("shuffled_or_reversed_use") is False,
        )
    ):
        raise ValueError("unsupported P1 primal capacity config")
    return config


def _condition(
    runtime: F0Runtime,
    *,
    owners: tuple[Any, ...],
    target_indices: tuple[int, ...],
    video_demo: int,
    chunk_size: int,
) -> tuple[Any, dict[str, object]]:
    sample = NaturalProgramSample(
        video_demos=(int(video_demo),),
        action_demos=(),
        k=1,
        robustness_view="g3_p1_task_local_primal",
    )
    packed = pack_shared_compiler_videos(
        task=runtime.task,
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.device,
    )
    language, mask = runtime.tokens[runtime.task.authority_id]
    condition = prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=owners,
        packed=packed,
        language_tokens=language,
        language_mask=mask,
        chunk_size=chunk_size,
    )
    video = condition.videos[0]
    video = replace(
        video,
        native=cache_native_video_readout(video.native),
        local_scene=video.local_scene[list(target_indices)],
        local_process=video.local_process[:, list(target_indices)],
        local_sigma=video.local_sigma[:, list(target_indices)],
    )
    return video, condition.metrics


def _teacher_rows(
    runtime: F0Runtime,
    *,
    video_demo: int,
    member_names: tuple[str, ...],
    target_indices: tuple[int, ...],
) -> tuple[Any, ...]:
    rows = runtime.teachers.lookup_members(
        authority_id=runtime.task.authority_id,
        k=1,
        video_demo=video_demo,
        member_names=member_names,
    )
    if rows is None:
        raise RuntimeError("P1 lost a fit-only native teacher")
    return tuple(subset_teacher(row, target_indices) for row in rows)


def _consensus_rows(
    store: FitConsensusTeacherStore,
    *,
    authority_id: int,
    video_demo: int,
    member_names: tuple[str, ...],
    target_indices: tuple[int, ...],
) -> tuple[Any, ...]:
    return tuple(
        subset_teacher(row, target_indices)
        for row in store.lookup_members(
            authority_id=authority_id,
            video_demo=video_demo,
            member_names=member_names,
        )
    )


def _fit_outputs(
    *,
    operator: PrimalDualVideoOperator,
    banks: Sequence[Any],
    code: TaskLocalPrimalCode,
    s_ref: torch.Tensor,
) -> tuple[Any, ...]:
    return tuple(
        task_local_output(
            operator=operator,
            prepared=bank,
            code=code,
            s_ref=s_ref,
        )
        for bank in banks
    )


def _fit_loss(
    *,
    outputs: tuple[Any, Any],
    teachers: tuple[tuple[Any, ...], tuple[Any, ...]],
    owners: tuple[Any, ...],
    temperature: float,
    consistency_weight: float,
    consistency_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    first = paired_mapping_loss(
        output=outputs[0],
        teachers=teachers[0],
        owners=owners,
        temperature=temperature,
    )
    second = paired_mapping_loss(
        output=outputs[1],
        teachers=teachers[1],
        owners=owners,
        temperature=temperature,
    )
    consistency = cross_video_consistency_loss(
        primary_output=outputs[0],
        companion_output=outputs[1],
        primary_teachers=teachers[0],
        companion_teachers=teachers[1],
        owners=owners,
        responsibilities=first.responsibilities,
        margin=consistency_margin,
    )
    total = 0.5 * (first.total + second.total)
    total = total + consistency_weight * consistency.total
    return total, {
        "total": float(total.detach()),
        "first_update_loss": float(first.total.detach()),
        "second_update_loss": float(second.total.detach()),
        "cross_video_loss": float(consistency.total.detach()),
    }


def _fit_records(
    outputs: Sequence[Any],
    teachers: Sequence[Sequence[Any]],
    owners: tuple[Any, ...],
    *,
    temperature: float,
) -> list[dict[str, object]]:
    return [
        recovery_record(
            output,
            rows,
            owners,
            temperature=temperature,
        )[1]
        for output, rows in zip(outputs, teachers, strict=True)
    ]


def _worker_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.output_dir.resolve()
    task_dir = root / f"task_{args.task:03d}"
    partial = root / f".task_{args.task:03d}.partial"
    if task_dir.exists() or partial.exists():
        raise RuntimeError("P1 task output already exists")
    root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return task_dir, partial


def _initialize_runtime(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[F0Runtime, Path, dict[str, Any]]:
    runtime = _prepare_runtime(
        argparse.Namespace(
            asset_root=args.asset_root,
            data_root=args.data_root,
            task=args.task,
        )
    )
    runtime.compiler.requires_grad_(False).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    seed = int(config["optimization"]["seed"]) + args.task
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    model_path = args.asset_root / str(config["model_config"])
    return runtime, model_path, load_shared_compiler_config(model_path)


def _task_teacher_sets(
    runtime: F0Runtime,
    *,
    task: int,
    target_indices: tuple[int, ...],
) -> tuple[Any, ...]:
    fit_rows = runtime.mapping_split.fit_by_task[task]
    held_rows = runtime.mapping_split.video_held_by_task[task]
    if len(fit_rows) < 2 or len(held_rows) != 1:
        raise RuntimeError("P1 task video split changed")
    fit_videos = tuple(row.video_demo for row in fit_rows[:2])
    held_video = held_rows[0].video_demo
    member_names = tuple(sorted(runtime.mapping_split.member_names[task]))
    consensus = FitConsensusTeacherStore(
        runtime.teachers, runtime.mapping_split, runtime.rank4_contract
    )
    all_videos = (*fit_videos, held_video)
    consensus_rows = {
        video: _consensus_rows(
            consensus,
            authority_id=task,
            video_demo=video,
            member_names=member_names,
            target_indices=target_indices,
        )
        for video in all_videos
    }
    native_rows = {
        video: _teacher_rows(
            runtime,
            video_demo=video,
            member_names=member_names,
            target_indices=target_indices,
        )
        for video in all_videos
    }
    return (
        fit_rows,
        fit_videos,
        held_video,
        member_names,
        consensus_rows,
        native_rows,
    )


def _new_operator(
    owners: tuple[Any, ...], model_config: dict[str, Any]
) -> PrimalDualVideoOperator:
    model = model_config["model"]
    return PrimalDualVideoOperator(
        owners,
        program_width=int(model["program_width"]),
        event_slots=int(model["event_slots"]),
        relative_eigenvalue_floor=float(model["relative_eigenvalue_floor"]),
        replay_score_rms=float(model["replay_score_rms"]),
        covariance_frame_chunk=int(model["frame_chunk_size"]),
    )


def _prepare_banks(
    runtime: F0Runtime,
    *,
    operator: PrimalDualVideoOperator,
    owners: tuple[Any, ...],
    target_indices: tuple[int, ...],
    videos: tuple[int, ...],
    frame_chunk_size: int,
) -> tuple[dict[int, Any], dict[str, object], float]:
    started = time.monotonic()
    prepared, capture_metrics = {}, {}
    for video in videos:
        value, metrics = _condition(
            runtime,
            owners=owners,
            target_indices=target_indices,
            video_demo=video,
            chunk_size=frame_chunk_size,
        )
        prepared[video] = operator.prepare(value)
        capture_metrics[str(video)] = metrics
    return prepared, capture_metrics, time.monotonic() - started


def _new_code_optimizer(
    *,
    owners: tuple[Any, ...],
    teachers: tuple[Any, ...],
    s_ref: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[TaskLocalPrimalCode, torch.optim.Optimizer]:
    code = TaskLocalPrimalCode(owners, teachers, s_ref=s_ref).to(device)
    optimizer_config = config["optimization"]
    optimizer = torch.optim.AdamW(
        (*tuple(code.input_code), *tuple(code.output_code)),
        lr=float(optimizer_config["primal_learning_rate"]),
        betas=tuple(optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    return code, optimizer


def _assemble_task_setup(
    args: argparse.Namespace,
    config: dict[str, Any],
    runtime: F0Runtime,
    model_path: Path,
    model_config: dict[str, Any],
) -> P1TaskSetup:
    target_indices = tuple(map(int, config["panel"]["target_indices"]))
    owners = tuple(runtime.owners[index] for index in target_indices)
    s_ref = runtime.ranks.s_ref[list(target_indices)]
    teacher_sets = _task_teacher_sets(
        runtime, task=args.task, target_indices=target_indices
    )
    fit_rows, fit_videos, held_video, member_names, consensus_rows, native_rows = (
        teacher_sets
    )
    operator = _new_operator(owners, model_config)
    prepared, capture_metrics, prepare_seconds = _prepare_banks(
        runtime,
        operator=operator,
        owners=owners,
        target_indices=target_indices,
        videos=(*fit_videos, held_video),
        frame_chunk_size=int(model_config["model"]["frame_chunk_size"]),
    )
    code, optimizer = _new_code_optimizer(
        owners=owners,
        teachers=consensus_rows[fit_videos[0]],
        s_ref=s_ref,
        config=config,
        device=runtime.device,
    )
    return P1TaskSetup(
        runtime=runtime,
        config=config,
        model_config=model_config,
        model_path=model_path,
        task=args.task,
        role=fit_rows[0].role,
        target_indices=target_indices,
        owners=owners,
        s_ref=s_ref,
        fit_rows=fit_rows,
        fit_videos=fit_videos,
        held_video=held_video,
        member_names=member_names,
        consensus_rows=consensus_rows,
        native_rows=native_rows,
        operator=operator,
        prepared=prepared,
        capture_metrics=capture_metrics,
        prepare_seconds=prepare_seconds,
        code=code,
        optimizer=optimizer,
        parameters=tuple(code.parameters()),
        temperature=float(config["optimization"]["teacher_temperature"]),
    )


def _prepare_task_setup(
    args: argparse.Namespace, config: dict[str, Any]
) -> P1TaskSetup:
    runtime, model_path, model_config = _initialize_runtime(args, config)
    try:
        return _assemble_task_setup(
            args, config, runtime, model_path, model_config
        )
    except BaseException:
        runtime.close()
        raise


def _setup_fit_loss(
    setup: P1TaskSetup,
) -> tuple[torch.Tensor, dict[str, float], tuple[Any, Any]]:
    outputs = _fit_outputs(
        operator=setup.operator,
        banks=setup.fit_banks,
        code=setup.code,
        s_ref=setup.s_ref,
    )
    loss, components = _fit_loss(
        outputs=outputs,
        teachers=setup.fit_teachers,
        owners=setup.owners,
        temperature=setup.temperature,
        consistency_weight=float(
            setup.config["optimization"]["cross_video_weight"]
        ),
        consistency_margin=float(
            setup.config["optimization"]["cross_video_margin"]
        ),
    )
    return loss, components, outputs


def _curve_record(
    setup: P1TaskSetup, *, step: int, gradient_norm: float | None
) -> dict[str, object]:
    with torch.no_grad():
        _, components, outputs = _setup_fit_loss(setup)
        rows = _fit_records(
            outputs,
            setup.fit_teachers,
            setup.owners,
            temperature=setup.temperature,
        )
    return {
        "step": step,
        "gradient_norm": gradient_norm,
        **components,
        "fit": rows,
        "fit_mean_recovery": statistics.fmean(
            float(row["mean_recovery"]) for row in rows
        ),
    }


def _train_task_code(
    setup: P1TaskSetup,
) -> tuple[list[dict[str, object]], float, float]:
    optimization = setup.config["optimization"]
    report_steps = set(map(int, optimization["report_steps"]))
    curve = [_curve_record(setup, step=0, gradient_norm=None)]
    started = time.monotonic()
    last_gradient = float("nan")
    for step in range(1, int(optimization["steps"]) + 1):
        setup.optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _setup_fit_loss(setup)
        loss.backward()
        if any(
            value.grad is None or not bool(torch.isfinite(value.grad).all())
            for value in setup.parameters
        ):
            raise RuntimeError("P1 task-local primal gradient is invalid")
        norm = torch.nn.utils.clip_grad_norm_(
            setup.parameters, float(optimization["gradient_clip_norm"])
        )
        last_gradient = float(norm)
        setup.optimizer.step()
        if step in report_steps:
            curve.append(
                _curve_record(setup, step=step, gradient_norm=last_gradient)
            )
    return curve, time.monotonic() - started, last_gradient


def _final_task_evaluation(setup: P1TaskSetup) -> dict[str, object]:
    with torch.no_grad():
        _, _, fit_outputs = _setup_fit_loss(setup)
        final_fit = _fit_records(
            fit_outputs,
            setup.fit_teachers,
            setup.owners,
            temperature=setup.temperature,
        )
        held_output = task_local_output(
            operator=setup.operator,
            prepared=setup.prepared[setup.held_video],
            code=setup.code,
            s_ref=setup.s_ref,
        )
        held = recovery_record(
            held_output,
            setup.consensus_rows[setup.held_video],
            setup.owners,
            temperature=setup.temperature,
        )[1]
        optimistic = {
            str(video): optimistic_recovery_record(
                setup.native_rows[video],
                setup.consensus_rows[video],
                setup.owners,
                temperature=setup.temperature,
            )
            for video in setup.all_videos
        }
    fit_mean = statistics.fmean(float(row["mean_recovery"]) for row in final_fit)
    held_mean = float(held["mean_recovery"])
    optimistic_held = float(
        optimistic[str(setup.held_video)]["mean_recovery"]
    )
    return {
        "final_fit": final_fit,
        "held": held,
        "optimistic_native_projection": optimistic,
        "fit_mean_recovery": fit_mean,
        "held_recovery": held_mean,
        "held_to_fit": held_mean / max(fit_mean, 1e-12),
        "held_relative_to_optimistic": held_mean / max(optimistic_held, 1e-12),
        "per_task_gate_passed": held_mean
        >= float(setup.config["gate"]["per_task_held_recovery_minimum"]),
    }


def _trainable_parameters(module: torch.nn.Module) -> int:
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def _save_task_code(
    setup: P1TaskSetup, partial: Path
) -> tuple[Path, dict[str, torch.Tensor]]:
    state = {
        name: value.detach().float().cpu().contiguous()
        for name, value in setup.code.state_dict().items()
    }
    checkpoint = partial / "task_code.safetensors"
    save_file(
        state,
        str(checkpoint),
        metadata={"schema_version": P1_TASK_SCHEMA, "task": str(setup.task)},
    )
    return checkpoint, state


def _task_report(
    setup: P1TaskSetup,
    *,
    args: argparse.Namespace,
    task_dir: Path,
    checkpoint: Path,
    state: Mapping[str, torch.Tensor],
    curve: list[dict[str, object]],
    final: dict[str, object],
    started: float,
    train_seconds: float,
    final_gradient: float,
) -> dict[str, object]:
    runtime = setup.runtime
    return {
        "schema_version": P1_TASK_SCHEMA,
        "status": "complete",
        "git": runtime.state,
        "config": str(args.config.resolve()),
        "model_config": str(setup.model_path.resolve()),
        "task": setup.task,
        "role": setup.role,
        "fit_videos": list(setup.fit_videos),
        "held_video": setup.held_video,
        "target_indices": list(setup.target_indices),
        "member_names": list(setup.member_names),
        "curve": curve,
        **final,
        "held_backward_calls": 0,
        "action_meta_modules": runtime.inventory["action_meta_module_count"],
        "action_meta_parameters": runtime.inventory["action_meta_parameter_count"],
        "source_trainable_parameters": _trainable_parameters(runtime.policy),
        "program_trainable_parameters": _trainable_parameters(runtime.program),
        "shared_compiler_trainable_parameters": _trainable_parameters(
            runtime.compiler
        ),
        "task_local_trainable_parameters": sum(
            value.numel() for value in setup.parameters
        ),
        "task_local_scale_trainable_parameters": 0,
        "teacher_tensor_file_reads": runtime.teachers.tensor_reads,
        "capture_metrics": setup.capture_metrics,
        "prepare_seconds": setup.prepare_seconds,
        "train_seconds": train_seconds,
        "total_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
        "final_gradient_norm": final_gradient,
        "checkpoint": {
            "path": str((task_dir / checkpoint.name).resolve()),
            "bytes": checkpoint.stat().st_size,
            "tensor_count": len(state),
            "fixed_final_step": int(setup.config["optimization"]["steps"]),
        },
        "information_wall": {
            "fit_teacher_only": True,
            "held_video_gradients": 0,
            "held_checkpoint_selection": False,
            "fit_only_fixed_scale": True,
            "validation_test_action_reward_reads": 0,
            "action_meta_installed": False,
            "shuffled_reversed_use": False,
        },
    }


def _publish_task_result(
    *,
    setup: P1TaskSetup,
    args: argparse.Namespace,
    task_dir: Path,
    partial: Path,
    started: float,
    curve: list[dict[str, object]],
    train_seconds: float,
    final_gradient: float,
) -> None:
    final = _final_task_evaluation(setup)
    checkpoint, state = _save_task_code(setup, partial)
    report = _task_report(
        setup,
        args=args,
        task_dir=task_dir,
        checkpoint=checkpoint,
        state=state,
        curve=curve,
        final=final,
        started=started,
        train_seconds=train_seconds,
        final_gradient=final_gradient,
    )
    write_json_atomic(partial / "report.json", report)
    os.replace(partial, task_dir)
    print(
        json.dumps(
            {
                "task": setup.task,
                "fit": report["fit_mean_recovery"],
                "held": report["held_recovery"],
                "relative": report["held_relative_to_optimistic"],
                "seconds": report["total_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_worker(args: argparse.Namespace) -> None:
    config = load_primal_capacity_config(args.config)
    if args.task not in tuple(map(int, config["panel"]["task_ids"])):
        raise ValueError("P1 worker task is outside the preregistered panel")
    task_dir, partial = _worker_paths(args)
    setup: P1TaskSetup | None = None
    started = time.monotonic()
    try:
        setup = _prepare_task_setup(args, config)
        curve, train_seconds, final_gradient = _train_task_code(setup)
        _publish_task_result(
            setup=setup,
            args=args,
            task_dir=task_dir,
            partial=partial,
            started=started,
            curve=curve,
            train_seconds=train_seconds,
            final_gradient=final_gradient,
        )
    except BaseException:
        if partial.exists():
            write_json_atomic(
                partial / "failure.json",
                {"task": args.task, "status": "failed"},
            )
        raise
    finally:
        if setup is not None:
            setup.runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--asset-root", type=Path, required=True)
    worker.add_argument("--data-root", type=Path, required=True)
    worker.add_argument("--task", type=int, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "worker":
        _run_worker(args)
    else:
        from ember.ecp.bank_conditioning.primal_capacity_aggregate import aggregate

        aggregate(args)
