#!/usr/bin/env python3
"""Test whether a successful fixed-route primal needs its own task's bank."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.joint_program_primal.evaluation import (
    _normalized,
    _panel_value,
    _positive_control_losses,
    _task_conditions,
    _wrong_task,
)
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_TASK_IDS,
    fixed_routing_program,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.routing_control_evaluation import (
    _checkpoint_authority,
    load_routing_control_gate,
    routing_task_assignments,
)
from ember.ecp.joint_program_primal.train_step import prepare_joint_condition
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.shared_compiler_span import _low_rank_geometry
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


SCHEMA = "ember_ecp_g3_cross_bank_upper_bound_v1"


class _FractionalSpectralOperator:
    """Diagnostic symmetric whitening without changing the retained operator."""

    def __init__(self, source: Any, inverse_power: float) -> None:
        self.source = source
        self.inverse_power = float(inverse_power)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def dual_and_score_rms(
        self, primal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.inverse_power == 1.0:
            return self.source.dual_and_score_rms(primal)
        if primal.ndim != 2 or primal.shape[-1] != self.native_width:
            raise ValueError("fractional-dual native width changed")
        basis = self.basis.to(primal).float()
        eigenvalues = self.eigenvalues.to(primal).float()
        coordinates = primal.float() @ basis
        relative = eigenvalues / eigenvalues[-1].clamp_min(1e-30)
        dual_coordinates = coordinates / relative.pow(self.inverse_power)[None]
        query = dual_coordinates @ basis.T
        score_rms = (
            dual_coordinates.square() * eigenvalues[None]
        ).sum(-1).clamp_min(0).sqrt()
        projected = coordinates @ basis.T
        projection = projected.norm(dim=-1) / primal.float().norm(
            dim=-1
        ).clamp_min(1e-30)
        return query.to(primal), score_rms.to(primal), projection.to(primal)


def _fractional_video(video: Any, inverse_power: float) -> Any:
    wrap = lambda row: _FractionalSpectralOperator(row, inverse_power)
    return replace(
        video,
        input_operators=tuple(wrap(row) for row in video.input_operators),
        output_operators=tuple(
            tuple(wrap(row) for row in groups)
            for groups in video.output_operators
        ),
    )


def _inverse_sqrt_transport(operator: Any, primal: torch.Tensor) -> torch.Tensor:
    basis = operator.basis.to(primal).float()
    eigenvalues = operator.eigenvalues.to(primal).float()
    relative = eigenvalues / eigenvalues[-1].clamp_min(1e-30)
    coordinates = primal.float() @ basis
    return ((coordinates / relative.sqrt()[None]) @ basis.T).to(primal)


def _fit_transport_primals(
    runtime: Any,
    *,
    program: Any,
    conditions: Sequence[Any],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]]:
    compiler = runtime.compiler
    state = compiler.primal_scorer.program_state(program)
    base_inputs = compiler.primal_scorer.input_primals(state)
    base_outputs = compiler.primal_scorer.output_primals(state)
    transported_inputs: list[tuple[torch.Tensor, ...]] = []
    transported_outputs: list[tuple[torch.Tensor, ...]] = []
    for condition in conditions:
        prepared, _ = prepare_joint_condition(runtime, condition)
        if len(prepared.videos) != 1:
            raise ValueError("fit-transport diagnostic escaped K1")
        video = prepared.videos[0]
        transported_inputs.append(
            tuple(
                _inverse_sqrt_transport(operator, primal)
                for operator, primal in zip(
                    video.input_operators, base_inputs, strict=True
                )
            )
        )
        transported_outputs.append(
            tuple(
                torch.stack(
                    tuple(
                        _inverse_sqrt_transport(operator, primal[group])
                        for group, operator in enumerate(operators)
                    )
                )
                for operators, primal in zip(
                    video.output_operators, base_outputs, strict=True
                )
            )
        )
        del prepared
        torch.cuda.empty_cache()
    if len(transported_inputs) != 2:
        raise ValueError("fit-transport diagnostic requires two fit videos")
    cosines = []
    for left, right in zip(
        (*transported_inputs[0], *transported_outputs[0]),
        (*transported_inputs[1], *transported_outputs[1]),
        strict=True,
    ):
        flat_left = left.float().reshape(left.shape[0], -1)
        flat_right = right.float().reshape(right.shape[0], -1)
        cosines.extend(
            torch.nn.functional.cosine_similarity(
                flat_left, flat_right, dim=-1
            ).tolist()
        )
    return (
        tuple(
            0.5 * (left + right)
            for left, right in zip(
                transported_inputs[0], transported_inputs[1], strict=True
            )
        ),
        tuple(
            0.5 * (left + right)
            for left, right in zip(
                transported_outputs[0], transported_outputs[1], strict=True
            )
        ),
        {
            "minimum": min(cosines),
            "median": statistics.median(cosines),
            "mean": statistics.fmean(cosines),
        },
    )


def _complete_state(
    runtime: Any,
    *,
    program: Any,
    bank: Any,
    inverse_power: float,
    input_primals: tuple[torch.Tensor, ...] | None = None,
    output_primals: tuple[torch.Tensor, ...] | None = None,
) -> tuple[dict[str, torch.Tensor], Any]:
    compiler = runtime.compiler
    state = compiler.primal_scorer.program_state(program)
    if input_primals is None:
        input_primals = compiler.primal_scorer.input_primals(state)
    if output_primals is None:
        output_primals = compiler.primal_scorer.output_primals(state)
    pooled = tuple(
        compiler.bank_operator.apply_compact(
            _fractional_video(video, inverse_power),
            input_primals,
            output_primals,
        )
        for video in bank.videos
    )
    output = compiler._output(state, pooled, s_ref=runtime.ranks.s_ref)
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=True
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    return complete, output


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    rows = tuple(map(float, values))
    if not rows:
        raise ValueError("cross-bank distribution is empty")
    ordered = sorted(rows)
    p10_index = max(0, int(0.1 * len(ordered)) - 1)
    return {
        "count": len(rows),
        "minimum": min(rows),
        "p10": ordered[p10_index],
        "median": statistics.median(rows),
        "mean": statistics.fmean(rows),
        "maximum": max(rows),
    }


def _r5_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    report = read_json(path.resolve())
    rows = {
        int(row["task"]): row
        for row in report.get("summary", {}).get("tasks", ())
    }
    if (
        report.get("gate_pass") is not True
        or int(report.get("checkpoint", {}).get("optimizer_step", -1)) != 110
        or set(rows) != set(ROUTING_TASK_IDS)
    ):
        raise ValueError("R5 passed aggregate authority changed")
    return rows


def _evaluate_task(
    runtime: Any,
    *,
    task_id: int,
    positive_root: Path,
    r5_row: Mapping[str, Any],
    inverse_power: float,
    primal_mode: str,
) -> dict[str, Any]:
    started = time.monotonic()
    task_conditions = _task_conditions(runtime, task_id)
    by_video = {int(row.video_demo): row for row in task_conditions}
    fit_videos = tuple(map(int, r5_row["fit_videos"]))
    held_video = int(r5_row["held_video"])
    if set((*fit_videos, held_video)) != set(by_video):
        raise ValueError("R5 fit/held video authority changed")
    first = by_video[fit_videos[0]]
    wrong_task = _wrong_task(runtime, task_id)
    wrong_first = _task_conditions(runtime, wrong_task)[0]
    free_reference, free_authority = _positive_control_losses(
        positive_root, task_id
    )
    correct_condition = first
    input_primals = output_primals = None
    transport_alignment: dict[str, float] | None = None
    with torch.inference_mode():
        program = fixed_routing_program(runtime, task_id)
        if primal_mode == "fit_transport":
            if inverse_power != 0.5:
                raise ValueError("fit transport requires symmetric whitening")
            input_primals, output_primals, transport_alignment = (
                _fit_transport_primals(
                    runtime,
                    program=program,
                    conditions=tuple(by_video[value] for value in fit_videos),
                )
            )
            correct_condition = by_video[held_video]

    correct_prepared, _ = prepare_joint_condition(runtime, correct_condition)
    wrong_prepared, _ = prepare_joint_condition(runtime, wrong_first)
    teacher_reads = runtime.native_teachers.tensor_reads
    with torch.inference_mode():
        correct_state, correct_output = _complete_state(
            runtime,
            program=program,
            bank=correct_prepared,
            inverse_power=inverse_power,
            input_primals=input_primals,
            output_primals=output_primals,
        )
        wrong_state, wrong_output = _complete_state(
            runtime,
            program=program,
            bank=wrong_prepared,
            inverse_power=inverse_power,
            input_primals=input_primals,
            output_primals=output_primals,
        )
    correct_record = _normalized(
        _panel_value(runtime, task_id=task_id, state=correct_state),
        free_reference[correct_condition.video_demo],
    )
    wrong_record = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_state),
        free_reference[correct_condition.video_demo],
    )
    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("cross-bank diagnostic read native teachers")

    correct_residual = residual_lora_state(
        correct_output.residual, runtime.rank4_contract, canonicalize=False
    )
    wrong_residual = residual_lora_state(
        wrong_output.residual, runtime.rank4_contract, canonicalize=False
    )
    geometry = _low_rank_geometry(wrong_residual, correct_residual, runtime)
    correct_recovery = float(correct_record["functional_recovery"])
    wrong_recovery = float(wrong_record["functional_recovery"])
    correct_benefit = float(correct_record["benefit_over_carrier"])
    wrong_benefit = float(wrong_record["benefit_over_carrier"])
    return {
        "task": task_id,
        "role": runtime.panels[task_id].role,
        "correct_video_demo": int(correct_condition.video_demo),
        "wrong_task": wrong_task,
        "wrong_video_demo": int(wrong_first.video_demo),
        "correct": dict(correct_record),
        "wrong_bank": wrong_record,
        "correct_minus_wrong_recovery": correct_recovery - wrong_recovery,
        "wrong_to_correct_benefit_retention": wrong_benefit
        / max(abs(correct_benefit), 1e-12),
        "wrong_to_correct_update_geometry": geometry,
        "free_primal_authority": free_authority,
        "native_teacher_tensor_reads": 0,
        "panel_b_backward_calls": 0,
        "action_meta_installed": False,
        "single_complete_rank16": True,
        "inverse_covariance_power": inverse_power,
        "primal_mode": primal_mode,
        "fit_transport_alignment": transport_alignment,
        "elapsed_seconds": time.monotonic() - started,
    }


def worker(args: argparse.Namespace) -> None:
    if args.inverse_power not in (0.5, 1.0):
        raise ValueError("cross-bank diagnostic inverse power changed")
    if args.primal_mode not in ("checkpoint", "fit_transport"):
        raise ValueError("cross-bank diagnostic primal mode changed")
    state = git_state(Path(__file__).resolve().parents[1])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("cross-bank evidence requires clean detached authority")
    gate = load_routing_control_gate(args.gate_config)
    positive_root = (
        args.asset_root / gate["authorities"]["positive_control_root"]
    ).resolve()
    r5_rows = _r5_rows(args.r5_aggregate)
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("cross-bank worker index changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("cross-bank workers must be independent single-GPU jobs")
    runtime_args = argparse.Namespace(
        config=args.config,
        base_config=args.base_config,
        mode="profile",
        phase="joint",
        task=None,
        asset_root=args.asset_root,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_root=args.data_root,
        output_dir=args.output_dir / f"worker_{args.worker_index:02d}_runtime",
        condition_cache_root=args.condition_cache_root,
        resume=None,
        stop_after_step=1,
        log_every=1,
        skip_routing_initialization=True,
    )
    runtime = None
    try:
        runtime = prepare_routing_control_runtime(runtime_args, context)
        checkpoint = _checkpoint_authority(
            runtime,
            compiler_run=args.compiler_run,
            compiler_checkpoint=args.compiler_checkpoint,
            gate=gate,
        )
        assignments = routing_task_assignments(
            args.worker_count, gate["evaluation"]["task_cost_seconds"]
        )
        rows = []
        for task_id in assignments[args.worker_index]:
            rows.append(
                _evaluate_task(
                    runtime,
                    task_id=task_id,
                    positive_root=positive_root,
                    r5_row=r5_rows[task_id],
                    inverse_power=args.inverse_power,
                    primal_mode=args.primal_mode,
                )
            )
            runtime.panel_batch_cache.clear()
            torch.cuda.empty_cache()
        worker_dir = args.output_dir / f"worker_{args.worker_index:02d}"
        if worker_dir.exists():
            raise ValueError("cross-bank worker output already exists")
        worker_dir.mkdir(parents=True)
        write_json_atomic(
            worker_dir / "result.json",
            {
                "schema_version": SCHEMA,
                "status": "complete",
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "assignments": [list(row) for row in assignments],
                "checkpoint": checkpoint,
                "tasks": rows,
                "git": {"commit": state["commit"], "branch": state["branch"]},
                "physical_visible_device": __import__("os").environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "inverse_covariance_power": args.inverse_power,
                "primal_mode": args.primal_mode,
            },
        )
        write_json_atomic(
            worker_dir / "completion.json",
            {
                "schema_version": SCHEMA,
                "worker_index": args.worker_index,
                "task_count": len(rows),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    commits = set()
    powers = set()
    primal_modes = set()
    for worker_index in range(args.worker_count):
        root = args.output_dir / f"worker_{worker_index:02d}"
        result = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        if (
            result.get("schema_version") != SCHEMA
            or result.get("status") != "complete"
            or result.get("worker_index") != worker_index
            or completion.get("schema_version") != SCHEMA
            or completion.get("worker_index") != worker_index
        ):
            raise ValueError("cross-bank worker evidence changed")
        rows.extend(result["tasks"])
        commits.add(result["git"]["commit"])
        powers.add(float(result.get("inverse_covariance_power", 1.0)))
        primal_modes.add(str(result.get("primal_mode", "checkpoint")))
    if len(rows) != len(ROUTING_TASK_IDS) or {row["task"] for row in rows} != set(
        ROUTING_TASK_IDS
    ) or len(commits) != 1 or powers != {args.inverse_power} or primal_modes != {
        args.primal_mode
    }:
        raise ValueError("cross-bank aggregate task or commit authority changed")
    rows.sort(key=lambda row: row["task"])
    margins = [float(row["correct_minus_wrong_recovery"]) for row in rows]
    retentions = [float(row["wrong_to_correct_benefit_retention"]) for row in rows]
    cosines = [
        float(row["wrong_to_correct_update_geometry"]["overall"]["update_cosine"])
        for row in rows
    ]
    summary = {
        "correct_recovery": _distribution(
            [float(row["correct"]["functional_recovery"]) for row in rows]
        ),
        "wrong_bank_recovery": _distribution(
            [float(row["wrong_bank"]["functional_recovery"]) for row in rows]
        ),
        "correct_minus_wrong_recovery": _distribution(margins),
        "wrong_to_correct_benefit_retention": _distribution(retentions),
        "wrong_to_correct_update_cosine": _distribution(cosines),
        "correct_bank_better_count": sum(value > 0 for value in margins),
        "bank_margin_at_least_0_10_count": sum(value >= 0.10 for value in margins),
        "wrong_bank_positive_count": sum(
            float(row["wrong_bank"]["functional_recovery"]) > 0 for row in rows
        ),
    }
    operator_bank_interaction_identifiable = (
        float(summary["correct_recovery"]["median"]) >= 0.75
        and float(summary["correct_minus_wrong_recovery"]["median"]) >= 0.10
        and int(summary["correct_bank_better_count"]) >= 8
    )
    write_json_atomic(
        args.output_dir / "aggregate.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "worker_count": args.worker_count,
            "git_commit": next(iter(commits)),
            "inverse_covariance_power": args.inverse_power,
            "primal_mode": args.primal_mode,
            "operator_bank_interaction_identifiable": (
                operator_bank_interaction_identifiable
            ),
            "decision_rule": {
                "median_correct_recovery": 0.75,
                "median_correct_minus_wrong_recovery": 0.10,
                "minimum_correct_bank_better_tasks": 8,
            },
            "summary": summary,
            "tasks": rows,
        },
    )
    print(
        {
            "operator_bank_interaction_identifiable": (
                operator_bank_interaction_identifiable
            ),
            "summary": summary,
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    work = commands.add_parser("worker")
    for name in (
        "config",
        "gate_config",
        "base_config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "compiler_run",
        "compiler_checkpoint",
        "condition_cache_root",
        "r5_aggregate",
        "output_dir",
    ):
        work.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    work.add_argument("--worker-index", type=int, required=True)
    work.add_argument("--worker-count", type=int, required=True)
    work.add_argument("--inverse-power", type=float, default=1.0)
    work.add_argument(
        "--primal-mode",
        choices=("checkpoint", "fit_transport"),
        default="checkpoint",
    )
    collect = commands.add_parser("aggregate")
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--worker-count", type=int, required=True)
    collect.add_argument("--inverse-power", type=float, default=1.0)
    collect.add_argument(
        "--primal-mode",
        choices=("checkpoint", "fit_transport"),
        default="checkpoint",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
