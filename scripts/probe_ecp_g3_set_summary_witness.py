#!/usr/bin/env python3
"""Train the fixed task93/q20 S2 set-summary free-code mechanism witness."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist

from ember.ecp.bank_conditioning.native_bank_runtime import (
    NativeCandidateBank,
    materialize_condition_banks,
    prepare_frozen_native_bank_runtime,
    prepare_k1_condition,
)
from ember.ecp.bank_conditioning.set_summary import (
    SetSummaryFactorSelector,
    TaskLocalSelectionCode,
)
from ember.ecp.native_factors import native_output_group_count
from ember.ecp.shared_compiler_native_teacher import (
    factor_subspace_loss,
    low_rank_update_direction_loss,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl


REPORT_SCHEMA = "ember_ecp_g3_set_summary_s2_witness_v2"
CHECKPOINT_SCHEMA = "ember_ecp_g3_set_summary_s2_witness_checkpoint_v2"


@dataclass(frozen=True)
class _Teacher:
    video_demo: int
    member_name: str
    a: torch.Tensor
    b: torch.Tensor
    scales: torch.Tensor


@dataclass(frozen=True)
class _Condition:
    video_demo: int
    input_bank: NativeCandidateBank
    output_banks: tuple[NativeCandidateBank, ...]
    teachers: tuple[_Teacher, ...]


@dataclass(frozen=True)
class _CapturedWitness:
    fit: tuple[_Condition, ...]
    held: _Condition
    teacher_pool: tuple[_Teacher, ...]
    device: torch.device
    output_groups: int
    inventory: dict[str, Any]
    actual_frozen: dict[str, int]
    candidate_encoder_authority: dict[str, Any]
    runtime_initialization_seconds: float
    capture_seconds: dict[str, float]
    capture_peak_reserved_bytes: int


@dataclass(frozen=True)
class _TrainingResult:
    selector: SetSummaryFactorSelector
    code_model: TaskLocalSelectionCode
    optimizer: torch.optim.Optimizer
    trainable: tuple[torch.nn.Parameter, ...]
    initial: dict[str, Any]
    final: dict[str, Any]
    last_loss: dict[str, float]
    training_seconds: float


def _teacher_rows(
    rows: Sequence[Any], *, target: int, video_demo: int, device: torch.device
) -> tuple[_Teacher, ...]:
    return tuple(
        _Teacher(
            video_demo=int(video_demo),
            member_name=str(row.member_name),
            a=row.a[target].detach().float().to(device),
            b=row.b[target].detach().float().to(device),
            scales=row.scales[target].detach().float().to(device),
        )
        for row in rows
    )


def _capture_condition(
    runtime: Any,
    *,
    task_id: int,
    video_demo: int,
    target: int,
    member_names: Sequence[str],
    fit: bool,
) -> tuple[_Condition, float]:
    started = time.perf_counter()
    condition = prepare_k1_condition(
        runtime,
        task_id=task_id,
        video_demo=video_demo,
        robustness_view="g3_set_summary_s2_witness_k1",
    )
    _state, banks = materialize_condition_banks(runtime, condition, (target,))
    groups = native_output_group_count(runtime.owners[target])
    teachers: tuple[_Teacher, ...] = ()
    if fit:
        rows = runtime.native_teachers.lookup_members(
            authority_id=task_id,
            k=1,
            video_demo=video_demo,
            member_names=tuple(map(str, member_names)),
        )
        if rows is None or tuple(row.member_name for row in rows) != tuple(
            map(str, member_names)
        ):
            raise RuntimeError("set-summary witness teacher lookup failed")
        teachers = _teacher_rows(
            rows,
            target=target,
            video_demo=video_demo,
            device=runtime.context.device,
        )
    result = _Condition(
        video_demo=int(video_demo),
        input_bank=banks[(target, "input", 0)],
        output_banks=tuple(
            banks[(target, "output", group)] for group in range(groups)
        ),
        teachers=teachers,
    )
    del condition, banks
    return result, time.perf_counter() - started


def _scaled_b(factor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (factor * scales[:, None]).transpose(0, 1)


def _update_loss(
    student_a: torch.Tensor,
    student_b: torch.Tensor,
    student_scales: torch.Tensor,
    teacher: _Teacher,
) -> torch.Tensor:
    return low_rank_update_direction_loss(
        student_a,
        _scaled_b(student_b, student_scales),
        teacher.a,
        _scaled_b(teacher.b, teacher.scales),
    )


def _soft_min(losses: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if losses.ndim != 1 or losses.numel() <= 0 or temperature <= 0:
        raise ValueError("set-summary member reduction changed")
    return -temperature * torch.logsumexp(-losses / temperature, dim=0) + (
        temperature * math.log(losses.numel())
    )


def _predict(
    selector: SetSummaryFactorSelector,
    code_model: TaskLocalSelectionCode,
    condition: _Condition,
) -> tuple[torch.Tensor, torch.Tensor]:
    code, event_weights = code_model()
    return selector(
        input_bank=condition.input_bank,
        output_banks=condition.output_banks,
        code=code,
        event_weights=event_weights,
    )


def _geometry(
    student_a: torch.Tensor,
    student_b: torch.Tensor,
    student_scales: torch.Tensor,
    teachers: Sequence[_Teacher],
) -> dict[str, Any]:
    rows = []
    for teacher in teachers:
        rows.append(
            {
                "reference_video_demo": teacher.video_demo,
                "member_name": teacher.member_name,
                "update_cosine": float(
                    1.0
                    - _update_loss(
                        student_a, student_b, student_scales, teacher
                    )
                ),
                "input_subspace_similarity": float(
                    1.0 - factor_subspace_loss(student_a, teacher.a)
                ),
                "output_subspace_similarity": float(
                    1.0 - factor_subspace_loss(student_b, teacher.b)
                ),
            }
        )
    if not rows:
        raise ValueError("set-summary geometry has no teacher reference")
    best = max(rows, key=lambda row: row["update_cosine"])
    return {"best": best, "members": rows}


def _prediction_cosine(
    first: tuple[torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor],
    scales: torch.Tensor,
) -> float:
    return float(
        1.0
        - low_rank_update_direction_loss(
            first[0], _scaled_b(first[1], scales), second[0],
            _scaled_b(second[1], scales),
        )
    )


@torch.no_grad()
def _evaluate(
    selector: SetSummaryFactorSelector,
    code_model: TaskLocalSelectionCode,
    fit: Sequence[_Condition],
    held: _Condition,
    teacher_pool: Sequence[_Teacher],
    scales: torch.Tensor,
    *,
    step: int,
) -> dict[str, Any]:
    selector.eval()
    code_model.eval()
    predictions = [_predict(selector, code_model, condition) for condition in fit]
    held_prediction = _predict(selector, code_model, held)
    fit_rows = [
        {
            "video_demo": condition.video_demo,
            **_geometry(a, b, scales, condition.teachers),
        }
        for condition, (a, b) in zip(fit, predictions, strict=True)
    ]
    held_row = {
        "video_demo": held.video_demo,
        **_geometry(*held_prediction, scales, teacher_pool),
    }
    fit_scores = torch.tensor(
        [row["best"]["update_cosine"] for row in fit_rows], dtype=torch.float64
    )
    fit_median = float(fit_scores.median())
    held_score = float(held_row["best"]["update_cosine"])
    consistency = [
        _prediction_cosine(prediction, held_prediction, scales)
        for prediction in predictions
    ]
    selector.train()
    code_model.train()
    return {
        "step": int(step),
        "fit": fit_rows,
        "held": held_row,
        "fit_update_median": fit_median,
        "held_update": held_score,
        "held_to_fit": (
            held_score / fit_median if fit_median > 0.0 else 0.0
        ),
        "held_to_fit_prediction_cosines": consistency,
    }


def _training_loss(
    selector: SetSummaryFactorSelector,
    code_model: TaskLocalSelectionCode,
    fit: Sequence[_Condition],
    scales: torch.Tensor,
    *,
    member_temperature: float,
    dispersion_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions = [_predict(selector, code_model, condition) for condition in fit]
    paired = []
    for condition, (a, b) in zip(fit, predictions, strict=True):
        member_losses = torch.stack(
            tuple(_update_loss(a, b, scales, teacher) for teacher in condition.teachers)
        )
        paired.append(_soft_min(member_losses, temperature=member_temperature))
    paired_loss = torch.stack(paired).mean()
    dispersion = low_rank_update_direction_loss(
        predictions[0][0],
        _scaled_b(predictions[0][1], scales),
        predictions[1][0],
        _scaled_b(predictions[1][1], scales),
    )
    total = paired_loss + dispersion_weight * dispersion
    return total, {
        "paired_update_loss": float(paired_loss.detach()),
        "cross_video_dispersion_loss": float(dispersion.detach()),
        "total_loss": float(total.detach()),
    }


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }


def _save_checkpoint(
    path: Path,
    *,
    selector: SetSummaryFactorSelector,
    code_model: TaskLocalSelectionCode,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "step": int(step),
            "selector": _cpu_state(selector),
            "task_local_capacity_code": _cpu_state(code_model),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _validate_config(config: Mapping[str, Any], *, formal: bool, steps: int | None) -> int:
    if config.get("schema_version") != "ember_ecp_set_summary_s2_v2":
        raise ValueError("set-summary witness config schema changed")
    witness = config["witness"]
    if (
        int(witness["authority_id"]) != 93
        or int(witness["target"]) != 20
        or tuple(map(int, witness["fit_videos"])) != (18, 48)
        or int(witness["held_video"]) != 0
    ):
        raise ValueError("set-summary witness selection changed")
    configured = int(config["optimization"]["steps"])
    if steps is None:
        return configured
    if formal or steps <= 0:
        raise ValueError("only non-formal smoke may override optimization steps")
    return int(steps)


def _frozen_counts(runtime: Any) -> dict[str, int]:
    return {
        "source_policy_trainable_parameter_count": sum(
            value.numel() for value in runtime.policy.parameters() if value.requires_grad
        ),
        "natural_program_trainable_parameter_count": sum(
            value.numel() for value in runtime.program.parameters() if value.requires_grad
        ),
        "compiler_trainable_parameter_count": sum(
            value.numel() for value in runtime.compiler.parameters() if value.requires_grad
        ),
    }


def _capture_witness(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> _CapturedWitness:
    started = time.perf_counter()
    reference = args.asset_root / config["authorities"]["reference_config"]
    runtime = prepare_frozen_native_bank_runtime(
        reference_config=reference,
        asset_root=args.asset_root,
        data_root=args.data_root,
        candidate_encoder_checkpoint=(
            args.asset_root
            / config["authorities"]["candidate_encoder_checkpoint"]
        ),
    )
    runtime_ready = time.perf_counter()
    try:
        if runtime.candidate_encoder_authority.get("authority_commit") != str(
            config["authorities"]["candidate_encoder_commit"]
        ):
            raise RuntimeError("set-summary candidate encoder authority changed")
        witness = config["witness"]
        task_id = int(witness["authority_id"])
        target = int(witness["target"])
        members = tuple(map(str, witness["member_names"]))
        fit = []
        capture_seconds = {}
        for video in map(int, witness["fit_videos"]):
            condition, elapsed = _capture_condition(
                runtime,
                task_id=task_id,
                video_demo=video,
                target=target,
                member_names=members,
                fit=True,
            )
            fit.append(condition)
            capture_seconds[str(video)] = elapsed
        held, elapsed = _capture_condition(
            runtime,
            task_id=task_id,
            video_demo=int(witness["held_video"]),
            target=target,
            member_names=members,
            fit=False,
        )
        capture_seconds[str(held.video_demo)] = elapsed
        frozen = _frozen_counts(runtime)
        inventory = dict(runtime.inventory)
        if inventory["action_meta_module_count"] != 0 or any(frozen.values()):
            raise RuntimeError("set-summary witness crossed the pure-Native wall")
        teacher_pool = tuple(
            teacher for condition in fit for teacher in condition.teachers
        )
        return _CapturedWitness(
            fit=tuple(fit),
            held=held,
            teacher_pool=teacher_pool,
            device=runtime.context.device,
            output_groups=native_output_group_count(runtime.owners[target]),
            inventory=inventory,
            actual_frozen=frozen,
            candidate_encoder_authority=dict(
                runtime.candidate_encoder_authority
            ),
            runtime_initialization_seconds=runtime_ready - started,
            capture_seconds=capture_seconds,
            capture_peak_reserved_bytes=torch.cuda.max_memory_reserved(
                runtime.context.device
            ),
        )
    finally:
        runtime.close()


def _build_student(
    config: Mapping[str, Any], captured: _CapturedWitness
) -> tuple[
    SetSummaryFactorSelector,
    TaskLocalSelectionCode,
    tuple[torch.nn.Parameter, ...],
    torch.optim.Optimizer,
]:
    seed = int(config["optimization"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model = config["model"]
    selector = SetSummaryFactorSelector(
        feature_width=int(model["feature_width"]),
        event_slots=int(model["event_slots"]),
        output_groups=captured.output_groups,
        global_events=True,
        hidden_width=int(model["hidden_width"]),
        logit_bound=float(model["logit_bound"]),
    ).to(captured.device)
    code_model = TaskLocalSelectionCode(
        events=int(model["event_slots"]), width=int(model["feature_width"])
    ).to(captured.device)
    trainable = tuple(selector.parameters()) + tuple(code_model.parameters())
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        betas=tuple(map(float, optimization["betas"])),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    return selector, code_model, trainable, optimizer


def _train_witness(
    config: Mapping[str, Any],
    captured: _CapturedWitness,
    *,
    steps: int,
    trace_path: Path,
) -> _TrainingResult:
    selector, code_model, trainable, optimizer = _build_student(config, captured)
    scales = torch.stack(
        tuple(row.scales for row in captured.teacher_pool)
    ).median(0).values
    initial = _evaluate(
        selector,
        code_model,
        captured.fit,
        captured.held,
        captured.teacher_pool,
        scales,
        step=0,
    )
    append_jsonl(trace_path, {"kind": "evaluation", **initial})
    optimization = config["optimization"]
    started = time.perf_counter()
    last_loss: dict[str, float] = {}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, last_loss = _training_loss(
            selector,
            code_model,
            captured.fit,
            scales,
            member_temperature=float(optimization["member_softmin_temperature"]),
            dispersion_weight=float(optimization["cross_video_dispersion_weight"]),
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("set-summary witness loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("set-summary witness gradient became non-finite")
        optimizer.step()
        if step % int(optimization["report_interval"]) == 0 or step == steps:
            append_jsonl(
                trace_path,
                {
                    "kind": "training",
                    "step": step,
                    **last_loss,
                    "gradient_norm": float(gradient_norm),
                },
            )
    training_seconds = time.perf_counter() - started
    final = _evaluate(
        selector,
        code_model,
        captured.fit,
        captured.held,
        captured.teacher_pool,
        scales,
        step=steps,
    )
    append_jsonl(trace_path, {"kind": "evaluation", **final})
    return _TrainingResult(
        selector=selector,
        code_model=code_model,
        optimizer=optimizer,
        trainable=trainable,
        initial=initial,
        final=final,
        last_loss=last_loss,
        training_seconds=training_seconds,
    )


def _gate_result(
    witness: Mapping[str, Any], final: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, bool]]:
    thresholds = {
        "fit_update_median": float(witness["fit_effective_update_median_minimum"]),
        "held_update": float(witness["held_effective_update_minimum"]),
        "held_input": float(witness["held_input_subspace_minimum"]),
        "held_output": float(witness["held_output_subspace_minimum"]),
        "held_to_fit": float(witness["held_to_fit_minimum"]),
    }
    held = final["held"]["best"]
    checks = {
        "fit_update": final["fit_update_median"] >= thresholds["fit_update_median"],
        "held_update": final["held_update"] >= thresholds["held_update"],
        "held_input": held["input_subspace_similarity"] >= thresholds["held_input"],
        "held_output": held["output_subspace_similarity"] >= thresholds["held_output"],
        "held_to_fit": final["held_to_fit"] >= thresholds["held_to_fit"],
    }
    return thresholds, checks


def _report_payload(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    captured: _CapturedWitness,
    training: _TrainingResult,
    steps: int,
    total_seconds: float,
) -> dict[str, Any]:
    witness = config["witness"]
    thresholds, checks = _gate_result(witness, training.final)
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "formal": bool(args.formal),
        "gate_pass": all(checks.values()),
        "claim_boundary": witness["claim_boundary"],
        "task": {
            "authority_id": int(witness["authority_id"]),
            "target": int(witness["target"]),
        },
        "videos": {
            "fit": [row.video_demo for row in captured.fit],
            "zero_gradient_held": captured.held.video_demo,
        },
        "initial": training.initial,
        "final": training.final,
        "thresholds": thresholds,
        "checks": checks,
        "optimization": {
            "steps": steps,
            "fixed_final_step_selection": True,
            "held_video_gradient_steps": 0,
            "held_video_checkpoint_selection": False,
            "trainable_parameter_count": sum(
                row.numel() for row in training.trainable
            ),
            "last_loss": training.last_loss,
        },
        "information_wall": {
            **captured.actual_frozen,
            "candidate_encoder_authority": captured.candidate_encoder_authority,
            "action_meta_module_count": captured.inventory["action_meta_module_count"],
            "deployment_task_video_frame_free_parameter_count": 0,
            "capacity_task_local_parameter_count": sum(
                value.numel() for value in training.code_model.parameters()
            ),
            "capacity_only_task_local_code": True,
            "teacher_tensors_in_checkpoint": False,
            "held_outcome_reads": 0,
            "shuffled_or_reversed_use": False,
        },
        "profile": {
            "runtime_initialization_seconds": captured.runtime_initialization_seconds,
            "capture_seconds_by_video": captured.capture_seconds,
            "capture_peak_reserved_bytes": captured.capture_peak_reserved_bytes,
            "training_seconds": training.training_seconds,
            "steps_per_second": steps / max(training.training_seconds, 1e-12),
            "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                captured.device
            ),
            "training_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                captured.device
            ),
            "total_seconds": total_seconds,
        },
        "git": dict(repository),
    }


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise ValueError("set-summary witness output already exists")
    config = read_json(args.config)
    steps = _validate_config(config, formal=args.formal, steps=args.steps)
    repository = git_state(REPO_ROOT)
    if args.formal and (
        not git_state_is_clean_pushed_or_frozen_authority(repository)
        or repository.get("branch") != ""
        or repository.get("upstream") is not None
    ):
        raise ValueError("formal set-summary witness requires clean detached authority")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        captured = _capture_witness(args, config)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(captured.device)
        training = _train_witness(
            config,
            captured,
            steps=steps,
            trace_path=args.output_dir / "metrics.jsonl",
        )
        _save_checkpoint(
            args.output_dir / "checkpoint.pt",
            selector=training.selector,
            code_model=training.code_model,
            optimizer=training.optimizer,
            step=steps,
        )
        report = _report_payload(
            args=args,
            config=config,
            repository=repository,
            captured=captured,
            training=training,
            steps=steps,
            total_seconds=time.perf_counter() - started,
        )
        write_json_atomic(args.output_dir / "report.json", report)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--formal", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("config", "asset_root", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
