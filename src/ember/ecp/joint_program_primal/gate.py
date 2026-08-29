"""J2 task-local functional positive control and qualification evidence."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from ember.ecp.bank_conditioning.primal_capacity import (
    TaskLocalPrimalCode,
    recovery_record,
    task_local_output,
)
from ember.ecp.joint_program_primal.train_step import (
    functional_loss_and_backward,
    functional_panel_batch,
    prepare_joint_condition,
)
from ember.ecp.joint_program_primal.runtime import (
    JointProgramPrimalRuntime,
    prepare_joint_program_primal_runtime,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_value,
)


POSITIVE_CONTROL_SCHEMA = "ember_ecp_j2_functional_positive_control_task_v1"


def _task_teachers(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> tuple[Any, ...]:
    member_names = tuple(sorted(runtime.mapping_split.member_names[task_id]))
    rows = []
    for condition in runtime.task_conditions[task_id].fit_views:
        values = runtime.native_teachers.lookup_members(
            authority_id=task_id,
            k=1,
            video_demo=condition.video_demo,
            member_names=member_names,
        )
        if values is None or tuple(row.member_name for row in values) != member_names:
            raise RuntimeError("J2 positive control lost its fit-only teacher set")
        rows.extend(values)
    return tuple(rows)


def _positive_control_code(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> tuple[TaskLocalPrimalCode, torch.optim.Optimizer]:
    runtime.writer_state.requires_grad_(False).eval()
    code = TaskLocalPrimalCode(
        runtime.owners,
        _task_teachers(runtime, task_id),
        s_ref=runtime.ranks.s_ref,
    ).to(runtime.context.device)
    cell = runtime.config["optimization"]["task_local_positive_control"]
    optimizer = torch.optim.AdamW(
        code.parameters(),
        lr=float(cell["learning_rate"]),
        betas=tuple(cell["betas"]),
        weight_decay=float(cell["weight_decay"]),
    )
    if (
        any(parameter.requires_grad for parameter in runtime.writer_state.parameters())
        or any(parameter.requires_grad for parameter in runtime.policy.parameters())
        or not all(parameter.requires_grad for parameter in code.parameters())
    ):
        raise RuntimeError("J2 positive-control parameter ownership changed")
    return code, optimizer


def _task_banks(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> tuple[dict[int, Any], dict[int, Mapping[str, Any]], float]:
    started = time.monotonic()
    conditions = (
        *runtime.task_conditions[task_id].fit_views,
        runtime.task_conditions[task_id].held_video,
    )
    banks, metrics = {}, {}
    for condition in conditions:
        prepared, condition_metrics = prepare_joint_condition(runtime, condition)
        if len(prepared.videos) != 1:
            raise RuntimeError("J2 positive control escaped K1")
        banks[condition.video_demo] = prepared.videos[0]
        metrics[condition.video_demo] = condition_metrics
        del prepared
    return banks, metrics, time.monotonic() - started


def _complete_state(
    runtime: JointProgramPrimalRuntime,
    *,
    code: TaskLocalPrimalCode,
    bank: Any,
    canonicalize: bool,
) -> tuple[dict[str, torch.Tensor], Any]:
    output = task_local_output(
        operator=runtime.compiler.bank_operator,
        prepared=bank,
        code=code,
        s_ref=runtime.ranks.s_ref,
    )
    residual = residual_lora_state(
        output.residual,
        runtime.rank4_contract,
        canonicalize=canonicalize,
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    return complete, output


def _train_positive_control(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    code: TaskLocalPrimalCode,
    optimizer: torch.optim.Optimizer,
    banks: Mapping[int, Any],
) -> tuple[list[dict[str, Any]], float]:
    cell = runtime.config["optimization"]["task_local_positive_control"]
    steps = 1 if runtime.args.mode == "profile" else int(cell["updates"])
    report_steps = {1, steps}
    if steps > 1:
        report_steps.update({steps // 4, steps // 2, 3 * steps // 4})
    conditions = runtime.task_conditions[task_id].fit_views
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        visit_index = (step - 1) % int(runtime.config["data"]["panel_visits"])
        batch, panel = functional_panel_batch(
            runtime,
            task_id=task_id,
            panel_name="a",
            visit_index=visit_index,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for condition in conditions:
            complete, output = _complete_state(
                runtime,
                code=code,
                bank=banks[condition.video_demo],
                canonicalize=False,
            )
            losses.append(
                functional_loss_and_backward(
                    runtime,
                    state=complete,
                    batch=batch,
                    policy_rng_seed=panel.policy_rng_seed,
                    loss_divisor=2.0,
                )
            )
            del complete, output
        parameters = tuple(code.parameters())
        if any(
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise RuntimeError("J2 positive-control gradient is invalid")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(cell["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("J2 positive-control gradient norm is non-finite")
        optimizer.step()
        if step in report_steps:
            curve.append(
                {
                    "optimizer_step": step,
                    "panel_visit": visit_index,
                    "mean_fit_functional_loss": statistics.fmean(losses),
                    "carrier_functional_loss": panel.flow_loss,
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )
    torch.cuda.synchronize(runtime.context.device)
    return curve, time.monotonic() - started


def _functional_value(
    runtime: JointProgramPrimalRuntime,
    *,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    seed: int,
) -> float:
    value, details = functional_lora_loss_value(
        runtime.policy,
        state,
        runtime.ranks.contract,
        batch=batch,
        policy_rng_seed=seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("J2 positive-control evaluation loss changed")
    return float(value)


def _verify_carrier_panel_authority(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> dict[str, float]:
    batch, panel = functional_panel_batch(
        runtime, task_id=task_id, panel_name="a", visit_index=0
    )
    observed = _functional_value(
        runtime,
        state=runtime.ranks.carrier_complete,
        batch=batch,
        seed=panel.policy_rng_seed,
    )
    error = abs(observed - panel.flow_loss)
    replay_tolerance = 1e-4
    if error > replay_tolerance:
        raise RuntimeError(
            "J2 sealed carrier panel loss no longer replays: "
            f"sealed={panel.flow_loss:.17g}, replayed={observed:.17g}, "
            f"absolute_error={error:.17g}"
        )
    return {
        "sealed": panel.flow_loss,
        "replayed": observed,
        "absolute_error": error,
        "absolute_tolerance": replay_tolerance,
    }


def _evaluate_video(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    video_demo: int,
    state: Mapping[str, torch.Tensor],
    visit_count: int,
) -> dict[str, Any]:
    result = {}
    for panel_name in ("a", "b"):
        rows = []
        for visit_index in range(visit_count):
            batch, panel = functional_panel_batch(
                runtime,
                task_id=task_id,
                panel_name=panel_name,
                visit_index=visit_index,
            )
            free = _functional_value(
                runtime, state=state, batch=batch, seed=panel.policy_rng_seed
            )
            rows.append(
                {
                    "visit": visit_index,
                    "carrier_loss": panel.flow_loss,
                    "free_primal_loss": free,
                    "benefit_over_carrier": panel.flow_loss - free,
                }
            )
        result[f"panel_{panel_name}"] = {
            "visits": rows,
            "carrier_loss": statistics.fmean(row["carrier_loss"] for row in rows),
            "free_primal_loss": statistics.fmean(
                row["free_primal_loss"] for row in rows
            ),
            "benefit_over_carrier": statistics.fmean(
                row["benefit_over_carrier"] for row in rows
            ),
        }
    return {"video_demo": video_demo, **result}


def _evaluate_positive_control(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    code: TaskLocalPrimalCode,
    banks: Mapping[int, Any],
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    visit_count = 1 if runtime.args.mode == "profile" else int(
        runtime.config["data"]["panel_visits"]
    )
    conditions = (
        *runtime.task_conditions[task_id].fit_views,
        runtime.task_conditions[task_id].held_video,
    )
    videos = {}
    held_output = None
    with torch.no_grad():
        for condition in conditions:
            state, output = _complete_state(
                runtime,
                code=code,
                bank=banks[condition.video_demo],
                canonicalize=True,
            )
            videos[condition.video_demo] = _evaluate_video(
                runtime,
                task_id=task_id,
                video_demo=condition.video_demo,
                state=state,
                visit_count=visit_count,
            )
            if condition.video_demo == runtime.task_conditions[task_id].held_video.video_demo:
                held_output = output
            del state
    held = runtime.task_conditions[task_id].held_video
    member_names = tuple(sorted(runtime.mapping_split.member_names[task_id]))
    teachers = runtime.native_teachers.lookup_members(
        authority_id=task_id,
        k=1,
        video_demo=held.video_demo,
        member_names=member_names,
    )
    if teachers is None or held_output is None:
        raise RuntimeError("J2 positive control lost held diagnostic authority")
    _, factor = recovery_record(
        held_output,
        teachers,
        runtime.owners,
        temperature=float(runtime.base_config["optimization"]["mapping"]["temperature"]),
    )
    fit_videos = [
        videos[condition.video_demo]
        for condition in runtime.task_conditions[task_id].fit_views
    ]
    held_video = videos[held.video_demo]
    return (
        {
            "fit_videos": fit_videos,
            "held_video": held_video,
            "held_factor_diagnostic": factor,
            "held_panel_b_benefit_over_carrier": held_video["panel_b"][
                "benefit_over_carrier"
            ],
            "every_video_panel_b_above_carrier": all(
                row["panel_b"]["benefit_over_carrier"] > 0
                for row in (*fit_videos, held_video)
            ),
        },
        time.monotonic() - started,
    )


def _save_code(
    output_dir: Path, code: TaskLocalPrimalCode, task_id: int
) -> Path:
    path = output_dir / "task_local_primal.safetensors"
    save_file(
        {
            name: value.detach().float().cpu().contiguous()
            for name, value in code.state_dict().items()
        },
        str(path),
        metadata={
            "schema_version": POSITIVE_CONTROL_SCHEMA,
            "task": str(task_id),
            "deployment_candidate": "false",
        },
    )
    return path


def run_positive_control(args: Any) -> None:
    if args.phase != "positive-control" or args.task is None or args.resume is not None:
        raise ValueError("J2 positive-control launch arguments changed")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    if context.world_size != 1:
        raise ValueError("J2 task-local positive control requires one process per GPU")
    runtime: JointProgramPrimalRuntime | None = None
    try:
        runtime = prepare_joint_program_primal_runtime(args, context)
        gradient_tasks = set(
            map(
                int,
                (
                    *runtime.config["task_split"]["gradient_meta"],
                    *runtime.config["task_split"]["gradient_target"],
                ),
            )
        )
        task_id = int(args.task)
        if task_id not in gradient_tasks:
            raise ValueError("J2 positive-control task is not a gradient task")
        started = time.monotonic()
        carrier_validation = (
            _verify_carrier_panel_authority(runtime, task_id)
            if args.mode == "profile"
            else None
        )
        code, optimizer = _positive_control_code(runtime, task_id)
        banks, bank_metrics, prepare_seconds = _task_banks(runtime, task_id)
        curve, train_seconds = _train_positive_control(
            runtime,
            task_id=task_id,
            code=code,
            optimizer=optimizer,
            banks=banks,
        )
        evaluation, evaluation_seconds = _evaluate_positive_control(
            runtime, task_id=task_id, code=code, banks=banks
        )
        checkpoint = _save_code(args.output_dir, code, task_id)
        report = {
            "schema_version": POSITIVE_CONTROL_SCHEMA,
            "status": "complete",
            "task": task_id,
            "role": runtime.task_conditions[task_id].fit_views[0].role,
            "fit_videos": [
                row.video_demo for row in runtime.task_conditions[task_id].fit_views
            ],
            "held_video": runtime.task_conditions[task_id].held_video.video_demo,
            "updates": 1 if args.mode == "profile" else int(
                runtime.config["optimization"]["task_local_positive_control"][
                    "updates"
                ]
            ),
            "curve": curve,
            "carrier_panel_authority_validation": carrier_validation,
            "evaluation": evaluation,
            "bank_metrics": bank_metrics,
            "prepare_seconds": prepare_seconds,
            "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "evaluation_to_training_wall": evaluation_seconds
            / max(train_seconds, 1e-12),
            "total_seconds": time.monotonic() - started,
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                context.device
            ),
            "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                context.device
            ),
            "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "deployment_candidate": False,
            },
            "held_backward_calls": 0,
            "panel_b_backward_calls": 0,
            "action_meta_installed": False,
            "single_complete_rank16": True,
        }
        write_json_atomic(args.output_dir / "result.json", report)
        write_json_atomic(
            args.output_dir / "completion.json",
            {"schema_version": POSITIVE_CONTROL_SCHEMA, "task": task_id},
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
