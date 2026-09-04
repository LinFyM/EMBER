"""Task-local functional qualification for the current-bank Composer."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.bank_conditioning.mapping import load_mapping_split
from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.model import PolicyResponseWriterOutput
from ember.ecp.policy_response_writer.process import PolicyResponseProcessOutput
from ember.ecp.policy_response_writer.tasklocal_contract import (
    build_tasklocal_result,
    build_tasklocal_run_contract,
    seal_or_validate_tasklocal_run_contract,
)
from ember.ecp.policy_response_writer.training import (
    PolicyResponseRuntime,
    capture_video,
    functional_panel_batch,
)
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    functional_lora_loss_value,
    writer_chain_rule_surrogate,
)


TASKLOCAL_STAGE = "policy_response_writer_tasklocal_composer"
TASKLOCAL_RUN_SCHEMA = "ember_policy_response_writer_tasklocal_run_v2"


def _video_split(
    runtime: PolicyResponseRuntime, task: int
) -> tuple[tuple[int, int], int]:
    split = load_mapping_split(runtime.base, asset_root=runtime.args.asset_root)
    fit = split.fit_by_task.get(task, ())
    held = split.video_held_by_task.get(task, ())
    if len(fit) < 2 or len(held) != 1:
        raise ValueError("Policy-Response Writer task-local video split changed")
    fit_demos = (int(fit[0].video_demo), int(fit[1].video_demo))
    held_demo = int(held[0].video_demo)
    panel_videos = set(runtime.panels[task].program_video_demos)
    if not {*fit_demos, held_demo} <= panel_videos:
        raise ValueError("task-local videos escaped the sealed functional panel")
    return fit_demos, held_demo


def _reference_losses(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    expected_videos: tuple[int, int, int],
) -> tuple[dict[int, float], Path]:
    root = (
        runtime.args.asset_root
        / str(runtime.config["authorities"]["task_local_reference_root"])
    ).resolve()
    path = root / f"task_{task:03d}" / "result.json"
    result = read_json(path)
    evaluation = result.get("evaluation", {})
    rows = (*evaluation.get("fit_videos", ()), evaluation.get("held_video", {}))
    values = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    if (
        result.get("status") != "complete"
        or int(result.get("task", -1)) != task
        or set(values) != set(expected_videos)
        or result.get("held_backward_calls") != 0
        or result.get("panel_b_backward_calls") != 0
        or any(not math.isfinite(value) for value in values.values())
    ):
        raise ValueError("task-local functional reference authority changed")
    return values, path


def _cache_evidence(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    videos: tuple[int, ...],
) -> tuple[
    dict[int, FrozenPolicyResponseVideo],
    dict[int, PolicyResponseProcessOutput],
    list[dict[str, Any]],
    float,
]:
    cache: dict[int, FrozenPolicyResponseVideo] = {}
    process_cache: dict[int, PolicyResponseProcessOutput] = {}
    records = []
    started = time.monotonic()
    for demo in videos:
        evidence, capture = capture_video(runtime, task_id=task, video_demo=demo)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            process = runtime.writer.process(
                evidence,
                representation=runtime.args.representation,
            )
        frozen = evidence.to("cpu")
        records.append({**capture, "tensor_bytes": frozen.tensor_bytes})
        cache[demo] = frozen
        process_cache[demo] = process
        del evidence, process
        torch.cuda.empty_cache()
    return cache, process_cache, records, time.monotonic() - started


def _materialized_state(
    runtime: PolicyResponseRuntime,
    video: FrozenPolicyResponseVideo,
    process: PolicyResponseProcessOutput,
    *,
    canonicalize: bool,
) -> dict[str, torch.Tensor]:
    residual = runtime.writer.composer(
        (video,),
        (process,),
        s_ref=runtime.ranks.s_ref,
    )
    output = PolicyResponseWriterOutput(
        residual=residual,
        processes=(process,),
    )
    return runtime.writer.materialize(
        output,
        carrier_state=runtime.ranks.carrier_rank12,
        rank4_contract=runtime.rank4_contract,
        rank16_contract=runtime.ranks.contract,
        canonicalize=canonicalize,
    )


def _functional_value(
    runtime: PolicyResponseRuntime,
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
            runtime.config["optimization"]["task_local"]["functional_microbatch"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("task-local functional value changed")
    return float(value)


def _functional_backward(
    runtime: PolicyResponseRuntime,
    *,
    video: FrozenPolicyResponseVideo,
    process: PolicyResponseProcessOutput,
    batch: Mapping[str, Any],
    seed: int,
) -> tuple[float, int, int]:
    # First evaluate the frozen policy from detached LoRA leaves.  The Writer is
    # then deterministically recomputed so its graph never overlaps the policy
    # graph.  This is the exact first-order chain rule used throughout EMBER.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        leaf_state = _materialized_state(
            runtime, video, process, canonicalize=False
        )
    value, details, leaf_gradients = functional_lora_loss_gradient(
        runtime.policy,
        leaf_state,
        runtime.ranks.contract,
        batch=batch,
        policy_rng_seed=seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            runtime.config["optimization"]["task_local"]["functional_microbatch"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("task-local functional gradient changed")
    del leaf_state
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_state = _materialized_state(
            runtime, video, process, canonicalize=False
        )
        surrogate = writer_chain_rule_surrogate(generated_state, leaf_gradients)
    surrogate.backward()
    torch.cuda.synchronize(runtime.context.device)
    return (
        float(value),
        int(torch.cuda.max_memory_allocated(runtime.context.device)),
        int(torch.cuda.max_memory_reserved(runtime.context.device)),
    )


def _optimizer(
    runtime: PolicyResponseRuntime,
) -> tuple[
    tuple[torch.nn.Parameter, ...],
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
]:
    runtime.writer.requires_grad_(False)
    runtime.writer.process.eval()
    runtime.writer.composer.requires_grad_(True).train()
    parameters = tuple(
        value for value in runtime.writer.composer.parameters() if value.requires_grad
    )
    cell = runtime.config["optimization"]["task_local"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cell["learning_rate"]),
        betas=tuple(cell["betas"]),
        weight_decay=float(cell["weight_decay"]),
    )
    warmup = int(cell["warmup_updates"])
    effective = int(cell["effective_updates"])
    floor = float(cell["decay_learning_rate"]) / float(cell["learning_rate"])

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(effective, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    if (
        not parameters
        or runtime.writer.composer.task_query is None
        or any(parameter.requires_grad for parameter in runtime.writer.process.parameters())
        or any(parameter.requires_grad for parameter in runtime.policy.parameters())
        or any(parameter.requires_grad for parameter in runtime.stage0.parameters())
    ):
        raise RuntimeError("task-local Composer parameter ownership changed")
    return parameters, optimizer, scheduler


def _composer_gradient_norms(runtime: PolicyResponseRuntime) -> dict[str, float]:
    groups = {
        "input_signed_attention": (
            "input_base_query",
            "input_contrast_query",
        ),
        "output_signed_attention": (
            "output_base_query",
            "output_contrast_query",
        ),
        "context_blocks": ("blocks", "query_seed"),
        "task_query": ("task_query",),
    }
    result = {}
    for label, prefixes in groups.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in runtime.writer.composer.named_parameters()
            if name.startswith(prefixes) and parameter.grad is not None
        ]
        result[label] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return result


def _evaluate_video(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    demo: int,
    evidence: FrozenPolicyResponseVideo,
    process: PolicyResponseProcessOutput,
    reference_loss: float,
    visits: int,
) -> dict[str, Any]:
    video = evidence.to(runtime.context.device)
    runtime.writer.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = _materialized_state(
            runtime, video, process, canonicalize=True
        )
    del video
    rows = []
    for visit_index in range(visits):
        batch, panel = functional_panel_batch(
            runtime,
            task_id=task,
            panel_name="b",
            visit_index=visit_index,
        )
        generated = _functional_value(
            runtime,
            state=state,
            batch=batch,
            seed=panel.policy_rng_seed,
        )
        rows.append(
            {
                "visit": visit_index,
                "carrier_loss": float(panel.flow_loss),
                "generated_loss": generated,
                "benefit_over_carrier": float(panel.flow_loss) - generated,
            }
        )
        del batch
    carrier = statistics.fmean(row["carrier_loss"] for row in rows)
    generated = statistics.fmean(row["generated_loss"] for row in rows)
    denominator = carrier - reference_loss
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("task-local reference functional denominator changed")
    del state
    return {
        "video_demo": demo,
        "visits": rows,
        "carrier_loss": carrier,
        "generated_loss": generated,
        "benefit_over_carrier": carrier - generated,
        "free_primal_loss": reference_loss,
        "free_primal_benefit": denominator,
        "functional_recovery": (carrier - generated) / denominator,
    }


def _evaluation(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    fit_demos: tuple[int, int],
    held_demo: int,
    cache: Mapping[int, FrozenPolicyResponseVideo],
    process_cache: Mapping[int, PolicyResponseProcessOutput],
    reference_losses: Mapping[int, float],
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    visits = (
        int(runtime.config["optimization"]["task_local"]["evaluation_visits"])
        if runtime.args.mode == "formal"
        else 1
    )
    rows = {
        demo: _evaluate_video(
            runtime,
            task=task,
            demo=demo,
            evidence=cache[demo],
            process=process_cache[demo],
            reference_loss=float(reference_losses[demo]),
            visits=visits,
        )
        for demo in (*fit_demos, held_demo)
    }
    fit = tuple(rows[demo] for demo in fit_demos)
    held = rows[held_demo]
    return (
        {
            "fit_videos": fit,
            "held_video": held,
            "fit_recovery_mean": statistics.fmean(
                float(row["functional_recovery"]) for row in fit
            ),
            "held_video_recovery": float(held["functional_recovery"]),
            "every_video_above_carrier": all(
                float(row["benefit_over_carrier"]) > 0 for row in (*fit, held)
            ),
            "panel_b_backward_calls": 0,
        },
        time.monotonic() - started,
    )


def _checkpoint_evaluations(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    fit_demos: tuple[int, int],
    held_demo: int,
    cache: Mapping[int, FrozenPolicyResponseVideo],
    process_cache: Mapping[int, PolicyResponseProcessOutput],
    reference_losses: Mapping[int, float],
) -> tuple[dict[str, dict[str, Any]], float]:
    rows = {}
    seconds = 0.0
    checkpoints = sorted(
        (runtime.args.output_dir / "checkpoints").glob("macro_*")
    )
    if len(checkpoints) != 2:
        raise RuntimeError("task-local Composer adjacent checkpoints changed")
    for checkpoint in checkpoints:
        runtime.writer.load_state_dict(
            load_file(
                str(checkpoint / "ecp.safetensors"),
                device=str(runtime.context.device),
            ),
            strict=True,
        )
        evaluation, elapsed = _evaluation(
            runtime,
            task=task,
            fit_demos=fit_demos,
            held_demo=held_demo,
            cache=cache,
            process_cache=process_cache,
            reference_losses=reference_losses,
        )
        rows[checkpoint.name] = evaluation
        seconds += elapsed
    return rows, seconds


def _train(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    fit_demos: tuple[int, int],
    cache: Mapping[int, FrozenPolicyResponseVideo],
    process_cache: Mapping[int, PolicyResponseProcessOutput],
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    start_step: int,
    stop: int,
    checkpoint_steps: set[int],
    metrics_rows: int,
) -> dict[str, Any]:
    cell = runtime.config["optimization"]["task_local"]
    curve = []
    checkpoints = []
    peak_allocated = 0
    peak_reserved = 0
    started = time.monotonic()
    for step in range(start_step + 1, stop + 1):
        tick = time.monotonic()
        demo = fit_demos[(step - 1) % len(fit_demos)]
        visit_index = (step - 1) % len(runtime.panels[task].panel_a)
        video = cache[demo].to(runtime.context.device)
        process = process_cache[demo]
        batch, panel = functional_panel_batch(
            runtime,
            task_id=task,
            panel_name="a",
            visit_index=visit_index,
            rows=int(cell["functional_rows"]),
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(runtime.context.device)
        functional_loss, allocated, reserved = _functional_backward(
            runtime,
            video=video,
            process=process,
            batch=batch,
            seed=panel.policy_rng_seed,
        )
        gradient_groups = _composer_gradient_norms(runtime)
        gradients = tuple(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        if (
            not gradients
            or not all(bool(torch.isfinite(value).all()) for value in gradients)
            or runtime.writer.composer.task_query.grad is None
        ):
            raise RuntimeError("task-local Composer gradient is invalid")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(cell["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("task-local Composer gradient norm is non-finite")
        optimizer.step()
        scheduler.step()
        peak_allocated = max(peak_allocated, allocated)
        peak_reserved = max(peak_reserved, reserved)
        row = {
            "optimizer_step": step,
            "task": task,
            "video_demo": demo,
            "panel_a_visit": visit_index,
            "functional_loss": functional_loss,
            "carrier_loss": float(panel.flow_loss),
            "benefit_over_carrier": float(panel.flow_loss) - functional_loss,
            "gradient_norm_before_clip": float(gradient_norm),
            "gradient_groups": gradient_groups,
            "next_lr": scheduler.get_last_lr()[0],
            "step_seconds": time.monotonic() - tick,
            "peak_cuda_allocated_bytes": allocated,
            "peak_cuda_reserved_bytes": reserved,
        }
        append_jsonl(runtime.args.output_dir / "metrics.jsonl", row)
        metrics_rows += 1
        if step == 1 or step % 10 == 0 or step == stop:
            curve.append(row)
            print(row, flush=True)
        if runtime.args.mode == "formal" and step in checkpoint_steps:
            checkpoint = save_ecp_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=step,
                stage=TASKLOCAL_STAGE,
                context=runtime.context,
                model=runtime.writer,
                optimizer=optimizer,
                scheduler=scheduler,
                run_contract_schema=TASKLOCAL_RUN_SCHEMA,
                metrics_rows=metrics_rows,
            )
            checkpoints.append(str(checkpoint))
        del video, batch
    return {
        "curve": curve,
        "checkpoints": checkpoints,
        "metrics_rows": metrics_rows,
        "train_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": peak_allocated,
        "max_cuda_reserved_bytes": peak_reserved,
    }


def _resume_cursor(
    runtime: PolicyResponseRuntime,
    *,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    stop: int,
) -> tuple[int, int]:
    if runtime.args.resume is None:
        return 0, 0
    start_step, expected_rows = load_ecp_checkpoint(
        checkpoint=runtime.args.resume,
        stage=TASKLOCAL_STAGE,
        context=runtime.context,
        model=runtime.writer,
        optimizer=optimizer,
        scheduler=scheduler,
        run_contract_schema=TASKLOCAL_RUN_SCHEMA,
    )
    metrics_rows = reconcile_metrics(
        runtime.args.output_dir / "metrics.jsonl",
        start_step,
        expected_rows,
        cursor_key="optimizer_step",
    )
    if not 0 <= start_step <= stop:
        raise ValueError("task-local Composer resume cursor changed")
    return start_step, metrics_rows


def run_task_local(runtime: PolicyResponseRuntime) -> dict[str, Any]:
    if runtime.context.world_size != 1:
        raise ValueError("task-local Composer qualification is one process per GPU")
    task = int(runtime.args.task)
    allowed = tuple(
        map(int, runtime.config["task_split"]["task_local_positive_control"])
    )
    if task not in allowed:
        raise ValueError("task-local Composer task is outside the positive control")
    cell = runtime.config["optimization"]["task_local"]
    total = int(cell["warmup_updates"]) + int(cell["effective_updates"])
    stop = int(
        runtime.args.stop_after_step
        or (int(cell["profile_updates"]) if runtime.args.mode == "profile" else total)
    )
    if (
        not 0 < stop <= total
        or (runtime.args.mode == "formal" and stop != total)
    ):
        raise ValueError("task-local Composer stop step changed")

    started = time.monotonic()
    fit_demos, held_demo = _video_split(runtime, task)
    all_demos = (*fit_demos, held_demo)
    references, reference_path = _reference_losses(
        runtime, task=task, expected_videos=all_demos
    )
    parameters, optimizer, scheduler = _optimizer(runtime)
    contract = build_tasklocal_run_contract(
        runtime,
        schema=TASKLOCAL_RUN_SCHEMA,
        stage=TASKLOCAL_STAGE,
        task=task,
        fit_demos=fit_demos,
        held_demo=held_demo,
        reference_path=reference_path,
        parameters=parameters,
        stop=stop,
    )
    seal_or_validate_tasklocal_run_contract(runtime, contract)
    start_step, metrics_rows = _resume_cursor(
        runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        stop=stop,
    )
    cache, process_cache, capture_records, capture_seconds = _cache_evidence(
        runtime, task=task, videos=all_demos
    )
    checkpoint_steps = {
        int(cell["warmup_updates"]) + int(value)
        for value in cell["checkpoint_effective_updates"]
    }
    training = _train(
        runtime,
        task=task,
        fit_demos=fit_demos,
        cache=cache,
        process_cache=process_cache,
        parameters=parameters,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=start_step,
        stop=stop,
        checkpoint_steps=checkpoint_steps,
        metrics_rows=metrics_rows,
    )
    if runtime.args.mode == "formal":
        evaluations, evaluation_seconds = _checkpoint_evaluations(
            runtime,
            task=task,
            fit_demos=fit_demos,
            held_demo=held_demo,
            cache=cache,
            process_cache=process_cache,
            reference_losses=references,
        )
        evaluation = evaluations[f"macro_{stop:08d}"]
    else:
        evaluation, evaluation_seconds = _evaluation(
            runtime,
            task=task,
            fit_demos=fit_demos,
            held_demo=held_demo,
            cache=cache,
            process_cache=process_cache,
            reference_losses=references,
        )
        evaluations = {"current": evaluation}
    return build_tasklocal_result(
        runtime,
        schema=TASKLOCAL_RUN_SCHEMA,
        task=task,
        fit_demos=fit_demos,
        held_demo=held_demo,
        stop=stop,
        start_step=start_step,
        total=total,
        training=training,
        evaluation=evaluation,
        evaluations=evaluations,
        capture_records=capture_records,
        capture_seconds=capture_seconds,
        cache=cache,
        reference_path=reference_path,
        parameters=parameters,
        evaluation_seconds=evaluation_seconds,
        started=started,
    )
