"""Zero-gradient Panel-B evaluation for shared Policy-Response Writer runs."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.policy_response_writer.shared import (
    SharedEvidenceCache,
    VideoSplit,
    _capture_missing,
    _gather_flat,
    _materialized_state,
)
from ember.ecp.policy_response_writer.shared_contract import reference_result_path
from ember.ecp.policy_response_writer.training import (
    PolicyResponseRuntime,
    functional_panel_batch,
)
from ember.pi05_source_checkpoint import barrier, read_json
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_value,
)


def _reference_losses(
    runtime: PolicyResponseRuntime,
    *,
    task: int,
    demos: Sequence[int],
) -> tuple[dict[int, float], str | None]:
    path = reference_result_path(runtime, task)
    if not path.is_file():
        return {}, None
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
        or result.get("held_backward_calls") != 0
        or result.get("panel_b_backward_calls") != 0
        or any(not math.isfinite(value) for value in values.values())
    ):
        raise ValueError("shared Writer task-local reference changed")
    if set(values) != set(map(int, demos)):
        return {}, None
    return values, str(path)


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
            runtime.config["optimization"]["shared"]["functional_microbatch"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("shared Writer functional evaluation changed")
    return float(value)


def _evaluate_video(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task: int,
    demo: int,
    reference_loss: float | None,
    visits: int,
) -> dict[str, Any]:
    video = cache.videos[(task, demo)].to(runtime.context.device)
    runtime.writer.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = _materialized_state(runtime, (video,), canonicalize=True)
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
        carrier = float(panel.flow_loss)
        rows.append(
            {
                "visit": visit_index,
                "carrier_loss": carrier,
                "generated_loss": generated,
                "benefit_over_carrier": carrier - generated,
                "free_primal_loss": reference_loss,
            }
        )
        del batch
    carrier = statistics.fmean(value["carrier_loss"] for value in rows)
    generated = statistics.fmean(value["generated_loss"] for value in rows)
    denominator = carrier - reference_loss if reference_loss is not None else None
    recovery = (
        (carrier - generated) / denominator
        if denominator is not None and denominator > 0
        else None
    )
    del state, video
    torch.cuda.empty_cache()
    return {
        "video_demo": demo,
        "visits": rows,
        "carrier_loss": carrier,
        "generated_loss": generated,
        "benefit_over_carrier": carrier - generated,
        "free_primal_loss": reference_loss,
        "functional_recovery": recovery,
    }


def _task_axis(runtime: PolicyResponseRuntime, task: int) -> tuple[str, str]:
    split = runtime.config["task_split"]
    if task in set(map(int, split["gradient_meta"])):
        return "meta", "gradient"
    if task in set(map(int, split["gradient_target"])):
        return "target", "gradient"
    if task in set(map(int, split["true_task_held_meta"])):
        return "meta", "true_task_held"
    if task in set(map(int, split["true_task_held_target"])):
        return "target", "true_task_held"
    raise ValueError("shared Writer evaluation task escaped its split")


def _evaluate_task(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task: int,
    split: VideoSplit,
    visits: int,
) -> dict[str, Any]:
    fit, held = split
    demos = (*fit, held)
    references, reference_path = _reference_losses(runtime, task=task, demos=demos)
    rows = {
        demo: _evaluate_video(
            runtime,
            cache,
            task=task,
            demo=demo,
            reference_loss=references.get(demo),
            visits=visits,
        )
        for demo in demos
    }
    role, task_split = _task_axis(runtime, task)
    return {
        "task": task,
        "role": role,
        "split": task_split,
        "fit_videos": [rows[value] for value in fit],
        "held_video": rows[held],
        "fit_benefit_mean": statistics.fmean(
            rows[value]["benefit_over_carrier"] for value in fit
        ),
        "held_benefit": rows[held]["benefit_over_carrier"],
        "fit_recovery_mean": (
            statistics.fmean(rows[value]["functional_recovery"] for value in fit)
            if all(rows[value]["functional_recovery"] is not None for value in fit)
            else None
        ),
        "held_recovery": rows[held]["functional_recovery"],
        "all_videos_above_carrier": all(
            rows[value]["benefit_over_carrier"] > 0 for value in demos
        ),
        "reference": reference_path,
        "panel_b_backward_calls": 0,
    }


def _mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gradient = [row for row in rows if row["split"] == "gradient"]
    held = [row for row in rows if row["split"] == "true_task_held"]
    return {
        "gradient_tasks": len(gradient),
        "true_task_held_tasks": len(held),
        "gradient_fit_benefit_mean": _mean_present(gradient, "fit_benefit_mean"),
        "gradient_held_benefit_mean": _mean_present(gradient, "held_benefit"),
        "gradient_fit_recovery_mean": _mean_present(gradient, "fit_recovery_mean"),
        "gradient_held_recovery_mean": _mean_present(gradient, "held_recovery"),
        "gradient_all_videos_above_carrier": sum(
            bool(row["all_videos_above_carrier"]) for row in gradient
        ),
        "true_task_held_fit_benefit_mean": _mean_present(held, "fit_benefit_mean"),
        "true_task_held_held_benefit_mean": _mean_present(held, "held_benefit"),
        "true_task_held_all_videos_above_carrier": sum(
            bool(row["all_videos_above_carrier"]) for row in held
        ),
        "panel_b_backward_calls": 0,
        "wrong_video_reads": 0,
    }


def evaluate_checkpoints(
    runtime: PolicyResponseRuntime,
    cache: SharedEvidenceCache,
    *,
    task_owners: Sequence[Sequence[int]],
    video_splits: Mapping[int, VideoSplit],
) -> tuple[dict[str, Any], float]:
    local_tasks = tuple(
        task
        for task in map(int, task_owners[runtime.context.rank])
        if task in video_splits
    )
    for task in local_tasks:
        fit, held = video_splits[task]
        _capture_missing(runtime, cache, task=task, demos=(*fit, held))
    barrier(runtime.context)
    if runtime.args.mode == "formal":
        checkpoints = sorted((runtime.args.output_dir / "checkpoints").glob("macro_*"))
        cell = runtime.config["optimization"]["shared"]
        expected = sorted(
            f"macro_{int(cell['warmup_updates']) + int(step):08d}"
            for step in cell["checkpoint_effective_updates"]
        )
        if not expected or [path.name for path in checkpoints] != expected:
            raise RuntimeError("shared Writer configured checkpoints changed")
        models = tuple((path.name, path / "ecp.safetensors") for path in checkpoints)
        visits = int(runtime.config["optimization"]["shared"]["evaluation_visits"])
    else:
        models = (("current", None),)
        visits = 1
    started = time.monotonic()
    evaluations = {}
    for name, model_path in models:
        if model_path is not None:
            runtime.writer.load_state_dict(
                load_file(str(model_path), device=str(runtime.context.device)), strict=True
            )
        barrier(runtime.context)
        local_rows = [
            _evaluate_task(
                runtime,
                cache,
                task=task,
                split=video_splits[task],
                visits=visits,
            )
            for task in local_tasks
        ]
        rows = sorted(_gather_flat(runtime, local_rows), key=lambda value: value["task"])
        if set(value["task"] for value in rows) != set(video_splits):
            raise RuntimeError("shared Writer evaluation lost a task")
        evaluations[name] = {"tasks": rows, "summary": _summary(rows)}
    return evaluations, time.monotonic() - started
