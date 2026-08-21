"""One task-equal optimization loop for functional-code inference."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.functional_adaptation.code_checkpoint import (
    code_writer_rng_state,
    save_code_writer_checkpoint,
)
from ember.functional_adaptation.code_schedule import controlled_process_input
from ember.functional_adaptation.code_training import CodeTrainingRuntime
from ember.functional_adaptation.objectives import functional_code_inference_loss
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.data import pack_teacher_condition


def _sync_gradients(runtime: CodeTrainingRuntime) -> torch.Tensor:
    if runtime.context.world_size > 1:
        for parameter in runtime.trainable:
            if parameter.grad is None:
                raise ValueError("functional-code parameter did not receive a gradient")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    return torch.nn.utils.clip_grad_norm_(
        runtime.trainable,
        float(runtime.settings["optimizer"]["gradient_clip_norm"]),
    )


def process_code_inference_task_loss(
    runtime: Any,
    *,
    task_id: int,
    demos: tuple[int, ...],
    action_demos: tuple[int, ...] | None,
    control_condition: str | None,
    control_seed: int,
    weights: Mapping[str, float],
) -> Any:
    """Run one fixed-code task loss with optional process/action supervision."""

    packed, _ = pack_teacher_condition(
        runtime.video_store,
        task_id=task_id,
        demos=demos,
        language=runtime.language[task_id],
        device=runtime.context.device,
    )
    (
        frames,
        frame_indices,
        video_offsets,
        condition_video_offsets,
        language_tokens,
        language_mask,
        task_span_mask,
    ) = packed
    action_phase_targets = None
    if action_demos is not None:
        if runtime.action_store is None:
            raise ValueError("process-supervised code training lacks action authority")
        action_phase_targets = runtime.action_store.phase_targets(
            task_id=task_id,
            video_demos=demos,
            action_demos=action_demos,
            frame_indices=frame_indices,
            video_offsets=video_offsets,
            device=runtime.context.device,
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features, frame_condition_ids = runtime.writer.encode_features(
            policy=runtime.policy,
            frames=frames,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
            language_tokens=language_tokens,
            language_mask=language_mask,
            task_span_mask=task_span_mask,
        )
        correct = runtime.writer.infer_features(
            features=features,
            frame_condition_ids=frame_condition_ids,
            frame_indices=frame_indices,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )
        control = None
        if control_condition is not None:
            controlled = controlled_process_input(
                features=features,
                frame_condition_ids=frame_condition_ids,
                frame_positions=frame_indices,
                video_offsets=video_offsets,
                condition=control_condition,
                order_seed=control_seed,
            )
            control = runtime.writer.infer_features(
                features=controlled.features,
                frame_condition_ids=controlled.frame_condition_ids,
                frame_indices=controlled.frame_positions,
                video_offsets=controlled.video_offsets,
                condition_video_offsets=condition_video_offsets,
            )
        return functional_code_inference_loss(
            correct,
            runtime.target_codes[task_id],
            weights=weights,
            control=control,
            action_phase_targets=action_phase_targets,
        )


def _checkpoint(runtime: CodeTrainingRuntime, macro: int) -> None:
    local_rng = code_writer_rng_state()
    if runtime.context.world_size > 1:
        states: list[Any] | None = (
            [None] * runtime.context.world_size if runtime.context.is_main else None
        )
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    if runtime.context.is_main:
        save_code_writer_checkpoint(
            output_dir=runtime.args.output_dir,
            macro=macro,
            world_size=runtime.context.world_size,
            writer=runtime.writer,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            metrics_rows=runtime.metrics_rows,
            rank_rng_states=states or (),
        )
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])


def train(runtime: CodeTrainingRuntime) -> None:
    metric_names = (
        "total",
        "combined_code",
        "language_code",
        "video_code",
        "correct_confidence",
        "control_confidence",
        "control_update",
        "same_task_consistency",
        "action_alignment",
    )
    task_count = len(runtime.tasks)
    started = time.monotonic()
    for macro in range(runtime.start_macro, runtime.stop_macro):
        runtime.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(len(metric_names), device=runtime.context.device)
        assignments = runtime.schedule.assignments(macro)
        control = runtime.schedule.control_for_macro(macro)
        for visit in assignments[runtime.context.rank]:
            loss = process_code_inference_task_loss(
                runtime,
                task_id=visit.task_id,
                demos=visit.demos,
                action_demos=visit.action_demos,
                control_condition=control,
                control_seed=int(runtime.settings["task_macro_seed"])
                + macro * 1009
                + visit.task_id * 9173,
                weights=runtime.settings["loss_weights"],
            )
            (loss.total / task_count).backward()
            totals += torch.stack(
                [getattr(loss, name).detach().float() for name in metric_names]
            )
        grad_norm = _sync_gradients(runtime)
        runtime.optimizer.step()
        runtime.scheduler.step()
        if runtime.context.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        completed = macro + 1
        row = {
            "macro": completed,
            "temporal_control": control,
            **{
                name: float((totals[index] / task_count).item())
                for index, name in enumerate(metric_names)
            },
            "gradient_norm_before_clip": float(grad_norm),
            "learning_rate": float(runtime.scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.monotonic() - started,
            "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        if runtime.context.is_main:
            append_jsonl(runtime.metrics_path, row)
            runtime.metrics_rows += 1
            print(json.dumps(row, sort_keys=True), flush=True)
        if runtime.context.world_size > 1:
            rows = torch.tensor(
                runtime.metrics_rows,
                device=runtime.context.device,
                dtype=torch.long,
            )
            dist.broadcast(rows, src=0)
            runtime.metrics_rows = int(rows.item())
        if completed in runtime.checkpoint_macros:
            _checkpoint(runtime, completed)
    if runtime.context.is_main:
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": "ember_functional_code_writer_completion_v1",
                "completed_macro": runtime.stop_macro,
                "metrics_rows": runtime.metrics_rows,
                "elapsed_seconds": time.monotonic() - started,
                "content_hash_policy": "disabled_by_owner",
            },
        )
    runtime.video_store.close()
    runtime.action_store.close()
    if runtime.context.world_size > 1:
        dist.barrier(device_ids=[runtime.context.local_rank])
