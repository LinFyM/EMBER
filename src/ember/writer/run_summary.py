"""Terminal run accounting for AS-Writer training."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.as_config import writer_stage

if TYPE_CHECKING:
    from ember.writer.training import WriterRuntime


def write_run_summary(runtime: WriterRuntime, *, started: float) -> None:
    stop = runtime.args.stop_after_step
    updates_per_cycle = runtime.sampler.optimizer_updates_per_task_cycle
    summary: dict[str, Any] = {
        "schema_version": "ember_pi05_as_writer_run_summary_v1",
        "contract_reference": runtime.contract_sha256,
        "completed_optimizer_steps": stop,
        "requested_optimizer_steps": runtime.total_steps,
        "stopped_early_for_profile": (
            runtime.args.mode == "profile" and stop < runtime.total_steps
        ),
        "selected_stage_stop": (
            runtime.args.mode == "formal" and stop < runtime.total_steps
        ),
        "metrics_rows": runtime.metrics_rows,
        "wall_seconds": time.monotonic() - started,
        "final_checkpoint": (
            str(runtime.args.output_dir / "checkpoints" / f"step_{stop:08d}")
            if stop in runtime.checkpoint_steps
            else None
        ),
        "train_tasks": len(runtime.task_ids),
        "teacher_action_episodes_available": len(runtime.task_ids) * 50,
        "global_policy_samples": (
            stop
            * runtime.context.world_size
            * runtime.batch_size
            * runtime.tasks_per_rank_per_update
        ),
        "global_writer_video_conditions": (
            stop
            * runtime.context.world_size
            * runtime.videos_per_task_visit
            * runtime.tasks_per_rank_per_update
        ),
        "optimizer_updates_per_task_cycle": updates_per_cycle,
        "completed_task_cycles": stop // updates_per_cycle,
        "scheduler_logical_updates": min(
            stop // updates_per_cycle,
            int(
                runtime.config["conditioning_training"].get(
                    "factor_decoder_train_through_macro", stop
                )
            ),
        ),
        "test_action_reads": 0,
        "test_video_value_reads": 0,
    }
    if writer_stage(runtime.config) == "final":
        summary["validation_action_episodes_available"] = 400
        summary["validation_video_episodes_available"] = 400
    else:
        summary["validation_action_reads"] = 0
    write_json_atomic(runtime.args.output_dir / "run_summary.json", summary)
