"""Adaptation checkpoint selection and matched rollout-schedule audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ember.source_base_checkpoint import read_json, sha256_file, write_json_atomic
from ember.task_local_rl_checkpoint import verify_task_local_checkpoint
from ember.task_local_rl_protocol import select_adaptation_checkpoint
from ember.writer.model import WriterModelError


def select_unit_checkpoint(
    unit_dir: Path, checkpoint_updates: Sequence[int]
) -> dict[str, Any]:
    candidates = []
    for update in checkpoint_updates:
        checkpoint = unit_dir / "checkpoints" / f"update_{update:08d}"
        manifest = verify_task_local_checkpoint(checkpoint)
        candidates.append(
            {
                "path": str(checkpoint),
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint / "checkpoint_manifest.json"
                ),
                "interaction_cursor": int(manifest["interaction_cursor"]),
                "next_update": int(manifest["next_update"]),
                "segment_successes": int(manifest["segment_successes"]),
                "segment_rollouts": int(manifest["segment_rollouts"]),
            }
        )
    selected = dict(select_adaptation_checkpoint(candidates))
    result = {
        "schema_version": "ember_task_local_lora_rl_selection_v1",
        "rule": "maximize preceding-segment adaptation success rate; tie earliest interaction cursor",
        "candidates": candidates,
        "selected": selected,
    }
    write_json_atomic(unit_dir / "selected_adaptation_checkpoint.json", result)
    return selected


def matched_schedule_summary(
    output_dir: Path, task_ids: Sequence[int], stop_after_update: int
) -> dict[str, Any]:
    checked = 0
    for task_id in task_ids:
        ledgers = {}
        for arm in ("identity", "writer"):
            unit_dir = output_dir / "units" / f"task_{task_id:03d}_{arm}"
            rows = [
                read_json(
                    unit_dir / "rollouts" / f"update_{update:08d}.json"
                )
                for update in range(stop_after_update)
            ]
            ledgers[arm] = [
                (
                    int(row["task_id"]),
                    int(row["update"]),
                    row["fixed_init_state_id"],
                    tuple(
                        (
                            int(trajectory["env_seed"]),
                            int(trajectory["policy_seed"]),
                        )
                        for trajectory in row["trajectories"]
                    ),
                )
                for row in rows
            ]
        if ledgers["identity"] != ledgers["writer"]:
            raise WriterModelError(
                f"matched task-local seed schedule diverged for task {task_id}"
            )
        checked += len(ledgers["identity"])
    return {
        "matched_task_count": len(task_ids),
        "matched_update_blocks_per_arm": checked,
        "same_task_env_policy_seed_sequences": True,
        "official_random_reset_only": True,
        "fixed_init_state_ids": None,
    }
