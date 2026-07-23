"""Sealed task-balanced functional-loss panel for PI05 development selection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file
from ember.writer.model import WriterModelError


PANEL_SCHEMA = "ember_pi05_validation_functional_loss_panel_v1"
PANEL_MANIFEST_SCHEMA = "ember_pi05_validation_functional_loss_manifest_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_validation_loss_panel(path: Path) -> dict[str, Any]:
    """Load the pre-action-read panel authority and fail closed on drift."""

    path = path.resolve()
    config = read_json(path)
    checksum = path.with_suffix(".sha256")
    expected_line = f"{sha256_file(path)}  {path.name}"
    if (
        config.get("schema_version") != PANEL_SCHEMA
        or config.get("status") != "sealed_before_validation_action_value_reads"
        or config.get("role") != "validation"
        or not checksum.is_file()
        or checksum.read_text(encoding="utf-8").strip() != expected_line
    ):
        raise WriterModelError("validation functional-loss panel is not sealed")
    authorities = config.get("authorities", {})
    if set(authorities) != {"target_data_manifest", "evaluation_config"}:
        raise WriterModelError("validation functional-loss authorities changed")
    for name, record in authorities.items():
        artifact = REPO_ROOT / str(record.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise WriterModelError(f"validation loss authority changed: {name}")
    wall = config.get("information_wall", {})
    if (
        wall.get("validation_action_gradient") is not False
        or wall.get("validation_actions_visible_to_writer") is not False
        or wall.get("validation_actions_visible_to_training_optimizer") is not False
        or int(wall.get("test_action_reads", -1)) != 0
        or int(wall.get("test_video_value_reads", -1)) != 0
    ):
        raise WriterModelError("validation loss information wall changed")
    sampling = config.get("sampling", {})
    expected = {
        "tasks": 8,
        "episodes_per_task": 50,
        "video_groups_per_task": 8,
        "action_queries_per_video_group": 8,
        "action_chunk_size": 50,
        "task_aggregation": "mean within each task then equal mean across eight tasks",
    }
    if (
        int(sampling.get("seed", -1)) < 0
        or any(sampling.get(key) != value for key, value in expected.items())
        or config.get("checkpoint_selection", {}).get("conflict_rule")
        != "closed-loop success overrides functional validation loss"
    ):
        raise WriterModelError("validation loss sampling or selection rule changed")
    target = read_json(
        REPO_ROOT / str(authorities["target_data_manifest"]["path"])
    )
    validation = target.get("summary", {}).get("roles", {}).get("validation", [])
    if len(validation) != 8 or len(set(map(int, validation))) != 8:
        raise WriterModelError("validation loss panel is not the sealed eight tasks")
    return config


def panel_task_ids(config: Mapping[str, Any]) -> tuple[int, ...]:
    target = read_json(
        REPO_ROOT
        / str(config["authorities"]["target_data_manifest"]["path"])
    )
    return tuple(
        sorted(int(value) for value in target["summary"]["roles"]["validation"])
    )


def _policy_seed(seed: int, task_id: int, group: int) -> int:
    payload = json.dumps(
        [PANEL_SCHEMA, seed, task_id, group], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def build_validation_loss_manifest(
    dataset: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive exact video/action identities without inspecting action values."""

    sampling = config["sampling"]
    seed = int(sampling["seed"])
    groups = int(sampling["video_groups_per_task"])
    queries = int(sampling["action_queries_per_video_group"])
    task_ids = panel_task_ids(config)
    frame_index = dataset.frame_index
    rows: list[dict[str, int]] = []
    for task_id in task_ids:
        episodes = dataset.task_episode_rows.get(task_id, {})
        demos = tuple(sorted(int(value) for value in episodes))
        if demos != tuple(range(int(sampling["episodes_per_task"]))):
            raise WriterModelError("validation task episode authority changed")
        videos = np.random.default_rng(
            np.random.SeedSequence([seed, task_id, 0x71D])
        ).permutation(demos)[:groups]
        for group, video_demo in enumerate(videos):
            candidates = tuple(value for value in demos if value != int(video_demo))
            action_demos = np.random.default_rng(
                np.random.SeedSequence([seed, task_id, group, 0xAC7])
            ).permutation(candidates)[:queries]
            group_seed = _policy_seed(seed, task_id, group)
            for query, action_demo in enumerate(action_demos):
                episode_rows = episodes[int(action_demo)]
                offset = int(
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, task_id, group, query, int(action_demo), 0xF4A]
                        )
                    ).integers(len(episode_rows))
                )
                flat = int(episode_rows[offset])
                row_task, row_demo, frame = frame_index[flat]
                if row_task != task_id or row_demo != int(action_demo):
                    raise WriterModelError("validation loss query crossed task authority")
                rows.append(
                    {
                        "ordinal": len(rows),
                        "global_task_id": task_id,
                        "video_group": group,
                        "teacher_demo_index": int(video_demo),
                        "query_ordinal": query,
                        "action_demo_index": int(action_demo),
                        "action_frame_index": int(frame),
                        "dataset_row_index": flat,
                        "policy_noise_seed": group_seed,
                    }
                )
    expected_rows = len(task_ids) * groups * queries
    if (
        len(rows) != expected_rows
        or any(row["teacher_demo_index"] == row["action_demo_index"] for row in rows)
    ):
        raise WriterModelError("validation functional-loss panel is incomplete")
    manifest = {
        "schema_version": PANEL_MANIFEST_SCHEMA,
        "panel_config_sha256": canonical_hash(dict(config)),
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "rows_per_task": groups * queries,
        "row_count": len(rows),
        "rows": rows,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    return manifest


def summarize_validation_losses(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_checkpoint: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        loss = float(row.get("loss", float("nan")))
        if not math.isfinite(loss):
            raise WriterModelError("validation functional loss is non-finite")
        by_checkpoint.setdefault(int(row["checkpoint_cursor"]), []).append(row)
    result: list[dict[str, Any]] = []
    for cursor, checkpoint_rows in sorted(by_checkpoint.items()):
        per_task: dict[str, dict[str, Any]] = {}
        for task_id in sorted({int(row["global_task_id"]) for row in checkpoint_rows}):
            values = np.asarray(
                [
                    float(row["loss"])
                    for row in checkpoint_rows
                    if int(row["global_task_id"]) == task_id
                ],
                dtype=np.float64,
            )
            per_task[str(task_id)] = {
                "rows": int(values.size),
                "mean_loss": float(values.mean()),
                "std_loss": float(values.std(ddof=0)),
            }
        task_means = np.asarray(
            [record["mean_loss"] for record in per_task.values()],
            dtype=np.float64,
        )
        result.append(
            {
                "checkpoint_cursor": cursor,
                "rows": len(checkpoint_rows),
                "tasks": len(per_task),
                "task_balanced_mean_loss": float(task_means.mean()),
                "task_mean_std": float(task_means.std(ddof=0)),
                "per_task": per_task,
            }
        )
    return {
        "schema_version": "ember_pi05_validation_functional_loss_summary_v1",
        "checkpoints": result,
    }
