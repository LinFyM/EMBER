"""Train-only admission for the bounded terminal-Writer reachability diagnostic.

Execution stays in the existing static LoRA adapter. These fitted variables are
never a Writer checkpoint, a validation candidate, or a future initialization.
Retire this admission when the registered diagnostic has closed; its frozen
evaluation commit and manifests preserve the evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json


PREFIX = "process_pullback_reachability_"
ARMS = frozenset(PREFIX + value for value in ("writer900", "free_q", "free_lora"))
TASK_SCHEMA = "train_pullback_reachability_task_v1"
ADAPTER_SCHEMA = "ember_process_pullback_reachability_adapter_v1"
STATES = [32, 33, 34, 35]


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _read_record(value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(value["path"])
    if dict(value) != _record(path):
        raise ValueError("diagnostic authority file changed")
    return read_json(path)


def _panel_valid(panel, *, support: bool) -> bool:
    pool, count, realizations = (list(range(17, 42)), 64, 2) if support else (list(range(42, 46)), 128, 1)
    demos, frames = panel.get("action_demos", ()), panel.get("action_frames", ())
    return all((
        panel.get("pool") == pool, panel.get("count") == count,
        len(demos) == len(frames) == count, set(demos) <= set(pool),
        all(type(frame) is int and frame >= 0 for frame in frames),
        panel.get("action_start_indices") == [frame + 1 for frame in frames],
        panel.get("action_chunk_size") == 50, panel.get("action_start_offset") == 1,
        panel.get("flow_realizations_per_position") == realizations,
        panel.get("prediction_count") == count * realizations,
        panel.get("gradients") is support,
        panel.get("full10_noise_independent_of_fm") is True,
    ))


def bank_provenance_valid(manifest: Mapping[str, Any]) -> bool:
    """Require the complete fixed train24 and the declared support-only fit."""
    from ember.writer.materialization import frozen_authority, inspect_writer_checkpoint

    try:
        run = _read_record(manifest["diagnostic_run"])
        root = Path(manifest["asset_root"])
        data = read_json(root / "configs/pi05_target_data_v1/manifest.json")
        tasks = {row["global_task_id"]: row for row in data["tasks"] if row["split_role"] == "train"}
        rows = manifest["tasks"]
        _, checkpoint = inspect_writer_checkpoint(Path(run["writer_checkpoint"]["path"]))
        wall = manifest["information_wall"]
        if not all((
            manifest["arm"] in ARMS, manifest["evaluation_role"] == "development_train",
            manifest.get("diagnostic_only") is True, manifest.get("checkpoint_selection") is False,
            manifest.get("future_initialization_allowed") is False,
            manifest.get("init_state_ids") == STATES, run.get("schema") == TASK_SCHEMA,
            run.get("smoke") is False, run.get("teacher_demo") == 16,
            run.get("support_pool") == list(range(17, 42)), run.get("heldout_pool") == list(range(42, 46)),
            run.get("support_positions") == 64, run.get("support_fm_realizations") == 2,
            run.get("heldout_positions") == 128, run.get("heldout_fm_realizations") == 1,
            run.get("source") == manifest["source"], run.get("writer_checkpoint") == checkpoint,
            checkpoint["macro"] == 900, manifest.get("writer_checkpoint") == checkpoint,
            manifest.get("shared_run_contract") == run, frozen_authority(run["diagnostic_git"]),
            manifest.get("training_commit") == run["diagnostic_git"]["commit"],
            len(rows) == len(tasks) == 24,
            [row["global_task_id"] for row in rows] == run["tasks"],
            set(run["tasks"]) == set(tasks),
            all(wall.get(key) == 0 for key in ("validation_action_or_reward_reads", "test_action_or_reward_reads")),
            wall.get("heldout_gradients") is False, wall.get("source_trainable") == 0,
            wall.get("writer_meta_trainable") == 0, wall.get("diagnostic_fit_only") is True,
        )):
            return False
        expected_optimizer = {"lr": 1., "max_iter": 32, "max_eval": 48, "history_size": 10,
                              "tolerance_grad": 1e-7, "tolerance_change": 1e-9, "line_search_fn": "strong_wolfe"}
        return run.get("optimizer") == expected_optimizer and all(
            all(row.get(key) == tasks[row["global_task_id"]][key] for key in ("suite", "task_id", "language"))
            and row.get("allowed_init_state_ids") == STATES for row in rows)
    except (KeyError, TypeError, ValueError, OSError):
        return False


def checkpoint_matches(arm, checkpoint, row, key, manifest) -> bool:
    """Keep fitted-arm provenance distinct from a generated Writer adapter."""
    from ember.writer.materialization import frozen_authority

    try:
        task = _read_record(row["diagnostic_task_manifest"])
        name = arm.removeprefix(PREFIX)
        output = task["arms"][name]
        source_adapter = output["adapter"]
        source_path = Path(source_adapter["path"])
        teacher = task["teacher_videos"]
        raw = teacher[0]["raw_frame_count"]
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        return all((
            arm in ARMS, checkpoint.get("schema_version") == ADAPTER_SCHEMA,
            checkpoint.get("arm") == arm,
            checkpoint.get("global_task_id") == row["global_task_id"] == task["global_task_id"],
            checkpoint.get("diagnostic_task_manifest") == row["diagnostic_task_manifest"],
            checkpoint.get("original_adapter") == source_adapter == _record(source_path),
            Path(row["adapter_path"]).samefile(source_path),
            Path(row["adapter_path"]).stat().st_size == source_adapter["bytes"],
            checkpoint.get("single_complete_rank16") is True,
            task.get("schema") == TASK_SCHEMA, task.get("smoke") is False,
            task.get("complete") is True, task.get("all_arms_usable") is True,
            task.get("offline_diagnostic_only") is True, task.get("future_initialization_allowed") is False,
            task.get("checkpoint_selection") is False, task.get("source_trainable") == 0,
            task.get("writer_meta_trainable") == 0, task.get("heldout_gradients") is False,
            task.get("source") == manifest["source"], task.get("writer_checkpoint") == manifest["writer_checkpoint"],
            task.get("diagnostic_git") == manifest["shared_run_contract"]["diagnostic_git"],
            frozen_authority(task["diagnostic_git"]),
            (task["suite"], task["task_id"]) == key, task["language"] == row["language"],
            task.get("teacher_demo_indices") == [16], len(teacher) == 1,
            teacher[0].get("demo_index") == 16, teacher[0].get("frame_indices") == indices,
            teacher[0].get("sampled_frame_count") == len(indices), teacher[0].get("camera_view") == "dual",
            _panel_valid(task["support"], support=True), _panel_valid(task["heldout"], support=False),
            output.get("arm") == name, output.get("single_complete_rank16") is True,
            output.get("rank") == output.get("alpha") == 16, output.get("target_count") == 38,
            output.get("factor_count") == 76, output.get("source_and_writer_frozen") is True,
            output.get("support_only_gradients") is True,
            output.get("only_q_optimized") is (name == "free_q"),
            output.get("only_lora_factors_optimized") is (name == "free_lora"),
            name != "writer900" or output.get("optimizer_updates") == 0,
        ))
    except (KeyError, TypeError, ValueError, OSError, IndexError):
        return False
