"""Mechanism-audit authority for the AS125 to Program-Credit transition."""

from __future__ import annotations

import argparse
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import canonical_hash, read_json, write_json_atomic
from ember.writer.as_contract import REPO_ROOT
from ember.writer.coldstart_analysis import (
    ACTION_PANEL_CONDITIONS,
    DEMO_INDICES,
    _action_panel,
    _distributed_context,
    _ownership,
)
from ember.writer.inference import task_video_mapping
from ember.writer.internal_analysis_sealing import finalize_internal_analysis
from ember.writer.model import WriterModelError
from ember.writer.program_credit_analysis_metrics import (
    PROGRAM_SKETCH_WIDTH,
    summary,
)
from ember.writer.program_credit_analysis_runtime import local_rows


RUN_SCHEMA = "ember_pi05_antithetic_program_credit_internal_audit_run_v1"
RESULT_SCHEMA = "ember_pi05_antithetic_program_credit_internal_audit_v1"
CHECKPOINT_LABELS = ("as125", "cycle1")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--as-training-run", type=Path, required=True)
    parser.add_argument("--as-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-training-run", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _task_rows(training: Mapping[str, Any], data_root: Path) -> list[dict[str, Any]]:
    manifest = read_json(
        REPO_ROOT / str(training["authorities"]["target_data_manifest"]["path"])
    )
    train_ids = set(int(value) for value in manifest["summary"]["roles"]["train"])
    rows = []
    for row in manifest["tasks"]:
        global_task_id = int(row["global_task_id"])
        if global_task_id not in train_ids:
            continue
        path = (data_root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise WriterModelError("program-credit analysis task data changed")
        rows.append(
            {
                "global_task_id": global_task_id,
                "suite": str(row["problem_folder"]),
                "task_id": int(row["task_id"]),
                "language": str(row["language"]),
                "path": str(path),
                "bytes": int(row["hdf5"]["bytes"]),
            }
        )
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != 24:
        raise WriterModelError("program-credit analysis lost the train24 panel")
    return rows


def _checkpoint_record(
    *,
    label: str,
    training_run: Path,
    training: Mapping[str, Any],
    checkpoint: Path,
    expected_cursor: int,
    axis: str,
    ordinal: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.parent != training_run:
        raise WriterModelError("program-credit analysis checkpoint crossed its run")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    consumed_key = "next_step" if axis == "optimizer_step" else "next_cycle"
    cursor = int(manifest.get("consumed", {}).get(consumed_key, -1))
    writer_record = manifest.get("files", {}).get("writer.safetensors", {})
    writer_path = checkpoint / "writer.safetensors"
    if (
        label not in CHECKPOINT_LABELS
        or cursor != expected_cursor
        or manifest.get("contract_sha256") != canonical_hash(training)
        or not writer_path.is_file()
        or int(writer_record.get("bytes", -1)) != writer_path.stat().st_size
    ):
        raise WriterModelError("program-credit analysis checkpoint seal changed")
    return {
        "label": label,
        "checkpoint_cursor": ordinal,
        "source_cursor": cursor,
        "source_cursor_axis": axis,
        "path": str(checkpoint),
        "checkpoint_manifest_payload_sha256": manifest["canonical_payload_sha256"],
        "writer_state_sha256": writer_record["sha256"],
    }


def _checkpoint_delta_audit(
    baseline: Path, candidate: Path
) -> dict[str, Any]:
    frozen_prefixes = ("semantic_encoder.", "factor_heads.", "template_")
    trainable_prefixes = (
        "semantic_core.",
        "visual_transition.",
        "procedure.",
        "compiler.",
    )
    totals = {prefix[:-1]: [0.0, 0.0, 0] for prefix in trainable_prefixes}
    frozen_tensors = frozen_mismatches = 0
    with (
        safe_open(str(baseline), framework="pt", device="cpu") as left,
        safe_open(str(candidate), framework="pt", device="cpu") as right,
    ):
        if set(left.keys()) != set(right.keys()):
            raise WriterModelError("Writer state keys changed across program credit")
        for name in left.keys():
            before = left.get_tensor(name)
            after = right.get_tensor(name)
            if name.startswith(frozen_prefixes):
                frozen_tensors += 1
                frozen_mismatches += int(not torch.equal(before, after))
                continue
            matched = next(
                (prefix for prefix in trainable_prefixes if name.startswith(prefix)),
                None,
            )
            if matched is None:
                raise WriterModelError(f"unowned Writer state tensor: {name}")
            key = matched[:-1]
            before = before.double()
            delta = after.double() - before
            totals[key][0] += float(before.square().sum())
            totals[key][1] += float(delta.square().sum())
            totals[key][2] += before.numel()
    if frozen_mismatches:
        raise WriterModelError("frozen Writer tensors changed during program credit")
    return {
        "frozen_tensor_count": frozen_tensors,
        "frozen_tensor_mismatches": frozen_mismatches,
        "trainable_blocks": {
            name: {
                "parameter_count": values[2],
                "relative_l2": math.sqrt(values[1] / max(values[0], 1e-30)),
                "delta_l2": math.sqrt(values[1]),
            }
            for name, values in totals.items()
        },
    }


def _analysis_records(
    args: argparse.Namespace,
    as_training: Mapping[str, Any],
    reward_training: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _checkpoint_record(
            label="as125",
            training_run=args.as_training_run,
            training=as_training,
            checkpoint=args.as_checkpoint,
            expected_cursor=125,
            axis="optimizer_step",
            ordinal=0,
        ),
        _checkpoint_record(
            label="cycle1",
            training_run=args.reward_training_run,
            training=reward_training,
            checkpoint=args.reward_checkpoint,
            expected_cursor=1,
            axis="outer_cycle",
            ordinal=1,
        ),
    ]


def _publish_contract(
    args: argparse.Namespace,
    context: Any,
    as_training: Mapping[str, Any],
    reward_training: Mapping[str, Any],
) -> None:
    if context.rank != 0:
        deadline = time.monotonic() + 300
        while not (args.output_dir / "run_contract.json").is_file():
            if time.monotonic() >= deadline:
                raise WriterModelError("program-credit analysis contract timed out")
            time.sleep(0.25)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise WriterModelError("program-credit analysis output is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (
        state["dirty_paths"] or state["commit"] != state["origin_main"]
    ):
        raise WriterModelError("formal program-credit analysis requires pushed clean code")
    tasks = _task_rows(as_training, args.data_root)
    panel = _action_panel(tasks)
    checkpoints = _analysis_records(args, as_training, reward_training)
    if (
        reward_training.get("coldstart", {}).get("source_writer_state_sha256")
        != checkpoints[0]["writer_state_sha256"]
    ):
        raise WriterModelError("reward run did not cold-start from the analyzed AS125")
    task_keys = [(str(row["suite"]), int(row["task_id"])) for row in tasks]
    wrong_mapping = task_video_mapping(
        task_keys, {key: "train" for key in task_keys}, "cross_suite_wrong"
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != context.world_size:
        raise WriterModelError("analysis visible devices differ from world size")
    write_json_atomic(
        args.output_dir / "run_contract.json",
        {
            "schema_version": RUN_SCHEMA,
            "mode": args.mode,
            "host": socket.gethostname(),
            "command": list(os.sys.argv),
            "git": state,
            "as_training_run": {
                "path": str(args.as_training_run),
                "contract_sha256": canonical_hash(as_training),
            },
            "reward_training_run": {
                "path": str(args.reward_training_run),
                "contract_sha256": canonical_hash(reward_training),
            },
            "checkpoints": checkpoints,
            "checkpoint_delta_audit": _checkpoint_delta_audit(
                args.as_checkpoint / "writer.safetensors",
                args.reward_checkpoint / "writer.safetensors",
            ),
            "source": as_training["source"],
            "tokenizer_path": str(args.tokenizer_path),
            "data_root": str(args.data_root),
            "world_size": context.world_size,
            "visible_gpu_ids": visible,
            "tasks": tasks,
            "action_panel_global_task_ids": sorted(panel),
            "ownership": _ownership(tasks, panel, context.world_size),
            "wrong_video_mapping": list(wrong_mapping),
            "conditions": {
                "same_task_demo_indices": list(DEMO_INDICES),
                "counterfactuals": ["wrong_0", "reversed_0", "shuffled_0"],
                "fixed_action": [
                    "demo_0", "demo_1", "wrong_0", *ACTION_PANEL_CONDITIONS[2:]
                ],
                "program_update_sketch_width": PROGRAM_SKETCH_WIDTH,
            },
            "information_wall": {
                "writer_inputs": "task language plus exactly one action-hidden teacher video",
                "training_actions_read": 0,
                "validation_or_test_data_read": 0,
                "fixed_action_probe_target_actions_read": 0,
            },
        },
    )


def main() -> None:
    args = _arguments()
    for name in (
        "as_training_run",
        "as_checkpoint",
        "reward_training_run",
        "reward_checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    context = _distributed_context()
    as_training = read_json(args.as_training_run / "run_contract.json")
    reward_training = read_json(args.reward_training_run / "run_contract.json")
    _publish_contract(args, context, as_training, reward_training)
    contract = read_json(args.output_dir / "run_contract.json")
    started = time.monotonic()
    rows = local_rows(args, context, contract, as_training)
    write_json_atomic(
        args.output_dir / f"rank_{context.rank:02d}_rows.json",
        {
            "rank": context.rank,
            "rows": rows,
            "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    )
    if context.rank != 0:
        return
    finalize_internal_analysis(
        args.output_dir,
        contract,
        result_schema=RESULT_SCHEMA,
        summary_function=summary,
        started=started,
        wait_for_ranks=True,
    )
