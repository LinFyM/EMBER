"""Focused internal audit for the v6 task-relative-flow cold-start trajectory."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_
from ember.pi05_eval_contract import git_state
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_setup import load_stats
from ember.writer.as_contract import REPO_ROOT
from ember.writer.adapter_analysis_metrics import (
    adapter_geometry,
    capture_policy_dictionary_mixing,
    distribution,
    effective_metrics,
    effective_variance,
    lora_pairs,
    policy_dictionary_batch_records,
    policy_dictionary_checkpoint_summary,
    state_row,
    tensor_metrics,
)
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority, _camera
from ember.writer.inference import writer_generation_seed, writer_shuffled_frame_permutation
from ember.writer.model import WriterModelError
from ember.writer.validation import _build_models


RUN_SCHEMA = "ember_pi05_v6_relative_flow_coldstart_internal_audit_run_v1"
RESULT_SCHEMA = "ember_pi05_v6_relative_flow_coldstart_internal_audit_v1"
DEMO_INDICES = (0, 1, 2, 3, 4)
ACTION_PANEL_CONDITIONS = ("demo_0", "demo_1", "reversed_0", "shuffled_0")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _distributed_context() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise WriterModelError("invalid cold-start analysis process topology")
    if not torch.cuda.is_available():
        raise WriterModelError("cold-start analysis requires CUDA")
    torch.cuda.set_device(local_rank)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _checkpoint_records(
    training_run: Path,
    training: Mapping[str, Any],
    checkpoints: Sequence[Path],
) -> list[dict[str, Any]]:
    contract_sha256 = canonical_hash(training)
    allowed = set(int(value) for value in training["runtime"]["checkpoint_steps"])
    records = []
    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve()
        if checkpoint.parent.parent != training_run:
            raise WriterModelError("analysis checkpoint crossed its training run")
        manifest = read_json(checkpoint / "checkpoint_manifest.json")
        cursor = int(manifest.get("consumed", {}).get("next_step", -1))
        writer_file = manifest.get("files", {}).get("writer.safetensors", {})
        if (
            manifest.get("contract_sha256") != contract_sha256
            or cursor not in allowed
            or checkpoint.name != f"step_{cursor:08d}"
            or not (checkpoint / "writer.safetensors").is_file()
            or int(writer_file.get("bytes", -1))
            != (checkpoint / "writer.safetensors").stat().st_size
        ):
            raise WriterModelError("analysis checkpoint seal changed")
        records.append(
            {
                "cursor": cursor,
                "path": str(checkpoint),
                "checkpoint_manifest_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
                "writer_state_sha256": writer_file["sha256"],
            }
        )
    records.sort(key=lambda row: int(row["cursor"]))
    if len(records) != len({int(row["cursor"]) for row in records}):
        raise WriterModelError("analysis checkpoints are duplicated")
    return records


def _task_rows(
    training: Mapping[str, Any], data_root: Path
) -> list[dict[str, Any]]:
    manifest = read_json(
        REPO_ROOT / str(training["authorities"]["target_data_manifest"]["path"])
    )
    train_ids = set(int(value) for value in manifest["summary"]["roles"]["train"])
    rows = []
    for row in manifest["tasks"]:
        task_id = int(row["global_task_id"])
        if task_id not in train_ids:
            continue
        path = (data_root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise WriterModelError("analysis task data path changed")
        rows.append(
            {
                "global_task_id": task_id,
                "suite": str(row["problem_folder"]),
                "task_id": int(row["task_id"]),
                "language": str(row["language"]),
                "path": str(path),
                "bytes": int(row["hdf5"]["bytes"]),
            }
        )
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != 24:
        raise WriterModelError("cold-start analysis did not resolve 24 train tasks")
    return rows


def _action_panel(tasks: Sequence[Mapping[str, Any]]) -> set[int]:
    by_suite: dict[str, list[int]] = {}
    for task in tasks:
        by_suite.setdefault(str(task["suite"]), []).append(
            int(task["global_task_id"])
        )
    panel = set()
    for values in by_suite.values():
        values.sort()
        panel.update((values[0], values[-1]))
    if len(by_suite) != 4 or len(panel) != 8:
        raise WriterModelError("fixed-action panel lost its two-per-suite topology")
    return panel


def _ownership(
    tasks: Sequence[Mapping[str, Any]], panel: set[int], world_size: int
) -> list[list[int]]:
    assignments = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    ordered = sorted(
        (int(task["global_task_id"]) for task in tasks),
        key=lambda task_id: (task_id not in panel, task_id),
    )
    for task_id in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[rank].append(task_id)
        loads[rank] += 5 if task_id in panel else 1
    return assignments


def _publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    training: Mapping[str, Any],
) -> None:
    if context.rank != 0:
        deadline = time.monotonic() + 300
        while not (args.output_dir / "run_contract.json").is_file():
            if time.monotonic() >= deadline:
                raise WriterModelError("analysis contract publication timed out")
            time.sleep(0.25)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise WriterModelError("analysis output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (
        state["dirty_paths"] or state["commit"] != state["origin_main"]
    ):
        raise WriterModelError("formal internal analysis requires pushed clean code")
    tasks = _task_rows(training, args.data_root)
    panel = _action_panel(tasks)
    records = _checkpoint_records(args.training_run, training, args.checkpoints)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != context.world_size:
        raise WriterModelError("visible GPU topology differs from world size")
    write_json_atomic(
        args.output_dir / "run_contract.json",
        {
            "schema_version": RUN_SCHEMA,
            "mode": args.mode,
            "host": socket.gethostname(),
            "command": list(os.sys.argv),
            "git": state,
            "training_run": {
                "path": str(args.training_run),
                "contract_sha256": canonical_hash(training),
            },
            "checkpoints": records,
            "source": training["source"],
            "tokenizer_path": str(args.tokenizer_path),
            "data_root": str(args.data_root),
            "world_size": context.world_size,
            "visible_gpu_ids": visible,
            "tasks": tasks,
            "action_panel_global_task_ids": sorted(panel),
            "ownership": _ownership(tasks, panel, context.world_size),
            "conditions": {
                "same_task_demo_indices": list(DEMO_INDICES),
                "fixed_action": list(ACTION_PANEL_CONDITIONS),
                "fixed_action_query": (
                    "task demo0 frame0 observation and state; no action dataset read"
                ),
                "policy_noise": "fixed per global task across all checkpoints and conditions",
            },
            "information_wall": {
                "writer_inputs": "task language plus exactly one action-hidden teacher video",
                "training_actions_read": 0,
                "validation_or_test_data_read": 0,
                "fixed_action_probe_target_actions_read": 0,
            },
        },
    )


def _condition_batch(
    task: WriterTaskAuthority,
    suite: str,
    suite_task_id: int,
    store: RawTeacherVideoStore,
    tokenizer: Pi05TeacherPrefixTokenizer,
    device: torch.device,
    include_action_conditions: bool,
    generation_seed: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    frames: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    names: list[str] = []
    videos = {demo: store.load(task.task_id, demo) for demo in DEMO_INDICES}

    def append(name: str, value: Any, order: torch.Tensor | None = None) -> None:
        selected = torch.from_numpy(value.frames)
        if order is not None:
            selected = selected.index_select(0, order)
        frames.append(selected)
        indices.append(torch.from_numpy(value.frame_indices))
        names.append(name)

    for demo in DEMO_INDICES:
        append(f"demo_{demo}", videos[demo])
    if include_action_conditions:
        count = int(videos[0].frames.shape[0])
        append("reversed_0", videos[0], torch.arange(count - 1, -1, -1))
        seed = writer_generation_seed(
            generation_seed,
            suite,
            suite_task_id,
            0,
            stream="frame_order",
        )
        append(
            "shuffled_0",
            videos[0],
            writer_shuffled_frame_permutation(count, seed, keep_first=False),
        )
    offsets = [0]
    for value in frames:
        offsets.append(offsets[-1] + value.shape[0])
    tokens, masks, spans = tokenizer([task.language] * len(names))
    return (
        {
            "frames": torch.cat(frames).to(device, non_blocking=True),
            "indices": torch.cat(indices).to(device, non_blocking=True),
            "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
            "tokens": tokens,
            "masks": masks,
            "spans": spans,
        },
        names,
    )


def _fixed_query(
    task: WriterTaskAuthority,
    processor: Pi05LiberoProcessor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with h5py.File(task.path, "r") as handle:
        observations = handle["data/demo_0/obs"]
        base = np.asarray(observations["agentview_rgb"][0])
        wrist = np.asarray(observations["eye_in_hand_rgb"][0])
        state = np.concatenate(
            (
                np.asarray(observations["ee_states"][0], dtype=np.float32),
                np.asarray(observations["gripper_states"][0], dtype=np.float32),
            )
        )
    base_tensor = torch.from_numpy(_camera(base))[None].to(device, dtype=torch.float32).div_(255)
    wrist_tensor = torch.from_numpy(_camera(wrist))[None].to(device, dtype=torch.float32).div_(255)
    state_tensor = torch.from_numpy(state)[None].to(device)
    tokens, masks = processor._tokenize_prompts(state_tensor, [task.language])
    return {
        "observation.images.base_0_rgb": base_tensor,
        "observation.images.left_wrist_0_rgb": wrist_tensor,
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": masks,
    }


@contextmanager
def _policy_attention_state(policy: torch.nn.Module) -> Any:
    bridge = policy.model.paligemma_with_expert
    language = bridge.paligemma.model.language_model.config
    expert = bridge.gemma_expert.model.config
    before = (language._attn_implementation, expert._attn_implementation)
    try:
        yield
    finally:
        language._attn_implementation, expert._attn_implementation = before


def _fixed_action(
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
    prepared: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    lora: Any,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    copy_task_lora_state_(policy, state, lora)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1,
        int(policy.model.config.chunk_size),
        int(policy.model.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with (
        torch.inference_mode(),
        _policy_attention_state(policy),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        value = policy.predict_action_chunk(dict(prepared), noise=noise, num_steps=10)
    return processor.unnormalize_action(value).detach().cpu()


def _local_rows(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    training: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks = {int(row["global_task_id"]): row for row in contract["tasks"]}
    owned = [int(value) for value in contract["ownership"][context.rank]]
    authorities = tuple(
        WriterTaskAuthority(
            task_id=task_id,
            language=str(tasks[task_id]["language"]),
            path=Path(tasks[task_id]["path"]),
            expected_bytes=int(tasks[task_id]["bytes"]),
            expected_sha256=None,
        )
        for task_id in owned
    )
    by_id = {task.task_id: task for task in authorities}
    policy, writer, lora, identity = _build_models(
        training=training,
        source=training["source"],
        context=context,
    )
    source_config = read_json(
        REPO_ROOT / str(training["authorities"]["source_base_config"]["path"])
    )
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    store = RawTeacherVideoStore(
        authorities,
        frame_stride=int(training["writer"]["frame_stride"]),
    )
    panel = set(int(value) for value in contract["action_panel_global_task_ids"])
    pairs = lora_pairs(writer)
    scale = float(lora.alpha) / float(lora.rank)
    fixed_queries = {
        task_id: _fixed_query(by_id[task_id], processor, context.device)
        for task_id in owned
        if task_id in panel
    }
    identity_actions = {
        task_id: _fixed_action(
            policy,
            processor,
            fixed_queries[task_id],
            identity,
            lora,
            202608050000 + task_id,
            context.device,
        )
        for task_id in fixed_queries
    }
    previous: dict[int, tuple[int, dict[str, torch.Tensor]]] = {}
    previous_actions: dict[int, tuple[int, torch.Tensor]] = {}
    rows = []
    try:
        for record in contract["checkpoints"]:
            writer.load_state_dict(
                load_file(
                    str(Path(record["path"]) / "writer.safetensors"),
                    device=str(context.device),
                ),
                strict=True,
            )
            for task_id in owned:
                task_row = tasks[task_id]
                task = by_id[task_id]
                copy_task_lora_state_(policy, identity, lora)
                packed, names = _condition_batch(
                    task,
                    str(task_row["suite"]),
                    int(task_row["task_id"]),
                    store,
                    tokenizer,
                    context.device,
                    task_id in panel,
                    int(training["writer"]["initialization_seed"]),
                )
                with capture_policy_dictionary_mixing(writer) as mixing_capture:
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        batched = writer(
                            packed["frames"],
                            packed["indices"],
                            packed["offsets"],
                            packed["tokens"],
                            packed["masks"],
                            packed["spans"],
                            policy=policy,
                        )
                states = {
                    name: state_row(batched, index)
                    for index, name in enumerate(names)
                }
                policy_dictionary = policy_dictionary_batch_records(
                    writer, mixing_capture, names
                )
                reference = states["demo_0"]
                churn = None
                if task_id in previous:
                    old_cursor, old_state = previous[task_id]
                    churn = {
                        "from_cursor": old_cursor,
                        **effective_metrics(pairs, old_state, reference),
                    }
                previous[task_id] = (int(record["cursor"]), reference)
                action = None
                action_churn = None
                if task_id in panel:
                    actions = {
                        name: _fixed_action(
                            policy,
                            processor,
                            fixed_queries[task_id],
                            states[name],
                            lora,
                            202608050000 + task_id,
                            context.device,
                        )
                        for name in ACTION_PANEL_CONDITIONS
                    }
                    action = {
                        "identity_to_demo_0": tensor_metrics(
                            identity_actions[task_id], actions["demo_0"]
                        ),
                        **{
                            f"demo_0_to_{name}": tensor_metrics(
                                actions["demo_0"], actions[name]
                            )
                            for name in ACTION_PANEL_CONDITIONS[1:]
                        },
                    }
                    if task_id in previous_actions:
                        old_cursor, old_action = previous_actions[task_id]
                        action_churn = {
                            "from_cursor": old_cursor,
                            **tensor_metrics(old_action, actions["demo_0"]),
                        }
                    previous_actions[task_id] = (
                        int(record["cursor"]),
                        actions["demo_0"],
                    )
                rows.append(
                    {
                        "checkpoint_cursor": int(record["cursor"]),
                        "global_task_id": task_id,
                        "suite": str(task_row["suite"]),
                        "suite_task_id": int(task_row["task_id"]),
                        "rank": context.rank,
                        "geometry_demo_0": adapter_geometry(
                            writer, pairs, reference, scale
                        ),
                        "same_task_video_variance": effective_variance(
                            pairs, [states[f"demo_{demo}"] for demo in DEMO_INDICES]
                        ),
                        "policy_dictionary": policy_dictionary,
                        "effective_ba_from_demo_0": {
                            name: effective_metrics(pairs, reference, states[name])
                            for name in names
                            if name != "demo_0"
                        },
                        "fixed_action": action,
                        "fixed_action_checkpoint_churn": action_churn,
                        "checkpoint_churn": churn,
                    }
                )
                print(
                    json.dumps(
                        {
                            "rank": context.rank,
                            "checkpoint": int(record["cursor"]),
                            "task": task_id,
                        }
                    ),
                    flush=True,
                )
    finally:
        copy_task_lora_state_(policy, identity, lora)
        store.close()
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    cursors = sorted({int(row["checkpoint_cursor"]) for row in rows})
    for cursor in cursors:
        selected = [row for row in rows if int(row["checkpoint_cursor"]) == cursor]
        geometry_keys = (
            "effective_lora_norm_scaled",
            "stable_rank_mean",
            "top_singular_energy_mean",
            "rank90_mean",
            "rank99_mean",
        )
        video_names = ("demo_1", "demo_2", "demo_3", "demo_4")
        action_rows = [row for row in selected if row["fixed_action"] is not None]
        result[str(cursor)] = {
            "task_count": len(selected),
            "geometry_demo_0": {
                key: distribution([float(row["geometry_demo_0"][key]) for row in selected])
                for key in geometry_keys
            },
            "q_v_component_top4_energy_fraction": {
                kind: distribution(
                    [
                        float(
                            row["geometry_demo_0"]
                            ["rank_coordinate_geometry_gauge_dependent"]
                            [kind]["top4_coordinate_energy_fraction"]
                        )
                        for row in selected
                    ]
                )
                for kind in ("q", "v")
            },
            "q_v_b_column_cosine": {
                kind: distribution(
                    [
                        float(
                            row["geometry_demo_0"]
                            ["rank_coordinate_geometry_gauge_dependent"]
                            [kind]["mean_absolute_b_column_cosine"]
                        )
                        for row in selected
                    ]
                )
                for kind in ("q", "v")
            },
            "same_task_video_centered_variance_over_sample_energy": distribution(
                [
                    float(
                        row["same_task_video_variance"]
                        ["centered_variance_over_sample_energy"]
                    )
                    for row in selected
                ]
            ),
            "same_task_video_effective_ba_relative_l2": {
                name: distribution(
                    [
                        float(row["effective_ba_from_demo_0"][name]["relative_l2"])
                        for row in selected
                    ]
                )
                for name in video_names
            },
            "action_panel": {
                key: distribution(
                    [float(row["fixed_action"][key]["relative_l2"]) for row in action_rows]
                )
                for key in (
                    "identity_to_demo_0",
                    "demo_0_to_demo_1",
                    "demo_0_to_reversed_0",
                    "demo_0_to_shuffled_0",
                )
            },
            "action_panel_effective_ba_relative_l2": {
                name: distribution(
                    [
                        float(row["effective_ba_from_demo_0"][name]["relative_l2"])
                        for row in action_rows
                    ]
                )
                for name in ("demo_1", "reversed_0", "shuffled_0")
            },
        }
        policy_dictionary = policy_dictionary_checkpoint_summary(selected)
        if policy_dictionary is not None:
            result[str(cursor)]["policy_dictionary"] = policy_dictionary
        churn = [row for row in selected if row["checkpoint_churn"] is not None]
        if churn:
            result[str(cursor)]["checkpoint_churn_effective_ba_relative_l2"] = (
                distribution(
                    [float(row["checkpoint_churn"]["relative_l2"]) for row in churn]
                )
            )
        action_churn = [
            row
            for row in action_rows
            if row["fixed_action_checkpoint_churn"] is not None
        ]
        if action_churn:
            result[str(cursor)]["checkpoint_churn_fixed_action_relative_l2"] = (
                distribution(
                    [
                        float(row["fixed_action_checkpoint_churn"]["relative_l2"])
                        for row in action_churn
                    ]
                )
            )
    return result


def main() -> None:
    args = _arguments()
    for name in ("training_run", "tokenizer_path", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.checkpoints = tuple(path.resolve() for path in args.checkpoints)
    context = _distributed_context()
    training = read_json(args.training_run / "run_contract.json")
    _publish_contract(args, context, training)
    contract = read_json(args.output_dir / "run_contract.json")
    started = time.monotonic()
    rows = _local_rows(args, context, contract, training)
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
    paths = [
        args.output_dir / f"rank_{rank:02d}_rows.json"
        for rank in range(context.world_size)
    ]
    deadline = time.monotonic() + 3600
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            raise WriterModelError("analysis rank sealing timed out")
        time.sleep(1)
    combined = []
    rank_payloads = []
    for path in paths:
        payload = read_json(path)
        rank_payloads.append(payload)
        combined.extend(payload["rows"])
    combined.sort(
        key=lambda row: (int(row["checkpoint_cursor"]), int(row["global_task_id"]))
    )
    expected = len(contract["checkpoints"]) * len(contract["tasks"])
    if len(combined) != expected:
        raise WriterModelError("analysis Cartesian result coverage changed")
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_contract_sha256": canonical_hash(contract),
        "rows": combined,
        "summary": _summary(combined),
        "completion": {
            "rows": len(combined),
            "tasks": len(contract["tasks"]),
            "checkpoints": len(contract["checkpoints"]),
            "world_size": context.world_size,
            "wall_seconds": time.monotonic() - started,
            "max_cuda_reserved_bytes": max(
                int(payload["max_cuda_reserved_bytes"])
                for payload in rank_payloads
            ),
            "target_action_reads": 0,
            "validation_or_test_reads": 0,
        },
    }
    write_json_atomic(args.output_dir / "results.json", result)
    write_json_atomic(
        args.output_dir / "completion.json",
        {**result["completion"], "results_payload_sha256": canonical_hash(result)},
    )
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
