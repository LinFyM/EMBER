"""Causal parameter-hybrid audit for the AS125 to progress-credit cycle2 update."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from ember.writer.adapter_analysis_metrics import (
    distribution,
    effective_delta_metrics,
    effective_metrics,
    effective_variance,
    lora_pairs,
    state_row,
    tensor_delta_metrics,
    tensor_metrics,
)
from ember.writer.as_contract import REPO_ROOT
from ember.writer.coldstart_analysis import (
    DEMO_INDICES,
    _action_panel,
    _condition_batch,
    _distributed_context,
    _fixed_action,
    _fixed_query,
    _ownership,
    _task_rows,
)
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.validation import _build_models


RUN_SCHEMA = "ember_pi05_progress_credit_parameter_hybrid_audit_run_v1"
RESULT_SCHEMA = "ember_pi05_progress_credit_parameter_hybrid_audit_v1"
ARM_NAMES = (
    "as125",
    "cycle2_full",
    "factor_output_only",
    "upstream_composition_only",
)
HYBRID_ARM_NAMES = ARM_NAMES[2:]
CONDITION_NAMES = tuple(f"demo_{index}" for index in DEMO_INDICES) + (
    "reversed_0",
    "shuffled_0",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--as-training-run", type=Path, required=True)
    parser.add_argument("--rl-training-run", type=Path, required=True)
    parser.add_argument("--as-checkpoint", type=Path, required=True)
    parser.add_argument("--rl-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_record(
    checkpoint: Path,
    training: Mapping[str, Any],
    *,
    cursor_key: str,
    expected_cursor: int,
) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    writer_path = checkpoint / "writer.safetensors"
    writer_file = manifest.get("files", {}).get("writer.safetensors", {})
    if (
        manifest.get("contract_sha256") != canonical_hash(training)
        or int(manifest.get("consumed", {}).get(cursor_key, -1)) != expected_cursor
        or not writer_path.is_file()
        or int(writer_file.get("bytes", -1)) != writer_path.stat().st_size
    ):
        raise WriterModelError("parameter-hybrid checkpoint seal changed")
    return {
        "path": str(checkpoint),
        "manifest_payload_sha256": manifest["canonical_payload_sha256"],
        "writer_state_sha256": writer_file["sha256"],
        cursor_key: expected_cursor,
    }


def _validate_authorities(
    args: argparse.Namespace,
    as_training: Mapping[str, Any],
    rl_training: Mapping[str, Any],
    as_record: Mapping[str, Any],
) -> None:
    coldstart = rl_training.get("coldstart", {})
    if (
        Path(str(coldstart.get("source_checkpoint", ""))).resolve()
        != args.as_checkpoint
        or coldstart.get("source_writer_state_sha256")
        != as_record["writer_state_sha256"]
        or coldstart.get("source_run_contract_sha256") != canonical_hash(as_training)
        or as_training["source"]["model_path"] != rl_training["source"]["model_path"]
    ):
        raise WriterModelError("AS125 and progress-credit authorities diverged")
    for name in (
        "lora_contract",
        "source_base_config",
        "target_data_manifest",
        "tokenizer_manifest",
    ):
        if as_training["authorities"][name] != rl_training["authorities"][name]:
            raise WriterModelError("parameter-hybrid authority changed")


def _publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    as_training: Mapping[str, Any],
    rl_training: Mapping[str, Any],
) -> None:
    if context.rank != 0:
        deadline = time.monotonic() + 300
        while not (args.output_dir / "run_contract.json").is_file():
            if time.monotonic() >= deadline:
                raise WriterModelError(
                    "parameter-hybrid contract publication timed out"
                )
            time.sleep(0.25)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise WriterModelError("parameter-hybrid output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(REPO_ROOT)
    if args.mode == "formal" and (
        state["dirty_paths"] or state["commit"] != state["origin_main"]
    ):
        raise WriterModelError(
            "formal parameter-hybrid audit requires pushed clean code"
        )
    as_record = _checkpoint_record(
        args.as_checkpoint,
        as_training,
        cursor_key="next_step",
        expected_cursor=125,
    )
    rl_record = _checkpoint_record(
        args.rl_checkpoint,
        rl_training,
        cursor_key="next_cycle",
        expected_cursor=2,
    )
    _validate_authorities(args, as_training, rl_training, as_record)
    tasks = _task_rows(as_training, args.data_root)
    panel = _action_panel(tasks)
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
            "as_training_run": {
                "path": str(args.as_training_run),
                "contract_sha256": canonical_hash(as_training),
            },
            "rl_training_run": {
                "path": str(args.rl_training_run),
                "contract_sha256": canonical_hash(rl_training),
            },
            "as_checkpoint": as_record,
            "rl_checkpoint": rl_record,
            "source": as_training["source"],
            "tokenizer_path": str(args.tokenizer_path),
            "data_root": str(args.data_root),
            "world_size": context.world_size,
            "visible_gpu_ids": visible,
            "tasks": tasks,
            "action_panel_global_task_ids": sorted(panel),
            "ownership": _ownership(tasks, panel, context.world_size),
            "parameter_arms": {
                "as125": "all Writer parameters from AS125",
                "cycle2_full": "all Writer parameters from progress-credit cycle2",
                "factor_output_only": (
                    "AS125 except factor_heads.*.network.2.weight from cycle2"
                ),
                "upstream_composition_only": (
                    "cycle2 except factor_heads.*.network.2.weight restored from AS125"
                ),
            },
            "conditions": {
                "names": list(CONDITION_NAMES),
                "same_task_demo_indices": list(DEMO_INDICES),
                "fixed_action_query": (
                    "task demo0 frame0 observation and state; no action dataset read"
                ),
                "policy_noise": "fixed per global task across all arms and conditions",
            },
            "information_wall": {
                "writer_inputs": (
                    "task language plus exactly one action-hidden teacher video"
                ),
                "training_actions_read": 0,
                "validation_or_test_data_read": 0,
                "fixed_action_probe_target_actions_read": 0,
            },
        },
    )


def _parameter_arms(
    writer: CompleteLoRAWriter,
    as_checkpoint: Path,
    rl_checkpoint: Path,
    device: torch.device,
) -> tuple[dict[str, Mapping[str, torch.Tensor]], tuple[str, ...]]:
    as_state = load_file(str(as_checkpoint / "writer.safetensors"), device=str(device))
    full_state = load_file(
        str(rl_checkpoint / "writer.safetensors"), device=str(device)
    )
    expected = set(writer.state_dict())
    if set(as_state) != expected or set(full_state) != expected:
        raise WriterModelError("parameter-hybrid Writer state keys changed")
    for name in expected:
        if as_state[name].shape != full_state[name].shape:
            raise WriterModelError("parameter-hybrid Writer state shape changed")
    factor_output_keys = tuple(
        sorted(
            name
            for name in expected
            if name.startswith("factor_heads.") and name.endswith(".network.2.weight")
        )
    )
    if len(factor_output_keys) != len(writer.FACTOR_WIDTHS):
        raise WriterModelError("factor-output basis selector changed")
    factor_set = set(factor_output_keys)
    factor_only = {
        name: full_state[name] if name in factor_set else as_state[name]
        for name in expected
    }
    upstream_only = {
        name: as_state[name] if name in factor_set else full_state[name]
        for name in expected
    }
    return (
        {
            "as125": as_state,
            "cycle2_full": full_state,
            "factor_output_only": factor_only,
            "upstream_composition_only": upstream_only,
        },
        factor_output_keys,
    )


def _generate_states(
    policy: torch.nn.Module,
    writer: CompleteLoRAWriter,
    identity: Mapping[str, torch.Tensor],
    lora: Any,
    packed: Mapping[str, torch.Tensor],
    names: Sequence[str],
    arms: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    result = {}
    for arm_name in ARM_NAMES:
        writer.load_state_dict(arms[arm_name], strict=True)
        copy_task_lora_state_(policy, identity, lora)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
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
        result[arm_name] = {
            name: state_row(batched, index) for index, name in enumerate(names)
        }
    return result


def _task_metrics(
    pairs: Mapping[str, Mapping[str, str]],
    states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    actions: Mapping[str, Mapping[str, torch.Tensor]] | None,
) -> dict[str, Any]:
    updates = {}
    action_updates = None if actions is None else {}
    for condition in CONDITION_NAMES:
        reference = states["as125"][condition]
        target = states["cycle2_full"][condition]
        updates[condition] = {
            "full_from_as": effective_metrics(pairs, reference, target),
            "hybrids": {
                arm: effective_delta_metrics(
                    pairs,
                    reference,
                    target,
                    states[arm][condition],
                )
                for arm in HYBRID_ARM_NAMES
            },
        }
        if actions is not None and action_updates is not None:
            action_updates[condition] = {
                "full_from_as": tensor_metrics(
                    actions["as125"][condition],
                    actions["cycle2_full"][condition],
                ),
                "hybrids": {
                    arm: tensor_delta_metrics(
                        actions["as125"][condition],
                        actions["cycle2_full"][condition],
                        actions[arm][condition],
                    )
                    for arm in HYBRID_ARM_NAMES
                },
            }
    conditioning = {}
    for arm in ARM_NAMES:
        conditioning[arm] = {
            "same_task_video_variance": effective_variance(
                pairs,
                [states[arm][f"demo_{demo}"] for demo in DEMO_INDICES],
            ),
            "effective_ba_from_demo_0": {
                condition: effective_metrics(
                    pairs,
                    states[arm]["demo_0"],
                    states[arm][condition],
                )
                for condition in CONDITION_NAMES[1:]
            },
            "fixed_action_from_demo_0": (
                None
                if actions is None
                else {
                    condition: tensor_metrics(
                        actions[arm]["demo_0"],
                        actions[arm][condition],
                    )
                    for condition in CONDITION_NAMES[1:]
                }
            ),
        }
    return {
        "effective_ba_update": updates,
        "fixed_action_update": action_updates,
        "video_conditioning": conditioning,
    }


def _local_rows(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    as_training: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
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
        training=as_training,
        source=as_training["source"],
        context=context,
    )
    source_config = read_json(
        REPO_ROOT / str(as_training["authorities"]["source_base_config"]["path"])
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
        frame_stride=int(as_training["writer"]["frame_stride"]),
    )
    arms, factor_output_keys = _parameter_arms(
        writer,
        args.as_checkpoint,
        args.rl_checkpoint,
        context.device,
    )
    pairs = lora_pairs(writer)
    panel = set(int(value) for value in contract["action_panel_global_task_ids"])
    fixed_queries = {
        task_id: _fixed_query(by_id[task_id], processor, context.device)
        for task_id in owned
        if task_id in panel
    }
    rows = []
    try:
        for task_id in owned:
            task_row = tasks[task_id]
            packed, names = _condition_batch(
                by_id[task_id],
                str(task_row["suite"]),
                int(task_row["task_id"]),
                store,
                tokenizer,
                context.device,
                True,
                int(as_training["writer"]["initialization_seed"]),
            )
            if tuple(names) != CONDITION_NAMES:
                raise WriterModelError("parameter-hybrid condition panel changed")
            states = _generate_states(
                policy,
                writer,
                identity,
                lora,
                packed,
                names,
                arms,
                context.device,
            )
            actions = None
            if task_id in fixed_queries:
                actions = {
                    arm: {
                        condition: _fixed_action(
                            policy,
                            processor,
                            fixed_queries[task_id],
                            states[arm][condition],
                            lora,
                            202608050000 + task_id,
                            context.device,
                        )
                        for condition in CONDITION_NAMES
                    }
                    for arm in ARM_NAMES
                }
            rows.append(
                {
                    "global_task_id": task_id,
                    "suite": str(task_row["suite"]),
                    "suite_task_id": int(task_row["task_id"]),
                    "rank": context.rank,
                    **_task_metrics(pairs, states, actions),
                }
            )
            print(json.dumps({"rank": context.rank, "task": task_id}), flush=True)
    finally:
        copy_task_lora_state_(policy, identity, lora)
        store.close()
    return rows, factor_output_keys


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    owner: str,
    arm: str,
    condition: str,
) -> dict[str, dict[str, float]]:
    metrics = (
        "candidate_over_target_delta_l2",
        "delta_cosine",
        "residual_over_target_delta_l2",
    )
    selected = [row for row in rows if row[owner] is not None]
    return {
        metric: distribution(
            [float(row[owner][condition]["hybrids"][arm][metric]) for row in selected]
        )
        for metric in metrics
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_count": len(rows),
        "action_panel_task_count": sum(
            row["fixed_action_update"] is not None for row in rows
        ),
        "effective_ba_update": {},
        "fixed_action_update": {},
        "same_task_video_variance_over_sample_energy": {},
        "video_order_effect": {},
    }
    for arm in HYBRID_ARM_NAMES:
        result["effective_ba_update"][arm] = {
            condition: _metric_summary(
                rows,
                owner="effective_ba_update",
                arm=arm,
                condition=condition,
            )
            for condition in CONDITION_NAMES
        }
        result["fixed_action_update"][arm] = {
            condition: _metric_summary(
                rows,
                owner="fixed_action_update",
                arm=arm,
                condition=condition,
            )
            for condition in CONDITION_NAMES
        }
    for arm in ARM_NAMES:
        result["same_task_video_variance_over_sample_energy"][arm] = distribution(
            [
                float(
                    row["video_conditioning"][arm]["same_task_video_variance"][
                        "centered_variance_over_sample_energy"
                    ]
                )
                for row in rows
            ]
        )
        action_rows = [row for row in rows if row["fixed_action_update"] is not None]
        result["video_order_effect"][arm] = {
            "effective_ba_relative_l2": {
                condition: distribution(
                    [
                        float(
                            row["video_conditioning"][arm]["effective_ba_from_demo_0"][
                                condition
                            ]["relative_l2"]
                        )
                        for row in rows
                    ]
                )
                for condition in ("reversed_0", "shuffled_0")
            },
            "fixed_action_relative_l2": {
                condition: distribution(
                    [
                        float(
                            row["video_conditioning"][arm]["fixed_action_from_demo_0"][
                                condition
                            ]["relative_l2"]
                        )
                        for row in action_rows
                    ]
                )
                for condition in ("reversed_0", "shuffled_0")
            },
        }
    return result


def main() -> None:
    args = _arguments()
    for name in (
        "as_training_run",
        "rl_training_run",
        "as_checkpoint",
        "rl_checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    context = _distributed_context()
    as_training = read_json(args.as_training_run / "run_contract.json")
    rl_training = read_json(args.rl_training_run / "run_contract.json")
    _publish_contract(args, context, as_training, rl_training)
    contract = read_json(args.output_dir / "run_contract.json")
    started = time.monotonic()
    rows, factor_output_keys = _local_rows(args, context, contract, as_training)
    write_json_atomic(
        args.output_dir / f"rank_{context.rank:02d}_rows.json",
        {
            "rank": context.rank,
            "rows": rows,
            "factor_output_keys": list(factor_output_keys),
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
            raise WriterModelError("parameter-hybrid rank sealing timed out")
        time.sleep(1)
    combined = []
    rank_payloads = []
    for path in paths:
        payload = read_json(path)
        rank_payloads.append(payload)
        combined.extend(payload["rows"])
    if any(
        payload["factor_output_keys"] != list(factor_output_keys)
        for payload in rank_payloads
    ):
        raise WriterModelError("parameter-hybrid factor selector differed by rank")
    combined.sort(key=lambda row: int(row["global_task_id"]))
    if (
        len(combined) != len(contract["tasks"])
        or len({int(row["global_task_id"]) for row in combined}) != len(combined)
        or sum(row["fixed_action_update"] is not None for row in combined)
        != len(contract["action_panel_global_task_ids"])
    ):
        raise WriterModelError("parameter-hybrid Cartesian result coverage changed")
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_contract_sha256": canonical_hash(contract),
        "factor_output_keys": list(factor_output_keys),
        "rows": combined,
        "summary": _summary(combined),
        "completion": {
            "rows": len(combined),
            "tasks": len(contract["tasks"]),
            "action_panel_tasks": len(contract["action_panel_global_task_ids"]),
            "conditions": len(CONDITION_NAMES),
            "parameter_arms": len(ARM_NAMES),
            "world_size": context.world_size,
            "wall_seconds": time.monotonic() - started,
            "max_cuda_reserved_bytes": max(
                int(payload["max_cuda_reserved_bytes"]) for payload in rank_payloads
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
