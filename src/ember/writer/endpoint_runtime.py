"""Runtime for the analysis-only PI05 endpoint validation mode."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from torch.utils.data import default_collate

from ember.batched_lora import BatchedLoRAInference
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
    load_run_contract,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    barrier,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_setup import initialize_distributed, load_policy, load_stats
from ember.writer.as_contract import REPO_ROOT
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.endpoint_validation import (
    ENDPOINT_NOISE_SCHEMA,
    ENDPOINT_RUN_SCHEMA,
    INFERENCE_TIMES,
    SEALED_PANEL_PAYLOAD_SHA256,
    EndpointCandidate,
    _load_candidates,
    _require_finite_tensor,
    _verify_lora_entry,
    endpoint_metric_rows,
    endpoint_noise,
    exact_endpoint_actions,
    parse_endpoint_candidate_specs,
)
from ember.writer.endpoint_results import (
    _aggregate,
    _validate_endpoint_output_rows,
)
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import WriterModelError
from ember.writer.validation_panel import (
    PANEL_MANIFEST_SCHEMA,
    PANEL_SCHEMA,
    build_validation_loss_manifest,
    load_validation_loss_panel,
)


@dataclass(frozen=True)
class _EndpointAssets:
    panel: Mapping[str, Any]
    authorities: Any
    source: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    dataset: FunctionalQueryDataset
    manifest: Mapping[str, Any]
    lora: Any
    candidates: tuple[EndpointCandidate, ...]


def _validate_device_scope(
    args: Any,
    context: Any | None = None,
) -> tuple[int, ...]:
    try:
        visible = tuple(
            int(value)
            for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        )
    except (KeyError, ValueError) as error:
        raise WriterModelError(
            "endpoint diagnostic requires explicit physical GPUs"
        ) from error
    if not visible or len(set(visible)) != len(visible) or any(
        device not in {4, 5, 6, 7} for device in visible
    ):
        raise WriterModelError("endpoint diagnostic escaped physical GPUs 4-7")
    if args.mode == "formal" and visible != (4, 5, 6, 7):
        raise WriterModelError(
            "formal endpoint diagnostic requires four ranks on GPUs 4-7"
        )
    if (
        args.mode == "formal"
        and context is not None
        and context.world_size != 4
    ):
        raise WriterModelError("formal endpoint diagnostic requires four ranks")
    return visible


def _teacher_bridge_grid_losses(
    policy: Any,
    batch: Mapping[str, torch.Tensor],
    teacher: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    _require_finite_tensor("teacher action chunk", teacher)
    _require_finite_tensor("noise", noise)
    images, image_masks = policy._preprocess_images(dict(batch))  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action({ACTION: teacher})
    losses = []
    for time_value in INFERENCE_TIMES:
        times = torch.full(
            (teacher.shape[0],),
            time_value,
            dtype=torch.float32,
            device=teacher.device,
        )
        value = policy.model.forward(
            images, image_masks, tokens, masks, actions, noise, times
        )
        if (
            value.ndim != 3
            or value.shape[:2] != teacher.shape[:2]
            or value.shape[-1] < 7
        ):
            raise WriterModelError("endpoint teacher-bridge output shape changed")
        _require_finite_tensor("teacher-bridge flow prediction", value)
        losses.append(value[:, :, :7].float())
    grid = torch.stack(losses).mean(dim=0)
    _require_finite_tensor("teacher-bridge grid", grid)
    return grid


def _predict_and_teacher_bridge(
    policy: Any,
    batch: Mapping[str, torch.Tensor],
    teacher: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the rollout sampler and functional control at their own precision.

    ``pi05_evaluation.rollout_shard`` runs recursive action prediction only
    under inference mode, with no autocast.  The teacher bridge is instead a
    direct functional-forward control and intentionally retains the bf16 CUDA
    autocast used by functional training and validation.
    """

    policy_batch = {key: value for key, value in batch.items() if key != ACTION}
    with torch.inference_mode():
        predicted = exact_endpoint_actions(policy, policy_batch, noise)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            grid = _teacher_bridge_grid_losses(
                policy, policy_batch, teacher, noise
            )
    return predicted, grid


def _endpoint_group(
    policy: Any,
    batched_lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    processor: Pi05LiberoProcessor,
    dataset: FunctionalQueryDataset,
    rows: Sequence[Mapping[str, Any]],
    panel_sha: str,
    device: torch.device,
) -> tuple[dict[str, Any], ...]:
    raw = default_collate(
        [dataset[int(row["dataset_row_index"])] for row in rows]
    )
    prepared = processor.training_batch(raw)
    teacher = prepared[ACTION]
    padding = raw["action_is_pad"].to(device=device, dtype=torch.bool)
    cpu_noise, seeds = endpoint_noise(panel_sha, rows)
    noise = cpu_noise.to(device=device)
    _require_finite_tensor("teacher action chunk", teacher)
    _require_finite_tensor("noise", noise)
    with batched_lora.activate([state] * len(rows)):
        predicted, grid = _predict_and_teacher_bridge(
            policy, prepared, teacher, noise, device
        )
    metrics = endpoint_metric_rows(predicted, teacher, padding, grid)
    return tuple(
        {**metric, "endpoint_noise_seed": seed}
        for metric, seed in zip(metrics, seeds, strict=True)
    )


def _task_authorities(
    panel: Mapping[str, Any],
    data_root: Path,
    *,
    verify_sha256: bool,
) -> tuple[WriterTaskAuthority, ...]:
    target = read_json(
        REPO_ROOT
        / str(panel["authorities"]["target_data_manifest"]["path"])
    )
    task_ids = set(map(int, target["summary"]["roles"]["validation"]))
    tasks = []
    for row in target["tasks"]:
        if int(row["global_task_id"]) not in task_ids:
            continue
        path = (data_root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root):
            raise WriterModelError("endpoint validation HDF5 escaped its data root")
        if (
            path.stat().st_size != int(row["hdf5"]["bytes"])
            or (
                verify_sha256
                and sha256_file(path) != row["hdf5"]["sha256"]
            )
        ):
            raise WriterModelError("endpoint validation HDF5 authority changed")
        tasks.append(
            WriterTaskAuthority(
                int(row["global_task_id"]),
                str(row["language"]),
                path,
                int(row["hdf5"]["bytes"]),
                None,
            )
        )
    if len(tasks) != 8:
        raise WriterModelError("endpoint validation did not resolve eight tasks")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _publish_contract(
    args: Any,
    context: Any,
    panel: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    candidates: Sequence[EndpointCandidate],
    physical_gpu_ids: Sequence[int],
) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise WriterModelError("endpoint output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "panel_manifest.json", manifest)
    write_json_atomic(
        args.output_dir / "run_contract.json",
        {
            "schema_version": ENDPOINT_RUN_SCHEMA,
            "mode": args.mode,
            "host": socket.gethostname(),
            "command": list(os.sys.argv),
            "git": state,
            "panel": {
                "schema_version": PANEL_SCHEMA,
                "path": str(args.panel_config),
                "file_sha256": sha256_file(args.panel_config),
                "manifest_payload_sha256": SEALED_PANEL_PAYLOAD_SHA256,
            },
            "source": source,
            "tokenizer": tokenizer,
            "candidates": [candidate.record() for candidate in candidates],
            "correct400_pairing": {
                "payload": dict(candidates[0].outcome_pairing_payload),
                "canonical_payload_sha256": candidates[0].outcome_pairing_sha256,
            },
            "schedule": {
                "times": list(INFERENCE_TIMES),
                "dt": -0.1,
                "steps": 10,
            },
            "noise": {
                "schema": ENDPOINT_NOISE_SCHEMA,
                "device": "cpu",
                "dtype": "float32",
                "shape": [50, 32],
                "draws_per_row": 1,
            },
            "information_wall": {
                **panel["information_wall"],
                "environment_constructed": False,
                "parameter_gradients_computed": False,
            },
            "world_size": context.world_size,
            "physical_gpu_ids": list(physical_gpu_ids),
            "max_groups_per_task": args.max_groups_per_task,
        },
    )


def _local_rows(
    context: Any,
    candidates: Sequence[EndpointCandidate],
    manifest: Mapping[str, Any],
    lora: Any,
    policy: Any,
    batched_lora: BatchedLoRAInference,
    processor: Pi05LiberoProcessor,
    dataset: FunctionalQueryDataset,
    max_groups_per_task: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in manifest["rows"]:
        grouped.setdefault(
            (int(row["global_task_id"]), int(row["video_group"])), []
        ).append(row)
    keys = sorted(grouped)
    if max_groups_per_task is not None:
        keys = [key for key in keys if key[1] < max_groups_per_task]
    first_contract = load_run_contract(
        candidates[0].evaluation_root / "run_contract.json"
    )
    task_suite = {
        int(row["language_global_task_id"]): (
            str(row["suite"]),
            int(row["task_id"]),
        )
        for row in first_contract["adapter"]["task_video_mapping"]
    }
    work = [(candidate, key) for candidate in candidates for key in keys]
    local = [
        item
        for ordinal, item in enumerate(work)
        if ordinal % context.world_size == context.rank
    ]
    output_rows = []
    for candidate, (task_id, video_group) in local:
        rows = grouped[(task_id, video_group)]
        demo = int(rows[0]["teacher_demo_index"])
        state_lora = _verify_lora_entry(
            candidate.entries[(task_id, demo)], lora, context.device
        )
        tick = time.monotonic()
        metrics = _endpoint_group(
            policy,
            batched_lora,
            state_lora,
            processor,
            dataset,
            rows,
            SEALED_PANEL_PAYLOAD_SHA256,
            context.device,
        )
        elapsed = time.monotonic() - tick
        suite, suite_task_id = task_suite[task_id]
        for row, metric in zip(rows, metrics, strict=True):
            output_rows.append(
                {
                    **row,
                    **metric,
                    "suite": suite,
                    "suite_task_id": suite_task_id,
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "checkpoint_cursor": candidate.checkpoint_cursor,
                    "group_wall_seconds": elapsed,
                    "rank": context.rank,
                }
            )
    return output_rows


def _broadcast_rank_zero_validation(
    validation: list[Any],
    context: Any,
) -> None:
    """Share rank-zero cache validation only when a process group exists."""

    if context.world_size == 1:
        if not context.is_main:
            raise WriterModelError(
                "single-rank endpoint profile has no main process"
            )
        return
    if not dist.is_available() or not dist.is_initialized():
        raise WriterModelError(
            "multi-rank endpoint diagnostic has no process group"
        )
    dist.broadcast_object_list(validation, src=0, device=context.device)


def _load_endpoint_assets(args: Any, context: Any) -> _EndpointAssets:
    panel = load_validation_loss_panel(args.panel_config)
    authorities = load_evaluation_authorities(
        REPO_ROOT / str(panel["authorities"]["evaluation_config"]["path"]),
        REPO_ROOT,
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.source_checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    task_validation: list[Any] = [None]
    tasks: tuple[WriterTaskAuthority, ...] | None = None
    if context.is_main:
        try:
            tasks = _task_authorities(
                panel,
                args.data_root,
                verify_sha256=True,
            )
            task_validation[0] = [
                {
                    "task_id": task.task_id,
                    "path": str(task.path),
                    "bytes": task.expected_bytes,
                }
                for task in tasks
            ]
        except Exception as error:
            task_validation[0] = {"error": repr(error)}
    _broadcast_rank_zero_validation(task_validation, context)
    if isinstance(task_validation[0], Mapping) and task_validation[0].get(
        "error"
    ):
        raise WriterModelError(str(task_validation[0]["error"]))
    if tasks is None:
        tasks = _task_authorities(
            panel,
            args.data_root,
            verify_sha256=False,
        )
        observed = [
            {
                "task_id": task.task_id,
                "path": str(task.path),
                "bytes": task.expected_bytes,
            }
            for task in tasks
        ]
        if observed != task_validation[0]:
            raise WriterModelError(
                "endpoint validation rank-local HDF5 authority changed"
            )
    dataset = FunctionalQueryDataset(
        tasks, demo_indices=range(50), action_chunk_size=50
    )
    try:
        manifest = build_validation_loss_manifest(dataset, panel)
        if (
            manifest.get("schema_version") != PANEL_MANIFEST_SCHEMA
            or manifest.get("canonical_payload_sha256")
            != SEALED_PANEL_PAYLOAD_SHA256
        ):
            raise WriterModelError(
                "endpoint diagnostic panel differs from the predeclared "
                "sealed512 panel"
            )
        lora = load_pi05_lora_contract(
            REPO_ROOT / "configs/pi05_lora_v1.json"
        )
        specs = parse_endpoint_candidate_specs(args.endpoint_candidates)
        validation: list[Any] = [None]
        if context.is_main:
            try:
                validation[0] = [
                    candidate.record()
                    for candidate in _load_candidates(
                        specs,
                        manifest,
                        source,
                        tokenizer,
                        lora,
                        args.data_root,
                        True,
                    )
                ]
            except Exception as error:
                validation[0] = {"error": repr(error)}
        _broadcast_rank_zero_validation(validation, context)
        if isinstance(validation[0], Mapping) and validation[0].get("error"):
            raise WriterModelError(str(validation[0]["error"]))
        candidates = _load_candidates(
            specs,
            manifest,
            source,
            tokenizer,
            lora,
            args.data_root,
            False,
        )
        if [candidate.record() for candidate in candidates] != validation[0]:
            raise WriterModelError(
                "endpoint candidate authority differs across ranks"
            )
        return _EndpointAssets(
            panel=panel,
            authorities=authorities,
            source=source,
            tokenizer=tokenizer,
            dataset=dataset,
            manifest=manifest,
            lora=lora,
            candidates=candidates,
        )
    except BaseException:
        dataset.close()
        raise


def _build_policy_runtime(
    assets: _EndpointAssets,
    args: Any,
    device: torch.device,
) -> tuple[Any, Pi05LiberoProcessor, BatchedLoRAInference]:
    source_config = assets.authorities.source_base_config
    policy = load_policy(
        Path(assets.source["model_path"]),
        source_config,
        device,
    )
    prepare_frozen_writer_policy(policy, assets.lora)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("endpoint diagnostic policy is trainable")
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    return policy, processor, BatchedLoRAInference(policy, assets.lora)


def _finalize_output(
    args: Any,
    context: Any,
    candidates: Sequence[EndpointCandidate],
    manifest: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    combined = []
    for rank in range(context.world_size):
        combined.extend(
            read_json(args.output_dir / f"rank_{rank:02d}_rows.json")["rows"]
        )
    combined.sort(key=lambda row: (row["candidate_id"], int(row["ordinal"])))
    expected_rows = _validate_endpoint_output_rows(
        combined,
        candidates,
        manifest,
        args.max_groups_per_task,
    )
    summary = _aggregate(combined, candidates)
    summary.update(
        {
            "wall_seconds": time.monotonic() - started,
            "row_count": len(combined),
            "validation_action_rows_read": expected_rows,
            "test_action_reads": 0,
            "environment_constructed": False,
            "parameter_gradients_computed": False,
        }
    )
    write_json_atomic(args.output_dir / "rows.json", {"rows": combined})
    write_json_atomic(args.output_dir / "summary.json", summary)
    return summary


def evaluate_endpoint(args: Any) -> None:
    physical_gpu_ids = _validate_device_scope(args)
    context = initialize_distributed(require_numa=args.mode == "formal")
    assets: _EndpointAssets | None = None
    batched_lora: BatchedLoRAInference | None = None
    try:
        _validate_device_scope(args, context)
        assets = _load_endpoint_assets(args, context)
        policy, processor, batched_lora = _build_policy_runtime(
            assets, args, context.device
        )
        state = git_state(REPO_ROOT)
        if args.mode == "formal" and (
            state["dirty_paths"] or state["commit"] != state["origin_main"]
        ):
            raise WriterModelError(
                "formal endpoint validation requires pushed clean code"
            )
        if context.is_main:
            _publish_contract(
                args,
                context,
                assets.panel,
                assets.manifest,
                state,
                assets.source,
                assets.tokenizer,
                assets.candidates,
                physical_gpu_ids,
            )
        barrier(context)
        started = time.monotonic()
        rows = _local_rows(
            context,
            assets.candidates,
            assets.manifest,
            assets.lora,
            policy,
            batched_lora,
            processor,
            assets.dataset,
            args.max_groups_per_task,
        )
        write_json_atomic(
            args.output_dir / f"rank_{context.rank:02d}_rows.json",
            {"rows": rows},
        )
        barrier(context)
        if context.is_main:
            summary = _finalize_output(
                args,
                context,
                assets.candidates,
                assets.manifest,
                started,
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if batched_lora is not None:
            batched_lora.close()
        if assets is not None:
            assets.dataset.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
