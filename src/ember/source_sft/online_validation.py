"""In-process validation-loss monitoring for Source-SFT training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    capture_rng,
    read_json,
    restore_rng,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.source_sft.contract import REPO_ROOT, Pi05SourceSFTError
from ember.source_sft.validation import (
    _group_loss,
    _validate_task_hashes,
    _validation_tasks,
)
from ember.writer.data import FunctionalQueryDataset
from ember.writer.validation_panel import (
    build_validation_loss_manifest,
    load_validation_loss_panel,
    summarize_validation_losses,
)


ONLINE_RUN_SCHEMA = "ember_pi05_source_sft_online_validation_loss_v1"
DEFAULT_PANEL = REPO_ROOT / "configs/pi05_validation_functional_loss_panel_v1.json"


@dataclass
class OnlineSourceSFTValidation:
    panel: dict[str, Any]
    manifest: dict[str, Any]
    dataset: FunctionalQueryDataset
    output_dir: Path
    local_keys: tuple[tuple[int, int], ...]

    def close(self) -> None:
        self.dataset.close()


def _broadcast_payload(
    context: DistributedContext,
    operation: Any,
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceSFTError(payload[0]["error"])
    return payload[0]


def _publish_contract(
    *,
    root: Path,
    training: Mapping[str, Any],
    panel: Mapping[str, Any],
    manifest: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    contract = {
        "schema_version": ONLINE_RUN_SCHEMA,
        "training_contract_sha256": canonical_hash(dict(training)),
        "panel": {
            "path": str(DEFAULT_PANEL),
            "file_sha256": sha256_file(DEFAULT_PANEL),
            "manifest_payload_sha256": manifest[
                "canonical_payload_sha256"
            ],
        },
        "data_validation": dict(data_validation),
        "information_wall": {
            **panel["information_wall"],
            "method": "shared_source_sft",
            "teacher_video_values_read": 0,
        },
        "cadence": "every retained training checkpoint",
        "world_size": world_size,
        "policy_reused_in_process": True,
        "optimizer_updates": 0,
        "parameter_gradients_computed": False,
        "physical_gpu_limit": [0, 1, 2, 3],
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_contract.json"
    if path.is_file() and read_json(path) != contract:
        raise Pi05SourceSFTError(
            "online Source-SFT validation contract changed during resume"
        )
    write_json_atomic(path, contract)
    manifest_path = root / "panel_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise Pi05SourceSFTError(
            "online Source-SFT validation manifest changed during resume"
        )
    write_json_atomic(manifest_path, dict(manifest))
    return {"ok": True}


def prepare_online_source_sft_validation(
    *,
    training: Mapping[str, Any],
    data_root: Path,
    context: DistributedContext,
    output_dir: Path,
) -> OnlineSourceSFTValidation:
    if training.get("stage") != "development":
        raise Pi05SourceSFTError(
            "online Source-SFT validation-action loss is development-only"
        )
    panel = load_validation_loss_panel(DEFAULT_PANEL)
    tasks = _validation_tasks(training, panel, data_root.resolve())
    data_validation = _broadcast_payload(
        context,
        lambda: _validate_task_hashes(tasks, training),
    )
    dataset = FunctionalQueryDataset(
        tasks,
        demo_indices=range(50),
        action_chunk_size=int(panel["sampling"]["action_chunk_size"]),
    )
    manifest = build_validation_loss_manifest(dataset, panel)
    grouped = {
        (int(row["global_task_id"]), int(row["video_group"]))
        for row in manifest["rows"]
    }
    local_keys = tuple(
        key
        for ordinal, key in enumerate(sorted(grouped))
        if ordinal % context.world_size == context.rank
    )
    root = output_dir / "validation_functional_loss"
    _broadcast_payload(
        context,
        lambda: _publish_contract(
            root=root,
            training=training,
            panel=panel,
            manifest=manifest,
            data_validation=data_validation,
            world_size=context.world_size,
        ),
    )
    barrier(context)
    return OnlineSourceSFTValidation(
        panel=panel,
        manifest=manifest,
        dataset=dataset,
        output_dir=root,
        local_keys=local_keys,
    )


def _local_rows(
    *,
    validation: OnlineSourceSFTValidation,
    context: DistributedContext,
    checkpoint_cursor: int,
    checkpoint_manifest_sha256: str,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in validation.manifest["rows"]:
        key = (int(row["global_task_id"]), int(row["video_group"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for key in validation.local_keys:
        rows = grouped[key]
        tick = time.monotonic()
        losses = _group_loss(
            policy=policy,
            processor=processor,
            dataset=validation.dataset,
            rows=rows,
            device=context.device,
        )
        elapsed = time.monotonic() - tick
        result.extend(
            {
                **row,
                "checkpoint_cursor": checkpoint_cursor,
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                "loss": loss,
                "group_wall_seconds": elapsed,
                "rank": context.rank,
            }
            for row, loss in zip(rows, losses, strict=True)
        )
    return result


def _summary(
    validation: OnlineSourceSFTValidation,
    checkpoint_cursor: int,
    world_size: int,
    started: float,
) -> dict[str, Any]:
    step_dir = validation.output_dir / f"step_{checkpoint_cursor:08d}"
    combined = []
    for rank in range(world_size):
        combined.extend(read_json(step_dir / f"rank_{rank:02d}_rows.json")["rows"])
    combined.sort(key=lambda row: int(row["ordinal"]))
    summary = summarize_validation_losses(combined)["checkpoints"][0]
    prior = []
    for path in sorted(validation.output_dir.glob("step_*/summary.json")):
        record = read_json(path)
        if int(record["checkpoint_cursor"]) < checkpoint_cursor:
            prior.append(record)
    previous = prior[-1] if prior else None
    previous_loss = (
        float(previous["task_balanced_mean_loss"]) if previous is not None else None
    )
    summary.update(
        {
            "schema_version": "ember_pi05_online_validation_checkpoint_v1",
            "wall_seconds": time.monotonic() - started,
            "row_count": len(combined),
            "previous_checkpoint_cursor": (
                int(previous["checkpoint_cursor"]) if previous is not None else None
            ),
            "loss_delta_from_previous": (
                float(summary["task_balanced_mean_loss"]) - previous_loss
                if previous_loss is not None
                else None
            ),
            "test_action_reads": 0,
            "teacher_video_value_reads": 0,
            "parameter_gradients_computed": False,
            "optimizer_updates": 0,
        }
    )
    write_json_atomic(step_dir / "rows.json", {"rows": combined})
    write_json_atomic(step_dir / "summary.json", summary)
    append_jsonl(validation.output_dir / "metrics.jsonl", summary)
    return summary


def evaluate_online_source_sft_checkpoint(
    *,
    validation: OnlineSourceSFTValidation,
    context: DistributedContext,
    checkpoint_cursor: int,
    checkpoint_dir: Path,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if (
        checkpoint_cursor <= 0
        or checkpoint_dir.name != f"step_{checkpoint_cursor:08d}"
        or not manifest_path.is_file()
    ):
        raise Pi05SourceSFTError(
            "online Source-SFT validation checkpoint is incomplete"
        )
    step_dir = validation.output_dir / f"step_{checkpoint_cursor:08d}"
    existing = _broadcast_payload(
        context,
        lambda: (
            read_json(step_dir / "summary.json")
            if (step_dir / "summary.json").is_file()
            else {"pending": True}
        ),
    )
    if not existing.get("pending"):
        return existing

    rng = capture_rng(context)
    was_training = policy.training
    started = time.monotonic()
    try:
        policy.eval()
        rows = _local_rows(
            validation=validation,
            context=context,
            checkpoint_cursor=checkpoint_cursor,
            checkpoint_manifest_sha256=sha256_file(manifest_path),
            policy=policy,
            processor=processor,
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            step_dir / f"rank_{context.rank:02d}_rows.json",
            {"rows": rows},
        )
        barrier(context)
        summary = _broadcast_payload(
            context,
            lambda: _summary(
                validation,
                checkpoint_cursor,
                context.world_size,
                started,
            ),
        )
        barrier(context)
        return summary
    finally:
        policy.train(was_training)
        restore_rng(rng, context)
