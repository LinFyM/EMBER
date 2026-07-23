"""In-process validation-loss monitoring for AS-Writer training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.pi05_processing import Pi05LiberoProcessor, Pi05PureLanguageTokenizer
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
from ember.writer.as_contract import REPO_ROOT
from ember.writer.data import ActionHiddenVideoStore, FunctionalQueryDataset
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.validation import (
    _condition_state,
    _group_loss,
    _task_authorities,
    _validate_task_hashes,
)
from ember.writer.validation_panel import (
    build_validation_loss_manifest,
    load_validation_loss_panel,
    summarize_validation_losses,
)


ONLINE_RUN_SCHEMA = "ember_pi05_as_writer_online_validation_loss_v1"
DEFAULT_PANEL = REPO_ROOT / "configs/pi05_validation_functional_loss_panel_v1.json"


@dataclass
class OnlineWriterValidation:
    """Validation data kept resident beside one AS-Writer training process."""

    panel: dict[str, Any]
    manifest: dict[str, Any]
    tasks: tuple[Any, ...]
    dataset: FunctionalQueryDataset
    store: ActionHiddenVideoStore
    tokenizer: Pi05PureLanguageTokenizer
    output_dir: Path
    local_keys: tuple[tuple[int, int], ...]

    def close(self) -> None:
        self.dataset.close()
        self.store.close()


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
        raise WriterModelError(payload[0]["error"])
    return payload[0]


def _online_contract(
    *,
    training: Mapping[str, Any],
    panel: Mapping[str, Any],
    manifest: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    return {
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
        "information_wall": dict(panel["information_wall"]),
        "cadence": "every retained training checkpoint",
        "world_size": world_size,
        "policy_and_writer_reused_in_process": True,
        "optimizer_updates": 0,
        "parameter_gradients_computed": False,
        "physical_gpu_limit": [0, 1, 2, 3],
    }


def _publish_online_contract(
    root: Path,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_contract.json"
    if path.is_file() and read_json(path) != contract:
        raise WriterModelError(
            "online validation-loss contract changed during resume"
        )
    write_json_atomic(path, dict(contract))
    manifest_path = root / "panel_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise WriterModelError(
            "online validation-loss manifest changed during resume"
        )
    write_json_atomic(manifest_path, dict(manifest))
    return {"ok": True}


def prepare_online_writer_validation(
    *,
    training: Mapping[str, Any],
    data_root: Path,
    tokenizer_path: Path,
    context: DistributedContext,
    output_dir: Path,
) -> OnlineWriterValidation:
    """Prepare the sealed panel once without loading another policy or Writer."""

    if training.get("stage") != "development":
        raise WriterModelError(
            "online validation-action loss is development-only"
        )
    panel = load_validation_loss_panel(DEFAULT_PANEL)
    tasks = _task_authorities(training, panel, data_root.resolve())
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
    store = ActionHiddenVideoStore(
        tasks,
        frame_stride=int(training["writer"]["frame_stride"]),
    )
    source_config = read_json(
        REPO_ROOT / str(training["authorities"]["source_base_config"]["path"])
    )
    tokenizer = Pi05PureLanguageTokenizer(
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
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
    contract = _online_contract(
        training=training,
        panel=panel,
        manifest=manifest,
        data_validation=data_validation,
        world_size=context.world_size,
    )
    _broadcast_payload(
        context,
        lambda: _publish_online_contract(root, contract, manifest),
    )
    barrier(context)
    return OnlineWriterValidation(
        panel=panel,
        manifest=manifest,
        tasks=tasks,
        dataset=dataset,
        store=store,
        tokenizer=tokenizer,
        output_dir=root,
        local_keys=local_keys,
    )


def _online_local_rows(
    *,
    validation: OnlineWriterValidation,
    context: DistributedContext,
    checkpoint_cursor: int,
    checkpoint_manifest_sha256: str,
    policy: torch.nn.Module,
    writer: CompleteLoRAWriter,
    identity: Mapping[str, torch.Tensor],
    lora: Any,
    processor: Pi05LiberoProcessor,
) -> list[dict[str, Any]]:
    tasks = {task.task_id: task for task in validation.tasks}
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in validation.manifest["rows"]:
        key = (int(row["global_task_id"]), int(row["video_group"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for task_id, video_group in validation.local_keys:
        rows = grouped[(task_id, video_group)]
        tick = time.monotonic()
        state = _condition_state(
            policy=policy,
            writer=writer,
            identity=identity,
            lora=lora,
            store=validation.store,
            tokenizer=validation.tokenizer,
            task=tasks[task_id],
            demo_index=int(rows[0]["teacher_demo_index"]),
            device=context.device,
        )
        losses = _group_loss(
            policy=policy,
            state=state,
            lora=lora,
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


def _online_summary(
    validation: OnlineWriterValidation,
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
    current_loss = float(summary["task_balanced_mean_loss"])
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
                current_loss - previous_loss
                if previous_loss is not None
                else None
            ),
            "test_action_reads": 0,
            "teacher_video_value_reads": len(
                {
                    (
                        int(row["global_task_id"]),
                        int(row["teacher_demo_index"]),
                    )
                    for row in combined
                }
            ),
            "parameter_gradients_computed": False,
            "optimizer_updates": 0,
        }
    )
    write_json_atomic(step_dir / "rows.json", {"rows": combined})
    write_json_atomic(step_dir / "summary.json", summary)
    append_jsonl(validation.output_dir / "metrics.jsonl", summary)
    return summary


def evaluate_online_writer_checkpoint(
    *,
    validation: OnlineWriterValidation,
    context: DistributedContext,
    checkpoint_cursor: int,
    checkpoint_dir: Path,
    policy: torch.nn.Module,
    writer: CompleteLoRAWriter,
    identity: Mapping[str, torch.Tensor],
    lora: Any,
    processor: Pi05LiberoProcessor,
) -> dict[str, Any]:
    """Evaluate one resident checkpoint while preserving the training RNG."""

    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if (
        checkpoint_cursor <= 0
        or checkpoint_dir.name != f"step_{checkpoint_cursor:08d}"
        or not manifest_path.is_file()
    ):
        raise WriterModelError("online validation checkpoint is incomplete")
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
    was_training = writer.training
    started = time.monotonic()
    try:
        writer.eval()
        rows = _online_local_rows(
            validation=validation,
            context=context,
            checkpoint_cursor=checkpoint_cursor,
            checkpoint_manifest_sha256=sha256_file(manifest_path),
            policy=policy,
            writer=writer,
            identity=identity,
            lora=lora,
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
            lambda: _online_summary(
                validation,
                checkpoint_cursor,
                context.world_size,
                started,
            ),
        )
        barrier(context)
        return summary
    finally:
        writer.train(was_training)
        restore_rng(rng, context)
