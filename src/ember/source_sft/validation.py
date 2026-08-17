"""Evaluate Source-SFT checkpoints on the sealed validation-loss panel."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import default_collate

from ember.lora import (
    copy_task_lora_state_,
    inject_task_lora,
    validate_lora_state,
)
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_setup import initialize_distributed, load_policy, load_stats
from ember.source_sft.checkpoint import validate_source_sft_checkpoint_files
from ember.source_sft.contract import (
    REPO_ROOT,
    SOURCE_SFT_LAUNCH_SCHEMA,
    Pi05SourceSFTError,
    authority_path,
    load_source_sft_config,
)
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.validation_panel import (
    PANEL_SCHEMA,
    build_validation_loss_manifest,
    load_validation_loss_panel,
    summarize_validation_losses,
)


RUN_SCHEMA = "ember_pi05_source_sft_validation_loss_run_v1"


def _checkpoint_records(
    training_run: Path,
    checkpoints: Sequence[Path],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    training = read_json(training_run / "run_contract.json")
    contract_sha256 = canonical_hash(training)
    world_size = int(training.get("runtime", {}).get("world_size", -1))
    if (
        training.get("schema_version") != SOURCE_SFT_LAUNCH_SCHEMA
        or training.get("mode") != "formal"
        or training.get("stage") != "development"
        or world_size <= 0
        or not checkpoints
    ):
        raise Pi05SourceSFTError(
            "validation loss requires a formal development Source-SFT run"
        )
    records = []
    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve()
        if checkpoint.parent.parent != training_run:
            raise Pi05SourceSFTError(
                "validation checkpoint crossed its Source-SFT run"
            )
        manifest = validate_source_sft_checkpoint_files(
            checkpoint,
            world_size=world_size,
            contract_sha256=contract_sha256,
        )
        cursor = int(manifest.get("consumed", {}).get("next_step", -1))
        lora_file = manifest.get("files", {}).get("lora.safetensors", {})
        if (
            cursor <= 0
            or checkpoint.name != f"step_{cursor:08d}"
            or cursor not in training["runtime"]["checkpoint_steps"]
        ):
            raise Pi05SourceSFTError(
                "validation Source-SFT checkpoint cursor changed"
            )
        records.append(
            {
                "path": str(checkpoint),
                "cursor": cursor,
                "manifest_file_sha256": sha256_file(
                    checkpoint / "checkpoint_manifest.json"
                ),
                "manifest_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
                "lora_file_sha256": lora_file["sha256"],
            }
        )
    if len({record["cursor"] for record in records}) != len(records):
        raise Pi05SourceSFTError("validation Source-SFT checkpoints are duplicated")
    return training, tuple(sorted(records, key=lambda row: row["cursor"]))


def _validated_config(
    training: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    config = load_source_sft_config(config_path)
    if (
        config.get("sealed_stage") != "development"
        or sha256_file(config_path) != training.get("config_sha256")
        or config.get("authorities") != training.get("authorities")
    ):
        raise Pi05SourceSFTError(
            "validation Source-SFT config differs from its training run"
        )
    return config


def _validation_tasks(
    training: Mapping[str, Any],
    panel: Mapping[str, Any],
    data_root: Path,
) -> tuple[WriterTaskAuthority, ...]:
    target_ref = training.get("authorities", {}).get("target_data_manifest", {})
    if target_ref != panel["authorities"]["target_data_manifest"]:
        raise Pi05SourceSFTError(
            "Source-SFT training and validation target authorities differ"
        )
    target = read_json(REPO_ROOT / str(target_ref["path"]))
    validation_ids = {
        int(value) for value in target["summary"]["roles"]["validation"]
    }
    tasks = []
    for row in target["tasks"]:
        task_id = int(row["global_task_id"])
        if task_id not in validation_ids:
            continue
        path = (data_root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root):
            raise Pi05SourceSFTError("validation HDF5 escaped its data root")
        tasks.append(
            WriterTaskAuthority(
                task_id=task_id,
                language=str(row["language"]),
                path=path,
                expected_bytes=int(row["hdf5"]["bytes"]),
                expected_sha256=None,
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    if len(tasks) != 8:
        raise Pi05SourceSFTError(
            "Source-SFT validation loss did not resolve eight tasks"
        )
    return tuple(tasks)


def _validate_task_hashes(
    tasks: Sequence[WriterTaskAuthority],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    target = read_json(
        REPO_ROOT
        / str(training["authorities"]["target_data_manifest"]["path"])
    )
    expected = {
        int(row["global_task_id"]): str(row["hdf5"]["sha256"])
        for row in target["tasks"]
    }
    records = []
    for task in tasks:
        observed = sha256_file(task.path)
        if (
            task.path.stat().st_size != task.expected_bytes
            or observed != expected[task.task_id]
        ):
            raise Pi05SourceSFTError("validation HDF5 identity changed")
        records.append([task.task_id, task.expected_bytes, observed])
    return {
        "task_count": len(records),
        "full_sha256_verified": True,
        "identity_sha256": canonical_hash(records),
    }


def _build_policy(
    *,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer_path: Path,
    context: DistributedContext,
) -> tuple[torch.nn.Module, Any, Pi05LiberoProcessor]:
    source_config = read_json(authority_path(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    inject_task_lora(policy, lora)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    return policy, lora, processor


def _group_loss(
    *,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
    dataset: FunctionalQueryDataset,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[float, ...]:
    batch = default_collate(
        [dataset[int(row["dataset_row_index"])] for row in rows]
    )
    prepared = processor.training_batch(batch)
    seed = int(rows[0]["policy_noise_seed"])
    fork_devices = [device] if device.type == "cuda" else []
    with (
        torch.inference_mode(),
        torch.random.fork_rng(devices=fork_devices),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        output = policy(prepared, reduction="none")
    if (
        not isinstance(output, tuple)
        or not isinstance(output[0], torch.Tensor)
        or output[0].shape != (len(rows),)
        or not bool(torch.isfinite(output[0]).all())
    ):
        raise Pi05SourceSFTError(
            "Source-SFT validation did not return finite per-query losses"
        )
    return tuple(float(value) for value in output[0].detach().cpu())


def _local_rows(
    *,
    context: DistributedContext,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    policy: torch.nn.Module,
    lora: Any,
    processor: Pi05LiberoProcessor,
    dataset: FunctionalQueryDataset,
    max_groups_per_task: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in manifest["rows"]:
        key = (int(row["global_task_id"]), int(row["video_group"]))
        grouped.setdefault(key, []).append(row)
    keys = sorted(grouped)
    if max_groups_per_task is not None:
        keys = [key for key in keys if key[1] < max_groups_per_task]
    keys = [
        key
        for ordinal, key in enumerate(keys)
        if ordinal % context.world_size == context.rank
    ]
    result = []
    for record in records:
        state = load_file(
            str(Path(record["path"]) / "lora.safetensors"),
            device=str(context.device),
        )
        validate_lora_state(state, lora)
        copy_task_lora_state_(policy, state, lora)
        for key in keys:
            rows = grouped[key]
            tick = time.monotonic()
            losses = _group_loss(
                policy=policy,
                processor=processor,
                dataset=dataset,
                rows=rows,
                device=context.device,
            )
            elapsed = time.monotonic() - tick
            for row, loss in zip(rows, losses, strict=True):
                result.append(
                    {
                        **row,
                        "checkpoint_cursor": int(record["cursor"]),
                        "checkpoint_manifest_sha256": record[
                            "manifest_file_sha256"
                        ],
                        "loss": loss,
                        "group_wall_seconds": elapsed,
                        "rank": context.rank,
                    }
                )
    return result


def _publish_run_contract(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    state: Mapping[str, Any],
    panel: Mapping[str, Any],
    manifest: Mapping[str, Any],
    training_run: Path,
    training: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    data_validation: Mapping[str, Any],
) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise Pi05SourceSFTError(
            "Source-SFT validation loss output directory is not empty"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "panel_manifest.json", manifest)
    write_json_atomic(
        args.output_dir / "run_contract.json",
        {
            "schema_version": RUN_SCHEMA,
            "mode": args.mode,
            "host": socket.gethostname(),
            "command": list(os.sys.argv),
            "git": state,
            "panel": {
                "schema_version": PANEL_SCHEMA,
                "path": str(args.panel_config),
                "file_sha256": sha256_file(args.panel_config),
                "manifest_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
            },
            "training_run": {
                "path": str(training_run),
                "contract_sha256": canonical_hash(training),
            },
            "checkpoints": list(records),
            "source": source,
            "tokenizer": tokenizer,
            "data_validation": data_validation,
            "information_wall": {
                **panel["information_wall"],
                "method": "shared_source_sft",
                "teacher_video_values_read": 0,
            },
            "world_size": context.world_size,
            "physical_gpu_limit": [0, 1, 2, 3],
            "max_groups_per_task": args.max_groups_per_task,
        },
    )


def _finalize_results(
    *,
    output_dir: Path,
    world_size: int,
    started: float,
) -> dict[str, Any]:
    combined = []
    for rank in range(world_size):
        combined.extend(
            read_json(output_dir / f"rank_{rank:02d}_rows.json")["rows"]
        )
    combined.sort(
        key=lambda row: (
            int(row["checkpoint_cursor"]),
            int(row["ordinal"]),
        )
    )
    summary = summarize_validation_losses(combined)
    summary.update(
        {
            "wall_seconds": time.monotonic() - started,
            "row_count": len(combined),
            "test_action_reads": 0,
            "teacher_video_value_reads": 0,
            "parameter_gradients_computed": False,
        }
    )
    write_json_atomic(output_dir / "rows.json", {"rows": combined})
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def evaluate(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=args.mode == "formal")
    dataset: FunctionalQueryDataset | None = None
    try:
        panel = load_validation_loss_panel(args.panel_config)
        training_run = args.training_run.resolve()
        training, records = _checkpoint_records(training_run, args.checkpoints)
        config = _validated_config(training, args.config)
        authorities = load_evaluation_authorities(
            authority_path(config, "evaluation_config"), REPO_ROOT
        )
        source = inspect_source_checkpoint(
            authorities,
            args.source_run,
            args.source_checkpoint,
            evaluation_mode="formal",
        )
        if training["source"] != source:
            raise Pi05SourceSFTError(
                "Source-SFT validation checkpoint and source policy differ"
            )
        tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
        tasks = _validation_tasks(training, panel, args.data_root)
        validation: list[Any] = [None]
        if context.is_main:
            try:
                validation[0] = _validate_task_hashes(tasks, training)
            except Exception as error:
                validation[0] = {"error": repr(error)}
        if context.world_size > 1:
            dist.broadcast_object_list(validation, src=0, device=context.device)
        if validation[0].get("error"):
            raise Pi05SourceSFTError(validation[0]["error"])
        dataset = FunctionalQueryDataset(
            tasks,
            demo_indices=range(50),
            action_chunk_size=int(panel["sampling"]["action_chunk_size"]),
        )
        manifest = build_validation_loss_manifest(dataset, panel)
        policy, lora, processor = _build_policy(
            config=config,
            source=source,
            tokenizer_path=args.tokenizer_path,
            context=context,
        )
        state = git_state(REPO_ROOT)
        if args.mode == "formal" and (
            state["dirty_paths"] or state["commit"] != state["origin_main"]
        ):
            raise Pi05SourceSFTError(
                "formal Source-SFT validation loss requires pushed clean code"
            )
        if context.is_main:
            _publish_run_contract(
                args=args,
                context=context,
                state=state,
                panel=panel,
                manifest=manifest,
                training_run=training_run,
                training=training,
                records=records,
                source=source,
                tokenizer=tokenizer,
                data_validation=validation[0],
            )
        barrier(context)
        started = time.monotonic()
        rows = _local_rows(
            context=context,
            records=records,
            manifest=manifest,
            policy=policy,
            lora=lora,
            processor=processor,
            dataset=dataset,
            max_groups_per_task=args.max_groups_per_task,
        )
        write_json_atomic(
            args.output_dir / f"rank_{context.rank:02d}_rows.json",
            {"rows": rows},
        )
        barrier(context)
        if context.is_main:
            summary = _finalize_results(
                output_dir=args.output_dir,
                world_size=context.world_size,
                started=started,
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if dataset is not None:
            dataset.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_validation_functional_loss_panel_v1.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_source_sft_development_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-groups-per-task", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "panel_config",
        "config",
        "training_run",
        "source_run",
        "source_checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.checkpoints = tuple(path.resolve() for path in args.checkpoints)
    if args.mode == "formal" and args.max_groups_per_task is not None:
        raise Pi05SourceSFTError(
            "formal Source-SFT validation loss cannot truncate the panel"
        )
    if args.max_groups_per_task is not None and not 0 < args.max_groups_per_task <= 8:
        raise Pi05SourceSFTError(
            "invalid Source-SFT validation profile group count"
        )
    return args
