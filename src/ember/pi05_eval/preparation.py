"""Fail-closed preparation and publication of one PI05 evaluation root."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    adapter_requests,
    inspect_dynamic_k_writer_adapter,
    inspect_source_sft_adapter,
    inspect_task_expert_adapter,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    build_run_contract,
    inspect_installed_target_tasks,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_eval_queue import (
    EvaluationTask,
    build_cost_balanced_shards,
    initialize_queue,
    publish_json_exclusive,
)


def shards_from_contract(contract: Mapping[str, Any]) -> tuple[Any, ...]:
    tasks = tuple(
        EvaluationTask(
            suite=row["suite"],
            task_id=int(row["task_id"]),
            horizon=int(row["horizon"]),
            init_state_ids=tuple(int(value) for value in row["init_state_ids"]),
        )
        for row in contract["tasks"]
    )
    return build_cost_balanced_shards(
        tasks,
        env_batch_size=int(contract["parallel"]["envs_per_replica"]),
        target_cost=int(contract["parallel"]["shard_target_cost"]),
        physical_gpu_count=int(contract["parallel"]["physical_gpu_count"]),
        replicas_per_gpu=int(contract["parallel"]["replicas_per_gpu"]),
    )


def parse_gpu_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise Pi05EvaluationError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise Pi05EvaluationError("GPU indices must be a non-empty unique sequence")
    return indices


def _inspect_adapter(
    args: Any,
    *,
    writer_kind: str | None,
    source_sft_requested: bool,
    authorities: Any,
    model: Mapping[str, Any],
    tasks: Sequence[Any],
) -> Mapping[str, Any] | None:
    if source_sft_requested:
        return inspect_source_sft_adapter(
            config_path=args.source_sft_config.resolve(),
            checkpoint=args.source_sft_checkpoint.resolve(),
            source=model,
            tasks=tasks,
            evaluation_role=args.role,
            require_formal=args.mode != "smoke",
        )
    if writer_kind == "task_expert":
        return inspect_task_expert_adapter(
            config_path=args.task_expert_config.resolve(),
            bank_root=args.task_expert_bank_root.resolve(),
            step=int(args.task_expert_step),
            source=model,
            tasks=tasks,
            evaluation_role=args.role,
            require_formal=args.mode != "smoke",
        )
    if writer_kind == DYNAMIC_K_WRITER_KIND:
        return inspect_dynamic_k_writer_adapter(
            config_path=args.dynamic_k_writer_config.resolve(),
            checkpoint=args.dynamic_k_writer_checkpoint.resolve(),
            video_data_root=args.dynamic_k_writer_video_data_root.resolve(),
            source=model,
            tasks=tasks,
            video_condition=str(args.dynamic_k_writer_video_condition),
            video_seed=int(authorities.config["rng"]["inference_seed"]),
            video_sampling_mode=str(args.dynamic_k_writer_video_sampling),
            require_formal=args.mode != "smoke",
            evaluation_k=int(getattr(args, "dynamic_k_writer_evaluation_k", 1)),
        )
    return None


def _prepared_payload(
    args: Any,
    *,
    output_dir: Path,
    staging: Path,
    repo_root: Path,
    command: Sequence[str],
    writer_kind: str | None,
    source_sft_requested: bool,
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    authorities = load_evaluation_authorities(args.config, repo_root)
    formal_count = int(authorities.config["environment"]["fixed_init_state_count"])
    if args.mode == "formal" and args.state_count != formal_count:
        raise Pi05EvaluationError("formal PI05 evaluation requires all 50 fixed states")
    if (
        args.mode == "screen"
        and writer_kind is None
        and not source_sft_requested
        and args.role != "all_targets"
    ):
        raise Pi05EvaluationError("source-base screen must cover all 40 target tasks")
    tasks, libero_paths = inspect_installed_target_tasks(
        authorities,
        role=args.role,
        state_count=args.state_count,
        libero_config_dir=staging / "libero_config",
    )
    model = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode=args.mode,
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    adapter = _inspect_adapter(
        args,
        writer_kind=writer_kind,
        source_sft_requested=source_sft_requested,
        authorities=authorities,
        model=model,
        tasks=tasks,
    )
    contract = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=libero_paths,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        role=args.role,
        mode=args.mode,
        replicas_per_gpu=args.replicas_per_gpu,
        physical_gpu_ids=parse_gpu_indices(args.gpu_indices),
        command=command,
        adapter=adapter,
        writer_generators_per_gpu=args.writer_generators_per_gpu,
        writer_generation_batch_size=args.writer_generation_batch_size,
        writer_cache_root=args.writer_lora_cache_root,
    )
    shards = shards_from_contract(contract)
    summary = {
        "event": "prepared",
        "contract_reference": contract["contract_reference"],
        "tasks": len(tasks),
        "states": sum(len(task.init_state_ids) for task in tasks),
        "shards": len(shards),
        "replicas_per_gpu": args.replicas_per_gpu,
        "writer_generators_per_gpu": contract["parallel"]["writer_generators_per_gpu"],
        "writer_generation_batch_size": contract["parallel"][
            "writer_generation_batch_size"
        ],
        "writer_lora_cache": contract["writer_lora_cache"],
        "physical_gpu_ids": contract["parallel"]["physical_gpu_ids"],
        "arm": contract["arm"],
        "output_dir": str(output_dir),
    }
    return contract, shards, summary


def _prepare_lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.prepare.lock"


def _claim_prepare_lock(output_dir: Path) -> Path:
    lock = _prepare_lock_path(output_dir)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise Pi05EvaluationError(
            f"another PI05 evaluator prepare owns the output lock: {output_dir}"
        ) from error
    return lock


def _publish_staging(staging: Path, output_dir: Path, *, lock: Path) -> None:
    if lock != _prepare_lock_path(output_dir) or not lock.is_dir():
        raise Pi05EvaluationError("PI05 evaluation publication lost its output lock")
    if output_dir.exists() or output_dir.is_symlink():
        raise Pi05EvaluationError(
            f"PI05 evaluation output already exists: {output_dir}"
        )
    staging.rename(output_dir)


def prepare_evaluation_run(
    args: Any,
    *,
    repo_root: Path,
    command: Sequence[str],
    create_evaluation_queue: bool = True,
) -> dict[str, Any]:
    """Validate into a private root, then publish the canonical root once."""

    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise Pi05EvaluationError(
            f"PI05 evaluation output already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = _claim_prepare_lock(output_dir)
    staging = output_dir.parent / (
        f".{output_dir.name}.prepare-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        writer_kind, source_sft_requested = adapter_requests(args)
        staging.mkdir()
        contract, shards, summary = _prepared_payload(
            args,
            output_dir=output_dir,
            staging=staging,
            repo_root=repo_root,
            command=command,
            writer_kind=writer_kind,
            source_sft_requested=source_sft_requested,
        )
        publish_json_exclusive(staging / "run_contract.json", contract)
        if create_evaluation_queue:
            initialize_queue(
                staging / "queue.sqlite3",
                shards,
                contract_reference=contract["contract_reference"],
            )
        _publish_staging(staging, output_dir, lock=lock)
        os.environ["LIBERO_CONFIG_PATH"] = str(output_dir / "libero_config")
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if lock.exists():
            lock.rmdir()
