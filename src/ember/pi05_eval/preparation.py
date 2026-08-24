"""Prepare and publish one immutable PI0.5 evaluation root."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.eval_adapters import (
    adapter_requests,
    inspect_source_sft_adapter,
    inspect_static_task_lora_adapter,
    inspect_task_expert_adapter,
    select_task_expert_adapter_tasks,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.occupancy_selection import (
    SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA,
    SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA,
    successful_expert_occupancy_tasks,
)
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
from ember.pi05_source_checkpoint import read_json


TASK_SUBSET_SELECTION_SCHEMA = "ember_pi05_task_subset_selection_v1"
TRAIN24_FOLD0_HELD = (
    (0, 0, "libero_spatial", 0),
    (5, 9, "libero_spatial", 9),
    (10, 18, "libero_object", 8),
    (15, 25, "libero_goal", 5),
    (20, 36, "libero_10", 6),
)
TRAIN24_FOLD0_PROFILE = ((1, 2, "libero_spatial", 2),)


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
        raise Pi05EvaluationError("GPU indices must be comma-separated integers") from error
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise Pi05EvaluationError("GPU indices must be a non-empty unique sequence")
    return indices


def _inspect_adapter(
    args: Any,
    *,
    adapter_kind: str | None,
    source_sft_requested: bool,
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
    if adapter_kind == "static_task_lora":
        return inspect_static_task_lora_adapter(
            manifest_path=args.static_task_lora_manifest.resolve(),
            source=model,
            tasks=tasks,
            evaluation_role=args.role,
            require_formal=args.mode != "smoke",
        )
    if adapter_kind != "task_expert":
        return None
    return inspect_task_expert_adapter(
        config_path=args.task_expert_config.resolve(),
        bank_root=args.task_expert_bank_root.resolve(),
        step=int(args.task_expert_step),
        source=model,
        tasks=tasks,
        evaluation_role=args.role,
        require_formal=args.mode != "smoke",
    )


def _task_subset_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    adapter_kind: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    path = getattr(args, "task_subset_selection", None)
    if path is None:
        return tuple(tasks), None
    if (
        getattr(args, "occupancy_capture_selection", None) is not None
        or (str(args.mode), int(args.state_count)) not in {("screen", 10), ("formal", 50)}
        or args.role != "development_train"
        or adapter_kind not in {None, "task_expert", "static_task_lora"}
    ):
        raise Pi05EvaluationError("formal task subset request changed")
    path = path.resolve()
    manifest = read_json(path)
    task_rows = tuple(manifest.get("tasks", ()))
    ordinals = tuple(int(value) for value in manifest.get("task_ordinals", ()))
    declared = tuple(
        (
            ordinal,
            int(row.get("global_task_id", -1)),
            str(row.get("suite")),
            int(row.get("task_id", -1)),
        )
        for ordinal, row in zip(ordinals, task_rows, strict=True)
    )
    panel = (
        "train24_fold0_held5"
        if declared == TRAIN24_FOLD0_HELD
        else "train24_fold0_profile1" if declared == TRAIN24_FOLD0_PROFILE else None
    )
    if (
        manifest.get("schema_version") != TASK_SUBSET_SELECTION_SCHEMA
        or manifest.get("role") != args.role
        or manifest.get("mode") != args.mode
        or int(manifest.get("state_count", -1)) != args.state_count
        or manifest.get("outcome_dependence") is not False
        or panel is None
    ):
        raise Pi05EvaluationError("formal task subset selection changed")
    by_key = {(str(task.suite), int(task.task_id)): task for task in tasks}
    keys = tuple((row[2], row[3]) for row in declared)
    if len(by_key) != len(tasks) or any(key not in by_key for key in keys):
        raise Pi05EvaluationError("task subset is outside development-train")
    selected = tuple(by_key[key] for key in keys)
    return selected, {
        "schema_version": TASK_SUBSET_SELECTION_SCHEMA,
        "selection_path": str(path),
        "selection_bytes": path.stat().st_size,
        "task_ordinals": [row[0] for row in declared],
        "global_task_ids": [row[1] for row in declared],
        "diagnostic_subset": panel,
        "outcome_dependence": False,
        "validation_use": False,
        "test_use": False,
    }


def _occupancy_capture_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    adapter_kind: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    path = getattr(args, "occupancy_capture_selection", None)
    if path is None:
        return tuple(tasks), None
    path = path.resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA:
        raise Pi05EvaluationError("unsupported occupancy capture selection")
    return successful_expert_occupancy_tasks(
        args,
        tasks,
        output_dir=output_dir,
        adapter_kind=adapter_kind,
        selection_path=path,
        manifest=manifest,
        rows=tuple(dict(row) for row in manifest.get("rows", ())),
    )


def _stage_predicate_capture(
    args: Any, occupancy_capture: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if not bool(getattr(args, "capture_stage_predicates", False)):
        return None
    if (
        args.mode != "formal"
        or args.role != "nonheld_meta_train"
        or occupancy_capture is None
        or occupancy_capture.get("schema_version")
        != SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA
    ):
        raise Pi05EvaluationError(
            "stage-predicate capture requires the successful-expert panel"
        )
    return {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "post_settling_then_every_executed_action_change_points",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "training_gradient_use": False,
        "checkpoint_selection_use": False,
        "validation_action_reads": 0,
        "validation_reward_reads": 0,
        "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals",
    }


def _prepared_payload(
    args: Any,
    *,
    output_dir: Path,
    staging: Path,
    repo_root: Path,
    command: Sequence[str],
    adapter_kind: str | None,
    source_sft_requested: bool,
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    authorities = load_evaluation_authorities(args.config, repo_root)
    formal_count = int(authorities.config["environment"]["fixed_init_state_count"])
    if args.mode == "formal" and args.state_count != formal_count:
        raise Pi05EvaluationError("formal PI05 evaluation requires all fixed states")
    if (
        args.mode == "screen"
        and adapter_kind is None
        and not source_sft_requested
        and args.role
        not in {
            "all_targets",
            "nonheld_meta",
            "nonheld_meta_train",
            "nonheld_meta_validation",
        }
    ):
        raise Pi05EvaluationError("source-base screen must cover its complete role")
    installed_tasks, libero_paths = inspect_installed_target_tasks(
        authorities,
        role=args.role,
        state_count=args.state_count,
        libero_config_dir=staging / "libero_config",
    )
    subset_tasks, task_subset = _task_subset_tasks(
        args, installed_tasks, adapter_kind=adapter_kind
    )
    tasks, occupancy_capture = _occupancy_capture_tasks(
        args,
        subset_tasks,
        output_dir=output_dir,
        adapter_kind=adapter_kind,
    )
    stage_predicates = _stage_predicate_capture(args, occupancy_capture)
    model = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode=args.mode,
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    diagnostic_subset = (
        "successful_on_policy_occupancy"
        if occupancy_capture is not None
        else str(task_subset["diagnostic_subset"]) if task_subset is not None else None
    )
    inspection_tasks = (
        installed_tasks
        if diagnostic_subset and adapter_kind == "task_expert"
        else tasks
    )
    adapter = _inspect_adapter(
        args,
        adapter_kind=adapter_kind,
        source_sft_requested=source_sft_requested,
        model=model,
        tasks=inspection_tasks,
    )
    if diagnostic_subset and adapter_kind == "task_expert":
        adapter = select_task_expert_adapter_tasks(
            adapter, tasks, diagnostic_subset=diagnostic_subset
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
    )
    contract["diagnostic_occupancy_capture"] = occupancy_capture
    contract["diagnostic_stage_predicates"] = stage_predicates
    contract["diagnostic_task_subset"] = task_subset
    shards = shards_from_contract(contract)
    summary = {
        "event": "prepared",
        "contract_reference": contract["contract_reference"],
        "tasks": len(tasks),
        "states": sum(len(task.init_state_ids) for task in tasks),
        "shards": len(shards),
        "replicas_per_gpu": args.replicas_per_gpu,
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
        raise Pi05EvaluationError(f"PI05 evaluation output already exists: {output_dir}")
    staging.rename(output_dir)


def prepare_evaluation_run(
    args: Any,
    *,
    repo_root: Path,
    command: Sequence[str],
    create_evaluation_queue: bool = True,
) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise Pi05EvaluationError(f"PI05 evaluation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = _claim_prepare_lock(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.prepare-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        adapter_kind, source_sft_requested = adapter_requests(args)
        staging.mkdir()
        contract, shards, summary = _prepared_payload(
            args,
            output_dir=output_dir,
            staging=staging,
            repo_root=repo_root,
            command=command,
            adapter_kind=adapter_kind,
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
