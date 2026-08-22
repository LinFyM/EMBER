"""Fail-closed preparation and publication of one PI05 evaluation root."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    FUNCTIONAL_CODE_WRITER_KIND,
    adapter_requests,
    inspect_dynamic_k_writer_adapter,
    inspect_functional_code_writer_adapter,
    inspect_source_sft_adapter,
    inspect_task_expert_adapter,
    select_task_expert_adapter_tasks,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    build_run_contract,
    inspect_installed_target_tasks,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_source_checkpoint import read_json
from ember.pi05_eval_queue import (
    EvaluationTask,
    build_cost_balanced_shards,
    initialize_queue,
    publish_json_exclusive,
)
from ember.pi05_eval.occupancy_selection import (
    MDCO_OCCUPANCY_SELECTION_SCHEMA,
    SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA,
    SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA,
    successful_expert_occupancy_tasks,
)


SUCCESSFUL_EXPERT_EQUIVALENCE_SELECTION_SCHEMA = (
    "ember_successful_expert_equivalence_selection_v1"
)
SUCCESSFUL_EXPERT_EQUIVALENCE_CAPTURE_SCHEMA = (
    "ember_successful_expert_equivalence_capture_v1"
)
PHASE_DECODER_FIT_OCCUPANCY_SELECTION_SCHEMA = (
    "ember_successful_expert_equivalence_phase_codes_v1"
)
PHASE_DECODER_FIT_OCCUPANCY_CAPTURE_SCHEMA = (
    "ember_phase_decoder_fit_projected_occupancy_capture_v1"
)
TASK_SUBSET_SELECTION_SCHEMA = "ember_pi05_task_subset_selection_v1"
PHASE_DECODER_HELD_ORDINALS = (0, 5, 10, 15, 20)
PHASE_DECODER_HELD_GLOBAL_IDS = (0, 9, 18, 25, 36)
PHASE_DECODER_HELD_KEYS = (
    ("libero_spatial", 0),
    ("libero_spatial", 9),
    ("libero_object", 8),
    ("libero_goal", 5),
    ("libero_10", 6),
)


def _task_expert_diagnostic_subset(
    occupancy_capture: Mapping[str, Any] | None,
) -> str | None:
    if occupancy_capture is None:
        return None
    schema = occupancy_capture.get("schema_version")
    if schema == SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA:
        return "successful_on_policy_occupancy"
    if schema == SUCCESSFUL_EXPERT_EQUIVALENCE_CAPTURE_SCHEMA:
        return "successful_expert_equivalence_occupancy"
    if schema == PHASE_DECODER_FIT_OCCUPANCY_CAPTURE_SCHEMA:
        return "phase_decoder_fit_projected_occupancy"
    return None


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
            projection_manifest=(
                args.task_expert_projection_manifest.resolve()
                if args.task_expert_projection_manifest is not None
                else None
            ),
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
    if writer_kind == FUNCTIONAL_CODE_WRITER_KIND:
        return inspect_functional_code_writer_adapter(
            config_path=args.functional_writer_config.resolve(),
            checkpoint=args.functional_writer_checkpoint.resolve(),
            video_data_root=args.functional_writer_video_data_root.resolve(),
            source=model,
            tasks=tasks,
            video_condition=str(args.functional_writer_video_condition),
            video_seed=int(authorities.config["rng"]["inference_seed"]),
            video_sampling_mode=str(args.functional_writer_video_sampling),
            require_formal=args.mode != "smoke",
            evaluation_k=int(getattr(args, "functional_writer_evaluation_k", 1)),
        )
    return None


def _occupancy_capture_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    writer_kind: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    path = getattr(args, "occupancy_capture_selection", None)
    if path is None:
        return tuple(tasks), None
    path = path.resolve()
    manifest = read_json(path)
    rows = tuple(dict(row) for row in manifest.get("rows", ()))
    if manifest.get("schema_version") in {
        SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA,
        MDCO_OCCUPANCY_SELECTION_SCHEMA,
    }:
        return successful_expert_occupancy_tasks(
            args,
            tasks,
            output_dir=output_dir,
            writer_kind=writer_kind,
            selection_path=path,
            manifest=manifest,
            rows=rows,
        )
    if (
        manifest.get("schema_version")
        == SUCCESSFUL_EXPERT_EQUIVALENCE_SELECTION_SCHEMA
    ):
        return _successful_expert_equivalence_tasks(
            args,
            tasks,
            output_dir=output_dir,
            writer_kind=writer_kind,
            selection_path=path,
            manifest=manifest,
            rows=rows,
        )
    if manifest.get("schema_version") == PHASE_DECODER_FIT_OCCUPANCY_SELECTION_SCHEMA:
        return _phase_decoder_fit_occupancy_tasks(
            args,
            tasks,
            output_dir=output_dir,
            writer_kind=writer_kind,
            selection_path=path,
            manifest=manifest,
        )
    counts = {
        category: sum(row.get("category") == category for row in rows)
        for category in ("lost", "gained", "retained")
    }
    keys = {
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in rows
    }
    if (
        args.mode != "formal"
        or args.role != "validation"
        or writer_kind != "layer_matched_memory_program_compiler_writer"
        or manifest.get("schema_version") != "ember_writer_occupancy_selection_v1"
        or counts != {"lost": 52, "gained": 13, "retained": 71}
        or len(keys) != len(rows)
        or len(rows) != 136
    ):
        raise Pi05EvaluationError("occupancy capture selection changed")
    selected = []
    covered = set()
    for task in tasks:
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (str(task.suite), int(task.task_id), int(state_id)) in keys
        )
        if state_ids:
            selected.append(replace(task, init_state_ids=state_ids))
            covered.update(
                (str(task.suite), int(task.task_id), int(state_id))
                for state_id in state_ids
            )
    if covered != keys:
        raise Pi05EvaluationError("occupancy selection is outside validation8")
    return tuple(selected), {
        "schema_version": "ember_writer_occupancy_capture_v1",
        "selection_path": str(path),
        "selection_bytes": path.stat().st_size,
        "category_counts": counts,
        "selected_rows": len(rows),
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": False,
        "validation_action_or_reward_gradient_use": False,
    }


def _phase_decoder_subset_manifest(
    manifest: Mapping[str, Any], args: Any
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[str, int], ...]]:
    ordinals = tuple(int(value) for value in manifest.get("task_ordinals", ()))
    global_ids = tuple(int(value) for value in manifest.get("global_task_ids", ()))
    declared_tasks = tuple(manifest.get("tasks", ()))
    declared_keys = tuple(
        (str(row.get("suite")), int(row.get("task_id", -1)))
        for row in declared_tasks
    )
    if (
        manifest.get("schema_version") != TASK_SUBSET_SELECTION_SCHEMA
        or manifest.get("role") != args.role
        or manifest.get("mode") != args.mode
        or int(manifest.get("state_count", -1)) != args.state_count
        or ordinals != PHASE_DECODER_HELD_ORDINALS
        or global_ids != PHASE_DECODER_HELD_GLOBAL_IDS
        or tuple(int(row.get("global_task_id", -1)) for row in declared_tasks)
        != global_ids
        or declared_keys != PHASE_DECODER_HELD_KEYS
        or manifest.get("outcome_dependence") is not False
    ):
        raise Pi05EvaluationError("formal task subset selection changed")
    return ordinals, global_ids, declared_keys


def _task_subset_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    writer_kind: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    path = getattr(args, "task_subset_selection", None)
    if path is None:
        return tuple(tasks), None
    if (
        getattr(args, "occupancy_capture_selection", None) is not None
        or args.mode != "formal"
        or args.role != "development_train"
        or args.state_count != 50
        or writer_kind not in {None, "task_expert", FUNCTIONAL_CODE_WRITER_KIND}
    ):
        raise Pi05EvaluationError("formal task subset request changed")
    path = path.resolve()
    ordinals, global_ids, declared_keys = _phase_decoder_subset_manifest(
        read_json(path), args
    )
    by_key = {(str(task.suite), int(task.task_id)): task for task in tasks}
    if len(by_key) != len(tasks) or any(key not in by_key for key in declared_keys):
        raise Pi05EvaluationError("installed target task identities overlap")
    selected = tuple(by_key[key] for key in declared_keys)
    if any(len(task.init_state_ids) != 50 for task in selected):
        raise Pi05EvaluationError("formal task subset lost fixed initial states")
    return selected, {
        "schema_version": TASK_SUBSET_SELECTION_SCHEMA,
        "selection_path": str(path),
        "selection_bytes": path.stat().st_size,
        "task_ordinals": list(ordinals),
        "global_task_ids": list(global_ids),
        "diagnostic_subset": "phase_aligned_decoder_held5",
        "outcome_dependence": False,
        "validation_use": False,
        "test_use": False,
    }


def _successful_expert_equivalence_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    writer_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    expected_by_step = {250: 21, 500: 2, 1000: 1, 1500: 0, 2000: 23}
    step = int(args.task_expert_step)
    selected_rows = tuple(
        row for row in rows if int(row.get("expert_step", -1)) == step
    )
    task_members: dict[tuple[str, int], set[str]] = {}
    ordinals: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row.get("suite")), int(row.get("task_id", -1)))
        task_members.setdefault(key, set()).add(str(row.get("member")))
        ordinals[key] = int(row.get("ordinal", -1))
    selected_keys = {
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in selected_rows
    }
    member_shapes = sorted(len(values) for values in task_members.values())
    if (
        args.mode != "formal"
        or args.role != "development_train"
        or writer_kind != "task_expert"
        or len(rows) != 47
        or len(task_members) != 24
        or member_shapes != [1] + [2] * 23
        or sorted(ordinals.values()) != list(range(24))
        or sum(values == {"only"} for values in task_members.values()) != 1
        or any(
            values not in ({"only"}, {"earliest", "latest"})
            for values in task_members.values()
        )
        or expected_by_step.get(step, 0) == 0
        or len(selected_rows) != expected_by_step[step]
        or len(selected_keys) != len(selected_rows)
        or any(
            row.get("fold_role")
            != ("held_transform_only" if int(row["ordinal"]) % 5 == 0 else "fit")
            for row in rows
        )
    ):
        raise Pi05EvaluationError("successful-expert equivalence selection changed")

    captured_tasks = []
    covered = set()
    for task in tasks:
        matching = [
            row
            for row in selected_rows
            if (str(row["suite"]), int(row["task_id"]))
            == (str(task.suite), int(task.task_id))
        ]
        if matching and any(
            row.get("language") != task.language for row in matching
        ):
            raise Pi05EvaluationError("successful-expert equivalence language changed")
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (str(task.suite), int(task.task_id), int(state_id)) in selected_keys
        )
        if state_ids:
            captured_tasks.append(replace(task, init_state_ids=state_ids))
            covered.update(
                (str(task.suite), int(task.task_id), int(state_id))
                for state_id in state_ids
            )
    if covered != selected_keys:
        raise Pi05EvaluationError(
            "successful-expert equivalence selection is outside development-train"
        )
    return tuple(captured_tasks), {
        "schema_version": SUCCESSFUL_EXPERT_EQUIVALENCE_CAPTURE_SCHEMA,
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "selected_step": step,
        "selected_rows": len(selected_rows),
        "selected_tasks": len(captured_tasks),
        "source_results": manifest.get("evaluation_results", {}).get(str(step)),
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "task_fold": manifest.get("selection_policy", {}).get("task_fold"),
        "training_gradient_use": False,
        "held_data_use": False,
        "claim_boundary": manifest.get("claim_boundary"),
    }


def _phase_decoder_fit_occupancy_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    writer_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    members = tuple(dict(row) for row in manifest.get("members", ()))
    fit_tasks = tuple(dict(row) for row in manifest.get("fit_tasks", ()))
    fit_members = tuple(row for row in members if row.get("fold_role") == "fit")
    keys = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in fit_members
    }
    fit_ordinals = tuple(int(row["ordinal"]) for row in fit_tasks)
    if (
        args.mode != "formal"
        or args.role != "development_train"
        or writer_kind != "task_expert"
        or getattr(args, "task_expert_projection_manifest", None) is None
        or len(members) != 47
        or len(fit_members) != 37
        or len(fit_tasks) != 19
        or len(keys) != 30
        or any(int(row["ordinal"]) % 5 == 0 for row in fit_members)
        or fit_ordinals
        != tuple(ordinal for ordinal in range(24) if ordinal % 5 != 0)
        or any(int(row["code_index"]) != index for index, row in enumerate(members))
    ):
        raise Pi05EvaluationError("phase-decoder fit occupancy selection changed")

    selected_tasks = []
    covered = set()
    for task in tasks:
        task_key = (str(task.suite), int(task.task_id))
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (*task_key, int(state_id)) in keys
        )
        if state_ids:
            selected_tasks.append(replace(task, init_state_ids=state_ids))
            covered.update((*task_key, int(state_id)) for state_id in state_ids)
    if covered != keys or len(selected_tasks) != 19:
        raise Pi05EvaluationError(
            "phase-decoder fit occupancy selection is outside development-train"
        )
    return tuple(selected_tasks), {
        "schema_version": PHASE_DECODER_FIT_OCCUPANCY_CAPTURE_SCHEMA,
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "selected_rows": len(keys),
        "selected_tasks": len(selected_tasks),
        "member_count": len(fit_members),
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": True,
        "gradient_scope": "fit19 phase decoder state aggregation only",
        "held_data_use": False,
        "validation_use": False,
        "test_use": False,
        "claim_boundary": (
            "Projected-policy trajectories provide fit-task learner occupancy; "
            "privileged experts are queried only after capture and held tasks are "
            "excluded."
        ),
    }


def _stage_predicate_capture(
    args: Any, occupancy_capture: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if not bool(getattr(args, "capture_stage_predicates", False)):
        return None
    diagnostic_subset = _task_expert_diagnostic_subset(occupancy_capture)
    if args.mode != "formal" or not (
        args.role == "validation"
        or (
            diagnostic_subset == "successful_on_policy_occupancy"
            and args.role == "nonheld_meta_train"
        )
        or (
            diagnostic_subset == "successful_expert_equivalence_occupancy"
            and args.role == "development_train"
        )
    ):
        raise Pi05EvaluationError(
            "stage-predicate capture requires a declared formal diagnostic panel"
        )
    return {
        "schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "post_settling_then_every_executed_action_change_points",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
        "training_gradient_use": False,
        "checkpoint_selection_use": False,
        "validation_action_reads": 0,
        "validation_reward_reads": 0,
        "held_data_use": args.role == "validation",
        "claim_boundary": (
            "BDDL goal predicates are a partial stage proxy, not a complete ordered "
            "procedure annotation"
        ),
    }


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
        and args.role
        not in {
            "all_targets",
            "nonheld_meta",
            "nonheld_meta_train",
            "nonheld_meta_validation",
        }
    ):
        raise Pi05EvaluationError("source-base screen must cover all 40 target tasks")
    installed_tasks, libero_paths = inspect_installed_target_tasks(
        authorities,
        role=args.role,
        state_count=args.state_count,
        libero_config_dir=staging / "libero_config",
    )
    subset_tasks, task_subset = _task_subset_tasks(
        args,
        installed_tasks,
        writer_kind=writer_kind,
    )
    tasks, occupancy_capture = _occupancy_capture_tasks(
        args,
        subset_tasks,
        output_dir=output_dir,
        writer_kind=writer_kind,
    )
    stage_predicate_capture = _stage_predicate_capture(args, occupancy_capture)
    model = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode=args.mode,
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    diagnostic_subset = _task_expert_diagnostic_subset(occupancy_capture)
    if diagnostic_subset is None and task_subset is not None:
        diagnostic_subset = str(task_subset["diagnostic_subset"])
    inspect_complete_bank = (
        diagnostic_subset is not None
        and writer_kind == "task_expert"
        and not (
            task_subset is not None
            and getattr(args, "task_expert_projection_manifest", None) is not None
        )
    )
    adapter = _inspect_adapter(
        args,
        writer_kind=writer_kind,
        source_sft_requested=source_sft_requested,
        authorities=authorities,
        model=model,
        tasks=installed_tasks if inspect_complete_bank else tasks,
    )
    if (
        diagnostic_subset is not None
        and adapter is not None
        and writer_kind == "task_expert"
    ):
        adapter = select_task_expert_adapter_tasks(
            adapter,
            tasks,
            diagnostic_subset=diagnostic_subset,
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
    contract["diagnostic_occupancy_capture"] = occupancy_capture
    contract["diagnostic_stage_predicates"] = stage_predicate_capture
    contract["diagnostic_task_subset"] = task_subset
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
