"""Static adapter inspection and evidence for canonical PI0.5 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError


STATIC_SOURCE_SFT_KIND = "shared_source_sft_lora"
STATIC_TASK_EXPERT_KIND = "task_local_expert_bank"
STATIC_TASK_LORA_KIND = "static_task_lora_bank"


def _all_or_none(values: Sequence[Any], label: str) -> bool:
    requested = tuple(value is not None for value in values)
    if any(requested) and not all(requested):
        raise Pi05EvaluationError(f"{label} evaluation requires all declared assets")
    return all(requested)


def source_sft_requested(args: Any) -> bool:
    return _all_or_none(
        (args.source_sft_config, args.source_sft_checkpoint), "Source-SFT"
    )


def task_expert_requested(args: Any) -> bool:
    requested = _all_or_none(
        (
            getattr(args, "task_expert_config", None),
            getattr(args, "task_expert_bank_root", None),
            getattr(args, "task_expert_step", None),
        ),
        "Task-Expert",
    )
    return requested


def static_task_lora_requested(args: Any) -> bool:
    return getattr(args, "static_task_lora_manifest", None) is not None


def adapter_requests(args: Any) -> tuple[str | None, bool]:
    source_requested = source_sft_requested(args)
    expert_requested = task_expert_requested(args)
    static_requested = static_task_lora_requested(args)
    if sum((source_requested, expert_requested, static_requested)) > 1:
        raise Pi05EvaluationError("PI05 evaluation adapters are mutually exclusive")
    kind = (
        "task_expert"
        if expert_requested
        else "static_task_lora" if static_requested else None
    )
    return kind, source_requested


def inspect_source_sft_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.source_sft.inference import inspect_source_sft_evaluation

    return inspect_source_sft_evaluation(
        config_path=config_path,
        checkpoint=checkpoint,
        source=source,
        task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
        evaluation_role=evaluation_role,
        require_formal=require_formal,
    )


def inspect_task_expert_adapter(
    *,
    config_path: Path,
    bank_root: Path,
    step: int,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.expert_manifold.contract import ExpertManifoldError
    from ember.expert_manifold.evaluation import inspect_task_expert_evaluation

    try:
        return inspect_task_expert_evaluation(
            config_path=config_path,
            bank_root=bank_root,
            step=step,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            evaluation_role=evaluation_role,
            require_formal=require_formal,
        )
    except ExpertManifoldError as error:
        raise Pi05EvaluationError(str(error)) from error


def inspect_static_task_lora_adapter(
    *,
    manifest_path: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.static_task_lora import inspect_static_task_lora_bank

    return inspect_static_task_lora_bank(
        manifest_path=manifest_path,
        source=source,
        task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
        evaluation_role=evaluation_role,
        require_formal=require_formal,
    )


def select_task_expert_adapter_tasks(
    adapter: Mapping[str, Any] | None,
    tasks: Sequence[Any],
    *,
    diagnostic_subset: str,
) -> dict[str, Any]:
    if adapter is None or adapter.get("kind") != STATIC_TASK_EXPERT_KIND:
        raise Pi05EvaluationError("diagnostic subset requires a task-expert adapter")
    task_rows = tuple(adapter.get("tasks", ()))
    records = {(str(row["suite"]), int(row["task_id"])): dict(row) for row in task_rows}
    keys = [(str(task.suite), int(task.task_id)) for task in tasks]
    if len(records) != len(task_rows) or any(key not in records for key in keys):
        raise Pi05EvaluationError("diagnostic task experts are incomplete")
    selected = dict(adapter)
    selected["tasks"] = [records[key] for key in keys]
    information_wall = dict(selected.get("information_wall", {}))
    information_wall.update(
        evaluated_task_count=len(keys),
        diagnostic_subset=diagnostic_subset,
        inspection_task_keys=[list(key) for key in records],
    )
    selected["information_wall"] = information_wall
    return selected


def load_evaluation_adapter(
    policy: Any,
    contract: Mapping[str, Any],
    *,
    device: Any,
) -> Any | None:
    adapter = contract.get("adapter")
    if adapter is None:
        return None
    common = {
        "policy": policy,
        "source": contract["model"],
        "evaluation_adapter": adapter,
        "task_keys": tuple(
            (str(row["suite"]), int(row["task_id"])) for row in contract["tasks"]
        ),
        "device": device,
        "require_formal": contract["mode"] != "smoke",
    }
    if adapter.get("kind") == STATIC_SOURCE_SFT_KIND:
        from ember.source_sft.inference import FrozenSourceSFTAdapter

        FrozenSourceSFTAdapter(**common)
        return None
    if adapter.get("kind") == STATIC_TASK_EXPERT_KIND:
        from ember.expert_manifold.evaluation import FrozenTaskExpertAdapter

        return FrozenTaskExpertAdapter(**common)
    if adapter.get("kind") == STATIC_TASK_LORA_KIND:
        from ember.static_task_lora import FrozenStaticTaskLoRAAdapter

        return FrozenStaticTaskLoRAAdapter(**common)
    raise Pi05EvaluationError("unsupported evaluation adapter kind")


def episode_adapter_fields(
    contract: Mapping[str, Any], task_adapter: Any | None, prepared: Any | None
) -> dict[str, Any]:
    if task_adapter is not None:
        if contract.get("adapter", {}).get("kind") == STATIC_TASK_LORA_KIND:
            return {"static_task_lora": dict(prepared.evidence)}
        return {"task_expert": dict(prepared.evidence)}
    adapter = contract.get("adapter")
    if adapter is not None and adapter.get("kind") == STATIC_SOURCE_SFT_KIND:
        return {"policy_adapter_sha256": adapter["lora_state_sha256"]}
    return {}


def validate_episode_adapter_fields(
    adapter: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if adapter is None:
        return (
            row.get("task_expert") is None
            and row.get("static_task_lora") is None
            and row.get("policy_adapter_sha256") is None
        )
    if adapter.get("kind") == STATIC_SOURCE_SFT_KIND:
        return (
            row.get("task_expert") is None
            and row.get("static_task_lora") is None
            and row.get("policy_adapter_sha256") == adapter.get("lora_state_sha256")
        )
    if adapter.get("kind") == STATIC_TASK_EXPERT_KIND:
        from ember.expert_manifold.evaluation import validate_task_expert_episode

        return (
            row.get("static_task_lora") is None
            and row.get("policy_adapter_sha256") is None
            and validate_task_expert_episode(
                adapter,
                row.get("task_expert"),
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
            )
        )
    if adapter.get("kind") == STATIC_TASK_LORA_KIND:
        from ember.static_task_lora import validate_static_task_lora_episode

        return (
            row.get("task_expert") is None
            and row.get("policy_adapter_sha256") is None
            and validate_static_task_lora_episode(
                adapter,
                row.get("static_task_lora"),
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
            )
        )
    return False
