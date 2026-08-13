"""Runtime dispatch and raw-row evidence for canonical PI05 evaluation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError


STATIC_SOURCE_SFT_KIND = "shared_source_sft_lora"
STATIC_TASK_EXPERT_KIND = "task_local_expert_bank"
EXPERT_MANIFOLD_WRITER_KIND = "expert_manifold_writer"
DYNAMIC_K_WRITER_KIND = "v6_dynamic_slot_set_writer"
WRITER_ADAPTER_KINDS = frozenset(
    {EXPERT_MANIFOLD_WRITER_KIND, DYNAMIC_K_WRITER_KIND}
)


def _all_or_none(values: Sequence[Any], label: str) -> bool:
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise Pi05EvaluationError(f"{label} evaluation requires all declared assets")
    return all(value is not None for value in values)


def source_sft_requested(args: Any) -> bool:
    return _all_or_none(
        (args.source_sft_config, args.source_sft_checkpoint), "Source-SFT"
    )


def task_expert_requested(args: Any) -> bool:
    return _all_or_none(
        (
            getattr(args, "task_expert_config", None),
            getattr(args, "task_expert_bank_root", None),
            getattr(args, "task_expert_step", None),
        ),
        "Task-Expert",
    )


def expert_manifold_writer_requested(args: Any) -> bool:
    if any(
        getattr(args, name, None) is not None
        for name in (
            "expert_manifold_expert_bank_root",
            "expert_manifold_feature_cache_root",
        )
    ):
        raise Pi05EvaluationError(
            "deployment expert-bank and feature-cache assets are retired"
        )
    return _all_or_none(
        (
            getattr(args, "expert_manifold_config", None),
            getattr(args, "expert_manifold_checkpoint", None),
            getattr(args, "expert_manifold_video_data_root", None),
            getattr(args, "expert_manifold_video_condition", None),
        ),
        "Expert-Manifold Writer",
    )


def dynamic_k_writer_requested(args: Any) -> bool:
    return _all_or_none(
        (
            getattr(args, "dynamic_k_writer_config", None),
            getattr(args, "dynamic_k_writer_checkpoint", None),
            getattr(args, "dynamic_k_writer_video_data_root", None),
            getattr(args, "dynamic_k_writer_video_condition", None),
        ),
        "Dynamic-K Writer",
    )


def adapter_requests(args: Any) -> tuple[str | None, bool]:
    sft_requested = source_sft_requested(args)
    expert_requested = task_expert_requested(args)
    manifold_requested = expert_manifold_writer_requested(args)
    dynamic_k_requested = dynamic_k_writer_requested(args)
    if sum(
        (sft_requested, expert_requested, manifold_requested, dynamic_k_requested)
    ) > 1:
        raise Pi05EvaluationError("PI05 evaluation adapters are mutually exclusive")
    kind = (
        "task_expert"
        if expert_requested
        else (
            EXPERT_MANIFOLD_WRITER_KIND
            if manifold_requested
            else DYNAMIC_K_WRITER_KIND if dynamic_k_requested else None
        )
    )
    return kind, sft_requested


def paired_writer_identity(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Return method-specific assets shared by correct/wrong Writer arms."""

    if (
        adapter.get("kind") not in WRITER_ADAPTER_KINDS
        or "video_data" not in adapter
    ):
        raise Pi05EvaluationError(
            "writer adapter lost its method-specific video authority"
        )
    keys = (
        "execution_backend",
        "config",
        "writer_asset",
        "evaluation_authority",
        "video_data",
        "lora_contract",
        "video_schedule",
        "pairing_reference",
    )
    return {key: adapter[key] for key in keys}


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
    from ember.expert_manifold.evaluation import inspect_task_expert_bank

    try:
        return inspect_task_expert_bank(
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


def inspect_expert_manifold_writer_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.expert_manifold.contract import ExpertManifoldError
    from ember.expert_manifold.inference import (
        inspect_expert_manifold_writer_evaluation,
    )

    try:
        return inspect_expert_manifold_writer_evaluation(
            config_path=config_path,
            checkpoint=checkpoint,
            video_data_root=video_data_root,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            video_condition=video_condition,
            video_seed=video_seed,
            video_sampling_mode=video_sampling_mode,
            require_formal=require_formal,
        )
    except ExpertManifoldError as error:
        raise Pi05EvaluationError(str(error)) from error


def inspect_dynamic_k_writer_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str,
    require_formal: bool,
    evaluation_k: int = 1,
) -> dict[str, Any]:
    from ember.writer.errors import WriterModelError
    from ember.writer.evaluation import inspect_dynamic_k_writer_evaluation

    try:
        return inspect_dynamic_k_writer_evaluation(
            config_path=config_path,
            checkpoint=checkpoint,
            video_data_root=video_data_root,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            video_condition=video_condition,
            video_seed=video_seed,
            video_sampling_mode=video_sampling_mode,
            require_formal=require_formal,
            evaluation_k=evaluation_k,
        )
    except WriterModelError as error:
        raise Pi05EvaluationError(str(error)) from error


def reinspect_writer_adapter(
    adapter: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    require_formal: bool,
) -> dict[str, Any]:
    """Rebuild one prepared Writer adapter from its immutable asset record."""

    kind = adapter.get("kind")
    common = {
        "config_path": Path(str(adapter["config"]["path"])),
        "checkpoint": Path(str(adapter["writer_asset"]["checkpoint"])),
        "video_data_root": Path(str(adapter["video_data"]["root"])),
        "source": source,
        "task_keys": task_keys,
        "video_condition": str(adapter["video_condition"]),
        "video_seed": int(adapter["video_schedule"]["seed"]),
        "video_sampling_mode": str(adapter["video_schedule"]["sampling_mode"]),
        "require_formal": require_formal,
    }
    if kind == EXPERT_MANIFOLD_WRITER_KIND:
        from ember.expert_manifold.inference import (
            inspect_expert_manifold_writer_evaluation,
        )

        return inspect_expert_manifold_writer_evaluation(**common)
    if kind == DYNAMIC_K_WRITER_KIND:
        from ember.writer.evaluation import inspect_dynamic_k_writer_evaluation

        common["evaluation_k"] = int(
            adapter.get("information_wall", {}).get("evaluation_k", 1)
        )
        return inspect_dynamic_k_writer_evaluation(**common)
    raise Pi05EvaluationError("retired Writer adapter kind")


def expected_writer_episode(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
    evidence_schema: str | None = None,
) -> dict[str, Any]:
    if adapter.get("kind") == EXPERT_MANIFOLD_WRITER_KIND:
        from ember.expert_manifold.inference import (
            expected_expert_manifold_episode_evidence,
        )

        result = expected_expert_manifold_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=lora_reference,
        )
    elif adapter.get("kind") == DYNAMIC_K_WRITER_KIND:
        from ember.writer.evaluation import expected_dynamic_k_episode_evidence

        result = expected_dynamic_k_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=lora_reference,
        )
    else:
        raise Pi05EvaluationError("retired Writer adapter kind")
    if evidence_schema is not None and result["schema_version"] != evidence_schema:
        raise Pi05EvaluationError("Writer episode evidence schema changed")
    return result


def validate_writer_episode(
    adapter: Mapping[str, Any],
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if adapter.get("kind") == EXPERT_MANIFOLD_WRITER_KIND:
        from ember.expert_manifold.inference import (
            validate_expert_manifold_episode_evidence,
        )

        return validate_expert_manifold_episode_evidence(
            adapter,
            row,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
        )
    if adapter.get("kind") == DYNAMIC_K_WRITER_KIND:
        from ember.writer.evaluation import validate_dynamic_k_episode_evidence

        return validate_dynamic_k_episode_evidence(
            adapter,
            row,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
        )
    return False


def writer_episode_schema(adapter: Mapping[str, Any]) -> str:
    if adapter.get("kind") == EXPERT_MANIFOLD_WRITER_KIND:
        from ember.expert_manifold.inference import expert_manifold_episode_schema

        return expert_manifold_episode_schema(adapter)
    if adapter.get("kind") == DYNAMIC_K_WRITER_KIND:
        from ember.writer.evaluation import dynamic_k_episode_schema

        return dynamic_k_episode_schema(adapter)
    raise Pi05EvaluationError("retired Writer adapter kind")


def load_evaluation_adapter(
    policy: Any,
    contract: Mapping[str, Any],
    *,
    device: Any,
    writer_generation: bool = False,
) -> Any | None:
    """Install one static shared adapter, or return a per-rollout Writer adapter."""

    adapter = contract.get("adapter")
    if adapter is None:
        return None
    task_keys = tuple(
        (str(row["suite"]), int(row["task_id"])) for row in contract["tasks"]
    )
    common = {
        "policy": policy,
        "source": contract["model"],
        "evaluation_adapter": adapter,
        "task_keys": task_keys,
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
    if adapter.get("kind") not in WRITER_ADAPTER_KINDS:
        raise Pi05EvaluationError("retired evaluation adapter kind")
    common["tokenizer_path"] = Path(contract["tokenizer"]["path"])
    if writer_generation:
        if adapter.get("kind") == EXPERT_MANIFOLD_WRITER_KIND:
            from ember.expert_manifold.live_adapter import (
                FrozenExpertManifoldTaskAdapter,
            )

            return FrozenExpertManifoldTaskAdapter(**common)
        from ember.writer.live_adapter import FrozenDynamicKTaskAdapter

        return FrozenDynamicKTaskAdapter(**common)
    from ember.writer.evaluation_runtime import FrozenCachedWriterTaskAdapter

    common["cache_contract"] = contract
    return FrozenCachedWriterTaskAdapter(**common)


def episode_adapter_fields(
    contract: Mapping[str, Any], task_adapter: Any | None, prepared: Any | None
) -> dict[str, Any]:
    if task_adapter is not None:
        if contract.get("adapter", {}).get("kind") == STATIC_TASK_EXPERT_KIND:
            return {"task_expert": dict(prepared.evidence)}
        return {"writer": dict(prepared.evidence)}
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
        return row.get("writer") is None and row.get("policy_adapter_sha256") is None
    if adapter.get("kind") == STATIC_SOURCE_SFT_KIND:
        return row.get("writer") is None and row.get(
            "policy_adapter_sha256"
        ) == adapter.get("lora_state_sha256")
    if adapter.get("kind") == STATIC_TASK_EXPERT_KIND:
        from ember.expert_manifold.evaluation import validate_task_expert_episode

        return (
            row.get("writer") is None
            and row.get("policy_adapter_sha256") is None
            and validate_task_expert_episode(
                adapter,
                row.get("task_expert"),
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
            )
        )
    return row.get("policy_adapter_sha256") is None and validate_writer_episode(
        adapter,
        row.get("writer"),
        suite=suite,
        task_id=task_id,
        init_state_id=init_state_id,
    )
