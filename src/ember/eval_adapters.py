"""Runtime dispatch and raw-row evidence for canonical PI05 evaluation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError


STATIC_SOURCE_SFT_KIND = "shared_source_sft_lora"


def _all_or_none(values: Sequence[Any], label: str) -> bool:
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise Pi05EvaluationError(f"{label} evaluation requires all declared assets")
    return all(value is not None for value in values)


def as_writer_requested(args: Any) -> bool:
    return _all_or_none(
        (
            args.as_writer_config,
            args.as_writer_checkpoint,
            args.writer_video_data_root,
            args.writer_video_condition,
        ),
        "AS-Writer",
    )


def rl_writer_requested(args: Any) -> bool:
    return _all_or_none(
        (
            args.rl_writer_config,
            args.rl_writer_checkpoint,
            args.rl_writer_feature_cache,
            args.rl_writer_video_condition,
        ),
        "RL-Writer",
    )


def source_sft_requested(args: Any) -> bool:
    return _all_or_none(
        (args.source_sft_config, args.source_sft_checkpoint), "Source-SFT"
    )


def adapter_requests(args: Any) -> tuple[str | None, bool]:
    as_requested = as_writer_requested(args)
    rl_requested = rl_writer_requested(args)
    sft_requested = source_sft_requested(args)
    if sum((as_requested, rl_requested, sft_requested)) > 1:
        raise Pi05EvaluationError("PI05 evaluation adapters are mutually exclusive")
    kind = "as_writer" if as_requested else "rl_writer" if rl_requested else None
    return kind, sft_requested


def paired_writer_identity(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Return method-specific assets shared by correct/wrong Writer arms."""

    data_key = {
        "as_writer": "video_data",
        "rl_writer": "feature_cache",
    }.get(str(adapter.get("kind")))
    if data_key is None or data_key not in adapter:
        raise Pi05EvaluationError(
            "writer adapter lost its method-specific video authority"
        )
    keys = (
        "execution_backend",
        "config",
        "training_run",
        "checkpoint",
        data_key,
        "lora_contract_sha256",
        "video_schedule",
        "pairing_sha256",
    )
    return {key: adapter[key] for key in keys}


def inspect_as_writer_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    video_condition: str,
    video_seed: int,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.lora import LoRAContractError
    from ember.writer.inference import inspect_as_writer_evaluation
    from ember.writer.model import WriterModelError

    try:
        return inspect_as_writer_evaluation(
            config_path=config_path,
            checkpoint=checkpoint,
            video_data_root=video_data_root,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            video_condition=video_condition,
            video_seed=video_seed,
            require_formal=require_formal,
        )
    except (LoRAContractError, WriterModelError) as error:
        raise Pi05EvaluationError(str(error)) from error


def inspect_rl_writer_adapter(
    *,
    config_path: Path,
    checkpoint: Path,
    feature_cache: Path,
    source: Mapping[str, Any],
    tasks: Sequence[Any],
    video_condition: str,
    video_seed: int,
    require_formal: bool,
) -> dict[str, Any]:
    from ember.reward.protocol import RewardProtocolError
    from ember.rl_writer.inference import inspect_rl_writer_evaluation
    from ember.writer.feature_cache import FeatureCacheError
    from ember.writer.model import WriterModelError

    try:
        return inspect_rl_writer_evaluation(
            config_path=config_path,
            checkpoint=checkpoint,
            feature_cache=feature_cache,
            source=source,
            task_keys=tuple((task.suite, int(task.task_id)) for task in tasks),
            video_condition=video_condition,
            video_seed=video_seed,
            require_formal=require_formal,
        )
    except (FeatureCacheError, RewardProtocolError, WriterModelError) as error:
        raise Pi05EvaluationError(str(error)) from error


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


def load_evaluation_adapter(
    policy: Any, contract: Mapping[str, Any], *, device: Any
) -> Any | None:
    """Install one static shared adapter, or return a per-rollout Writer adapter."""

    adapter = contract.get("adapter")
    if adapter is None:
        return None
    task_keys = tuple((str(row["suite"]), int(row["task_id"])) for row in contract["tasks"])
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
    from ember.writer.inference import FrozenWriterTaskAdapter

    common["tokenizer_path"] = Path(contract["tokenizer"]["path"])
    return FrozenWriterTaskAdapter(**common)


def episode_adapter_fields(
    contract: Mapping[str, Any], task_adapter: Any | None, prepared: Any | None
) -> dict[str, Any]:
    if task_adapter is not None:
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
        return (
            row.get("writer") is None
            and row.get("policy_adapter_sha256") == adapter.get("lora_state_sha256")
        )
    from ember.writer.inference import validate_writer_episode_evidence

    return row.get("policy_adapter_sha256") is None and validate_writer_episode_evidence(
        adapter,
        row.get("writer"),
        suite=suite,
        task_id=task_id,
        init_state_id=init_state_id,
    )
