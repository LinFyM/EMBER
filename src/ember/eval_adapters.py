"""Runtime dispatch and raw-row evidence for canonical PI05 evaluation adapters."""

from __future__ import annotations

from typing import Any, Mapping


STATIC_SOURCE_SFT_KIND = "shared_source_sft_lora"


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
