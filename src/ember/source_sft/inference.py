"""Inspect and install one static shared PI05 Source-SFT LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    canonical_contract_sha256,
    copy_task_lora_state_,
    inject_task_lora,
    lora_state_sha256,
    validate_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file
from ember.source_sft.checkpoint import validate_source_sft_checkpoint_files
from ember.source_sft.contract import (
    SOURCE_SFT_LAUNCH_SCHEMA,
    Pi05SourceSFTError,
    REPO_ROOT,
    authority_path,
    load_source_sft_config,
)


STATIC_ADAPTER_KIND = "shared_source_sft_lora"
STATIC_ADAPTER_SCHEMA = "ember_pi05_source_sft_eval_adapter_v1"


def _validate_evaluation_role(stage: str, role: str) -> None:
    allowed = {
        "development": {"development_train", "seen_panel", "validation"},
        "final": {"final_source", "seen_panel", "test"},
    }
    if role not in allowed.get(stage, set()):
        raise Pi05SourceSFTError(
            f"Source-SFT {stage} artifact cannot be evaluated on role {role}"
        )


def _validate_task_keys(
    config: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    evaluation_role: str,
) -> None:
    if evaluation_role == "seen_panel":
        panel = read_json(REPO_ROOT / "configs/pi05_seen_panel_v1.json")
        panel_authority = panel.get("authority", {})
        manifest_authority = config["authorities"]["target_data_manifest"]
        if (
            panel.get("schema_version") != "ember_pi05_seen_panel_v1"
            or panel_authority.get("target_data_manifest")
            != manifest_authority["path"]
            or panel_authority.get("target_data_manifest_sha256")
            != manifest_authority["sha256"]
        ):
            raise Pi05SourceSFTError("Source-SFT seen-panel authority changed")
        expected = {
            (str(row["suite"]), int(row["task_id"])) for row in panel["tasks"]
        }
        observed = {(str(suite), int(task_id)) for suite, task_id in task_keys}
        if (
            len(expected) != len(observed)
            or len(observed) != len(task_keys)
            or observed != expected
        ):
            raise Pi05SourceSFTError("Source-SFT evaluation task panel changed")
        return
    role_splits = {
        "development_train": {"train"},
        "validation": {"validation"},
        "final_source": {"train", "validation"},
        "test": {"test"},
    }
    manifest = read_json(authority_path(config, "target_data_manifest"))
    expected = {
        (str(row["suite"]), int(row["task_id"]))
        for row in manifest["tasks"]
        if row["split_role"] in role_splits[evaluation_role]
    }
    observed = {(str(suite), int(task_id)) for suite, task_id in task_keys}
    if len(observed) != len(task_keys) or observed != expected:
        raise Pi05SourceSFTError("Source-SFT evaluation task panel changed")


def _validate_run_contract(
    *,
    run_contract: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    source: Mapping[str, Any],
    stage: str,
    lora: Any,
) -> None:
    valid = (
        run_contract.get("schema_version") == SOURCE_SFT_LAUNCH_SCHEMA
        and run_contract.get("config_sha256") == sha256_file(config_path)
        and stage == config.get("sealed_stage")
        and run_contract.get("source") == dict(source)
        and run_contract.get("authorities") == config["authorities"]
        and run_contract.get("information_wall") == config["information_wall"]
        and run_contract.get("stage_contract") == config["stages"][stage]
        and run_contract.get("trainable", {}).get("object")
        == "one_shared_multitask_pi05_lora_only"
        and int(run_contract.get("trainable", {}).get("per_task_adapters", -1)) == 0
        and run_contract.get("trainable", {}).get("lora_contract_sha256")
        == canonical_contract_sha256(lora)
    )
    if not valid:
        raise Pi05SourceSFTError("Source-SFT training authority or source linkage changed")


def _validate_checkpoint_contract(
    *,
    checkpoint: Path,
    run_contract: Mapping[str, Any],
    run_contract_sha: str,
    stage: str,
) -> tuple[dict[str, Any], int]:
    world_size = int(run_contract.get("runtime", {}).get("world_size", -1))
    manifest = validate_source_sft_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=run_contract_sha,
    )
    step = int(manifest.get("consumed", {}).get("next_step", -1))
    declared_steps = tuple(
        int(value) for value in run_contract.get("runtime", {}).get("checkpoint_steps", ())
    )
    if checkpoint.name != f"step_{step:08d}" or step not in declared_steps:
        raise Pi05SourceSFTError("Source-SFT evaluation checkpoint is not declared")
    if manifest.get("stage") != stage:
        raise Pi05SourceSFTError("Source-SFT checkpoint stage changed")
    return manifest, step


def _formal_summary_sha(
    *,
    require_formal: bool,
    run_root: Path,
    run_contract: Mapping[str, Any],
    run_contract_sha: str,
    config: Mapping[str, Any],
    stage: str,
    step: int,
) -> str | None:
    if not require_formal:
        return None
    world_size = int(run_contract.get("runtime", {}).get("world_size", -1))
    formal = config["stages"][stage]["formal_run"]
    if (
        run_contract.get("mode") != "formal"
        or formal.get("status") != "sealed"
        or world_size != 8
    ):
        raise Pi05SourceSFTError("formal evaluation requires a sealed formal Source-SFT run")
    summary_path = run_root / "run_summary.json"
    summary = read_json(summary_path)
    if (
        summary.get("schema_version") != "ember_pi05_source_sft_run_summary_v1"
        or summary.get("contract_sha256") != run_contract_sha
        or summary.get("stage") != stage
        or int(summary.get("completed_optimizer_steps", -1)) < step
        or int(summary.get("test_action_reads", -1)) != 0
    ):
        raise Pi05SourceSFTError("Source-SFT run summary changed")
    return sha256_file(summary_path)


def inspect_source_sft_evaluation(
    *,
    config_path: Path,
    checkpoint: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    checkpoint = checkpoint.resolve()
    config = load_source_sft_config(config_path)
    run_root = checkpoint.parent.parent
    if checkpoint.parent.name != "checkpoints":
        raise Pi05SourceSFTError("Source-SFT checkpoint is not owned by a training run")
    run_contract_path = run_root / "run_contract.json"
    run_contract = read_json(run_contract_path)
    run_contract_sha = canonical_hash(run_contract)
    stage = str(run_contract.get("stage", ""))
    _validate_evaluation_role(stage, evaluation_role)
    _validate_task_keys(config, task_keys, evaluation_role)
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    _validate_run_contract(
        run_contract=run_contract,
        config=config,
        config_path=config_path,
        source=source,
        stage=stage,
        lora=lora,
    )
    manifest, step = _validate_checkpoint_contract(
        checkpoint=checkpoint,
        run_contract=run_contract,
        run_contract_sha=run_contract_sha,
        stage=stage,
    )
    summary_sha = _formal_summary_sha(
        require_formal=require_formal,
        run_root=run_root,
        run_contract=run_contract,
        run_contract_sha=run_contract_sha,
        config=config,
        stage=stage,
        step=step,
    )
    lora_path = checkpoint / "lora.safetensors"
    state = load_file(str(lora_path), device="cpu")
    validate_lora_state(state, lora)
    state_sha = lora_state_sha256(state)
    return {
        "schema_version": STATIC_ADAPTER_SCHEMA,
        "kind": STATIC_ADAPTER_KIND,
        "arm": "source_sft",
        "execution_backend": "materialized_once_per_worker_batched_replan",
        "stage": stage,
        "evaluation_role": evaluation_role,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "training_run": {
            "root": str(run_root.resolve()),
            "run_contract_file_sha256": sha256_file(run_contract_path),
            "run_contract_sha256": run_contract_sha,
            "run_summary_sha256": summary_sha,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "step": step,
            "manifest_file_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "manifest_payload_sha256": manifest["canonical_payload_sha256"],
            "lora_file_sha256": sha256_file(lora_path),
        },
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "lora_state_sha256": state_sha,
        "shared_adapter_count": 1,
        "per_task_adapter_count": 0,
        "teacher_video_reads": 0,
        "test_action_reads": 0,
    }


class FrozenSourceSFTAdapter:
    """Install the static shared adapter exactly once at worker initialization."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        source: Mapping[str, Any],
        evaluation_adapter: Mapping[str, Any],
        task_keys: Sequence[tuple[str, int]],
        device: torch.device,
        require_formal: bool,
    ) -> None:
        observed = inspect_source_sft_evaluation(
            config_path=Path(evaluation_adapter["config"]["path"]),
            checkpoint=Path(evaluation_adapter["checkpoint"]["path"]),
            source=source,
            task_keys=task_keys,
            evaluation_role=str(evaluation_adapter["evaluation_role"]),
            require_formal=require_formal,
        )
        if observed != dict(evaluation_adapter):
            raise Pi05SourceSFTError("Source-SFT evaluation adapter changed after prepare")
        config = load_source_sft_config(Path(observed["config"]["path"]))
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        inject_task_lora(policy, lora)
        state = load_file(
            str(Path(observed["checkpoint"]["path"]) / "lora.safetensors"),
            device=str(device),
        )
        validate_lora_state(state, lora)
        if lora_state_sha256(state) != observed["lora_state_sha256"]:
            raise Pi05SourceSFTError("Source-SFT LoRA state changed during worker load")
        copy_task_lora_state_(policy, state, lora)
        policy.eval()
        self.evidence = observed
