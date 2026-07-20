"""Load one frozen Writer checkpoint and materialize task-local LoRA states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    canonical_contract_sha256,
    copy_task_lora_state_,
    lora_state_sha256,
    load_lora_contract,
    validate_lora_state,
)
from ember.source_base_checkpoint import canonical_hash, read_json, sha256_file
from ember.writer.feature_cache import WriterFeatureStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)
from ember.writer.training import load_writer_config
from ember.writer_rl_protocol import load_writer_rl_config


def _verify_checkpoint_files(checkpoint: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files", {})
    if not isinstance(files, Mapping) or "writer.safetensors" not in files:
        raise WriterModelError("Writer checkpoint manifest is incomplete")
    for name, record in files.items():
        path = checkpoint / str(name)
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise WriterModelError(f"Writer checkpoint file changed: {name}")


def _resolve_writer_stage(
    *,
    training_contract: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    writer_config: Mapping[str, Any],
    writer_config_path: Path,
    writer_rl_config_path: Path | None,
    policy_files: Mapping[str, str],
    lora_contract_sha256: str,
    require_formal: bool,
) -> tuple[str, str, int]:
    schema = training_contract.get("schema_version")
    common_changed = (
        training_contract.get("source_policy_files") != dict(policy_files)
        or training_contract.get("trainable", {}).get("lora_contract_sha256")
        != lora_contract_sha256
        or checkpoint_manifest.get("contract_sha256")
        != canonical_hash(training_contract)
    )
    if schema == "ember_writer_cold_start_launch_v1":
        if (
            writer_rl_config_path is not None
            or training_contract.get("writer") != writer_config["writer"]
            or common_changed
        ):
            raise WriterModelError("cold-start Writer checkpoint authority changed")
        config_sha256 = sha256_file(writer_config_path)
        cursor = int(checkpoint_manifest.get("consumed", {}).get("next_step", -1))
        stage = "cold_start"
    elif schema == "ember_writer_only_rl_launch_v1":
        if writer_rl_config_path is None:
            raise WriterModelError(
                "Writer-only RL checkpoint requires its sealed training config"
            )
        writer_rl_config = load_writer_rl_config(writer_rl_config_path)
        trainable = training_contract.get("trainable", {})
        if (
            common_changed
            or training_contract.get("protocol") != writer_rl_config["protocol"]
            or training_contract.get("algorithm") != writer_rl_config["algorithm"]
            or writer_rl_config["protocol"].get("writer_config_sha256")
            != sha256_file(writer_config_path)
            or trainable.get("object") != "shared_writer_only"
            or int(trainable.get("source_policy_parameter_count", -1)) != 0
            or int(trainable.get("generated_lora_in_place_parameter_count", -1))
            != 0
            or int(trainable.get("critic_parameter_count", -1)) != 0
            or training_contract.get("runtime", {}).get(
                "generated_lora_in_place_updates"
            )
            is not False
            or training_contract.get("environment", {}).get(
                "fixed_pruned_init_states"
            )
            is not False
            or checkpoint_manifest.get("schema_version")
            != "ember_writer_only_rl_checkpoint_v1"
        ):
            raise WriterModelError("Writer-only RL checkpoint authority changed")
        config_sha256 = sha256_file(writer_rl_config_path)
        cursor = int(checkpoint_manifest.get("consumed", {}).get("next_update", -1))
        stage = "writer_only_rl"
    else:
        raise WriterModelError("unsupported Writer checkpoint stage")
    if cursor <= 0:
        raise WriterModelError("Writer checkpoint cursor is invalid")
    if require_formal and (
        training_contract.get("mode") != "formal"
        or training_contract.get("config_sha256") != config_sha256
    ):
        raise WriterModelError("formal evaluation requires a formal Writer checkpoint")
    return stage, config_sha256, cursor


class FrozenWriterTaskAdapter:
    """Generate and install one complete Writer LoRA for each evaluated task."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        policy_files: Mapping[str, str],
        writer_config_path: Path,
        writer_checkpoint: Path,
        feature_cache: Path,
        task_ids: Sequence[int],
        device: torch.device,
        require_formal: bool,
        writer_rl_config_path: Path | None = None,
    ) -> None:
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise WriterModelError("Writer inference task IDs must be unique")
        config = load_writer_config(writer_config_path)
        contract = load_lora_contract(
            writer_config_path.parents[1] / config["protocol"]["lora_contract"]
        )
        template = prepare_frozen_writer_policy(policy, contract)
        writer = CompleteLoRAWriter(
            build_lora_tensor_specs(template),
            template_state=template,
            **config["writer"],
        ).to(device)

        checkpoint_manifest = read_json(
            writer_checkpoint / "checkpoint_manifest.json"
        )
        training_contract_path = writer_checkpoint.parent.parent / "run_contract.json"
        training_contract = read_json(training_contract_path)
        stage, training_config_sha256, checkpoint_cursor = _resolve_writer_stage(
            training_contract=training_contract,
            checkpoint_manifest=checkpoint_manifest,
            writer_config=config,
            writer_config_path=writer_config_path,
            writer_rl_config_path=writer_rl_config_path,
            policy_files=policy_files,
            lora_contract_sha256=canonical_contract_sha256(contract),
            require_formal=require_formal,
        )
        _verify_checkpoint_files(writer_checkpoint, checkpoint_manifest)
        writer.load_state_dict(
            load_file(
                str(writer_checkpoint / "writer.safetensors"), device=str(device)
            ),
            strict=True,
        )
        writer.eval()
        for parameter in writer.parameters():
            parameter.requires_grad_(False)

        cache_contract_path = feature_cache / "run_contract.json"
        cache_manifest_path = feature_cache / "cache_manifest.json"
        cache_contract = read_json(cache_contract_path)
        cache_manifest = read_json(cache_manifest_path)
        expected_task_ids = tuple(int(value) for value in task_ids)
        cached_task_ids = tuple(
            int(value) for value in cache_contract.get("task_ids", [])
        )
        record_task_ids = tuple(
            sorted(
                int(record["task_id"])
                for record in cache_manifest.get("task_records", [])
            )
        )
        if (
            cache_contract.get("schema_version")
            != "ember_writer_feature_cache_launch_v1"
            or cache_contract.get("mode") != "formal"
            or not set(expected_task_ids).issubset(cached_task_ids)
            or tuple(cache_contract.get("demo_indices", [])) != tuple(range(50))
            or cache_contract.get("policy_files")
            != {
                name: policy_files[name]
                for name in ("config.json", "model.safetensors")
            }
            or cache_manifest.get("schema_version")
            != "ember_writer_feature_cache_manifest_v1"
            or cache_manifest.get("contract_sha256")
            != cache_contract.get("contract_sha256")
            or cache_manifest.get("extraction_sha256")
            != cache_contract.get("extraction_sha256")
            or int(cache_manifest.get("task_count", -1)) != len(cached_task_ids)
            or int(cache_manifest.get("episode_count", -1))
            != 50 * len(cached_task_ids)
            or record_task_ids != tuple(sorted(cached_task_ids))
        ):
            raise WriterModelError("Writer validation feature-cache authority changed")

        self.policy = policy
        self.contract = contract
        self.writer = writer
        self.device = device
        self.store = WriterFeatureStore(
            feature_cache,
            task_ids=cached_task_ids,
            expected_extraction_sha256=str(cache_contract["extraction_sha256"]),
            max_cached_tasks=1,
            expected_dim=int(config["writer"]["vision_feature_dim"]),
        )
        self.evidence = {
            "stage": stage,
            "checkpoint_cursor": checkpoint_cursor,
            "checkpoint_manifest_sha256": sha256_file(
                writer_checkpoint / "checkpoint_manifest.json"
            ),
            "training_contract_sha256": canonical_hash(training_contract),
            "training_config_sha256": training_config_sha256,
            "writer_state_sha256": checkpoint_manifest["files"]["writer.safetensors"][
                "sha256"
            ],
            "feature_cache_contract_sha256": cache_contract["contract_sha256"],
            "feature_cache_manifest_sha256": sha256_file(cache_manifest_path),
            "feature_cache_extraction_sha256": cache_contract["extraction_sha256"],
            "lora_contract_sha256": canonical_contract_sha256(contract),
        }

    @torch.no_grad()
    def apply(self, task_id: int) -> str:
        cached = self.store.load(task_id)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.writer(
                cached.language_features.to(self.device),
                cached.video_features.to(self.device),
                cached.episode_offsets.to(self.device),
            )
        validate_lora_state(state, self.contract)
        copy_task_lora_state_(self.policy, state, self.contract)
        return lora_state_sha256(state)
