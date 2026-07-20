"""Load frozen direct-LoRA task artifacts into the shared source policy."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.direct_lora_checkpoint import verify_checkpoint_files
from ember.direct_lora_protocol import load_direct_lora_config
from ember.lora import (
    canonical_contract_sha256,
    copy_task_lora_state_,
    inject_task_lora,
    load_lora_contract,
    lora_state_sha256,
    validate_lora_state,
)
from ember.source_base_checkpoint import canonical_hash, read_json, sha256_file


class DirectLoRAInferenceError(RuntimeError):
    """Raised when a direct-LoRA evaluation artifact changed."""


class FrozenDirectLoRAAdapter:
    """Install one independently trained task-local LoRA before each rollout."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        policy_files: Mapping[str, str],
        config_path: Path,
        run_root: Path,
        task_ids: Sequence[int],
        device: torch.device,
        require_formal: bool,
    ) -> None:
        config = load_direct_lora_config(config_path)
        lora_contract = load_lora_contract(
            config_path.parents[1] / config["protocol"]["lora_contract"]
        )
        run_contract_path = run_root / "run_contract.json"
        run_contract = read_json(run_contract_path)
        run_task_ids = tuple(int(value) for value in run_contract.get("task_ids", []))
        requested_task_ids = tuple(int(value) for value in task_ids)
        if (
            run_contract.get("schema_version")
            != "ember_direct_lora_sft_launch_v1"
            or run_contract.get("source_policy_files") != dict(policy_files)
            or not set(requested_task_ids).issubset(run_task_ids)
            or run_contract.get("trainable", {}).get("lora_contract_sha256")
            != canonical_contract_sha256(lora_contract)
        ):
            raise DirectLoRAInferenceError("direct-LoRA run authority changed")
        if require_formal and (
            run_contract.get("mode") != "formal"
            or run_contract.get("config_sha256") != sha256_file(config_path)
        ):
            raise DirectLoRAInferenceError(
                "formal evaluation requires a formal direct-LoRA run"
            )
        inject_task_lora(policy, lora_contract)
        policy.eval()
        self.policy = policy
        self.contract = lora_contract
        self.run_root = run_root
        self.run_contract = run_contract
        self.run_contract_sha256 = canonical_hash(run_contract)
        self.device = device
        self.require_formal = require_formal
        summary_path = run_root / "run_summary.json"
        self.evidence = {
            "run_contract_sha256": self.run_contract_sha256,
            "run_contract_file_sha256": sha256_file(run_contract_path),
            "run_summary_sha256": (
                sha256_file(summary_path) if summary_path.is_file() else None
            ),
            "per_task_total_steps": int(
                run_contract["runtime"]["per_task_total_steps"]
            ),
            "per_task_consumed_queries": int(
                run_contract["runtime"]["per_task_consumed_queries"]
            ),
        }

    @torch.no_grad()
    def apply(self, task_id: int) -> str:
        task_dir = self.run_root / "tasks" / f"task_{task_id:03d}"
        task_contract = read_json(task_dir / "task_contract.json")
        latest = read_json(task_dir / "latest_checkpoint.json")
        latest_path = latest.get("path")
        if not isinstance(latest_path, str) or not latest_path:
            raise DirectLoRAInferenceError(
                f"direct-LoRA latest checkpoint is invalid: {task_id}"
            )
        checkpoint = Path(latest_path).resolve(strict=True)
        if checkpoint.parent != (task_dir / "checkpoints").resolve(strict=True):
            raise DirectLoRAInferenceError(
                f"direct-LoRA checkpoint escaped task directory: {task_id}"
            )
        manifest = verify_checkpoint_files(checkpoint)
        expected_step = int(self.run_contract["runtime"]["per_task_total_steps"])
        actual_step = int(manifest.get("consumed", {}).get("next_step", -1))
        if (
            task_contract.get("run_contract_sha256")
            != self.run_contract_sha256
            or int(task_contract.get("task_id", -1)) != task_id
            or manifest.get("task_contract_sha256")
            != canonical_hash(task_contract)
            or int(latest.get("step", -1)) != actual_step
            or (self.require_formal and actual_step != expected_step)
        ):
            raise DirectLoRAInferenceError(
                f"direct-LoRA task artifact changed: {task_id}"
            )
        state = load_file(
            str(checkpoint / "lora.safetensors"), device=str(self.device)
        )
        validate_lora_state(state, self.contract)
        copy_task_lora_state_(self.policy, state, self.contract)
        return lora_state_sha256(state)
