"""Load one selected frozen task-local reward-adapted LoRA for fresh evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    canonical_contract_sha256,
    copy_task_lora_state_,
    inject_task_lora,
    load_lora_contract,
    lora_state_sha256,
    validate_lora_state,
)
from ember.source_base_checkpoint import canonical_hash, read_json, sha256_file
from ember.task_local_rl_checkpoint import verify_task_local_checkpoint
from ember.task_local_rl_protocol import (
    load_task_local_rl_config,
    select_adaptation_checkpoint,
)
from ember.writer.model import WriterModelError


class FrozenTaskLocalRLAdapter:
    """Install the adaptation-reward-selected LoRA for one task and arm."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        policy_files: Mapping[str, str],
        config_path: Path,
        run_root: Path,
        arm: str,
        task_ids: Sequence[int],
        device: torch.device,
        require_formal: bool,
    ) -> None:
        config = load_task_local_rl_config(config_path)
        if arm not in config["arms"]:
            raise WriterModelError(f"invalid task-local evaluation arm: {arm}")
        contract = load_lora_contract(
            config_path.parents[1] / config["protocol"]["lora_contract"]
        )
        run_contract_path = run_root / "run_contract.json"
        summary_path = run_root / "run_summary.json"
        run_contract = read_json(run_contract_path)
        summary = read_json(summary_path)
        requested = tuple(int(value) for value in task_ids)
        run_tasks = tuple(
            int(value) for value in run_contract.get("role", {}).get("task_ids", [])
        )
        if (
            run_contract.get("schema_version")
            != "ember_task_local_lora_rl_launch_v1"
            or run_contract.get("source_policy_files") != dict(policy_files)
            or not set(requested).issubset(run_tasks)
            or run_contract.get("trainable", {}).get("lora_contract_sha256")
            != canonical_contract_sha256(contract)
            or run_contract.get("runtime", {}).get("fixed_init_state_sampling")
            is not False
            or run_contract.get("runtime", {}).get(
                "matched_seed_schedule_excludes_arm"
            )
            is not True
            or summary.get("schema_version")
            != "ember_task_local_lora_rl_summary_v1"
            or summary.get("run_contract_sha256") != canonical_hash(run_contract)
        ):
            raise WriterModelError("task-local RL run authority changed")
        if require_formal and (
            run_contract.get("mode") != "formal"
            or run_contract.get("config_sha256") != sha256_file(config_path)
            or summary.get("complete") is not True
        ):
            raise WriterModelError(
                "formal evaluation requires a complete formal task-local RL run"
            )
        inject_task_lora(policy, contract)
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        self.policy = policy
        self.contract = contract
        self.run_root = run_root
        self.run_contract_sha256 = canonical_hash(run_contract)
        self.arm = arm
        self.device = device
        self.evidence = {
            "arm": arm,
            "run_contract_sha256": self.run_contract_sha256,
            "run_contract_file_sha256": sha256_file(run_contract_path),
            "run_summary_sha256": sha256_file(summary_path),
            "training_config_sha256": sha256_file(config_path),
            "writer_initialization": run_contract["writer_initialization"],
            "checkpoint_selection": run_contract["checkpoint_selection"],
            "official_random_reset_only": True,
            "fresh_fixed_state_evaluation_is_separate": True,
        }

    def _selected_checkpoint(self, task_id: int) -> Path:
        unit_dir = self.run_root / "units" / f"task_{task_id:03d}_{self.arm}"
        unit_contract = read_json(unit_dir / "unit_contract.json")
        selection = read_json(unit_dir / "selected_adaptation_checkpoint.json")
        candidates = selection.get("candidates", [])
        selected = selection.get("selected")
        if (
            unit_contract.get("run_contract_sha256")
            != self.run_contract_sha256
            or int(unit_contract.get("task_id", -1)) != task_id
            or unit_contract.get("arm") != self.arm
            or selection.get("schema_version")
            != "ember_task_local_lora_rl_selection_v1"
            or not isinstance(candidates, list)
            or not candidates
            or not isinstance(selected, Mapping)
            or canonical_hash(select_adaptation_checkpoint(candidates))
            != canonical_hash(selected)
        ):
            raise WriterModelError(
                f"task-local RL selection authority changed: {task_id} {self.arm}"
            )
        value = selected.get("path")
        if not isinstance(value, str) or not value:
            raise WriterModelError("task-local RL selected path is invalid")
        checkpoint = Path(value).resolve(strict=True)
        if checkpoint.parent != (unit_dir / "checkpoints").resolve(strict=True):
            raise WriterModelError("task-local RL checkpoint escaped its unit")
        manifest = verify_task_local_checkpoint(checkpoint)
        if (
            manifest.get("unit_contract_sha256") != canonical_hash(unit_contract)
            or sha256_file(checkpoint / "checkpoint_manifest.json")
            != selected.get("checkpoint_manifest_sha256")
            or int(manifest.get("next_update", -1))
            != int(selected.get("next_update", -2))
            or int(manifest.get("interaction_cursor", -1))
            != int(selected.get("interaction_cursor", -2))
        ):
            raise WriterModelError("task-local RL selected checkpoint changed")
        return checkpoint

    @torch.no_grad()
    def apply(self, task_id: int) -> str:
        checkpoint = self._selected_checkpoint(task_id)
        state = load_file(
            str(checkpoint / "lora.safetensors"), device=str(self.device)
        )
        validate_lora_state(state, self.contract)
        copy_task_lora_state_(self.policy, state, self.contract)
        return lora_state_sha256(state)
