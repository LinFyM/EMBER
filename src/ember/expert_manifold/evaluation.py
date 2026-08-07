"""Inspect and execute one complete development-train task-expert bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.checkpoint import CHECKPOINT_SCHEMA
from ember.expert_manifold.contract import (
    CONFIG_SCHEMA,
    ExpertManifoldError,
    authority_path,
    load_expert_manifold_config,
)
from ember.lora import (
    copy_task_lora_state_,
    inject_task_lora,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json


TASK_EXPERT_ADAPTER_KIND = "task_local_expert_bank"
TASK_EXPERT_ADAPTER_SCHEMA = "ember_pi05_task_expert_eval_adapter_v1"
TASK_EXPERT_EPISODE_SCHEMA = "ember_pi05_task_expert_episode_v1"


def _train_task_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = [dict(row) for row in manifest.get("tasks", []) if row.get("split_role") == "train"]
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != int(config["task_experts"]["task_count"]):
        raise ExpertManifoldError("task-expert evaluation did not resolve train24")
    return tuple(rows)


def _source_paths_match(
    worker_source: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    return (
        Path(str(worker_source.get("run", ""))).resolve()
        == Path(str(source.get("source_run", ""))).resolve()
        and Path(str(worker_source.get("checkpoint", ""))).resolve()
        == Path(str(source.get("checkpoint", ""))).resolve()
        and Path(str(worker_source.get("model_path", ""))).resolve()
        == Path(str(source.get("model_path", ""))).resolve()
    )


def inspect_task_expert_bank(
    *,
    config_path: Path,
    bank_root: Path,
    step: int,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    """Seal path/schema/size evidence for all 24 task-local expert adapters."""

    config_path = config_path.resolve()
    bank_root = bank_root.resolve()
    config = load_expert_manifold_config(config_path)
    formal = config["task_experts"]["formal_run"]
    if evaluation_role != "development_train":
        raise ExpertManifoldError("task experts may only evaluate development_train")
    if require_formal and formal.get("status") != "sealed":
        raise ExpertManifoldError("formal task-expert evaluation requires a sealed profile")
    checkpoints = tuple(int(value) for value in formal["checkpoint_steps"])
    if step not in checkpoints or step <= 0:
        raise ExpertManifoldError("task-expert evaluation step is not declared")

    expected_rows = _train_task_rows(config)
    expected_by_key = {
        (str(row["suite"]), int(row["task_id"])): (ordinal, row)
        for ordinal, row in enumerate(expected_rows)
    }
    observed_keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(observed_keys)) != len(observed_keys) or set(observed_keys) != set(expected_by_key):
        raise ExpertManifoldError("task-expert evaluation panel differs from train24")

    workers = tuple(sorted(path for path in bank_root.glob("worker_*") if path.is_dir()))
    expected_worker_count = int(formal["allowed_worker_count"])
    if len(workers) != expected_worker_count:
        raise ExpertManifoldError("task-expert bank worker count is incomplete")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    task_records: dict[tuple[str, int], dict[str, Any]] = {}
    training_commits: set[str] = set()
    physical_devices: list[dict[str, Any]] = []
    expected_config_path = config_path.resolve()

    for worker in workers:
        contract_path = worker / "run_contract.json"
        summary_path = worker / "worker_summary.json"
        contract = read_json(contract_path)
        summary = read_json(summary_path)
        worker_tasks = tuple(contract.get("tasks", ()))
        runtime = contract.get("runtime", {})
        if (
            contract.get("schema_version") != "ember_pi05_task_expert_worker_launch_v1"
            or contract.get("mode") != "formal"
            or contract.get("content_hash_policy") != "disabled_by_owner"
            or Path(str(contract.get("config", {}).get("path", ""))).resolve()
            != expected_config_path
            or contract.get("config", {}).get("schema") != CONFIG_SCHEMA
            or not _source_paths_match(contract.get("source", {}), source)
            or len(worker_tasks) != int(formal["tasks_per_worker"])
            or int(runtime.get("per_task_batch_size", -1))
            != int(formal["per_task_batch_size"])
            or runtime.get("task_parameter_sharing") != "none"
            or summary.get("schema_version")
            != "ember_pi05_task_expert_worker_summary_v1"
            or int(summary.get("completed_task_count", -1)) != len(worker_tasks)
            or int(summary.get("selected_stop_step", -1)) < step
        ):
            raise ExpertManifoldError("task-expert worker formal contract changed")
        summary_rows = {
            int(row.get("task_ordinal", -1)): row for row in summary.get("tasks", ())
        }
        if len(summary_rows) != len(worker_tasks):
            raise ExpertManifoldError("task-expert worker summary ownership changed")
        training_commits.add(str(contract.get("git", {}).get("commit", "")))
        physical_devices.append(
            {
                "worker": worker.name,
                "host": runtime.get("host"),
                "physical_gpu": runtime.get("cuda_visible_device"),
                "device": runtime.get("device_name"),
            }
        )
        for declared in worker_tasks:
            key = (str(declared["suite"]), int(declared["task_id"]))
            expected = expected_by_key.get(key)
            if expected is None or key in task_records:
                raise ExpertManifoldError("task-expert bank task ownership overlaps")
            ordinal, row = expected
            global_task_id = int(row["global_task_id"])
            summary_row = summary_rows.get(ordinal, {})
            checkpoint = (
                worker
                / f"task_{ordinal:02d}_global_{global_task_id:02d}"
                / "checkpoints"
                / f"step_{step:08d}"
            )
            manifest_path = checkpoint / "manifest.json"
            manifest = read_json(manifest_path)
            files = manifest.get("files", {})
            valid_declared = (
                int(declared.get("ordinal", -1)) == ordinal
                and int(declared.get("global_task_id", -1)) == global_task_id
                and declared.get("split_role") == "train"
                and declared.get("language") == row["language"]
                and int(summary_row.get("global_task_id", -1)) == global_task_id
                and int(summary_row.get("completed_steps", -1)) >= step
            )
            valid_checkpoint = (
                manifest.get("schema_version") == CHECKPOINT_SCHEMA
                and int(manifest.get("step", -1)) == step
                and int(manifest.get("task_ordinal", -1)) == ordinal
                and int(manifest.get("global_task_id", -1)) == global_task_id
                and int(manifest.get("state_tensor_count", -1)) == lora.state_tensor_count
                and int(manifest.get("state_parameter_count", -1)) == lora.parameter_count
                and manifest.get("content_hash_policy") == "disabled_by_owner"
            )
            if not valid_declared or not valid_checkpoint:
                raise ExpertManifoldError("task-expert checkpoint ownership changed")
            for name in ("adapter.safetensors", "trainer.pt"):
                path = checkpoint / name
                if not path.is_file() or path.stat().st_size != int(files.get(name, -1)):
                    raise ExpertManifoldError("task-expert checkpoint file size changed")
            task_records[key] = {
                "suite": key[0],
                "task_id": key[1],
                "ordinal": ordinal,
                "global_task_id": global_task_id,
                "language": str(row["language"]),
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "manifest_bytes": manifest_path.stat().st_size,
                "adapter_bytes": (checkpoint / "adapter.safetensors").stat().st_size,
            }

    if len(training_commits) != 1 or "" in training_commits or set(task_records) != set(expected_by_key):
        raise ExpertManifoldError("task-expert bank is not one complete formal family")
    return {
        "schema_version": TASK_EXPERT_ADAPTER_SCHEMA,
        "kind": TASK_EXPERT_ADAPTER_KIND,
        "arm": f"task_expert_step_{step}",
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": CONFIG_SCHEMA,
        },
        "bank_root": str(bank_root),
        "training_commit": next(iter(training_commits)),
        "step": step,
        "source": {
            "source_run": str(Path(str(source["source_run"])).resolve()),
            "checkpoint": str(Path(str(source["checkpoint"])).resolve()),
            "model_path": str(Path(str(source["model_path"])).resolve()),
        },
        "lora_contract": {
            "reference": (
                f"{authority_path(config, 'lora_contract').relative_to(config_path.parents[1])}:"
                f"{lora.state_tensor_count}tensors:{lora.parameter_count}parameters"
            ),
            "rank": lora.rank,
            "target_count": len(lora.targets),
        },
        "tasks": [task_records[key] for key in sorted(task_records)],
        "workers": physical_devices,
        "information_wall": {
            "evaluation_role": "development_train",
            "validation_experts": 0,
            "test_experts": 0,
            "validation_actions_read": 0,
            "test_actions_read": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }


@dataclass(frozen=True)
class PreparedTaskExpert:
    key: tuple[str, int]
    evidence: dict[str, Any]


class FrozenTaskExpertAdapter:
    """Install the declared task-local expert only when a worker changes task."""

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
        observed = inspect_task_expert_bank(
            config_path=Path(str(evaluation_adapter["config"]["path"])),
            bank_root=Path(str(evaluation_adapter["bank_root"])),
            step=int(evaluation_adapter["step"]),
            source=source,
            task_keys=task_keys,
            evaluation_role="development_train",
            require_formal=require_formal,
        )
        if observed != evaluation_adapter:
            raise ExpertManifoldError("task-expert evaluation adapter changed at runtime")
        config = load_expert_manifold_config(Path(str(observed["config"]["path"])))
        self.lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.policy = policy
        self.device = device
        self.records = {
            (str(row["suite"]), int(row["task_id"])): dict(row)
            for row in observed["tasks"]
        }
        self._states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._installed: tuple[str, int] | None = None

    def _state(self, key: tuple[str, int]) -> dict[str, torch.Tensor]:
        if key not in self._states:
            path = Path(self.records[key]["checkpoint"]) / "adapter.safetensors"
            state = load_file(str(path), device="cpu")
            validate_lora_state(state, self.lora)
            self._states[key] = state
        return self._states[key]

    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedTaskExpert:
        key = (str(suite), int(task_id))
        if key not in self.records:
            raise ExpertManifoldError("rollout task is outside the expert bank")
        row = self.records[key]
        return PreparedTaskExpert(
            key=key,
            evidence={
                "schema_version": TASK_EXPERT_EPISODE_SCHEMA,
                **row,
                "init_state_id": int(init_state_id),
            },
        )

    @torch.no_grad()
    def install(self, prepared: PreparedTaskExpert) -> None:
        if prepared.key == self._installed:
            return
        copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)
        self._installed = prepared.key


def validate_task_expert_episode(
    adapter: Mapping[str, Any],
    evidence: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    records = {
        (str(row["suite"]), int(row["task_id"])): row for row in adapter.get("tasks", ())
    }
    row = records.get((str(suite), int(task_id)))
    expected = (
        {
            "schema_version": TASK_EXPERT_EPISODE_SCHEMA,
            **dict(row),
            "init_state_id": int(init_state_id),
        }
        if row is not None
        else None
    )
    return dict(evidence) == expected
