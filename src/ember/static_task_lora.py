"""Inspect and execute a sealed bank of complete task-local PI0.5 LoRAs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    copy_task_lora_state_,
    inject_task_lora,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json


STATIC_TASK_LORA_KIND = "static_task_lora_bank"
STATIC_TASK_LORA_MANIFEST_SCHEMA = "ember_pi05_static_task_lora_bank_v1"
STATIC_TASK_LORA_ADAPTER_SCHEMA = "ember_pi05_static_task_lora_eval_adapter_v1"
STATIC_TASK_LORA_EPISODE_SCHEMA = "ember_pi05_static_task_lora_episode_v1"
G3_MATERIALIZED_ADAPTER_SCHEMA = (
    "ember_ecp_shared_compiler_g3_materialized_adapter_v1"
)
G3_STATIC_ARMS = {
    "ecp_shared_compiler_g3_correct_full": "correct_full",
    "ecp_shared_compiler_g3_first_final": "first_final",
    "ecp_shared_compiler_g3_same_task_other": "same_task_other",
}
G3_LANGUAGE_ARM = "ecp_shared_compiler_g3_learned_language_only"
G3_LANGUAGE_ADAPTER_SCHEMA = "ember_ecp_g3_language_only_adapter_v1"


def _source_matches(declared: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return all(
        Path(str(declared.get(name, ""))).resolve()
        == Path(str(observed.get(name, ""))).resolve()
        for name in ("source_run", "checkpoint", "model_path")
    )


def inspect_static_task_lora_bank(
    *,
    manifest_path: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    evaluation_role: str,
    require_formal: bool,
) -> dict[str, Any]:
    """Reinspect one immutable complete-rank16 task bank without file hashes."""

    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    rows = tuple(dict(row) for row in manifest.get("tasks", ()))
    records = {
        (str(row.get("suite")), int(row.get("task_id", -1))): row for row in rows
    }
    requested = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    lora_cell = manifest.get("lora_contract", {})
    lora_path = Path(str(lora_cell.get("path", ""))).resolve()
    information_wall = manifest.get("information_wall", {})
    arm = str(manifest.get("arm", ""))
    provenance_valid = True
    if arm == "ecp_native_factor_g1_free_code":
        training_commit = manifest.get("training_commit")
        shared_contract = manifest.get("shared_run_contract")
        provenance_valid = (
            isinstance(training_commit, str)
            and len(training_commit) == 40
            and isinstance(shared_contract, Mapping)
            and shared_contract.get("schema_version")
            == "ember_ecp_native_factor_g1_task_run_v1"
            and shared_contract.get("mode") == "formal"
        )
    elif arm in G3_STATIC_ARMS:
        training_commit = manifest.get("training_commit")
        materialization_commit = manifest.get("materialization_commit")
        shared_contract = manifest.get("shared_run_contract")
        condition = manifest.get("condition", {})
        checkpoint = manifest.get("compiler_checkpoint", {})
        provenance_valid = (
            isinstance(training_commit, str)
            and len(training_commit) == 40
            and isinstance(materialization_commit, str)
            and len(materialization_commit) == 40
            and isinstance(shared_contract, Mapping)
            and shared_contract.get("schema_version")
            == "ember_ecp_shared_compiler_g3_run_v1"
            and shared_contract.get("stage") == "g3_shared_compiler"
            and shared_contract.get("mode") == "formal"
            and condition.get("name") == G3_STATIC_ARMS[arm]
            and int(condition.get("K", -1)) == 4
            and isinstance(checkpoint.get("path"), str)
            and int(checkpoint.get("macro", -1)) > 0
        )
    elif arm == G3_LANGUAGE_ARM:
        training_commit = manifest.get("training_commit")
        materialization_commit = manifest.get("materialization_commit")
        shared_contract = manifest.get("shared_run_contract")
        condition = manifest.get("condition", {})
        provenance_valid = (
            isinstance(training_commit, str)
            and len(training_commit) == 40
            and isinstance(materialization_commit, str)
            and len(materialization_commit) == 40
            and isinstance(shared_contract, Mapping)
            and shared_contract.get("schema_version")
            == "ember_ecp_g3_language_only_baseline_v1"
            and shared_contract.get("stage") == "g3_learned_language_only"
            and shared_contract.get("mode") == "formal"
            and condition.get("name") == "learned_language_only"
            and int(condition.get("K", -1)) == 0
            and condition.get("video_demos") == []
        )
    valid = (
        manifest.get("schema_version") == STATIC_TASK_LORA_MANIFEST_SCHEMA
        and (not require_formal or manifest.get("status") == "sealed")
        and evaluation_role == "development_train"
        and len(requested) == len(set(requested)) == len(rows) == len(records)
        and set(requested) == set(records)
        and manifest.get("single_complete_rank16") is True
        and manifest.get("rank_partition") == {"carrier": [0, 12], "task": [12, 16]}
        and information_wall.get("action_meta_installed") is False
        and information_wall.get("second_adapter_deployed") is False
        and information_wall.get("teacher_video_runtime_reads") == 0
        and provenance_valid
        and _source_matches(manifest.get("source", {}), source)
        and lora_path.is_file()
        and lora_path.stat().st_size == int(lora_cell.get("bytes", -1))
    )
    if not valid:
        raise Pi05EvaluationError("static task-LoRA manifest changed")
    lora = load_pi05_lora_contract(lora_path)
    if lora.rank != 16 or len(lora.targets) != 38:
        raise Pi05EvaluationError("static task-LoRA contract is not complete rank16")

    inspected_rows = []
    for key in requested:
        row = records[key]
        checkpoint = Path(str(row.get("checkpoint", ""))).resolve()
        adapter_path = Path(str(row.get("adapter_path", ""))).resolve()
        checkpoint_manifest = checkpoint / "manifest.json"
        valid_row = (
            adapter_path == checkpoint / "adapter.safetensors"
            and checkpoint_manifest.is_file()
            and checkpoint_manifest.stat().st_size
            == int(row.get("checkpoint_manifest_bytes", -1))
            and adapter_path.is_file()
            and adapter_path.stat().st_size == int(row.get("adapter_bytes", -1))
            and row.get("single_complete_rank16") is True
        )
        if not valid_row:
            raise Pi05EvaluationError("static task-LoRA checkpoint changed")
        checkpoint_cell = read_json(checkpoint_manifest)
        g1_checkpoint = (
            arm not in {*G3_STATIC_ARMS, G3_LANGUAGE_ARM}
            and checkpoint_cell.get("schema_version")
            == "ember_ecp_native_factor_g1_checkpoint_v1"
            and int(checkpoint_cell.get("task_ordinal", -1))
            == int(row.get("ordinal", -2))
            and int(checkpoint_cell.get("global_task_id", -1))
            == int(row.get("global_task_id", -2))
            and int(checkpoint_cell.get("step", -1)) == int(row.get("step", -2))
            and checkpoint_cell.get("single_complete_rank16") is True
        )
        g3_checkpoint = (
            arm in G3_STATIC_ARMS
            and checkpoint_cell.get("schema_version")
            == G3_MATERIALIZED_ADAPTER_SCHEMA
            and checkpoint_cell.get("condition") == G3_STATIC_ARMS[arm]
            and int(checkpoint_cell.get("authority_id", -1))
            == int(row.get("natural_program_authority_id", -2))
            and int(checkpoint_cell.get("global_task_id", -1))
            == int(row.get("global_task_id", -2))
            and str(checkpoint_cell.get("suite")) == key[0]
            and int(checkpoint_cell.get("task_id", -1)) == key[1]
            and int(checkpoint_cell.get("compiler_macro", -1))
            == int(row.get("compiler_macro", -2))
            and checkpoint_cell.get("single_complete_rank16") is True
        )
        language_checkpoint = (
            arm == G3_LANGUAGE_ARM
            and checkpoint_cell.get("schema_version")
            == G3_LANGUAGE_ADAPTER_SCHEMA
            and checkpoint_cell.get("condition") == "learned_language_only"
            and int(checkpoint_cell.get("authority_id", -1))
            == int(row.get("natural_program_authority_id", -2))
            and int(checkpoint_cell.get("global_task_id", -1))
            == int(row.get("global_task_id", -2))
            and str(checkpoint_cell.get("suite")) == key[0]
            and int(checkpoint_cell.get("task_id", -1)) == key[1]
            and checkpoint_cell.get("single_complete_rank16") is True
        )
        if not (g1_checkpoint or g3_checkpoint or language_checkpoint):
            raise Pi05EvaluationError("static task-LoRA checkpoint authority changed")
        state = load_file(str(adapter_path), device="cpu")
        validate_lora_state(state, lora)
        inspected_rows.append(dict(row))

    return {
        "schema_version": STATIC_TASK_LORA_ADAPTER_SCHEMA,
        "kind": STATIC_TASK_LORA_KIND,
        "arm": arm,
        "evaluation_role": evaluation_role,
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "source": dict(manifest["source"]),
        "lora_contract": {"path": str(lora_path), "bytes": lora_path.stat().st_size},
        "rank_partition": dict(manifest["rank_partition"]),
        "single_complete_rank16": True,
        "tasks": inspected_rows,
        "information_wall": dict(information_wall),
        "training_commit": manifest.get("training_commit"),
        "materialization_commit": manifest.get("materialization_commit"),
        "shared_run_contract": manifest.get("shared_run_contract"),
        "compiler_checkpoint": manifest.get("compiler_checkpoint"),
        "condition": manifest.get("condition"),
        "content_hash_policy": "disabled_by_owner",
    }


@dataclass(frozen=True)
class PreparedStaticTaskLoRA:
    key: tuple[str, int]
    evidence: dict[str, Any]


class FrozenStaticTaskLoRAAdapter:
    """Install one sealed complete task LoRA when a worker changes task."""

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
        del source, device, require_formal
        rows = tuple(dict(row) for row in evaluation_adapter.get("tasks", ()))
        records = {(str(row["suite"]), int(row["task_id"])): row for row in rows}
        expected = {(str(suite), int(task_id)) for suite, task_id in task_keys}
        if (
            evaluation_adapter.get("kind") != STATIC_TASK_LORA_KIND
            or len(records) != len(rows)
            or set(records) != expected
            or evaluation_adapter.get("single_complete_rank16") is not True
        ):
            raise Pi05EvaluationError("static task-LoRA runtime panel changed")
        self.lora = load_pi05_lora_contract(
            Path(str(evaluation_adapter["lora_contract"]["path"]))
        )
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.policy = policy
        self.records = records
        self._states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._installed: tuple[str, int] | None = None

    def _state(self, key: tuple[str, int]) -> dict[str, torch.Tensor]:
        if key not in self._states:
            row = self.records[key]
            path = Path(str(row["adapter_path"]))
            if not path.is_file() or path.stat().st_size != int(row["adapter_bytes"]):
                raise Pi05EvaluationError("static task-LoRA runtime adapter changed")
            state = load_file(str(path), device="cpu")
            validate_lora_state(state, self.lora)
            self._states[key] = state
        return self._states[key]

    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedStaticTaskLoRA:
        key = (str(suite), int(task_id))
        row = self.records.get(key)
        if row is None:
            raise Pi05EvaluationError("rollout task is outside static task-LoRA bank")
        return PreparedStaticTaskLoRA(
            key=key,
            evidence={
                "schema_version": STATIC_TASK_LORA_EPISODE_SCHEMA,
                **row,
                "init_state_id": int(init_state_id),
            },
        )

    @torch.no_grad()
    def install(self, prepared: PreparedStaticTaskLoRA) -> None:
        if prepared.key == self._installed:
            return
        copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)
        self._installed = prepared.key


def validate_static_task_lora_episode(
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
        (str(row["suite"]), int(row["task_id"])): row
        for row in adapter.get("tasks", ())
    }
    row = records.get((str(suite), int(task_id)))
    expected = (
        {
            "schema_version": STATIC_TASK_LORA_EPISODE_SCHEMA,
            **dict(row),
            "init_state_id": int(init_state_id),
        }
        if row is not None
        else None
    )
    return dict(evidence) == expected
