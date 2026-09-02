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
POLICY_RESPONSE_WRITER_ADAPTER_SCHEMA = (
    "ember_ecp_policy_response_writer_materialized_adapter_v1"
)
POLICY_RESPONSE_WRITER_ARM_PREFIX = "ecp_policy_response_writer_"


def _source_matches(declared: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return all(
        Path(str(declared.get(name, ""))).resolve()
        == Path(str(observed.get(name, ""))).resolve()
        for name in ("source_run", "checkpoint", "model_path")
    )


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40


def _g1_provenance(manifest: Mapping[str, Any]) -> bool:
    contract = manifest.get("shared_run_contract")
    return all(
        (
            _is_commit(manifest.get("training_commit")),
            isinstance(contract, Mapping),
            contract.get("schema_version")
            == "ember_ecp_native_factor_g1_task_run_v1",
            contract.get("mode") == "formal",
        )
    )


def _policy_response_provenance(
    manifest: Mapping[str, Any], arm: str
) -> bool:
    contract = manifest.get("shared_run_contract")
    condition = manifest.get("condition", {})
    checkpoint = manifest.get("writer_checkpoint", {})
    representation = str(condition.get("representation", ""))
    return all(
        (
            arm
            == f"{POLICY_RESPONSE_WRITER_ARM_PREFIX}{representation}_correct_k1",
            representation in {"full", "coarse"},
            _is_commit(manifest.get("training_commit")),
            _is_commit(manifest.get("materialization_commit")),
            isinstance(contract, Mapping),
            contract.get("schema_version")
            == "ember_policy_response_writer_shared_run_v1",
            contract.get("stage")
            == "policy_response_writer_shared_positive_only",
            contract.get("mode") == "formal",
            contract.get("representation") == representation,
            condition.get("name") == "correct_k1",
            condition.get("video_demos") == [5],
            int(condition.get("K", -1)) == 1,
            condition.get("outcome_dependence") is False,
            condition.get("gradient_use") is False,
            isinstance(checkpoint.get("path"), str),
            int(checkpoint.get("macro", -1)) in {70, 110},
        )
    )


def _g3_provenance(manifest: Mapping[str, Any], arm: str) -> bool:
    contract = manifest.get("shared_run_contract")
    condition = manifest.get("condition", {})
    checkpoint = manifest.get("compiler_checkpoint", {})
    return all(
        (
            _is_commit(manifest.get("training_commit")),
            _is_commit(manifest.get("materialization_commit")),
            isinstance(contract, Mapping),
            contract.get("schema_version") == "ember_ecp_shared_compiler_g3_run_v2",
            contract.get("stage") == "g3_shared_compiler",
            contract.get("mode") == "formal",
            condition.get("name") == G3_STATIC_ARMS[arm],
            int(condition.get("K", -1)) == 4,
            isinstance(checkpoint.get("path"), str),
            int(checkpoint.get("macro", -1)) > 0,
        )
    )


def _language_provenance(manifest: Mapping[str, Any]) -> bool:
    contract = manifest.get("shared_run_contract")
    condition = manifest.get("condition", {})
    return all(
        (
            _is_commit(manifest.get("training_commit")),
            _is_commit(manifest.get("materialization_commit")),
            isinstance(contract, Mapping),
            contract.get("schema_version")
            == "ember_ecp_g3_language_only_baseline_v1",
            contract.get("stage") == "g3_learned_language_only",
            contract.get("mode") == "formal",
            condition.get("name") == "learned_language_only",
            int(condition.get("K", -1)) == 0,
            condition.get("video_demos") == [],
        )
    )


def _manifest_provenance_valid(manifest: Mapping[str, Any], arm: str) -> bool:
    if arm == "ecp_native_factor_g1_free_code":
        return _g1_provenance(manifest)
    if arm.startswith(POLICY_RESPONSE_WRITER_ARM_PREFIX):
        return _policy_response_provenance(manifest, arm)
    if arm in G3_STATIC_ARMS:
        return _g3_provenance(manifest, arm)
    if arm == G3_LANGUAGE_ARM:
        return _language_provenance(manifest)
    return True


def _g1_checkpoint_matches(
    arm: str, checkpoint: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    return all(
        (
            arm not in {*G3_STATIC_ARMS, G3_LANGUAGE_ARM},
            checkpoint.get("schema_version")
            == "ember_ecp_native_factor_g1_checkpoint_v1",
            int(checkpoint.get("task_ordinal", -1)) == int(row.get("ordinal", -2)),
            int(checkpoint.get("global_task_id", -1))
            == int(row.get("global_task_id", -2)),
            int(checkpoint.get("step", -1)) == int(row.get("step", -2)),
            checkpoint.get("single_complete_rank16") is True,
        )
    )


def _g3_checkpoint_matches(
    arm: str,
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    key: tuple[str, int],
) -> bool:
    return all(
        (
            arm in G3_STATIC_ARMS,
            checkpoint.get("schema_version") == G3_MATERIALIZED_ADAPTER_SCHEMA,
            checkpoint.get("condition") == G3_STATIC_ARMS.get(arm),
            int(checkpoint.get("authority_id", -1))
            == int(row.get("natural_program_authority_id", -2)),
            int(checkpoint.get("global_task_id", -1))
            == int(row.get("global_task_id", -2)),
            str(checkpoint.get("suite")) == key[0],
            int(checkpoint.get("task_id", -1)) == key[1],
            int(checkpoint.get("compiler_macro", -1))
            == int(row.get("compiler_macro", -2)),
            checkpoint.get("single_complete_rank16") is True,
        )
    )


def _language_checkpoint_matches(
    arm: str,
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    key: tuple[str, int],
) -> bool:
    return all(
        (
            arm == G3_LANGUAGE_ARM,
            checkpoint.get("schema_version") == G3_LANGUAGE_ADAPTER_SCHEMA,
            checkpoint.get("condition") == "learned_language_only",
            int(checkpoint.get("authority_id", -1))
            == int(row.get("natural_program_authority_id", -2)),
            int(checkpoint.get("global_task_id", -1))
            == int(row.get("global_task_id", -2)),
            str(checkpoint.get("suite")) == key[0],
            int(checkpoint.get("task_id", -1)) == key[1],
            checkpoint.get("single_complete_rank16") is True,
        )
    )


def _policy_response_checkpoint_matches(
    arm: str,
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    key: tuple[str, int],
    manifest: Mapping[str, Any],
) -> bool:
    return all(
        (
            arm.startswith(POLICY_RESPONSE_WRITER_ARM_PREFIX),
            checkpoint.get("schema_version") == POLICY_RESPONSE_WRITER_ADAPTER_SCHEMA,
            checkpoint.get("condition") == "correct_k1",
            checkpoint.get("representation")
            == manifest.get("condition", {}).get("representation"),
            str(checkpoint.get("writer_checkpoint", ""))
            == str(manifest.get("writer_checkpoint", {}).get("path", "")),
            int(checkpoint.get("authority_id", -1))
            == int(row.get("natural_program_authority_id", -2)),
            int(checkpoint.get("global_task_id", -1))
            == int(row.get("global_task_id", -2)),
            str(checkpoint.get("suite")) == key[0],
            int(checkpoint.get("task_id", -1)) == key[1],
            int(checkpoint.get("writer_macro", -1))
            == int(row.get("writer_macro", -2)),
            checkpoint.get("single_complete_rank16") is True,
        )
    )


def _policy_response_wall_matches(
    arm: str, information_wall: Mapping[str, Any]
) -> bool:
    if not arm.startswith(POLICY_RESPONSE_WRITER_ARM_PREFIX):
        return True
    return all(
        (
            information_wall.get("writer_invocations_per_task_condition") == 1,
            information_wall.get("validation_action_or_reward_reads") == 0,
            information_wall.get("test_action_or_reward_reads") == 0,
            information_wall.get("shuffled_or_reversed_use") is False,
            information_wall.get("wrong_video_use") is False,
        )
    )


def _checkpoint_authority_matches(
    arm: str,
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    key: tuple[str, int],
    manifest: Mapping[str, Any],
) -> bool:
    return any(
        (
            _g1_checkpoint_matches(arm, checkpoint, row),
            _g3_checkpoint_matches(arm, checkpoint, row, key),
            _language_checkpoint_matches(arm, checkpoint, row, key),
            _policy_response_checkpoint_matches(
                arm, checkpoint, row, key, manifest
            ),
        )
    )


def _inspect_static_task_row(
    *,
    row: Mapping[str, Any],
    key: tuple[str, int],
    arm: str,
    manifest: Mapping[str, Any],
    lora: Any,
) -> dict[str, Any]:
    checkpoint = Path(str(row.get("checkpoint", ""))).resolve()
    adapter_path = Path(str(row.get("adapter_path", ""))).resolve()
    checkpoint_manifest = checkpoint / "manifest.json"
    valid = all(
        (
            adapter_path == checkpoint / "adapter.safetensors",
            checkpoint_manifest.is_file(),
            checkpoint_manifest.stat().st_size
            == int(row.get("checkpoint_manifest_bytes", -1)),
            adapter_path.is_file(),
            adapter_path.stat().st_size == int(row.get("adapter_bytes", -1)),
            row.get("single_complete_rank16") is True,
        )
    )
    if not valid:
        raise Pi05EvaluationError("static task-LoRA checkpoint changed")
    checkpoint_cell = read_json(checkpoint_manifest)
    if not _checkpoint_authority_matches(
        arm, checkpoint_cell, row, key, manifest
    ):
        raise Pi05EvaluationError("static task-LoRA checkpoint authority changed")
    state = load_file(str(adapter_path), device="cpu")
    validate_lora_state(state, lora)
    return dict(row)


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
    provenance_valid = _manifest_provenance_valid(manifest, arm)
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
        and _policy_response_wall_matches(arm, information_wall)
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

    inspected_rows = [
        _inspect_static_task_row(
            row=records[key], key=key, arm=arm, manifest=manifest, lora=lora
        )
        for key in requested
    ]

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
        "writer_checkpoint": manifest.get("writer_checkpoint"),
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
