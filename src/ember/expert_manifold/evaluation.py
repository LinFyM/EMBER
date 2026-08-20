"""Inspect and execute one complete development-train task-expert bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.checkpoint import CHECKPOINT_SCHEMA
from ember.expert_manifold.contract import (
    ExpertManifoldError,
    authority_path,
    load_task_expert_config,
)
from ember.expert_manifold.meta_contract import (
    META_EXPERT_CONFIG_SCHEMA,
    meta_expert_rows,
    meta_worker_assignments,
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
PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_writer_fixed_head_projected_task_expert_eval_adapter_v1"
)
PROJECTED_TASK_EXPERT_MANIFEST_SCHEMA = "ember_writer_fixed_head_reachability_oracle_v1"
FUNCTIONAL_DECODER_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_functional_decoder_projected_task_expert_eval_adapter_v1"
)
FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_functional_decoder_train24_projection_v1"
)
FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_functional_decoder_nonheld_meta_projection_v1"
)


def _projection_file(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    record = manifest.get(name, {})
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ExpertManifoldError("functional-decoder projection asset changed")
    return {"path": str(path), "bytes": path.stat().st_size}


def _projection_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema = manifest.get("schema_version")
    if schema == PROJECTED_TASK_EXPERT_MANIFEST_SCHEMA:
        if manifest.get("optimization", {}).get("factor_heads_frozen") is not True:
            raise ExpertManifoldError("fixed-head projection manifest changed")
        return {
            "adapter_schema": PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA,
            "arm": "macro25_fixed_factor_heads_free_program_projection",
            "asset": {
                "writer_checkpoint": manifest.get("writer_checkpoint"),
                "factor_heads_frozen": True,
            },
        }
    if schema in {
        FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA,
        FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA,
    }:
        optimization = manifest.get("optimization", {})
        code_condition = optimization.get("code_condition", "task_fingerprint")
        if (
            manifest.get("projection_kind")
            != "fixed_functional_decoder_code_projection"
            or optimization.get("decoder_frozen_for_held_code_fit") is not True
            or code_condition not in {"task_fingerprint", "shared_zero"}
        ):
            raise ExpertManifoldError("functional-decoder projection manifest changed")
        meta_surface = schema == FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA
        return {
            "adapter_schema": FUNCTIONAL_DECODER_TASK_EXPERT_ADAPTER_SCHEMA,
            "arm": (
                "functional_decoder_nonheld_meta_shared_zero_carrier"
                if meta_surface and code_condition == "shared_zero"
                else "functional_decoder_nonheld_meta_projection"
                if meta_surface
                else "functional_decoder_train24_projection"
            ),
            "asset": {
                "decoder_checkpoint": _projection_file(
                    manifest, "decoder_checkpoint"
                ),
                "held_codes": _projection_file(manifest, "held_codes"),
                "profile_result": _projection_file(manifest, "profile_result"),
                "decoder_frozen_for_held_code_fit": True,
                "code_condition": code_condition,
            },
        }
    raise ExpertManifoldError("projected task-expert manifest schema changed")


def _expert_task_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if config.get("schema_version") == META_EXPERT_CONFIG_SCHEMA:
        return meta_expert_rows(config)
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = [
        dict(row)
        for row in manifest.get("tasks", [])
        if row.get("split_role") == "train"
    ]
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != int(config["task_experts"]["task_count"]):
        raise ExpertManifoldError("task-expert evaluation did not resolve train24")
    return tuple({"ordinal": ordinal, **row} for ordinal, row in enumerate(rows))


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


def _evaluation_task_rows(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    is_meta: bool,
    evaluation_role: str,
) -> tuple[Mapping[str, Any], ...]:
    if not is_meta:
        if evaluation_role != "development_train":
            raise ExpertManifoldError(
                "task-expert evaluation role differs from its bank"
            )
        return tuple(expected_rows)

    split_roles = {
        "nonheld_meta": None,
        "nonheld_meta_train": "meta_train",
        "nonheld_meta_validation": "meta_validation_oracle",
    }
    if evaluation_role not in split_roles:
        raise ExpertManifoldError("task-expert evaluation role differs from its bank")
    split_role = split_roles[evaluation_role]
    selected = tuple(
        row
        for row in expected_rows
        if split_role is None or row.get("split_role") == split_role
    )
    expected_count = {
        "nonheld_meta": 71,
        "nonheld_meta_train": 56,
        "nonheld_meta_validation": 15,
    }[evaluation_role]
    if len(selected) != expected_count:
        raise ExpertManifoldError("task-expert evaluation split changed")
    return selected


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
    """Seal one complete train24 or non-held meta task-expert bank."""

    config_path = config_path.resolve()
    bank_root = bank_root.resolve()
    config = load_task_expert_config(config_path)
    formal = config["task_experts"]["formal_run"]
    is_meta = config.get("schema_version") == META_EXPERT_CONFIG_SCHEMA
    if require_formal and formal.get("status") != "sealed":
        raise ExpertManifoldError(
            "formal task-expert evaluation requires a sealed profile"
        )
    checkpoints = tuple(int(value) for value in formal["checkpoint_steps"])
    if step not in checkpoints or step <= 0:
        raise ExpertManifoldError("task-expert evaluation step is not declared")

    expected_rows = _expert_task_rows(config)
    evaluation_rows = _evaluation_task_rows(
        expected_rows,
        is_meta=is_meta,
        evaluation_role=evaluation_role,
    )
    expected_by_key = {
        (str(row["suite"]), int(row["task_id"])): (int(row["ordinal"]), row)
        for row in expected_rows
    }
    evaluation_keys = {
        (str(row["suite"]), int(row["task_id"])) for row in evaluation_rows
    }
    observed_keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(observed_keys)) != len(observed_keys) or set(observed_keys) != set(
        evaluation_keys
    ):
        raise ExpertManifoldError("task-expert panel differs from its evaluation role")

    workers = tuple(
        sorted(path for path in bank_root.glob("worker_*") if path.is_dir())
    )
    expected_worker_count = int(formal["allowed_worker_count"])
    if len(workers) != expected_worker_count:
        raise ExpertManifoldError("task-expert bank worker count is incomplete")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    task_records: dict[tuple[str, int], dict[str, Any]] = {}
    training_commits: set[str] = set()
    physical_devices: list[dict[str, Any]] = []
    expected_config_suffix = config_path.parts[-2:]
    allowed_assignments = (
        set(meta_worker_assignments(formal)) if is_meta else None
    )
    observed_assignments: set[tuple[int, ...]] = set()

    for worker in workers:
        contract_path = worker / "run_contract.json"
        summary_path = worker / "worker_summary.json"
        contract = read_json(contract_path)
        summary = read_json(summary_path)
        worker_tasks = tuple(contract.get("tasks", ()))
        runtime = contract.get("runtime", {})
        worker_assignment = tuple(
            sorted(int(row.get("ordinal", -1)) for row in worker_tasks)
        )
        assignment_valid = (
            worker_assignment in allowed_assignments
            if allowed_assignments is not None
            else len(worker_tasks) == int(formal["tasks_per_worker"])
        )
        if (
            contract.get("schema_version") != "ember_pi05_task_expert_worker_launch_v1"
            or contract.get("mode") != "formal"
            or contract.get("content_hash_policy") != "disabled_by_owner"
            or Path(str(contract.get("config", {}).get("path", ""))).parts[-2:]
            != expected_config_suffix
            or contract.get("config", {}).get("schema")
            != config.get("schema_version")
            or not _source_paths_match(contract.get("source", {}), source)
            or not assignment_valid
            or int(runtime.get("per_task_batch_size", -1))
            != int(formal["per_task_batch_size"])
            or runtime.get("task_parameter_sharing") != "none"
            or summary.get("schema_version")
            != "ember_pi05_task_expert_worker_summary_v1"
            or int(summary.get("completed_task_count", -1)) != len(worker_tasks)
            or int(summary.get("selected_stop_step", -1)) < step
        ):
            raise ExpertManifoldError("task-expert worker formal contract changed")
        if worker_assignment in observed_assignments:
            raise ExpertManifoldError("task-expert worker assignment is duplicated")
        observed_assignments.add(worker_assignment)
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
                and declared.get("split_role") == row["split_role"]
                and declared.get("language") == row["language"]
                and int(summary_row.get("global_task_id", -1)) == global_task_id
                and int(summary_row.get("completed_steps", -1)) >= step
            )
            valid_checkpoint = (
                manifest.get("schema_version") == CHECKPOINT_SCHEMA
                and int(manifest.get("step", -1)) == step
                and int(manifest.get("task_ordinal", -1)) == ordinal
                and int(manifest.get("global_task_id", -1)) == global_task_id
                and int(manifest.get("state_tensor_count", -1))
                == lora.state_tensor_count
                and int(manifest.get("state_parameter_count", -1))
                == lora.parameter_count
                and manifest.get("content_hash_policy") == "disabled_by_owner"
            )
            if not valid_declared or not valid_checkpoint:
                raise ExpertManifoldError("task-expert checkpoint ownership changed")
            for name in ("adapter.safetensors", "trainer.pt"):
                path = checkpoint / name
                if not path.is_file() or path.stat().st_size != int(
                    files.get(name, -1)
                ):
                    raise ExpertManifoldError(
                        "task-expert checkpoint file size changed"
                    )
            task_record = {
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
            if is_meta:
                task_record["split_role"] = str(row["split_role"])
            task_records[key] = task_record

    if (
        len(training_commits) != 1
        or "" in training_commits
        or set(task_records) != set(expected_by_key)
        or (
            allowed_assignments is not None
            and observed_assignments != allowed_assignments
        )
    ):
        raise ExpertManifoldError("task-expert bank is not one complete formal family")
    return {
        "schema_version": TASK_EXPERT_ADAPTER_SCHEMA,
        "kind": TASK_EXPERT_ADAPTER_KIND,
        "arm": (
            f"nonheld_meta_task_expert_step_{step}"
            if is_meta
            else f"task_expert_step_{step}"
        ),
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": str(config["schema_version"]),
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
                f"{config['authorities']['lora_contract']['path']}:"
                f"{lora.state_tensor_count}tensors:{lora.parameter_count}parameters"
            ),
            "rank": lora.rank,
            "target_count": len(lora.targets),
        },
        "tasks": [task_records[key] for key in sorted(evaluation_keys)],
        "workers": physical_devices,
        "information_wall": (
            {
                "evaluation_role": evaluation_role,
                "bank_role": "nonheld_meta",
                "meta_train_experts": 56,
                "meta_validation_oracles": 15,
                "evaluated_task_count": len(evaluation_keys),
                "target40_action_reads": 0,
                "deployment_uses_privileged_experts": False,
            }
            if is_meta
            else {
                "evaluation_role": "development_train",
                "validation_experts": 0,
                "test_experts": 0,
                "validation_actions_read": 0,
                "test_actions_read": 0,
            }
        ),
        "content_hash_policy": "disabled_by_owner",
    }


def inspect_projected_task_expert_bank(
    base: Mapping[str, Any], projection_manifest: Path
) -> dict[str, Any]:
    """Bind one complete functional-decoder projection to its expert authority."""

    projection_manifest = projection_manifest.resolve()
    manifest = read_json(projection_manifest)
    projection_contract = _projection_contract(manifest)
    projected = {
        (str(row.get("suite")), int(row.get("task_id", -1))): dict(row)
        for row in manifest.get("tasks", ())
    }
    base_records = {
        (str(row["suite"]), int(row["task_id"])): dict(row) for row in base["tasks"]
    }
    evaluation_role = str(base.get("information_wall", {}).get("evaluation_role"))
    expected_oracle_role = (
        "nonheld_meta_oracle_only"
        if evaluation_role == "nonheld_meta"
        else "development_train_oracle_only"
    )
    if (
        manifest.get("repository", {}).get("dirty_paths") != []
        or manifest.get("information_wall", {}).get("role")
        != expected_oracle_role
        or manifest.get("information_wall", {}).get("deployment_carrier") is not False
        or set(projected) != set(base_records)
    ):
        raise ExpertManifoldError("fixed-head projection manifest changed")
    tasks = []
    for key in sorted(base_records):
        source = base_records[key]
        row = projected[key]
        path = Path(str(row.get("projected_adapter", ""))).resolve()
        if (
            int(row.get("ordinal", -1)) != int(source["ordinal"])
            or int(row.get("global_task_id", -1)) != int(source["global_task_id"])
            or Path(str(row.get("expert_checkpoint", ""))).resolve()
            != Path(str(source["checkpoint"])).resolve()
            or not path.is_file()
            or path.stat().st_size != int(row.get("projected_adapter_bytes", -1))
        ):
            raise ExpertManifoldError("fixed-head projected task adapter changed")
        tasks.append(
            {
                **source,
                "projected_adapter": str(path),
                "projected_adapter_bytes": path.stat().st_size,
            }
        )
    return {
        **dict(base),
        "schema_version": projection_contract["adapter_schema"],
        "arm": projection_contract["arm"],
        "tasks": tasks,
        "projection": {
            "manifest_path": str(projection_manifest),
            "manifest_bytes": projection_manifest.stat().st_size,
            "schema": manifest["schema_version"],
            **projection_contract["asset"],
            "deployment_carrier": False,
        },
    }


def inspect_task_expert_evaluation(
    *,
    projection_manifest: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    evaluation_role = str(kwargs["evaluation_role"])
    config = (
        load_task_expert_config(Path(kwargs["config_path"]))
        if projection_manifest is not None
        else None
    )
    if (
        projection_manifest is not None
        and config is not None
        and config.get("schema_version") == META_EXPERT_CONFIG_SCHEMA
        and evaluation_role in {"nonheld_meta_train", "nonheld_meta_validation"}
    ):
        all_rows = _expert_task_rows(config)
        full_kwargs = {
            **kwargs,
            "task_keys": [
                (str(row["suite"]), int(row["task_id"])) for row in all_rows
            ],
            "evaluation_role": "nonheld_meta",
        }
        projected = inspect_projected_task_expert_bank(
            inspect_task_expert_bank(**full_kwargs), projection_manifest
        )
        requested = tuple(
            (str(suite), int(task_id)) for suite, task_id in kwargs["task_keys"]
        )
        requested_keys = set(requested)
        selected_rows = _evaluation_task_rows(
            projected["tasks"], is_meta=True, evaluation_role=evaluation_role
        )
        selected_keys = {
            (str(row["suite"]), int(row["task_id"])) for row in selected_rows
        }
        if len(requested_keys) != len(requested) or requested_keys != selected_keys:
            raise ExpertManifoldError(
                "projected task-expert panel differs from its evaluation role"
            )
        information_wall = dict(projected["information_wall"])
        information_wall.update(
            evaluation_role=evaluation_role,
            evaluated_task_count=len(selected_rows),
        )
        return {
            **projected,
            "tasks": [dict(row) for row in selected_rows],
            "information_wall": information_wall,
        }
    base = inspect_task_expert_bank(**kwargs)
    if projection_manifest is None:
        return base
    return inspect_projected_task_expert_bank(base, projection_manifest)


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
        del source, require_formal
        observed = dict(evaluation_adapter)
        task_rows = tuple(observed.get("tasks", ()))
        records = {
            (str(row["suite"]), int(row["task_id"])): dict(row)
            for row in task_rows
        }
        expected_keys = {(str(suite), int(task_id)) for suite, task_id in task_keys}
        if len(records) != len(task_rows) or set(records) != expected_keys:
            raise ExpertManifoldError("task-expert runtime panel changed")
        config = load_task_expert_config(Path(str(observed["config"]["path"])))
        self.lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.policy = policy
        self.device = device
        self.records = records
        self._states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self._installed: tuple[str, int] | None = None

    def _state(self, key: tuple[str, int]) -> dict[str, torch.Tensor]:
        if key not in self._states:
            record = self.records[key]
            path = (
                Path(str(record["projected_adapter"]))
                if "projected_adapter" in record
                else Path(str(record["checkpoint"])) / "adapter.safetensors"
            )
            expected_bytes = int(
                record.get("projected_adapter_bytes", record.get("adapter_bytes", -1))
            )
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ExpertManifoldError("task-expert runtime adapter changed")
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
        (str(row["suite"]), int(row["task_id"])): row
        for row in adapter.get("tasks", ())
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
