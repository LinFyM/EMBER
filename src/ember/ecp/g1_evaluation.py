"""Publish sealed G1 task LoRAs for the canonical strict evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file

from ember.ecp.g1_assets import authority_path, load_g1_config
from ember.ecp.g1_runtime import G1_CHECKPOINT_SCHEMA, G1_RUN_SCHEMA, REPO_ROOT
from ember.lora import validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.static_task_lora import STATIC_TASK_LORA_MANIFEST_SCHEMA


def _shared_scientific_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Extract task-invariant scientific choices that one sealed bank must share."""

    shared = {
        "schema_version": contract.get("schema_version"),
        "mode": contract.get("mode"),
        "content_hash_policy": contract.get("content_hash_policy"),
        "authorities": contract.get("authorities"),
        "video_contract": contract.get("video_contract"),
        "functional_query": contract.get("functional_query"),
        "native_factor": contract.get("native_factor"),
        "optimization": contract.get("optimization"),
        "information_wall": contract.get("information_wall"),
    }
    if any(
        not isinstance(shared[key], Mapping)
        for key in shared
        if key
        not in {
            "schema_version",
            "mode",
            "content_hash_policy",
        }
    ):
        raise ValueError("G1 task run lacks its shared scientific contract")
    return shared


def _task_record(
    *, run_root: Path, step: int, lora: Any
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    run_root = run_root.resolve()
    contract = read_json(run_root / "run_contract.json")
    completion = read_json(run_root / "segment_completion.json")
    task = contract.get("task", {})
    native = contract.get("pure_native_stage0", {})
    repository = contract.get("repository", {})
    runtime = contract.get("runtime", {})
    checkpoint = run_root / "checkpoints" / f"step_{step:08d}"
    manifest_path = checkpoint / "manifest.json"
    manifest = read_json(manifest_path)
    adapter_path = checkpoint / "adapter.safetensors"
    valid = (
        contract.get("schema_version") == G1_RUN_SCHEMA
        and contract.get("mode") == "formal"
        and contract.get("content_hash_policy") == "disabled_by_owner"
        and int(contract.get("video", {}).get("K", -1)) == 1
        and contract.get("video", {}).get("cross_video_weight") == "identity_k1"
        and native.get("action_meta_module_count") == 0
        and native.get("action_meta_parameter_count") == 0
        and native.get("policy_trainable_parameter_count") == 0
        and native.get("stage0_trainable_parameter_count") == 0
        and repository.get("dirty_paths") == []
        and repository.get("branch") == ""
        and repository.get("upstream") is None
        and int(runtime.get("world_size", -1)) == 1
        and runtime.get("torch_device") == contract.get("device")
        and isinstance(runtime.get("cuda_visible_devices"), str)
        and bool(runtime.get("cuda_visible_devices"))
        and "," not in runtime.get("cuda_visible_devices")
        and runtime.get("device_name") == "NVIDIA A40"
        and completion.get("status") == "segment_complete"
        and int(completion.get("completed_steps", -1)) >= step
        and manifest.get("schema_version") == G1_CHECKPOINT_SCHEMA
        and int(manifest.get("step", -1)) == step
        and int(manifest.get("task_ordinal", -1)) == int(task.get("ordinal", -2))
        and int(manifest.get("global_task_id", -1))
        == int(task.get("global_task_id", -2))
        and manifest.get("rank_partition") == {"carrier": [0, 12], "task": [12, 16]}
        and manifest.get("single_complete_rank16") is True
        and manifest.get("content_hash_policy") == "disabled_by_owner"
        and adapter_path.is_file()
        and adapter_path.stat().st_size
        == int(manifest.get("files", {}).get("adapter.safetensors", -1))
    )
    if not valid:
        raise ValueError(
            f"G1 task run is not a sealed pure-Native rank16 result: {run_root}"
        )
    state = load_file(str(adapter_path), device="cpu")
    validate_lora_state(state, lora)
    return (
        {
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "ordinal": int(task["ordinal"]),
            "global_task_id": int(task["global_task_id"]),
            "language": str(task["language"]),
            "step": step,
            "run_root": str(run_root),
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_bytes": manifest_path.stat().st_size,
            "adapter_path": str(adapter_path),
            "adapter_bytes": adapter_path.stat().st_size,
            "single_complete_rank16": True,
        },
        str(repository["commit"]),
        _shared_scientific_contract(contract),
    )


def _single_training_authority(
    records: Sequence[tuple[dict[str, Any], str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    commits = {_commit for _row, _commit, _shared in records}
    if len(commits) != 1 or "" in commits:
        raise ValueError("G1 evaluation bank cannot mix training commits")
    shared_contracts = [_shared for _row, _commit, _shared in records]
    if not shared_contracts or any(
        contract != shared_contracts[0] for contract in shared_contracts[1:]
    ):
        raise ValueError("G1 evaluation bank cannot mix scientific run contracts")
    return next(iter(commits)), shared_contracts[0]


def publish_g1_evaluation_bank(
    *,
    config_path: Path,
    asset_root: Path,
    task_runs: Sequence[Path],
    step: int,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    asset_root = asset_root.resolve()
    output_path = output_path.resolve()
    config = load_g1_config(config_path)
    expected_ordinals = tuple(int(value) for value in config["tasks"]["held_ordinals"])
    if step < 0 or len(task_runs) != len(expected_ordinals):
        raise ValueError(
            "G1 evaluation bank requires one non-negative shared step per held task"
        )
    lora_path = authority_path(config, "lora_contract", asset_root=asset_root)
    lora = load_pi05_lora_contract(lora_path)
    if lora.rank != 16 or len(lora.targets) != 38:
        raise ValueError("G1 evaluation requires the complete rank16 contract")
    records_and_commits = [
        _task_record(run_root=path, step=step, lora=lora) for path in task_runs
    ]
    rows = [row for row, _commit, _shared in records_and_commits]
    by_ordinal = {int(row["ordinal"]): row for row in rows}
    if len(by_ordinal) != len(rows) or set(by_ordinal) != set(expected_ordinals):
        raise ValueError("G1 task-run bank differs from held5")
    ordered = [by_ordinal[ordinal] for ordinal in expected_ordinals]
    expected_globals = tuple(int(value) for value in config["tasks"]["global_task_ids"])
    if tuple(int(row["global_task_id"]) for row in ordered) != expected_globals:
        raise ValueError("G1 task-run global IDs changed")
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=asset_root
    )
    source_run = source_checkpoint.parent.parent
    training_commit, shared_contract = _single_training_authority(records_and_commits)
    payload = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": "ecp_native_factor_g1_free_code",
        "source": {
            "source_run": str(source_run),
            "checkpoint": str(source_checkpoint),
            "model_path": str(source_checkpoint / "policy"),
        },
        "lora_contract": {"path": str(lora_path), "bytes": lora_path.stat().st_size},
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": training_commit,
        "shared_run_contract": shared_contract,
        "tasks": ordered,
        "information_wall": {
            "task_local_free_code_capacity_oracle": True,
            "shared_program_attention_claim": False,
            "action_meta_installed": False,
            "teacher_video_runtime_reads": 0,
            "second_adapter_deployed": False,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    if output_path.exists():
        if read_json(output_path) != payload:
            raise ValueError("existing G1 evaluation manifest differs")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--task-run", type=Path, action="append", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = publish_g1_evaluation_bank(
        config_path=args.config,
        asset_root=args.asset_root,
        task_runs=args.task_run,
        step=args.step,
        output_path=args.output,
    )
    print(
        f"sealed {len(payload['tasks'])} G1 rank16 adapters at {args.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
