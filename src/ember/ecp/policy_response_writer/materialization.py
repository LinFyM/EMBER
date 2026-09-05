"""Freeze one shared Writer checkpoint into registered evaluation task LoRAs."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.natural_program_data import NaturalProgramTask
from ember.ecp.policy_response_writer.shared import SHARED_RUN_SCHEMA, SHARED_STAGE
from ember.ecp.policy_response_writer.training import (
    PolicyResponseRuntime,
    capture_video,
    prepare_runtime,
)
from ember.ecp.shared_compiler_assets import authority_path
from ember.ecp.stage0_training import stage0_source_authority
from ember.lora import validate_lora_state
from ember.pi05_eval_contract import SUITE_ORDER, git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything
from ember.static_task_lora import (
    STATIC_TASK_LORA_MANIFEST_SCHEMA,
    policy_response_video_demos,
    validation_task_keys,
)
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[4]
HELD5_EVALUATION_SCHEMA = "ember_ecp_policy_response_writer_held5_eval_v4"
VALIDATION_EVALUATION_SCHEMA = "ember_ecp_policy_response_writer_validation_eval_v1"
MATERIALIZED_ADAPTER_SCHEMA = (
    "ember_ecp_policy_response_writer_materialized_adapter_v1"
)


@dataclass
class WriterMaterializationRuntime:
    runtime: PolicyResponseRuntime
    evaluation: dict[str, Any]
    shared_contract: dict[str, Any]
    writer_macro: int
    tasks: tuple[Any, ...]
    target_keys: dict[int, tuple[str, int]]
    source: dict[str, Any]
    state: dict[str, Any]
    wall: dict[str, Any]

    def close(self) -> None:
        self.runtime.close()


def _evaluation_scope_valid(config: Mapping[str, Any]) -> bool:
    condition = config.get("condition", {})
    if config.get("schema_version") == VALIDATION_EVALUATION_SCHEMA:
        return all((
            config.get("status") == "active_correct_only_validation_materialization",
            config.get("evaluation_role") == "validation",
            config.get("task_subset") is None,
            config.get("target_global_ids") == [
                SUITE_ORDER.index(suite) * 10 + task for suite, task in validation_task_keys()
            ],
            config.get("require_training_completion", True) is True,
            condition.get("selection") == "predeclared_fixed_validation8_correct_video",
            condition.get("checkpoint_selection_use") is True,
            "video_demos_by_global_task" in condition,
            "video_demos" not in condition,
        ))
    if config.get("schema_version") == HELD5_EVALUATION_SCHEMA:
        return all((
            config.get("status") == "active_correct_only_held5_materialization",
            config.get("task_subset") == "configs/pi05_train24_fold0_held5_eval_v1.json",
            config.get("target_held_global_ids") == [0, 9, 18, 25, 36],
            condition.get("video_demos") == [5],
            "video_demos_by_global_task" not in condition,
            condition.get("selection") == "fixed_first_member_of_existing_correct_5_6_7_8_panel",
            condition.get("checkpoint_selection_use") is True,
        ))
    return all((
        config.get("schema_version") == "ember_ecp_policy_response_writer_train_diagnostic_eval_v1",
        config.get("status") == "active_correct_only_train_diagnostic_materialization",
        condition.get("selection") == "registered_train_side_first_fit_or_held_video",
        condition.get("video_role") in {"first_fit_video", "held_video"},
        "video_demos" not in condition,
        condition.get("checkpoint_selection_use") is False,
    ))


def _evaluation_information_wall(validation: bool) -> dict[str, Any]:
    return {
        "validation_or_test_use": validation,
        **({"test_use": False} if validation else {}),
        "held_action_or_reward_reads": 0,
        "shuffled_or_reversed_use": False,
        "wrong_video_use": False,
        "language_only_use": False,
        "writer_invocations_per_task_condition": 1,
        "single_complete_rank16": True,
    }


def load_writer_evaluation_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    condition = config.get("condition", {})
    wall = config.get("information_wall", {})
    validation = config.get("schema_version") == VALIDATION_EVALUATION_SCHEMA
    candidates = tuple(map(int, config.get("checkpoint_candidates", ())))
    target_ids = tuple(map(int, config.get(
        "target_global_ids", config.get("target_held_global_ids", ())
    )))
    by_task = condition.get("video_demos_by_global_task")
    demos = tuple(policy_response_video_demos(condition, task) for task in target_ids)
    if (
        not _evaluation_scope_valid(config)
        or not str(config.get("training_config", "")).startswith("configs/")
        or not target_ids or tuple(sorted(set(target_ids))) != target_ids
        or min(target_ids) < 0 or max(target_ids) >= 40
        or (
            "target_held_global_ids" in config
            and tuple(config["target_held_global_ids"]) != target_ids
        )
        or (by_task is not None and set(by_task) != set(map(str, target_ids)))
        or not candidates or tuple(sorted(set(candidates))) != candidates
        or min(candidates) <= 0
        or condition.get("name") != "correct_k1" or condition.get("K") != 1
        or any(len(row) != 1 or not 0 <= row[0] < 50 for row in demos)
        or condition.get("outcome_dependence") is not False
        or condition.get("gradient_use") is not False
        or not isinstance(config.get("require_training_completion", True), bool)
        or wall != _evaluation_information_wall(validation)
    ):
        raise ValueError("unsupported Policy-Response Writer evaluation config")
    return {**config, "target_global_ids": list(target_ids),
            "evaluation_role": "validation" if validation else "development_train"}


def _validation_deployment_tasks(
    args: argparse.Namespace, evaluation: Mapping[str, Any],
) -> tuple[NaturalProgramTask, ...] | None:
    if evaluation["evaluation_role"] != "validation":
        return None
    config = read_json(args.config)
    base = read_json(args.asset_root / config["authorities"]["base_g3_config"])
    manifest = read_json(authority_path(base, "target_manifest", asset_root=args.asset_root))
    records = sorted(
        (row for row in manifest["tasks"] if row["split_role"] == "validation"),
        key=lambda row: int(row["global_task_id"]),
    )
    if (
        [int(row["global_task_id"]) for row in records] != evaluation["target_global_ids"]
        or {(int(row["global_task_id"]), str(row["suite"]), int(row["task_id"]))
            for row in records} != {
                (SUITE_ORDER.index(suite) * 10 + task, suite, task)
                for suite, task in validation_task_keys()
            } or len(records) != 8
    ):
        raise ValueError("Writer deployment escaped fixed validation8")
    # Negative lookup keys cannot collide with training authorities; the Writer sees no IDs.
    tasks = tuple(
        NaturalProgramTask(
            authority_id=-1-int(row["global_task_id"]),
            domain="target_validation8", domain_task_id=int(row["global_task_id"]),
            role="target_validation", suite=str(row["suite"]),
            language=str(row["language"]), task_name=str(row["task_name"]),
            problem_folder=str(row["problem_folder"]), bddl_file=str(row["bddl"]["filename"]),
            path=args.data_root / str(row["hdf5"]["relative_path"]),
            expected_bytes=int(row["hdf5"]["bytes"]),
            episode_lengths=tuple(map(int, row["demonstrations"]["episode_lengths"])),
        ) for row in records
    )
    if any(not task.path.is_file() or task.path.stat().st_size != task.expected_bytes
           or len(task.episode_lengths) != 50 for task in tasks):
        raise ValueError("Writer validation video authority changed")
    return tasks


def _shared_contract_matches(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> bool:
    config = contract.get("config", {})
    candidates = tuple(map(int, evaluation["checkpoint_candidates"]))
    topology = tuple(contract.get("world_topology", ()))
    return all(
        (
            contract.get("schema_version") == SHARED_RUN_SCHEMA,
            contract.get("stage") == SHARED_STAGE,
            contract.get("mode") == "formal",
            contract.get("representation") == "full",
            contract.get("initialization_request") in {"component", "random"},
            int(contract.get("stop_step", -1)) >= max(candidates),
            1 <= len(topology) <= 6,
            Path(str(config.get("path", ""))).name == args.config.name,
            int(config.get("bytes", -1)) == args.config.stat().st_size,
        )
    )


def _checkpoint_manifest_matches(
    manifest: Mapping[str, Any], *, macro: int, world_size: int
) -> bool:
    expected_files = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    return all(
        (
            manifest.get("schema_version") == ECP_CHECKPOINT_SCHEMA,
            manifest.get("stage") == SHARED_STAGE,
            manifest.get("run_contract_schema") == SHARED_RUN_SCHEMA,
            int(manifest.get("next_macro", -1)) == macro,
            int(manifest.get("world_size", -1)) == world_size,
            set(manifest.get("files", {})) == expected_files,
        )
    )


def _completion_matches(
    completion: Mapping[str, Any], *, expected_stop: int
) -> bool:
    return all(
        (
            completion.get("status") == "complete",
            completion.get("phase") == "shared",
            int(completion.get("optimizer_steps", -1)) == expected_stop,
        )
    )


def _load_writer_checkpoint(
    args: argparse.Namespace,
    evaluation: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    run_root = args.writer_run.resolve()
    checkpoint = args.writer_checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    contract = read_json(run_root / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    completion_path = run_root / "completion.json"
    completion = read_json(completion_path) if completion_path.is_file() else None
    world_size = len(contract.get("world_topology", ()))
    expected_stop = int(contract.get("stop_step", -1))
    completion_required = bool(evaluation.get("require_training_completion", True))
    completion_valid = completion is not None and _completion_matches(
        completion, expected_stop=expected_stop
    )
    if not all(
        (
            checkpoint.parent.parent == run_root,
            macro in set(map(int, evaluation["checkpoint_candidates"])),
            _shared_contract_matches(args, contract, evaluation),
            _checkpoint_manifest_matches(manifest, macro=macro, world_size=world_size),
            completion_valid if completion_required or completion is not None else True,
            (checkpoint / "ecp.safetensors").is_file(),
        )
    ):
        raise ValueError("Policy-Response Writer checkpoint authority changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Policy-Response Writer checkpoint file changed: {name}")
    return macro, contract


def _evaluation_tasks(
    runtime: PolicyResponseRuntime, evaluation: Mapping[str, Any]
) -> tuple[Any, ...]:
    expected = tuple(map(int, evaluation["target_global_ids"]))
    tasks = tuple(
        sorted(
            (
                task
                for task in runtime.task_by_id.values()
                if task.role in {"target_fit", "target_held", "target_validation"}
                and task.domain_task_id in set(expected)
            ),
            key=lambda task: task.domain_task_id,
        )
    )
    if tuple(task.domain_task_id for task in tasks) != expected:
        raise ValueError("Policy-Response Writer evaluation task authority changed")
    return tasks


def _target_keys(runtime: PolicyResponseRuntime) -> dict[int, tuple[str, int]]:
    path = authority_path(
        runtime.base, "target_manifest", asset_root=runtime.args.asset_root
    )
    manifest = read_json(path)
    return {
        int(row["global_task_id"]): (str(row["suite"]), int(row["task_id"]))
        for row in manifest["tasks"]
        if row["split_role"] in {"train", "validation"}
    }


def _freeze_and_inspect(runtime: PolicyResponseRuntime) -> dict[str, Any]:
    modules = (
        (runtime.policy, "policy"),
        (runtime.stage0, "stage0"),
        (runtime.writer, "writer"),
    )
    for module, _ in modules:
        module.requires_grad_(False).eval()
    action_meta = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in modules
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    trainable = [
        f"{prefix}.{name}"
        for root, prefix in modules
        for name, value in root.named_parameters()
        if value.requires_grad
    ]
    if action_meta or trainable or any(root.training for root, _ in modules):
        raise ValueError("Policy-Response Writer materialization wall changed")
    return {
        "action_meta_module_instances": action_meta,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": 0,
        "held_action_reads": 0,
        "held_reward_reads": 0,
        "held_state_reads": 0,
    }


def prepare_materialization_runtime(
    args: argparse.Namespace,
    resident: PolicyResponseRuntime | None = None,
) -> WriterMaterializationRuntime:
    if any(
        value is None
        for value in (
            args.evaluation_config,
            args.writer_run,
            args.writer_checkpoint,
        )
    ):
        raise ValueError("Policy-Response Writer materialization assets are required")
    evaluation = load_writer_evaluation_config(args.evaluation_config)
    if args.config != (REPO_ROOT / evaluation["training_config"]).resolve():
        raise ValueError("Policy-Response Writer materializer training config changed")
    macro, shared_contract = _load_writer_checkpoint(args, evaluation)
    deployment_tasks = _validation_deployment_tasks(args, evaluation)
    args.phase = "materialize"
    args.task = None
    args.video_demo = None
    args.representation = str(shared_contract["representation"])
    args.initialization = str(shared_contract["initialization_request"])
    args.mode = "formal"
    args.stop_after_step = None
    args.resume = None
    context = (
        resident.context if resident is not None else
        initialize_distributed(require_numa=True, defer_process_group=True)
    )
    if context.world_size != 1:
        raise ValueError("Policy-Response Writer materialization requires one GPU")
    if resident is not None:
        for field in ("config", "asset_root", "data_root", "writer_run"):
            if getattr(resident.args, field) != getattr(args, field):
                raise ValueError(f"resident materialization changed {field}")
        runtime = resident
        runtime.args = args
        seed_everything(int(runtime.config["optimization"]["seed"]), context)
    else:
        runtime = prepare_runtime(
            args, context,
            deployment_global_ids=tuple(map(int, evaluation["target_global_ids"])),
            deployment_tasks=deployment_tasks,
        )
    tasks = _evaluation_tasks(runtime, evaluation)
    if set(runtime.language_tokens) != {task.authority_id for task in tasks}:
        raise ValueError("resident materialization changed its task set")
    if evaluation["condition"].get("checkpoint_selection_use") is False:
        gradient = set(shared_contract["task_split"]["gradient_target"])
        for task in tasks:
            split = shared_contract["video_splits"].get(str(task.authority_id), {})
            role = evaluation["condition"]["video_role"]
            expected_demo = (
                split.get("fit", [None])[0]
                if role == "first_fit_video" else split.get("held")
            )
            if task.authority_id not in gradient or policy_response_video_demos(
                evaluation["condition"], task.domain_task_id
            ) != (expected_demo,):
                raise ValueError("train diagnostic escaped its registered task/video split")
    source_checkpoint = authority_path(
        runtime.base, "source_checkpoint", asset_root=args.asset_root
    )
    if str(source_checkpoint) != str(shared_contract.get("source_checkpoint")):
        raise ValueError("Policy-Response Writer materializer source changed")
    args.checkpoint = source_checkpoint
    args.source_run = source_checkpoint.parent.parent
    args.tokenizer_path = authority_path(
        runtime.base, "tokenizer", asset_root=args.asset_root
    )
    source = stage0_source_authority(args)
    if runtime.query_dataset is not None or runtime.query_processor is not None:
        raise ValueError("Policy-Response Writer materialization opened functional data")
    runtime.writer.load_state_dict(
        load_file(
            str(args.writer_checkpoint / "ecp.safetensors"),
            device=str(context.device),
        ),
        strict=True,
    )
    return WriterMaterializationRuntime(
        runtime=runtime,
        evaluation=evaluation,
        shared_contract=shared_contract,
        writer_macro=macro,
        tasks=tasks,
        target_keys=_target_keys(runtime),
        source=source,
        state=git_state(REPO_ROOT),
        wall=_freeze_and_inspect(runtime),
    )


def _adapter_record(
    runtime: WriterMaterializationRuntime,
    *,
    task: Any,
    checkpoint: Path,
    adapter_path: Path,
    adapter_bytes: int,
    manifest_path: Path,
) -> dict[str, Any]:
    suite, task_id = runtime.target_keys[task.domain_task_id]
    return {
        "suite": suite,
        "task_id": task_id,
        "natural_program_authority_id": task.authority_id,
        "global_task_id": task.domain_task_id,
        "language": task.language,
        "condition": "correct_k1",
        "representation": runtime.runtime.args.representation,
        "writer_macro": runtime.writer_macro,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_bytes": manifest_path.stat().st_size,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_bytes,
        "single_complete_rank16": True,
    }


def _capture_and_materialize(
    prepared: WriterMaterializationRuntime,
    *,
    task: Any,
    demos: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    runtime = prepared.runtime
    videos = []
    captures = []
    for demo in demos:
        video, capture = capture_video(
            runtime, task_id=task.authority_id, video_demo=demo
        )
        videos.append(video)
        captures.append(capture)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.writer(
            tuple(videos),
            s_ref=runtime.ranks.s_ref,
            representation=runtime.args.representation,
        )
        complete = runtime.writer.materialize(
            output,
            carrier_state=runtime.ranks.carrier_rank12,
            rank4_contract=runtime.rank4_contract,
            rank16_contract=runtime.ranks.contract,
            canonicalize=True,
        )
    validate_lora_state(complete, runtime.ranks.contract)
    return complete, captures


def _materialize_task(
    prepared: WriterMaterializationRuntime,
    *,
    task: Any,
    demos: tuple[int, ...],
    partial_root: Path,
    final_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete, captures = _capture_and_materialize(
        prepared, task=task, demos=demos
    )
    suite, task_id = prepared.target_keys[task.domain_task_id]
    relative = Path("adapters") / f"{suite}_task_{task_id:02d}"
    write_root = partial_root / relative
    final_checkpoint = final_root / relative
    write_root.mkdir(parents=True)
    adapter_path = write_root / "adapter.safetensors"
    save_file(
        {
            name: value.detach().float().cpu().contiguous()
            for name, value in complete.items()
        },
        str(adapter_path),
    )
    manifest_path = write_root / "manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": MATERIALIZED_ADAPTER_SCHEMA,
            "condition": "correct_k1",
            "representation": prepared.runtime.args.representation,
            "writer_macro": prepared.writer_macro,
            "writer_checkpoint": str(prepared.runtime.args.writer_checkpoint),
            "authority_id": task.authority_id,
            "global_task_id": task.domain_task_id,
            "suite": suite,
            "task_id": task_id,
            "language": task.language,
            "video_demos": list(demos),
            "capture": captures,
            "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
            "single_complete_rank16": True,
            "files": {"adapter.safetensors": adapter_path.stat().st_size},
        },
    )
    record = _adapter_record(
        prepared,
        task=task,
        checkpoint=final_checkpoint,
        adapter_path=final_checkpoint / "adapter.safetensors",
        adapter_bytes=adapter_path.stat().st_size,
        manifest_path=manifest_path,
    )
    del complete
    torch.cuda.empty_cache()
    return record, captures


def _bank_payload(
    prepared: WriterMaterializationRuntime,
    *,
    records: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    lora_path: Path,
) -> dict[str, Any]:
    representation = str(prepared.runtime.args.representation)
    return {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "evaluation_role": prepared.evaluation["evaluation_role"],
        "arm": f"ecp_policy_response_writer_{representation}_correct_k1",
        "source": prepared.source,
        "lora_contract": {"path": str(lora_path), "bytes": lora_path.stat().st_size},
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": str(prepared.shared_contract["git"]["commit"]),
        "materialization_commit": str(prepared.state["commit"]),
        "shared_run_contract": prepared.shared_contract,
        "writer_checkpoint": {
            "path": str(prepared.runtime.args.writer_checkpoint),
            "macro": prepared.writer_macro,
        },
        "condition": {
            **prepared.evaluation["condition"],
            "representation": representation,
        },
        "tasks": records,
        "information_wall": {
            "deployment_inputs": [
                "exact language",
                "same-task action-hidden internally ordered videos",
            ],
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
            "writer_invocations_per_task_condition": 1,
            "total_writer_invocations": len(records),
            "materialization_teacher_video_count": len(captures),
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "shuffled_or_reversed_use": False,
            "wrong_video_use": False,
            **prepared.wall,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _seal_bank(
    *,
    partial_root: Path,
    final_root: Path,
    payload: Mapping[str, Any],
    representation: str,
    writer_macro: int,
) -> None:
    write_json_atomic(partial_root / "manifest.json", payload)
    write_json_atomic(
        partial_root / "completion.json",
        {
            "schema_version": "ember_ecp_policy_response_writer_materialization_completion_v1",
            "condition": "correct_k1",
            "representation": representation,
            "tasks": len(payload["tasks"]),
            "writer_macro": writer_macro,
        },
    )
    partial_root.rename(final_root)


def _materialize_prepared_bank(
    prepared: WriterMaterializationRuntime,
) -> dict[str, Any]:
    args = prepared.runtime.args
    final_root = args.output_dir
    partial_root = final_root.parent / f".{final_root.name}.partial-{os.getpid()}"
    if final_root.exists() or partial_root.exists():
        raise ValueError("Policy-Response Writer materialization output already exists")
    partial_root.mkdir(parents=True)
    condition = prepared.evaluation["condition"]
    records: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for task in prepared.tasks:
            demos = policy_response_video_demos(condition, task.domain_task_id)
            record, task_captures = _materialize_task(
                prepared,
                task=task,
                demos=demos,
                partial_root=partial_root,
                final_root=final_root,
            )
            records.append(record)
            captures.extend(task_captures)
    lora_path = authority_path(
        prepared.runtime.base, "lora_contract", asset_root=args.asset_root
    )
    payload = _bank_payload(
        prepared, records=records, captures=captures, lora_path=lora_path
    )
    representation = str(prepared.runtime.args.representation)
    _seal_bank(
        partial_root=partial_root,
        final_root=final_root,
        payload=payload,
        representation=representation,
        writer_macro=prepared.writer_macro,
    )
    return payload


def materialize_writer_evaluation_bank(args: argparse.Namespace) -> dict[str, Any]:
    """Seal registered conditions while loading the frozen runtime only once."""
    jobs = [args]
    for evaluation, checkpoint, output in getattr(args, "additional_materialization", ()):
        jobs.append(argparse.Namespace(**{
            **vars(args), "evaluation_config": Path(evaluation).resolve(),
            "writer_checkpoint": Path(checkpoint).resolve(),
            "output_dir": Path(output).resolve(),
        }))
    if len({job.output_dir for job in jobs}) != len(jobs):
        raise ValueError("duplicate materialization output in batch")
    if any(job.output_dir.exists() for job in jobs):
        raise ValueError("materialization output already exists")
    resident = None
    first_payload = None
    try:
        for job in jobs:
            started = time.monotonic()
            prepared = prepare_materialization_runtime(job, resident)
            resident = prepared.runtime
            ready = time.monotonic()
            payload = _materialize_prepared_bank(prepared)
            if first_payload is None:
                first_payload = payload
            print({"event": "materialization_sealed", "output": str(job.output_dir),
                   "runtime_prepare_seconds": ready - started,
                   "materialize_seconds": time.monotonic() - ready,
                   "tasks": len(payload["tasks"])}, flush=True)
    finally:
        if resident is not None:
            resident.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return first_payload
