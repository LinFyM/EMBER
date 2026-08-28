"""Frozen single-GPU workers for G3 mapping qualification."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_data import NaturalProgramTask, load_natural_program_tasks
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    G3_CONFIG_SCHEMA,
    SharedCompilerRankAssets,
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.shared_compiler_authority import MAPPING_RUN_SCHEMA
from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSplit,
    load_mapping_split,
    paired_mapping_loss,
)
from ember.ecp.bank_conditioning.mapping_step import (
    load_mapping_condition_teachers,
    mapping_condition_output,
    mapping_recovery_record,
    prepare_mapping_condition,
    prepare_mapping_condition_program,
)
from ember.ecp.bank_conditioning.program_causality import (
    ProgramCausalityPair,
    load_program_causality_contract,
    program_causality_extra_costs,
    program_causality_pairs,
)
from ember.ecp.shared_compiler_native_teacher import NativeTeacherStore
from ember.ecp.stage0_training import stage0_source_authority, tokenize_stage0_languages
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_setup import (
    initialize_distributed,
    load_config,
    load_policy,
    seed_everything,
)
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_SCHEMA = "ember_ecp_shared_compiler_mapping_evaluation_v2"
FAMILY_NAMES = ("q", "v", "action_in", "action_out")
SPLIT_NAMES = ("fit", "video_holdout", "task_holdout")


@dataclass
class MappingEvaluationRuntime:
    config: dict[str, Any]
    context: DistributedContext
    task_by_id: dict[int, NaturalProgramTask]
    mapping_split: SharedCompilerMappingSplit
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    owners: tuple[TargetOwner, ...]
    ranks: SharedCompilerRankAssets
    native_teachers: NativeTeacherStore
    query_points: int
    checkpoint_macro: int
    training_contract: dict[str, Any]
    program_causality_contract: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()


def load_mapping_tasks(
    config: Mapping[str, Any], *, asset_root: Path, data_root: Path
) -> tuple[NaturalProgramTask, ...]:
    fold = config["fold"]
    return load_natural_program_tasks(
        meta_protocol_path=authority_path(
            config, "meta_protocol", asset_root=asset_root
        ),
        source_manifest_path=authority_path(
            config, "source_manifest", asset_root=asset_root
        ),
        target_manifest_path=authority_path(
            config, "target_manifest", asset_root=asset_root
        ),
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _load_compiler_checkpoint(
    compiler: SharedNativeFactorCompiler,
    *,
    phase: str,
    config_path: Path,
    training_run: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    training_run = training_run.resolve()
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    contract = read_json(training_run / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    completion = read_json(training_run / "segment_completion.json")
    tensor = checkpoint / "ecp.safetensors"
    tensor_record = manifest.get("files", {}).get("ecp.safetensors", {})
    expected_stage = f"g3_mapping_{phase}"
    if (
        checkpoint.parent.parent != training_run
        or contract.get("schema_version") != MAPPING_RUN_SCHEMA
        or contract.get("stage") != expected_stage
        or contract.get("phase") != phase
        or contract.get("mode") != "formal"
        or contract.get("config", {}).get("bytes") != config_path.stat().st_size
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != expected_stage
        or manifest.get("run_contract_schema") != MAPPING_RUN_SCHEMA
        or int(manifest.get("next_macro", -1)) != macro
        or int(completion.get("completed_macros", -1)) < macro
        or not tensor.is_file()
        or tensor.stat().st_size != int(tensor_record.get("bytes", -1))
    ):
        raise ValueError("G3 mapping evaluation checkpoint authority changed")
    compiler.load_state_dict(load_file(str(tensor), device=str(device)), strict=True)
    compiler.requires_grad_(False).eval()
    return macro, contract


def _evaluation_wall(
    *,
    policy: torch.nn.Module,
    program: torch.nn.Module,
    compiler: torch.nn.Module,
) -> dict[str, Any]:
    action_meta = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (program, "program"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    trainable = [
        f"{prefix}.{name}"
        for root, prefix in (
            (policy, "policy"),
            (program, "program"),
            (compiler, "compiler"),
        )
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    ]
    signature = list(inspect.signature(compiler.forward).parameters)
    if (
        action_meta
        or trainable
        or policy.training
        or program.training
        or compiler.training
        or signature != ["program", "videos", "s_ref"]
    ):
        raise ValueError("G3 mapping evaluation information wall changed")
    return {
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "trainable_parameter_count": 0,
        "compiler_forward_signature": signature,
        "validation_or_test_gradient_count": 0,
        "shuffled_or_reversed_use": False,
    }


def prepare_mapping_evaluation_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> MappingEvaluationRuntime:
    if context.world_size != 1:
        raise ValueError("mapping evaluation workers are independent single-GPU jobs")
    state = git_state(REPO_ROOT)
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal mapping evaluation requires clean detached authority")
    config = load_shared_compiler_config(args.config)
    if config.get("schema_version") != G3_CONFIG_SCHEMA:
        raise ValueError("mapping evaluation requires the active G3 config")
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = load_mapping_tasks(
        config, asset_root=args.asset_root, data_root=args.data_root
    )
    mapping_split = load_mapping_split(config, asset_root=args.asset_root)
    source_args = SimpleNamespace(
        checkpoint=args.source_checkpoint, source_run=args.source_run
    )
    expected_source = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    expected_tokenizer = authority_path(config, "tokenizer", asset_root=args.asset_root)
    if (
        args.source_checkpoint != expected_source
        or args.source_run != expected_source.parent.parent
        or args.tokenizer_path != expected_tokenizer
    ):
        raise ValueError("mapping evaluation source authority changed")
    source = stage0_source_authority(source_args)
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        config,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    program = build_frozen_g2_program(
        config, asset_root=args.asset_root, owners=owners, device=context.device
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(config["model"]["program_width"]),
        event_slots=int(config["model"]["event_slots"]),
        relative_eigenvalue_floor=float(
            config["model"]["relative_eigenvalue_floor"]
        ),
        replay_score_rms=float(config["model"]["replay_score_rms"]),
    ).to(context.device)
    macro, training_contract = _load_compiler_checkpoint(
        compiler,
        phase=args.phase,
        config_path=args.config,
        training_run=args.training_run,
        checkpoint=args.compiler_checkpoint,
        device=context.device,
    )
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    teacher_root_path = authority_path(
        config, "native_teacher_manifest", asset_root=args.asset_root
    )
    teacher_root = read_json(teacher_root_path)
    native_teachers = NativeTeacherStore(
        teacher_root_path,
        contract=rank4_contract,
        expected_fit_task_ids=set(map(int, teacher_root["coverage"]["task_ids"])),
        expected_full_fit_task_ids=set(
            map(int, teacher_root["fit_authority_task_ids"])
        ),
        device=context.device,
    )
    video_store = RawTeacherVideoStore(
        tuple(task.writer_authority() for task in tasks),
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=8,
    )
    language_tokens = tokenize_stage0_languages(
        tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    g2 = read_json(authority_path(config, "g2_config", asset_root=args.asset_root))
    program_causality_contract = load_program_causality_contract(
        args.program_causality_contract
    )
    _evaluation_wall(policy=policy, program=program, compiler=compiler)
    return MappingEvaluationRuntime(
        config=config,
        context=context,
        task_by_id={task.authority_id: task for task in tasks},
        mapping_split=mapping_split,
        video_store=video_store,
        language_tokens=language_tokens,
        policy=policy,
        program=program,
        compiler=compiler,
        owners=owners,
        ranks=ranks,
        native_teachers=native_teachers,
        query_points=int(g2["data"]["query_points"]),
        checkpoint_macro=macro,
        training_contract=training_contract,
        program_causality_contract=program_causality_contract,
    )


def labeled_mapping_conditions(
    split: SharedCompilerMappingSplit,
) -> tuple[tuple[str, MappingCondition], ...]:
    return tuple(
        (name, condition)
        for name, rows in (
            ("fit", split.fit),
            ("video_holdout", split.video_held),
            ("task_holdout", split.task_held),
        )
        for condition in rows
    )


def balanced_mapping_assignments(
    rows: Sequence[tuple[str, MappingCondition]],
    worker_count: int,
    *,
    extra_costs: Mapping[tuple[int, int], int] | None = None,
) -> tuple[tuple[tuple[str, MappingCondition], ...], ...]:
    if not 1 <= worker_count <= 6:
        raise ValueError("mapping evaluation worker count must be in [1, 6]")
    assignments: list[list[tuple[str, MappingCondition]]] = [
        [] for _ in range(worker_count)
    ]
    loads = [0] * worker_count
    extras = extra_costs or {}
    cost = lambda value: value.sampled_frames + int(
        extras.get((value.authority_id, value.video_demo), 0)
    )
    for row in sorted(
        rows,
        key=lambda value: (
            -cost(value[1]),
            value[0],
            value[1].authority_id,
            value[1].video_demo,
        ),
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(row)
        loads[worker] += cost(row[1])
    return tuple(tuple(values) for values in assignments)


def _worker_contract(
    runtime: MappingEvaluationRuntime,
    args: argparse.Namespace,
    assigned: Sequence[tuple[str, MappingCondition]],
    causal_pairs: Sequence[ProgramCausalityPair],
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA,
        "phase": args.phase,
        "checkpoint": str(args.compiler_checkpoint),
        "checkpoint_macro": runtime.checkpoint_macro,
        "training_commit": runtime.training_contract["git"]["commit"],
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "condition_count": len(assigned),
        "program_causality_condition_count": len(causal_pairs),
        "sampled_frame_cost": sum(row.sampled_frames for _, row in assigned),
        "program_causality_extra_cost": sum(
            pair.primary.sampled_frames + pair.wrong.sampled_frames
            for pair in causal_pairs
        ),
        "program_causality_contract": str(args.program_causality_contract),
        "program_causality_contract_bytes": (
            args.program_causality_contract.stat().st_size
        ),
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "information_wall": _evaluation_wall(
            policy=runtime.policy,
            program=runtime.program,
            compiler=runtime.compiler,
        ),
    }


def _assigned_program_pairs(
    split: SharedCompilerMappingSplit,
    assigned: Sequence[tuple[str, MappingCondition]],
) -> tuple[ProgramCausalityPair, ...]:
    keys = {
        (condition.authority_id, condition.video_demo)
        for _, condition in assigned
    }
    return tuple(
        pair
        for pair in program_causality_pairs(split)
        if (pair.primary.authority_id, pair.primary.video_demo) in keys
    )


def _evaluate_condition(
    runtime: MappingEvaluationRuntime,
    *,
    split_name: str,
    condition: MappingCondition,
    causal_pair: ProgramCausalityPair | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    tick = time.monotonic()
    temperature = float(runtime.config["optimization"]["mapping"]["temperature"])
    with torch.no_grad():
        prepared = prepare_mapping_condition(runtime, condition)
        output, metrics = mapping_condition_output(runtime, condition, prepared)
        teachers = load_mapping_condition_teachers(runtime, condition)
        loss = paired_mapping_loss(
            output=output,
            teachers=teachers,
            owners=runtime.owners,
            temperature=temperature,
        )
    record = {
        "split": split_name,
        "authority_id": condition.authority_id,
        "role": condition.role,
        "video_demo": condition.video_demo,
        "sampled_frames": condition.sampled_frames,
        **mapping_recovery_record(loss),
        "condition_metrics": metrics,
        "condition_seconds": time.monotonic() - tick,
    }
    if causal_pair is None:
        return record, None

    causal_tick = time.monotonic()
    with torch.no_grad():
        wrong_program = prepare_mapping_condition_program(
            runtime, causal_pair.wrong
        )
        wrong_output, _ = mapping_condition_output(
            runtime, condition, prepared, program=wrong_program
        )
        wrong_loss = paired_mapping_loss(
            output=wrong_output,
            teachers=teachers,
            owners=runtime.owners,
            temperature=temperature,
        )
    causal_record = {
        "authority_id": condition.authority_id,
        "role": condition.role,
        "video_demo": condition.video_demo,
        "wrong_authority_id": causal_pair.wrong.authority_id,
        "wrong_video_demo": causal_pair.wrong.video_demo,
        "correct": mapping_recovery_record(loss),
        "wrong": mapping_recovery_record(wrong_loss),
        "causal_seconds": time.monotonic() - causal_tick,
    }
    return record, causal_record


def evaluate_mapping_worker(args: argparse.Namespace) -> None:
    context = initialize_distributed(require_numa=True)
    runtime: MappingEvaluationRuntime | None = None
    try:
        runtime = prepare_mapping_evaluation_runtime(args, context)
        rows = labeled_mapping_conditions(runtime.mapping_split)
        assigned = balanced_mapping_assignments(
            rows,
            args.worker_count,
            extra_costs=program_causality_extra_costs(runtime.mapping_split),
        )[args.worker_index]
        causal_pairs = _assigned_program_pairs(runtime.mapping_split, assigned)
        causal_by_key = {
            (pair.primary.authority_id, pair.primary.video_demo): pair
            for pair in causal_pairs
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / f"worker_{args.worker_index:02d}_contract.json"
        rows_path = args.output_dir / f"worker_{args.worker_index:02d}_rows.jsonl"
        causal_path = (
            args.output_dir
            / f"worker_{args.worker_index:02d}_program_causality_rows.jsonl"
        )
        completion_path = (
            args.output_dir / f"worker_{args.worker_index:02d}_completion.json"
        )
        if any(
            path.exists()
            for path in (contract_path, rows_path, causal_path, completion_path)
        ):
            raise ValueError("mapping evaluation worker output already exists")
        contract = _worker_contract(runtime, args, assigned, causal_pairs)
        write_json_atomic(contract_path, contract)
        started = time.monotonic()
        with (
            rows_path.open("x", encoding="utf-8") as handle,
            causal_path.open("x", encoding="utf-8") as causal_handle,
        ):
            for split_name, condition in assigned:
                record, causal_record = _evaluate_condition(
                    runtime,
                    split_name=split_name,
                    condition=condition,
                    causal_pair=causal_by_key.get(
                        (condition.authority_id, condition.video_demo)
                    ),
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if causal_record is not None:
                    causal_handle.write(
                        json.dumps(causal_record, sort_keys=True) + "\n"
                    )
                    causal_handle.flush()
        write_json_atomic(
            completion_path,
            {
                **contract,
                "elapsed_seconds": time.monotonic() - started,
                "native_teacher_tensor_reads": runtime.native_teachers.tensor_reads,
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    context.device
                ),
                "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                    context.device
                ),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
