#!/usr/bin/env python3
"""Seal the macro1-40 fit-task K1 native-feasible teacher union for G3."""

from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.natural_program_data import (
    NaturalProgramSample,
    NaturalProgramSchedule,
    load_natural_program_tasks,
)
from ember.ecp.shared_compiler_assets import (
    authority_path,
    load_shared_compiler_config,
    load_shared_task_members,
)
from ember.ecp.shared_compiler_native_teacher import (
    G3_NATIVE_TEACHER_FORMAL_MACROS,
    G3_NATIVE_TEACHER_TASK_SCHEMA,
    G3_NATIVE_TEACHER_WORKER_SCHEMA,
    native_teacher_from_lora_state,
    publish_native_teacher_root,
    write_native_teacher_task_shard,
)
from ember.ecp.shared_compiler_span import (
    capture_k1_native_readout,
    project_member_into_k1_native_span,
)
from ember.ecp.shared_compiler_training import prepare_runtime
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tasks_and_members(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    fold = config["fold"]
    tasks = load_natural_program_tasks(
        meta_protocol_path=authority_path(
            config, "meta_protocol", asset_root=args.asset_root
        ),
        source_manifest_path=authority_path(
            config, "source_manifest", asset_root=args.asset_root
        ),
        target_manifest_path=authority_path(
            config, "target_manifest", asset_root=args.asset_root
        ),
        data_root=args.data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )
    members = load_shared_task_members(config, tasks, asset_root=args.asset_root)
    return tasks, members


def _k1_union(
    tasks: Sequence[Any], schedule: NaturalProgramSchedule
) -> dict[int, dict[int, tuple[int, ...]]]:
    """Return task -> video -> one-based macros, deduplicating repeated K1 visits."""

    fit = {
        task.authority_id: task
        for task in tasks
        if task.role in {"meta_fit", "target_fit"}
    }
    visits: dict[int, dict[int, list[int]]] = {}
    for macro in range(G3_NATIVE_TEACHER_FORMAL_MACROS):
        for task_id in schedule.training_task_ids(macro):
            sample = schedule.sample(task_id, macro)
            if sample.k != 1:
                continue
            if len(sample.video_demos) != 1 or task_id not in fit:
                raise ValueError("G3 K1 teacher schedule crossed its fit authority")
            visits.setdefault(task_id, {}).setdefault(sample.video_demos[0], []).append(
                macro + 1
            )
    role_counts = {
        role: sum(fit[task_id].role == role for task_id in visits)
        for role in ("meta_fit", "target_fit")
    }
    if len(visits) != 50 or role_counts != {"meta_fit": 31, "target_fit": 19}:
        raise ValueError(
            "G3 macro1-40 K1-covered fit subset changed: "
            f"tasks={len(visits)} roles={role_counts}"
        )
    return {
        task_id: {
            demo: tuple(macros) for demo, macros in sorted(rows.items())
        }
        for task_id, rows in sorted(visits.items())
    }


def _shards(
    tasks: Sequence[Any], union: Mapping[int, Mapping[int, Sequence[int]]], count: int
) -> tuple[tuple[int, ...], ...]:
    if not 0 < count <= len(union):
        raise ValueError("invalid G3 native-teacher shard count")
    by_id = {task.authority_id: task for task in tasks}
    costs = {
        task_id: sum(by_id[task_id].episode_lengths[demo] for demo in demos)
        for task_id, demos in union.items()
    }
    groups: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    for task_id in sorted(costs, key=lambda row: (-costs[row], row)):
        shard = min(range(count), key=lambda row: (loads[row], row))
        groups[shard].append(task_id)
        loads[shard] += costs[task_id]
    return tuple(tuple(sorted(group)) for group in groups)


def _projection_scales(
    initialization: Mapping[str, Any], runtime: Any
) -> torch.Tensor:
    rows = tuple(initialization.get("targets", ()))
    if len(rows) != len(runtime.rank4_contract.targets):
        raise ValueError("G3 native teacher projection scale rows changed")
    ratios = []
    for target, row in zip(runtime.rank4_contract.targets, rows, strict=True):
        values = torch.tensor(
            row.get("scale_to_s_ref", ()),
            device=runtime.context.device,
            dtype=torch.float32,
        )
        if row.get("target") != target.name or values.shape != (4,):
            raise ValueError("G3 native teacher projection scale target changed")
        ratios.append(values)
    return torch.stack(ratios) * runtime.ranks.s_ref[:, None].float()


def _projection_summary(
    initialization: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    targets = tuple(initialization["targets"])
    scale_ratios = [
        float(value) for row in targets for value in row["scale_to_s_ref"]
    ]
    return {
        "kind": initialization["kind"],
        "reference_member": initialization["reference_member"],
        "relative_singular_threshold": initialization[
            "relative_singular_threshold"
        ],
        "probability_floor_mass": initialization["probability_floor_mass"],
        "minimum_input_direction_cosine": min(
            float(value)
            for row in targets
            for value in row["input_direction_cosine"]
        ),
        "minimum_output_direction_cosine": min(
            float(value)
            for row in targets
            for value in row["output_direction_cosine"]
        ),
        "median_scale_to_s_ref": statistics.median(scale_ratios),
        "scale_cap_fraction": sum(value >= 1.0 - 2e-6 for value in scale_ratios)
        / len(scale_ratios),
        "update_geometry": dict(geometry),
    }


def _runtime_args(args: argparse.Namespace, worker_dir: Path) -> argparse.Namespace:
    runtime = copy.copy(args)
    runtime.mode = "profile"
    runtime.output_dir = (worker_dir / "runtime").resolve()
    runtime.stop_after_macro = 1
    runtime.resume = None
    runtime.log_every = 1
    return runtime


def _actual_native_only(runtime: Any) -> None:
    action_meta = [
        module
        for root in (runtime.policy, runtime.program)
        for module in root.modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    if (
        action_meta
        or runtime.config["information_wall"].get("action_meta_installed") is not False
        or any(parameter.requires_grad for parameter in runtime.policy.parameters())
        or any(parameter.requires_grad for parameter in runtime.program.parameters())
    ):
        raise ValueError("G3 native teacher sealer did not load pure Native Stage 0")


def _seal_task(
    *,
    runtime: Any,
    worker_dir: Path,
    task: Any,
    member_assets: Any,
    demos: Mapping[int, Sequence[int]],
    threshold: float,
    floor: float,
) -> dict[str, Any]:
    bank = runtime.effect_banks.get(task.authority_id)
    if tuple(member.name for member in member_assets.members) != bank.member_names:
        raise ValueError("G3 native teacher member authority changed")
    teachers = []
    for video_demo, macros in demos.items():
        sample = NaturalProgramSample(
            video_demos=(video_demo,),
            action_demos=(),
            k=1,
            robustness_view="g3_native_teacher_k1",
        )
        native, video_metrics = capture_k1_native_readout(runtime, task, sample)
        for member_index, (member, reference) in enumerate(
            zip(member_assets.members, bank.projections, strict=True)
        ):
            state, initialized, geometry = project_member_into_k1_native_span(
                runtime,
                task_id=task.authority_id,
                member_index=member_index,
                member_name=member.name,
                native=native,
                reference=reference,
                relative_singular_threshold=threshold,
                probability_floor_mass=floor,
            )
            teachers.append(
                native_teacher_from_lora_state(
                    authority_id=task.authority_id,
                    video_demo=video_demo,
                    member_name=member.name,
                    state=state,
                    scales=_projection_scales(initialized, runtime),
                    contract=runtime.rank4_contract,
                    provenance={
                        "scheduled_macros": list(macros),
                        "member": {
                            "name": member.name,
                            "step": member.step,
                            "adapter": str(member.adapter),
                            "adapter_bytes": member.adapter_bytes,
                            "successes": member.successes,
                        },
                        "video_capture": dict(video_metrics),
                        "projection": _projection_summary(initialized, geometry),
                    },
                )
            )
            del state
        del native
        torch.cuda.empty_cache()
    return write_native_teacher_task_shard(
        worker_dir=worker_dir,
        task={
            "authority_id": task.authority_id,
            "domain": task.domain,
            "domain_task_id": task.domain_task_id,
            "role": task.role,
            "suite": task.suite,
            "language": task.language,
        },
        teachers=teachers,
        contract=runtime.rank4_contract,
        provenance={
            "solver": "g1_fp64_stable_span_reference_projection",
            "formal_schedule_macros": G3_NATIVE_TEACHER_FORMAL_MACROS,
            "K": 1,
            "video_action_reads": 0,
            "action_meta_installed": False,
        },
    )


def worker(args: argparse.Namespace) -> None:
    config = load_shared_compiler_config(args.config)
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("G3 native-teacher sharding uses independent single-GPU workers")
    runtime = None
    worker_dir = (
        args.output_dir / "workers" / f"worker_{args.shard_index:03d}"
    ).resolve()
    if worker_dir.exists() or (args.output_dir / "manifest.json").exists():
        raise ValueError("G3 native-teacher worker output already exists")
    try:
        runtime = prepare_runtime(_runtime_args(args, worker_dir), context)
        _actual_native_only(runtime)
        union = _k1_union(runtime.tasks, runtime.schedule)
        assigned = _shards(runtime.tasks, union, args.shard_count)[args.shard_index]
        members_by_id = {
            row.task.authority_id: row for row in runtime.members
        }
        task_by_id = {task.authority_id: task for task in runtime.tasks}
        g1 = read_json(args.g1_config)
        initialization = g1.get("optimization", {}).get("initialization", {})
        if g1.get("schema_version") != "ember_ecp_native_factor_g1_v1":
            raise ValueError("G3 native teacher lost the G1 projection authority")
        threshold = float(initialization["relative_singular_threshold"])
        floor = float(initialization["probability_floor_mass"])
        records = []
        with torch.no_grad():
            for task_id in assigned:
                records.append(
                    _seal_task(
                        runtime=runtime,
                        worker_dir=worker_dir,
                        task=task_by_id[task_id],
                        member_assets=members_by_id[task_id],
                        demos=union[task_id],
                        threshold=threshold,
                        floor=floor,
                    )
                )
        write_json_atomic(
            worker_dir / "completion.json",
            {
                "schema_version": G3_NATIVE_TEACHER_WORKER_SCHEMA,
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "task_ids": list(assigned),
                "task_count": len(records),
                "records": records,
                "git": git_state(REPO_ROOT),
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    context.device
                ),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _validate_task_manifest(
    record: Mapping[str, Any],
    *,
    task: Any,
    demos: Mapping[int, Sequence[int]],
    member_names: Sequence[str],
) -> None:
    path = Path(str(record.get("manifest", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(
        record.get("manifest_bytes", -1)
    ):
        raise ValueError("G3 native-teacher aggregate lost a task manifest")
    manifest = read_json(path)
    tensor = manifest.get("tensor_file", {})
    tensor_path = Path(str(tensor.get("path", ""))).resolve()
    expected = len(demos) * len(member_names)
    if (
        manifest.get("schema_version") != G3_NATIVE_TEACHER_TASK_SCHEMA
        or manifest.get("status") != "complete"
        or int(manifest.get("task", {}).get("authority_id", -1))
        != task.authority_id
        or manifest.get("task", {}).get("role") != task.role
        or manifest.get("video_demos") != sorted(demos)
        or manifest.get("member_names") != sorted(member_names)
        or int(manifest.get("teacher_count", -1)) != expected
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != int(tensor.get("bytes", -1))
    ):
        raise ValueError("G3 native-teacher aggregate found a changed task shard")


def aggregate(args: argparse.Namespace) -> None:
    config = load_shared_compiler_config(args.config)
    tasks, members = _tasks_and_members(args, config)
    schedule = NaturalProgramSchedule(
        tasks, seed=int(config["optimization"]["seed"]), query_points=2
    )
    union = _k1_union(tasks, schedule)
    shards = _shards(tasks, union, args.shard_count)
    task_by_id = {task.authority_id: task for task in tasks}
    members_by_id = {row.task.authority_id: row for row in members}
    records = []
    worker_git = None
    for shard_index, expected_ids in enumerate(shards):
        path = (
            args.output_dir
            / "workers"
            / f"worker_{shard_index:03d}"
            / "completion.json"
        )
        completion = read_json(path)
        if (
            completion.get("schema_version") != G3_NATIVE_TEACHER_WORKER_SCHEMA
            or completion.get("status") != "complete"
            or int(completion.get("shard_index", -1)) != shard_index
            or int(completion.get("shard_count", -1)) != args.shard_count
            or completion.get("task_ids") != list(expected_ids)
            or int(completion.get("task_count", -1)) != len(expected_ids)
        ):
            raise ValueError("G3 native-teacher worker completion changed")
        if worker_git is None:
            worker_git = completion.get("git")
        elif completion.get("git") != worker_git:
            raise ValueError("G3 native-teacher workers used different Git authorities")
        worker_records = tuple(completion.get("records", ()))
        if len(worker_records) != len(expected_ids):
            raise ValueError("G3 native-teacher worker record count changed")
        by_id = {int(row.get("authority_id", -1)): row for row in worker_records}
        if set(by_id) != set(expected_ids):
            raise ValueError("G3 native-teacher worker task ownership changed")
        for task_id in expected_ids:
            names = tuple(
                member.name for member in members_by_id[task_id].members
            )
            _validate_task_manifest(
                by_id[task_id],
                task=task_by_id[task_id],
                demos=union[task_id],
                member_names=names,
            )
            records.append(by_id[task_id])
    if set(int(row["authority_id"]) for row in records) != set(union):
        raise ValueError("G3 native-teacher aggregate lost fit-task coverage")
    rank16 = load_pi05_lora_contract(
        authority_path(config, "lora_contract", asset_root=args.asset_root)
    )
    publish_native_teacher_root(
        output_dir=args.output_dir,
        records=records,
        contract=derive_pi05_lora_rank(rank16, rank=4),
        fit_authority_roles={
            task.authority_id: task.role
            for task in tasks
            if task.role in {"meta_fit", "target_fit"}
        },
        provenance={
            "config": str(args.config),
            "g1_config": str(args.g1_config),
            "formal_schedule_macros": G3_NATIVE_TEACHER_FORMAL_MACROS,
            "unique_K1_task_video_pairs": sum(len(rows) for rows in union.values()),
            "worker_git": worker_git,
            "publisher_git": git_state(REPO_ROOT),
            "claim_boundary": (
                "fit-task K1 training supervision only; no shared mapping or "
                "closed-loop Gate claim"
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v1.json",
    )
    parser.add_argument(
        "--g1-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--effect-bank-root", type=Path)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "g1_config",
        "asset_root",
        "data_root",
        "output_dir",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "effect_bank_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    worker_values = (
        args.shard_index,
        args.source_run,
        args.checkpoint,
        args.tokenizer_path,
        args.effect_bank_root,
    )
    if args.shard_count <= 0 or (
        args.command == "worker"
        and (
            any(value is None for value in worker_values)
            or not 0 <= args.shard_index < args.shard_count
        )
    ):
        raise ValueError("invalid G3 native-teacher shard request")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
