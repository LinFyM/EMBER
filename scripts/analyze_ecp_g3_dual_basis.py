#!/usr/bin/env python3
"""Fit and replay the fit-only G3 K1 analytic-dual basis capacity probe.

The four commands form one retained evidence path: workers seal compact FP64
minimum-norm dual labels, aggregation fits task-balanced leave-one-task-out
bases, replay returns those bases to the real native banks, and report applies
the preregistered four-family capacity thresholds.  This never alters the active
compiler and cannot prove the shared Program-to-coefficient mapping.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.native_factors import (
    NativeOutputBankState,
    native_output_group_count,
)
from ember.ecp.natural_program_data import (
    NaturalProgramSample,
    NaturalProgramSchedule,
)
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.ecp.shared_compiler_native_teacher import (
    G3_NATIVE_TEACHER_FORMAL_MACROS,
    NativeTeacherFactors,
)
from ember.ecp.shared_compiler_dual_basis import (
    BASIS_SCHEMA,
    DEFAULT_BASIS_DIMENSIONS,
    DEFAULT_TARGETS,
    REPLAY_SCHEMA,
    WORKER_SCHEMA,
    factor_cosines as _factor_cosines,
    projected_duals as _projected_queries,
    signed_pool_output_groups as _signed_pool_output_groups,
    signed_pool_queries as _signed_pool_queries,
    stable_dual_factor as _stable_factor,
    tensor_prefix as _tensor_prefix,
    update_geometry as _update_geometry,
)
from ember.ecp.shared_compiler_dual_basis_artifacts import (
    aggregate as _aggregate_artifacts,
    capture_workers as _capture_workers,
    report as _report_artifacts,
    save_safetensors_atomic as _save_safetensors_atomic,
    validate_formal_arguments as _validate_formal_arguments,
    validate_formal_git as _validate_formal_git,
)
from ember.ecp.shared_compiler_training import prepare_runtime
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from scripts.seal_ecp_g3_native_teachers import (
    _actual_native_only,
    _k1_union,
    _runtime_args,
    _shards,
)


SELECTION_RULE = (
    "per task, retain distinct K1 videos by first occurrence in the existing "
    "formal macro1-40 schedule; ties use video_demo only as a deterministic "
    "fallback; never select by teacher, dual, or outcome"
)


def _selected_union(
    tasks: Sequence[Any],
    schedule: NaturalProgramSchedule,
    *,
    max_videos_per_task: int,
    task_ids: Sequence[int] | None,
) -> dict[int, dict[int, tuple[int, ...]]]:
    full = _k1_union(tasks, schedule)
    if max_videos_per_task <= 0:
        raise ValueError("--max-videos-per-task must be positive")
    requested = set(full) if task_ids is None else set(map(int, task_ids))
    if not requested or not requested <= set(full):
        raise ValueError("--task-ids crossed or missed the formal K1 fit subset")
    selected: dict[int, dict[int, tuple[int, ...]]] = {}
    for task_id in sorted(requested):
        ordered = sorted(
            full[task_id].items(), key=lambda row: (min(row[1]), row[0])
        )
        selected[task_id] = dict(ordered[:max_videos_per_task])
    if any(not rows for rows in selected.values()):
        raise RuntimeError("deterministic K1 selection produced an empty task")
    return selected


def _selection_record(
    full: Mapping[int, Mapping[int, Sequence[int]]],
    selected: Mapping[int, Mapping[int, Sequence[int]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "rule": SELECTION_RULE,
        "formal_schedule_macros": G3_NATIVE_TEACHER_FORMAL_MACROS,
        "full_K1_covered_task_count": len(full),
        "full_K1_unique_video_count": sum(len(rows) for rows in full.values()),
        "selected_task_ids": sorted(selected),
        "selected_task_count": len(selected),
        "selected_unique_video_count": sum(len(rows) for rows in selected.values()),
        "max_videos_per_task": args.max_videos_per_task,
        "explicit_task_ids": None if args.task_ids is None else sorted(args.task_ids),
        "dirty_exploratory_subset": args.task_ids is not None,
    }


def _validate_targets(owners: Sequence[Any], indices: Sequence[int]) -> tuple[int, ...]:
    result = tuple(map(int, indices))
    if not result or len(set(result)) != len(result):
        raise ValueError("--target-indices must be a nonempty unique list")
    if min(result) < 0 or max(result) >= len(owners):
        raise ValueError("--target-indices crossed the 38-target contract")
    return result


def _prepare_condition(runtime: Any, task_id: int, video_demo: int) -> Any:
    sample = NaturalProgramSample(
        video_demos=(video_demo,),
        action_demos=(),
        k=1,
        robustness_view="g3_dual_basis_k1",
    )
    packed = pack_shared_compiler_videos(
        task=runtime.task_by_id[task_id],
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[task_id]
    condition = prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=runtime.owners,
        packed=packed,
        language_tokens=tokens,
        language_mask=mask,
        chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
    )
    if len(condition.videos) != 1 or condition.metrics.get("K") != 1:
        raise RuntimeError("dual-basis capture lost exact K1")
    return condition


def _fixed_measure_banks(
    runtime: Any, condition: Any, target: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real X, grouped dynamic Y, and fixed G2-rho candidate masses."""

    video = condition.videos[0]
    owner = runtime.owners[target]
    groups = native_output_group_count(owner)
    group_width = owner.out_features // groups
    input_values = []
    output_values = []
    boundary = NativeOutputBankState(final=video.native.final_outputs[target].detach())
    next_frame = 0
    for chunk in video.native.chunks():
        input_values.append(chunk.inputs[target].detach())
        dynamic = boundary.build(chunk.outputs[target].detach(), start_frame=next_frame)
        output_values.append(
            dynamic.reshape(*dynamic.shape[:-1], groups, group_width).movedim(-2, 0)
        )
        next_frame += chunk.frame_count
    if next_frame != video.native.frame_count or boundary.next_frame != next_frame:
        raise RuntimeError("dual-basis native chunks ended early")
    x = torch.cat(input_values, dim=0)
    y = torch.cat(output_values, dim=1)

    frames = video.native.frame_count
    assignment = video.canonical_assignment.double().clamp_min(0)
    assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-30)
    rho = condition.program.rho.double().clamp_min(1e-12)
    rho = rho / rho.sum()
    frame_mass = (assignment @ rho) * runtime.compiler._quadrature(
        video.frame_positions
    ).double()
    frame_mass = frame_mass / frame_mass.sum().clamp_min(1e-30)
    if x.shape[:3] != (frames, 2, 50) or y.shape[1:5] != (frames, 2, 50, 4):
        raise RuntimeError("dual-basis native candidate axes changed")
    input_mass = frame_mass[:, None, None].expand(frames, 2, 50)
    output_mass = frame_mass[:, None, None, None].expand(frames, 2, 50, 4)
    return x, y, input_mass, output_mass


def _teacher_rows(
    teachers: Sequence[NativeTeacherFactors], target: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not teachers:
        raise RuntimeError("dual-basis condition has no native teachers")
    a = torch.stack([teacher.a[target] for teacher in teachers], dim=0)
    b = torch.stack([teacher.b[target] for teacher in teachers], dim=0)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2]:
        raise RuntimeError("dual-basis teacher factor rank changed")
    return a, b


def _runtime_for(args: argparse.Namespace, runtime_dir: Path) -> Any:
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1 or context.rank != 0:
        raise ValueError("dual-basis capture uses independent single-GPU workers")
    runtime_args = _runtime_args(args, runtime_dir)
    runtime = prepare_runtime(runtime_args, context, load_native_teachers=True)
    _actual_native_only(runtime)
    if runtime.native_teachers is None:
        raise RuntimeError("dual-basis worker did not load native-teacher authority")
    return runtime


def _close_runtime(runtime: Any | None) -> None:
    if runtime is not None:
        runtime.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _g1_threshold(args: argparse.Namespace) -> float:
    g1 = read_json(args.g1_config)
    initialization = g1.get("optimization", {}).get("initialization", {})
    if g1.get("schema_version") != "ember_ecp_native_factor_g1_v1":
        raise ValueError("dual-basis scan lost G1 stable-span authority")
    return float(initialization["relative_singular_threshold"])


def worker(args: argparse.Namespace) -> None:
    worker_dir = args.output_dir / "workers" / f"worker_{args.shard_index:03d}"
    if worker_dir.exists() or (args.output_dir / "basis_manifest.json").exists():
        raise ValueError("dual-basis worker output already exists")
    runtime = None
    try:
        runtime = _runtime_for(args, worker_dir / "runtime")
        targets = _validate_targets(runtime.owners, args.target_indices)
        full = _k1_union(runtime.tasks, runtime.schedule)
        selected = _selected_union(
            runtime.tasks,
            runtime.schedule,
            max_videos_per_task=args.max_videos_per_task,
            task_ids=args.task_ids,
        )
        shards = _shards(runtime.tasks, selected, args.shard_count)
        assigned = shards[args.shard_index]
        threshold = _g1_threshold(args)
        tensors: dict[str, torch.Tensor] = {}
        records = []
        task_by_id = runtime.task_by_id
        record_index = 0
        for task_id in assigned:
            task = task_by_id[task_id]
            if task.role not in {"meta_fit", "target_fit"}:
                raise RuntimeError("dual-basis worker crossed held authority")
            effect_bank = runtime.effect_banks.get(task_id)
            for video_demo, macros in selected[task_id].items():
                condition = _prepare_condition(runtime, task_id, video_demo)
                before = runtime.native_teachers.tensor_reads
                teachers = runtime.native_teachers.lookup_members(
                    authority_id=task_id,
                    k=1,
                    video_demo=video_demo,
                    member_names=effect_bank.member_names,
                )
                if teachers is None or tuple(
                    teacher.member_name for teacher in teachers
                ) != effect_bank.member_names:
                    raise RuntimeError("dual-basis exact teacher lookup changed")
                if runtime.native_teachers.tensor_reads - before not in (0, 1):
                    raise RuntimeError("dual-basis lookup read an unexpected shard")
                target_rows = []
                for target in targets:
                    owner = runtime.owners[target]
                    groups = native_output_group_count(owner)
                    x, y, input_mass, output_mass = _fixed_measure_banks(
                        runtime, condition, target
                    )
                    teacher_a, teacher_b = _teacher_rows(teachers, target)
                    prefix = _tensor_prefix(record_index, target)

                    input_factor = _stable_factor(
                        x,
                        input_mass,
                        relative_singular_threshold=threshold,
                    )
                    unit, gain, projection, input_geometry = input_factor.solve(
                        teacher_a.to(device=x.device)
                    )
                    members, ranks = teacher_a.shape[:2]
                    tensors[f"{prefix}/input/unit"] = unit.reshape(
                        members, ranks, -1
                    )
                    tensors[f"{prefix}/input/dual_l2_norm"] = gain.reshape(
                        members, ranks
                    )
                    tensors[f"{prefix}/input/projection_cosine"] = projection.reshape(
                        members, ranks
                    ).float()
                    del input_factor

                    group_width = owner.out_features // groups
                    desired_blocks = teacher_b.to(device=y.device).reshape(
                        members * ranks, groups, group_width
                    )
                    output_geometry = []
                    for group in range(groups):
                        output_factor = _stable_factor(
                            y[group],
                            output_mass,
                            relative_singular_threshold=threshold,
                        )
                        unit, gain, projection, geometry = output_factor.solve(
                            desired_blocks[:, group]
                        )
                        base = f"{prefix}/output/g{group:03d}"
                        tensors[f"{base}/unit"] = unit.reshape(
                            members, ranks, -1
                        )
                        tensors[f"{base}/dual_l2_norm"] = gain.reshape(
                            members, ranks
                        )
                        tensors[f"{base}/projection_cosine"] = projection.reshape(
                            members, ranks
                        ).float()
                        output_geometry.append({"group": group, **geometry})
                        del output_factor
                    teacher_scales = torch.stack(
                        [teacher.scales[target] for teacher in teachers]
                    ).float()
                    adjacent_gap = (
                        (teacher_scales[:, :-1] - teacher_scales[:, 1:]).abs()
                        / teacher_scales[:, :-1].abs().clamp_min(1e-12)
                    )
                    target_rows.append(
                        {
                            "target": target,
                            "target_name": runtime.rank4_contract.targets[target].name,
                            "family": owner.family.value,
                            "groups": groups,
                            "input_width": owner.in_features,
                            "output_width": owner.out_features,
                            "group_width": group_width,
                            "canonical_rank_gauge": {
                                "authority": "balanced_small_core_svd",
                                "nonincreasing_scales": bool(
                                    torch.all(
                                        teacher_scales[:, :-1]
                                        >= teacher_scales[:, 1:] - 1e-7
                                    )
                                ),
                                "minimum_relative_adjacent_gap": float(
                                    adjacent_gap.min()
                                ),
                                "near_degenerate_adjacent_fraction_at_0.05": float(
                                    (adjacent_gap < 0.05).float().mean()
                                ),
                            },
                            "input": input_geometry,
                            "output": output_geometry,
                        }
                    )
                    del x, y, input_mass, output_mass, teacher_a, teacher_b
                    torch.cuda.empty_cache()
                records.append(
                    {
                        "record_index": record_index,
                        "authority_id": task_id,
                        "role": task.role,
                        "suite": task.suite,
                        "video_demo": video_demo,
                        "scheduled_macros": list(macros),
                        "first_scheduled_macro": min(macros),
                        "member_names": list(effect_bank.member_names),
                        "member_count": len(effect_bank.member_names),
                        "frame_count": condition.videos[0].native.frame_count,
                        "measure": "fixed_g2_rho_times_temporal_quadrature",
                        "targets": target_rows,
                    }
                )
                record_index += 1
                del condition, teachers
                torch.cuda.empty_cache()

        tensor_path = worker_dir / "duals.safetensors"
        _save_safetensors_atomic(tensor_path, tensors)
        write_json_atomic(
            worker_dir / "completion.json",
            {
                "schema_version": WORKER_SCHEMA,
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "task_ids": list(assigned),
                "target_indices": list(targets),
                "relative_singular_threshold": threshold,
                "selection": _selection_record(full, selected, args),
                "records": records,
                "record_count": len(records),
                "tensor_file": {
                    "path": str(tensor_path.resolve()),
                    "bytes": tensor_path.stat().st_size,
                },
                "dual_payload": (
                    "unit minimum-norm dual direction plus L2 norm; "
                    "no native bank, covariance, logits, selection weights, or outcomes"
                ),
                "information_wall": {
                    "roles": sorted({runtime.task_by_id[row].role for row in assigned}),
                    "held_authority_reads": 0,
                    "action_meta_installed": False,
                    "action_meta_module_count": 0,
                    "action_meta_parameter_count": 0,
                    "source_policy_trainable_parameter_count": 0,
                    "natural_program_trainable_parameter_count": 0,
                    "deployment_use": False,
                },
                "git": git_state(REPO_ROOT),
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    runtime.context.device
                ),
            },
        )
    finally:
        _close_runtime(runtime)


def _capture_record_sources(
    captures: Sequence[
        tuple[Path, Mapping[str, Any], Mapping[str, torch.Tensor]]
    ],
) -> tuple[dict[str, Any], ...]:
    sources = []
    for worker_dir, completion, tensors in captures:
        for record in completion["records"]:
            sources.append(
                {
                    "worker_dir": worker_dir,
                    "completion": completion,
                    "tensors": tensors,
                    "record": record,
                }
            )
    return tuple(sources)


def _replay_task_shards(
    sources: Sequence[Mapping[str, Any]], count: int
) -> tuple[tuple[int, ...], ...]:
    by_task: dict[int, int] = defaultdict(int)
    for source in sources:
        record = source["record"]
        groups = sum(int(row["groups"]) + 1 for row in record["targets"])
        by_task[int(record["authority_id"])] += (
            int(record["frame_count"])
            * int(record["member_count"])
            * groups
        )
    if not 0 < count <= len(by_task):
        raise ValueError("invalid dual-basis replay shard count")
    shards: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    for task, cost in sorted(by_task.items(), key=lambda row: (-row[1], row[0])):
        shard = min(range(count), key=lambda index: (loads[index], index))
        shards[shard].append(task)
        loads[shard] += cost
    return tuple(tuple(sorted(shard)) for shard in shards)


def replay_worker(args: argparse.Namespace) -> None:
    """Replay each task through bases fit without that task's dual labels."""

    replay_dir = args.output_dir / "replay_workers" / f"worker_{args.shard_index:03d}"
    if (replay_dir / "results.json").exists():
        raise ValueError("dual-basis replay worker output already exists")
    basis_manifest = read_json(args.output_dir / "basis_manifest.json")
    if (
        basis_manifest.get("schema_version") != BASIS_SCHEMA
        or basis_manifest.get("status") != "complete"
        or basis_manifest.get("fit", {}).get("source_shard_count")
        != args.shard_count
        or abs(
            float(basis_manifest.get("fit", {}).get("score_bound", -1.0))
            - args.score_bound
        )
        > 1e-12
    ):
        raise ValueError("dual-basis replay lost its fitted basis authority")
    formal_probe = not bool(basis_manifest["fit"]["dirty_exploratory_subset"])
    if formal_probe:
        basis_git = basis_manifest.get("git", {})
        if (
            basis_git.get("dirty_paths")
            or basis_git.get("authority_contains_commit") is not True
            or basis_git.get("commit")
            != basis_manifest["fit"]["source_commit"]
        ):
            raise ValueError("formal dual-basis aggregate authority changed")
        _validate_formal_arguments(args, repo_root=REPO_ROOT)
        replay_git = _validate_formal_git(
            REPO_ROOT,
            expected_commit=str(basis_manifest["fit"]["source_commit"]),
        )
    else:
        replay_git = git_state(REPO_ROOT)
    basis_path = args.output_dir / "bases.safetensors"
    basis_file = basis_manifest.get("basis_file", {})
    if (
        Path(str(basis_file.get("path"))).resolve() != basis_path.resolve()
        or basis_file.get("bytes") != basis_path.stat().st_size
    ):
        raise ValueError("dual-basis replay basis artifact changed")
    dimensions = tuple(map(int, basis_manifest["fit"]["basis_dimensions"]))
    targets = tuple(map(int, basis_manifest["fit"]["target_indices"]))
    bases = load_file(str(basis_path))
    captures = _capture_workers(args.output_dir, args.shard_count)
    sources = _capture_record_sources(captures)
    shards = _replay_task_shards(sources, args.shard_count)
    assigned = shards[args.shard_index]
    selected = [
        source
        for source in sources
        if int(source["record"]["authority_id"]) in assigned
    ]
    runtime = None
    try:
        runtime = _runtime_for(args, replay_dir / "runtime")
        _validate_targets(runtime.owners, targets)
        rows = []
        record_summaries = []
        with torch.no_grad():
            for source in selected:
                record = source["record"]
                tensors = source["tensors"]
                task_id = int(record["authority_id"])
                video_demo = int(record["video_demo"])
                condition = _prepare_condition(runtime, task_id, video_demo)
                effect_bank = runtime.effect_banks.get(task_id)
                teachers = runtime.native_teachers.lookup_members(
                    authority_id=task_id,
                    k=1,
                    video_demo=video_demo,
                    member_names=effect_bank.member_names,
                )
                if teachers is None or tuple(
                    teacher.member_name for teacher in teachers
                ) != tuple(record["member_names"]):
                    raise RuntimeError("dual-basis replay exact teacher lookup changed")
                target_summaries = []
                for target_row in record["targets"]:
                    target = int(target_row["target"])
                    if target not in targets:
                        continue
                    x, y, input_mass, output_mass = _fixed_measure_banks(
                        runtime, condition, target
                    )
                    prefix = _tensor_prefix(int(record["record_index"]), target)
                    input_unit = tensors[f"{prefix}/input/unit"].to(
                        device=x.device, dtype=torch.float64
                    )
                    input_norm = tensors[f"{prefix}/input/dual_l2_norm"].to(
                        device=x.device, dtype=torch.float64
                    )
                    input_dual = input_unit * input_norm[..., None]
                    input_basis = bases[
                        f"held/{task_id:03d}/target/{target:02d}/input"
                    ].to(device=x.device, dtype=torch.float64)
                    input_queries, input_effective, input_projection = (
                        _projected_queries(input_dual, input_basis, dimensions)
                    )
                    student_a = _signed_pool_queries(
                        input_queries,
                        x,
                        input_mass,
                        score_bound=args.score_bound,
                    )

                    groups = int(target_row["groups"])
                    output_basis = bases[
                        f"held/{task_id:03d}/target/{target:02d}/output"
                    ].to(device=y.device, dtype=torch.float64)
                    output_queries = []
                    output_effective: tuple[int, ...] | None = None
                    output_projection_values = []
                    for group in range(groups):
                        unit = tensors[
                            f"{prefix}/output/g{group:03d}/unit"
                        ].to(device=y.device, dtype=torch.float64)
                        norm = tensors[
                            f"{prefix}/output/g{group:03d}/dual_l2_norm"
                        ].to(device=y.device, dtype=torch.float64)
                        dual = unit * norm[..., None]
                        queries, effective, projection = _projected_queries(
                            dual, output_basis, dimensions
                        )
                        if output_effective is None:
                            output_effective = effective
                        elif output_effective != effective:
                            raise RuntimeError("dual-basis output group ranks changed")
                        output_projection_values.append(projection)
                        output_queries.append(queries)
                    student_b = _signed_pool_output_groups(
                        output_queries,
                        y,
                        output_mass,
                        score_bound=args.score_bound,
                    )
                    if output_effective is None:
                        raise RuntimeError("dual-basis output groups disappeared")
                    teacher_a, teacher_b = _teacher_rows(teachers, target)
                    teacher_a = teacher_a.to(student_a)
                    teacher_b = teacher_b.to(student_b)
                    labels: tuple[str | int, ...] = ("full", *dimensions)
                    output_projection = tuple(
                        sum(values[index] for values in output_projection_values)
                        / groups
                        for index in range(len(labels))
                    )
                    for variant, label in enumerate(labels):
                        for member, teacher in enumerate(teachers):
                            input_mean, input_minimum = _factor_cosines(
                                student_a[variant, member], teacher_a[member]
                            )
                            output_mean, output_minimum = _factor_cosines(
                                student_b[variant, member], teacher_b[member]
                            )
                            update_geometry = _update_geometry(
                                student_a[variant, member],
                                student_b[variant, member],
                                teacher_a[member],
                                teacher_b[member],
                                teacher.scales[target].to(student_a),
                            )
                            rows.append(
                                {
                                    "authority_id": task_id,
                                    "role": record["role"],
                                    "suite": record["suite"],
                                    "video_demo": video_demo,
                                    "member_name": teacher.member_name,
                                    "target": target,
                                    "target_name": target_row["target_name"],
                                    "family": target_row["family"],
                                    "dimension": label,
                                    "effective_input_dimension": input_effective[
                                        variant
                                    ],
                                    "effective_output_dimension": output_effective[
                                        variant
                                    ],
                                    "input_dual_projection_cosine": input_projection[
                                        variant
                                    ],
                                    "output_dual_projection_cosine": output_projection[
                                        variant
                                    ],
                                    "input_factor_cosine_mean": input_mean,
                                    "input_factor_cosine_minimum": input_minimum,
                                    "output_factor_cosine_mean": output_mean,
                                    "output_factor_cosine_minimum": output_minimum,
                                    **update_geometry,
                                }
                            )
                    target_summaries.append(
                        {
                            "target": target,
                            "groups": groups,
                            "input_candidates": int(input_mass.numel()),
                            "output_candidates_per_group": int(output_mass.numel()),
                        }
                    )
                    del x, y, input_mass, output_mass, student_a, student_b
                    torch.cuda.empty_cache()
                record_summaries.append(
                    {
                        "authority_id": task_id,
                        "video_demo": video_demo,
                        "frame_count": record["frame_count"],
                        "member_count": record["member_count"],
                        "targets": target_summaries,
                    }
                )
                del condition, teachers
                torch.cuda.empty_cache()
        replay_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            replay_dir / "results.json",
            {
                "schema_version": REPLAY_SCHEMA,
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "task_ids": list(assigned),
                "score_bound": args.score_bound,
                "pooling": (
                    "fixed-G2-rho measure plus antithetic exp softmax at bounded "
                    "small centered logits; one common gain across every output "
                    "group of a target/member/rank"
                ),
                "claim_boundary": (
                    "the score-bound calibration reads the complete current bank "
                    "and is an oracle replay statistic, not a demonstrated chunked "
                    "deployment mechanism"
                ),
                "basis_authority": {
                    "source_commit": basis_manifest["fit"]["source_commit"],
                    "path": str(basis_path.resolve()),
                    "bytes": basis_path.stat().st_size,
                },
                "rows": rows,
                "row_count": len(rows),
                "records": record_summaries,
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    runtime.context.device
                ),
                "git": replay_git,
            },
        )
    finally:
        _close_runtime(runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("worker", "aggregate", "replay-worker", "report")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v2.json",
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
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--effect-bank-root", type=Path, required=True)
    parser.add_argument(
        "--target-indices", type=int, nargs="+", default=list(DEFAULT_TARGETS)
    )
    parser.add_argument("--max-videos-per-task", type=int, default=2)
    parser.add_argument(
        "--basis-dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_BASIS_DIMENSIONS),
    )
    parser.add_argument("--aggregate-device", default="cpu")
    parser.add_argument("--score-bound", type=float, default=0.1)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="dirty exploratory smoke subset only; omission means formal K1-covered50",
    )
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
        setattr(args, name, getattr(args, name).resolve())
    args.target_indices = tuple(map(int, args.target_indices))
    args.basis_dimensions = tuple(map(int, args.basis_dimensions))
    if (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or args.max_videos_per_task <= 0
        or not 0.0 < args.score_bound <= 0.25
    ):
        raise ValueError("invalid dual-basis worker shard contract")
    if args.command != "worker" and args.task_ids is not None:
        raise ValueError("--task-ids is only valid for an exploratory worker")
    if args.command == "worker" and args.task_ids is None:
        _validate_formal_arguments(args, repo_root=REPO_ROOT)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    if args.command == "worker":
        worker(args)
    elif args.command == "aggregate":
        _aggregate_artifacts(args, repo_root=REPO_ROOT)
    elif args.command == "replay-worker":
        replay_worker(args)
    else:
        _report_artifacts(args, repo_root=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
