#!/usr/bin/env python3
"""Probe low-dimensional bank-adaptive functional sketches on real G3 banks."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist

from ember.ecp.bank_conditioning import (
    StreamingProjectedFunctionalStatistics,
    StreamingSignedPool,
    StreamingSketchCrossImage,
    bank_adaptive_basis,
    fixed_nested_projection,
    functional_target_queries,
    materialized_signed_pool,
)
from ember.ecp.bank_conditioning.anchor_solve import candidate_mass
from ember.ecp.bank_conditioning.mapping_eval_runtime import load_mapping_tasks
from ember.ecp.bank_conditioning.mapping_training import _load_training_assets
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    NativeOutputBankState,
    native_output_group_count,
    rms_normalize,
)
from ember.ecp.shared_compiler_assets import (
    authority_path,
    load_shared_compiler_config,
)
from ember.ecp.shared_compiler_authority import pure_shared_compiler_inventory
from ember.ecp.shared_compiler_dual_basis import update_geometry
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, seed_everything
from scripts.analyze_ecp_bank_conditioned_operator import (
    _dual_sources,
    _task_shards,
)
from scripts.analyze_ecp_g3_dual_basis import (
    _prepare_condition,
    _teacher_rows,
)


WORKER_SCHEMA = "ember_ecp_g3_functional_sketch_s1_worker_v1"
REPORT_SCHEMA = "ember_ecp_g3_functional_sketch_s1_report_v1"


@dataclass(frozen=True)
class _GroupBank:
    values: torch.Tensor
    keys: torch.Tensor
    base_mass: torch.Tensor
    event_mass: torch.Tensor
    replay_mass: torch.Tensor


@dataclass
class _SketchRuntime:
    context: Any
    config: dict[str, Any]
    task_by_id: dict[int, Any]
    video_store: Any
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: torch.nn.Module
    compiler: torch.nn.Module
    owners: tuple[Any, ...]
    native_teachers: Any
    query_points: int
    inventory: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()


def _prepare_runtime(args: argparse.Namespace) -> _SketchRuntime:
    """Load only the current frozen v4 bank path, without the retired trainer."""

    context = initialize_distributed(require_numa=False, defer_process_group=True)
    if context.world_size != 1 or context.rank != 0:
        raise ValueError("functional-sketch workers are independent single-GPU jobs")
    config = load_shared_compiler_config(args.reference_config)
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = load_mapping_tasks(
        config, asset_root=args.asset_root, data_root=args.data_root
    )
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    tokenizer = authority_path(config, "tokenizer", asset_root=args.asset_root)
    assets = _load_training_assets(
        SimpleNamespace(
            asset_root=args.asset_root,
            checkpoint=source_checkpoint,
            source_run=source_checkpoint.parent.parent,
            tokenizer_path=tokenizer,
        ),
        config,
        context,
        tasks,
    )
    inventory = pure_shared_compiler_inventory(
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        owners=assets.owners,
    )
    assets.compiler.requires_grad_(False).eval()
    if any(
        parameter.requires_grad
        for module in (assets.policy, assets.program, assets.compiler)
        for parameter in module.parameters()
    ):
        assets.video_store.close()
        raise RuntimeError("functional-sketch S1 unexpectedly loaded trainable state")
    torch.cuda.reset_peak_memory_stats(context.device)
    return _SketchRuntime(
        context=context,
        config=config,
        task_by_id={task.authority_id: task for task in tasks},
        video_store=assets.video_store,
        language_tokens=assets.language_tokens,
        policy=assets.policy,
        program=assets.program,
        compiler=assets.compiler,
        owners=assets.owners,
        native_teachers=assets.native_teachers,
        query_points=assets.query_points,
        inventory=inventory,
    )


def _candidate_keys(
    scorer: Any,
    value: torch.Tensor,
    metadata: torch.Tensor,
    *,
    target: int,
    output: bool,
) -> torch.Tensor:
    if output:
        raw = scorer.output_keys(value, metadata, target=target)
        return scorer.output_projected_keys(raw, target=target)
    raw = scorer.input_keys(value, metadata, target=target)
    return scorer.input_projected_keys(raw, target=target)


def _condition_banks(
    runtime: Any,
    condition: Any,
    targets: Sequence[int],
) -> tuple[Any, dict[tuple[int, str, int], _GroupBank]]:
    video = condition.videos[0]
    scorer = runtime.compiler.anchor_scorer
    state = scorer.program_state(condition.program)
    base_frame, event_frame, frame_measure = runtime.compiler._video_measures(
        video, state.event_weights
    )
    events = event_frame.shape[0]
    input_values: dict[int, list[torch.Tensor]] = {target: [] for target in targets}
    input_keys: dict[int, list[torch.Tensor]] = {target: [] for target in targets}
    output_values: dict[tuple[int, int], list[torch.Tensor]] = {}
    output_keys: dict[tuple[int, int], list[torch.Tensor]] = {}
    boundaries = {
        target: NativeOutputBankState(
            final=video.native.final_outputs[target].detach()
        )
        for target in targets
    }
    next_frame = 0
    with torch.no_grad():
        for chunk in video.native.chunks():
            stop = next_frame + chunk.frame_count
            if chunk.start_frame != next_frame or stop > video.native.frame_count:
                raise RuntimeError("functional-sketch native chunks changed")
            assignment = video.canonical_assignment[next_frame:stop].float()
            frame_metadata = scorer.frame_metadata(
                assignment, video.frame_positions[next_frame:stop]
            )
            x_metadata = scorer.candidate_metadata(frame_metadata, output=False)
            y_metadata = scorer.candidate_metadata(frame_metadata, output=True)
            for target in targets:
                owner = runtime.owners[target]
                x = chunk.inputs[target].detach()
                input_values[target].append(x)
                key = _candidate_keys(
                    scorer, x, x_metadata, target=target, output=False
                )
                input_keys[target].append(key[None].expand(events, *key.shape))

                dynamic = boundaries[target].build(
                    chunk.outputs[target].detach(), start_frame=next_frame
                )
                groups = native_output_group_count(owner)
                grouped = dynamic.reshape(
                    *dynamic.shape[:-1], groups, owner.out_features // groups
                ).movedim(-2, 0)
                keys = _candidate_keys(
                    scorer,
                    grouped,
                    y_metadata[None],
                    target=target,
                    output=True,
                )
                for group in range(groups):
                    output_values.setdefault((target, group), []).append(grouped[group])
                    event_key = keys[group][None].expand(events, *keys[group].shape)
                    output_keys.setdefault((target, group), []).append(event_key)
            next_frame = stop
    if next_frame != video.native.frame_count or any(
        boundary.next_frame != next_frame for boundary in boundaries.values()
    ):
        raise RuntimeError("functional-sketch native stream ended early")

    x_base = candidate_mass(base_frame, output=False)
    x_event = candidate_mass(event_frame, output=False)
    y_base = candidate_mass(base_frame, output=True)
    y_event = candidate_mass(event_frame, output=True)
    banks: dict[tuple[int, str, int], _GroupBank] = {}
    for target in targets:
        banks[(target, "input", 0)] = _GroupBank(
            values=torch.cat(input_values[target], dim=0),
            keys=torch.cat(input_keys[target], dim=1),
            base_mass=x_base,
            event_mass=x_event,
            replay_mass=candidate_mass(frame_measure[target], output=False),
        )
        for group in range(native_output_group_count(runtime.owners[target])):
            banks[(target, "output", group)] = _GroupBank(
                values=torch.cat(output_values[(target, group)], dim=0),
                keys=torch.cat(output_keys[(target, group)], dim=1),
                base_mass=y_base,
                event_mass=y_event,
                replay_mass=candidate_mass(frame_measure[target], output=True),
            )
    return state, banks


def _statistics(
    bank: _GroupBank,
    projection: torch.Tensor,
    *,
    rank: int,
    mode: str,
    frame_chunk: int,
    singular_floor: float,
    dtype: torch.dtype,
    streamed: bool,
) -> Any:
    cross = StreamingSketchCrossImage(
        native_width=bank.values.shape[-1],
        key_width=bank.keys.shape[-1],
        events=bank.keys.shape[0],
        device=bank.values.device,
        dtype=dtype,
    )
    spans = (
        tuple(
            (start, min(start + frame_chunk, bank.values.shape[0]))
            for start in range(0, bank.values.shape[0], frame_chunk)
        )
        if streamed
        else ((0, bank.values.shape[0]),)
    )
    for start, stop in spans:
        cross.add(
            bank.values[start:stop],
            bank.base_mass[start:stop],
            bank.event_mass[:, start:stop],
            bank.keys[:, start:stop],
        )
    cross_image = cross.finalize()
    basis = bank_adaptive_basis(
        cross_image,
        projection,
        requested_rank=rank,
        mode=mode,
        relative_singular_floor=singular_floor,
    )
    projected = StreamingProjectedFunctionalStatistics(
        basis,
        ranks=G1_RESIDUAL_RANK,
        device=bank.values.device,
        dtype=dtype,
    )
    for start, stop in spans:
        projected.add(
            bank.values[start:stop],
            bank.base_mass[start:stop],
            bank.replay_mass[:, start:stop],
        )
    return projected.finalize(cross_image)


def _score_rms(queries: Any, statistics: Any) -> torch.Tensor:
    return torch.einsum(
        "mrs,rsd,mrd->mr",
        queries.reduced.float(),
        statistics.replay_covariances.float(),
        queries.reduced.float(),
    ).clamp_min(0).sqrt()


def _pool(
    query: torch.Tensor,
    bank: _GroupBank,
    *,
    frame_chunk: int,
    streamed: bool,
) -> torch.Tensor:
    members = query.shape[0]
    ratio = bank.replay_mass / bank.base_mass[None].clamp_min(1e-12)
    bias = ratio.clamp_min(1e-12).log()[None].expand(members, *ratio.shape)
    if not streamed:
        return materialized_signed_pool(
            query,
            bank.values,
            bank.base_mass,
            dtype=torch.float32,
            logit_bias=bias,
        )
    accumulator = StreamingSignedPool(query, dtype=torch.float32)
    for start in range(0, bank.values.shape[0], frame_chunk):
        stop = min(start + frame_chunk, bank.values.shape[0])
        accumulator.add(
            bank.values[start:stop],
            bank.base_mass[start:stop],
            bias[..., start:stop, *([slice(None)] * (bank.base_mass.ndim - 1))],
        )
    return accumulator.signed_mean()


def _target_rank_result(
    *,
    runtime: Any,
    target: int,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    teacher_scales: torch.Tensor,
    state: Any,
    banks: Mapping[tuple[int, str, int], _GroupBank],
    rank: int,
    config: Mapping[str, Any],
    projections: Mapping[str, torch.Tensor],
    frame_chunk: int,
) -> list[dict[str, Any]]:
    owner = runtime.owners[target]
    mode = config["model"]["family_mode"][owner.family.value]
    floor = float(config["model"]["relative_covariance_floor"])
    singular_floor = float(config["model"]["relative_singular_floor"])
    score_rms = float(config["model"]["replay_score_rms"])
    event_weights = state.event_weights[target]

    input_bank = banks[(target, "input", 0)]
    input_stats = {}
    input_queries = {}
    input_factors = {}
    for path in ("materialized", "streaming"):
        streamed = path == "streaming"
        statistics = _statistics(
            input_bank,
            projections[mode],
            rank=rank,
            mode=mode,
            frame_chunk=frame_chunk,
            singular_floor=singular_floor,
            dtype=torch.float32,
            streamed=streamed,
        )
        solved = functional_target_queries(
            teacher_a.to(statistics.basis),
            statistics,
            relative_floor=floor,
        )
        scale = score_rms / _score_rms(solved, statistics).clamp_min(1e-12)
        query = solved.native * scale[..., None].to(solved.native)
        factor = rms_normalize(
            _pool(query, input_bank, frame_chunk=frame_chunk, streamed=streamed)
        )
        input_stats[path] = statistics
        input_queries[path] = solved
        input_factors[path] = factor

    groups = native_output_group_count(owner)
    output_stats: dict[str, list[Any]] = {"materialized": [], "streaming": []}
    output_queries: dict[str, list[Any]] = {"materialized": [], "streaming": []}
    for group in range(groups):
        bank = banks[(target, "output", group)]
        desired = teacher_b.reshape(
            teacher_b.shape[0], teacher_b.shape[1], groups, -1
        )[:, :, group]
        for path in ("materialized", "streaming"):
            streamed = path == "streaming"
            statistics = _statistics(
                bank,
                projections[mode],
                rank=rank,
                mode=mode,
                frame_chunk=frame_chunk,
                singular_floor=singular_floor,
                dtype=torch.float32,
                streamed=streamed,
            )
            solved = functional_target_queries(
                desired.to(statistics.basis),
                statistics,
                relative_floor=floor,
            )
            output_stats[path].append(statistics)
            output_queries[path].append(solved)

    output_factors = {}
    for path in ("materialized", "streaming"):
        maxima = torch.stack(
            tuple(
                _score_rms(query, statistics)
                for query, statistics in zip(
                    output_queries[path], output_stats[path], strict=True
                )
            )
        ).amax(0)
        scale = score_rms / maxima.clamp_min(1e-12)
        blocks = []
        for group, solved in enumerate(output_queries[path]):
            query = solved.native * scale[..., None].to(solved.native)
            blocks.append(
                _pool(
                    query,
                    banks[(target, "output", group)],
                    frame_chunk=frame_chunk,
                    streamed=path == "streaming",
                )
            )
        output_factors[path] = rms_normalize(torch.cat(blocks, dim=-1))

    rows = []
    for member in range(teacher_a.shape[0]):
        materialized = update_geometry(
            input_factors["materialized"][member],
            output_factors["materialized"][member],
            teacher_a[member].to(input_factors["materialized"]),
            teacher_b[member].to(output_factors["materialized"]),
            teacher_scales[member].to(input_factors["materialized"]),
        )
        chunk = update_geometry(
            input_factors["streaming"][member],
            output_factors["streaming"][member],
            input_factors["materialized"][member],
            output_factors["materialized"][member],
            teacher_scales[member].to(input_factors["materialized"]),
        )
        rows.append(
            {
                "target": target,
                "target_name": owner.target_name,
                "family": owner.family.value,
                "depth": (
                    "action"
                    if target >= 36
                    else "shallow"
                    if target <= 1
                    else "middle"
                    if target <= 21
                    else "deep"
                ),
                "member_index": member,
                "sketch_rank": rank,
                "mode": mode,
                "materialized_to_teacher_update_cosine": materialized[
                    "update_cosine"
                ],
                "materialized_input_subspace_similarity": materialized[
                    "input_subspace_similarity"
                ],
                "materialized_output_subspace_similarity": materialized[
                    "output_subspace_similarity"
                ],
                "streaming_to_materialized_update_cosine": chunk["update_cosine"],
                "input_linear_recovery_minimum": float(
                    input_queries["materialized"].linear_recovery[member].min()
                ),
                "output_linear_recovery_minimum": min(
                    float(value.linear_recovery[member].min())
                    for value in output_queries["materialized"]
                ),
                "input_retained_rank": int(
                    input_stats["materialized"].basis.shape[1]
                ),
                "output_retained_rank_minimum": min(
                    int(value.basis.shape[1]) for value in output_stats["materialized"]
                ),
            }
        )
    return rows


def _condition_rows(
    runtime: Any,
    source: Mapping[str, Any],
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    started = time.perf_counter()
    record = source["record"]
    task_id = int(record["authority_id"])
    video_demo = int(record["video_demo"])
    condition = _prepare_condition(runtime, task_id, video_demo)
    captured = time.perf_counter()
    targets = args.target_indices
    state, banks = _condition_banks(runtime, condition, targets)
    bank_ready = time.perf_counter()
    member_names = tuple(map(str, record["member_names"]))
    teachers = runtime.native_teachers.lookup_members(
        authority_id=task_id,
        k=1,
        video_demo=video_demo,
        member_names=member_names,
    )
    if teachers is None or tuple(row.member_name for row in teachers) != member_names:
        raise RuntimeError("functional-sketch teacher lookup failed")
    events = int(config["model"]["event_slots"])
    key_width = int(config["model"]["key_width"])
    maximum_rank = max(map(int, config["model"]["rank_curve"]))
    projections = {
        mode: fixed_nested_projection(
            events=events,
            key_width=key_width,
            maximum_rank=maximum_rank,
            mode=mode,
            seed=int(config["model"]["fixed_nested_projection_seed"]),
            device=runtime.context.device,
            dtype=torch.float32,
        )
        for mode in ("global", "per_event")
    }
    rows = []
    for target in targets:
        teacher_a, teacher_b = _teacher_rows(teachers, target)
        scales = torch.stack(tuple(teacher.scales[target] for teacher in teachers))
        for rank in map(int, config["model"]["rank_curve"]):
            target_rows = _target_rank_result(
                runtime=runtime,
                target=target,
                teacher_a=teacher_a,
                teacher_b=teacher_b,
                teacher_scales=scales,
                state=state,
                banks=banks,
                rank=rank,
                config=config,
                projections=projections,
                frame_chunk=args.frame_chunk,
            )
            rows.extend(
                {
                    "authority_id": task_id,
                    "role": record["role"],
                    "suite": record["suite"],
                    "video_demo": video_demo,
                    "member_name": teachers[row["member_index"]].member_name,
                    **row,
                }
                for row in target_rows
            )
    complete = time.perf_counter()
    return rows, {
        "pass_a_and_native_capture_seconds": captured - started,
        "bank_key_and_measure_seconds": bank_ready - captured,
        "rank_curve_and_exact_replay_seconds": complete - bank_ready,
        "total_seconds": complete - started,
    }


def _reference_rows(config: Mapping[str, Any], asset_root: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    report = asset_root / str(config["authorities"]["f1_operator_report"])
    root = report.parent
    summary = read_json(report)
    if summary.get("status") != "complete" or not summary.get("gate_pass"):
        raise ValueError("functional-sketch lost its sealed F1 positive reference")
    rows = {}
    for path in sorted((root / "workers").glob("worker_*/results.json")):
        payload = read_json(path)
        for row in payload["rows"]:
            key = (
                int(row["authority_id"]),
                int(row["video_demo"]),
                str(row["member_name"]),
                int(row["target"]),
            )
            if key in rows:
                raise ValueError("F1 positive reference contains a duplicate row")
            rows[key] = row
    if len(rows) != int(summary["row_count"]):
        raise ValueError("F1 positive reference row count changed")
    return rows


def _attach_reference(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key = (
            int(row["authority_id"]),
            int(row["video_demo"]),
            str(row["member_name"]),
            int(row["target"]),
        )
        positive = reference.get(key)
        if positive is None:
            raise ValueError("functional-sketch row has no exact F1 positive reference")
        result.append(
            {
                **row,
                "reference_analytic_to_teacher_update_cosine": float(
                    positive["analytic_to_teacher_update_cosine"]
                ),
                "reference_materialized_to_analytic_update_cosine": float(
                    positive["materialized_to_analytic_update_cosine"]
                ),
            }
        )
    return result


def worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    worker_dir = args.output_dir / "workers" / f"worker_{args.shard_index:03d}"
    if worker_dir.exists():
        raise ValueError("functional-sketch worker output already exists")
    config = read_json(args.sketch_config)
    if config.get("schema_version") != "ember_ecp_functional_sketch_s1_v1":
        raise ValueError("functional-sketch config schema changed")
    repository = git_state(REPO_ROOT)
    if args.formal and (
        not git_state_is_clean_pushed_or_frozen_authority(repository)
        or repository.get("branch") != ""
        or repository.get("upstream") is not None
    ):
        raise ValueError("formal functional-sketch requires clean detached authority")
    if args.early_disqualifier and (
        not args.formal
        or args.shard_count != 1
        or args.shard_index != 0
        or args.task_ids != (93,)
        or args.target_indices != (20,)
    ):
        raise ValueError("S1 early disqualifier is fixed to task93/q20")
    sources = _dual_sources(args.dual_authority)
    if args.task_ids is not None:
        requested = set(args.task_ids)
        sources = tuple(
            source
            for source in sources
            if int(source["record"]["authority_id"]) in requested
        )
        if not sources or {int(row["record"]["authority_id"]) for row in sources} != requested:
            raise ValueError("functional-sketch task subset left its authority")
    assigned = set(_task_shards(sources, args.shard_count)[args.shard_index])
    selected = tuple(
        source
        for source in sources
        if int(source["record"]["authority_id"]) in assigned
    )
    runtime = None
    try:
        runtime = _prepare_runtime(args)
        runtime_ready = time.perf_counter()
        reference = _reference_rows(config, args.asset_root)
        rows = []
        condition_profiles = []
        for source in selected:
            condition_rows, profile = _condition_rows(runtime, source, args, config)
            rows.extend(_attach_reference(condition_rows, reference))
            condition_profiles.append(profile)
        complete = time.perf_counter()
        worker_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            worker_dir / "results.json",
            {
                "schema_version": WORKER_SCHEMA,
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "task_ids": sorted(assigned),
                "record_count": len(selected),
                "row_count": len(rows),
                "rows": rows,
                "profile": {
                    "runtime_initialization_seconds": runtime_ready - started,
                    "condition_seconds": condition_profiles,
                    "condition_seconds_mean": sum(
                        row["total_seconds"] for row in condition_profiles
                    ) / max(len(condition_profiles), 1),
                    "analysis_seconds": complete - runtime_ready,
                    "total_seconds": complete - started,
                    "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                        runtime.context.device
                    ),
                    "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                        runtime.context.device
                    ),
                },
                "information_wall": {
                    "training": False,
                    "held_outcome_reads": 0,
                    "action_meta_installed": bool(
                        runtime.inventory["action_meta_module_count"]
                    ),
                    "source_policy_trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in runtime.policy.parameters()
                        if parameter.requires_grad
                    ),
                    "natural_program_trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in runtime.program.parameters()
                        if parameter.requires_grad
                    ),
                    "compiler_trainable_parameter_count": sum(
                        parameter.numel()
                        for parameter in runtime.compiler.parameters()
                        if parameter.requires_grad
                    ),
                    "shared_model_claim": False,
                },
                "reference_boundary": (
                    "The sketch is compared with the exact same fit-only native "
                    "teachers as sealed F1 analytic/current-bank positive rows."
                ),
                "formal": bool(args.formal),
                "early_disqualifier": bool(args.early_disqualifier),
                "git": repository,
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def report(args: argparse.Namespace) -> None:
    rows = []
    workers = []
    for index in range(args.shard_count):
        payload = read_json(
            args.output_dir / "workers" / f"worker_{index:03d}" / "results.json"
        )
        if (
            payload.get("schema_version") != WORKER_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("shard_count") != args.shard_count
        ):
            raise ValueError("functional-sketch report received an invalid worker")
        rows.extend(payload["rows"])
        workers.append(
            {
                "shard_index": index,
                "task_ids": payload["task_ids"],
                "record_count": payload["record_count"],
                "profile": payload["profile"],
            }
        )
    rank64 = [row for row in rows if int(row["sketch_rank"]) == 64]
    threshold = float(read_json(args.sketch_config)["gate"]["each_family_row_minimum"])
    capacity_values = [
        float(row["materialized_to_teacher_update_cosine"]) for row in rank64
    ]
    chunk_values = [
        float(row["streaming_to_materialized_update_cosine"]) for row in rank64
    ]
    reference_values = [
        float(row["reference_analytic_to_teacher_update_cosine"])
        for row in rank64
    ]
    formal = all(bool(payload.get("formal")) for payload in (
        read_json(args.output_dir / "workers" / f"worker_{index:03d}" / "results.json")
        for index in range(args.shard_count)
    ))
    early = all(bool(payload.get("early_disqualifier")) for payload in (
        read_json(args.output_dir / "workers" / f"worker_{index:03d}" / "results.json")
        for index in range(args.shard_count)
    ))
    capacity_pass = bool(capacity_values) and min(capacity_values) >= threshold
    write_json_atomic(
        args.output_dir / "report.json",
        {
            "schema_version": REPORT_SCHEMA,
            "status": (
                "complete_capacity_disqualifier"
                if formal and early and not capacity_pass
                else "exploratory_or_incomplete"
            ),
            "formal_gate_pass": bool(formal and not early and capacity_pass),
            "formal": formal,
            "early_disqualifier": early,
            "row_count": len(rows),
            "task_count": len({int(row["authority_id"]) for row in rows}),
            "condition_count": len(
                {(int(row["authority_id"]), int(row["video_demo"])) for row in rows}
            ),
            "rank_rows": rows,
            "rank64_counterexample": {
                "row_count": len(rank64),
                "required_minimum": threshold,
                "sketch_minimum": min(capacity_values),
                "sketch_mean": sum(capacity_values) / len(capacity_values),
                "streaming_to_materialized_minimum": min(chunk_values),
                "sealed_F1_analytic_to_teacher_minimum": min(reference_values),
                "capacity_pass": capacity_pass,
            },
            "workers": workers,
            "claim_boundary": (
                "A formal early counterexample can disqualify the mandatory row "
                "minimum but cannot estimate full-panel distributions."
            ),
            "git": git_state(REPO_ROOT),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "report"))
    parser.add_argument("--sketch-config", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dual-authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-indices", type=int, nargs="+")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--frame-chunk", type=int, default=4)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--early-disqualifier", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "sketch_config",
        "reference_config",
        "asset_root",
        "data_root",
        "dual_authority",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    config = read_json(args.sketch_config)
    default_targets = tuple(
        config["panel"]["primary_targets"] + config["panel"]["depth_targets"]
    )
    args.target_indices = tuple(
        default_targets if args.target_indices is None else args.target_indices
    )
    args.task_ids = None if args.task_ids is None else tuple(map(int, args.task_ids))
    if (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or args.frame_chunk <= 0
        or len(set(args.target_indices)) != len(args.target_indices)
    ):
        raise ValueError("invalid functional-sketch analyzer contract")
    if args.command == "worker":
        worker(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
