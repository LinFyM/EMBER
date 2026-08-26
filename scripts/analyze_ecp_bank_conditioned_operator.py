#!/usr/bin/env python3
"""Validate the streamed bank-conditioned G3 operator on the sealed F1 authority."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.bank_conditioning import (
    BankStatistics,
    StreamingBankStatistics,
    StreamingSignedPool,
    batched_spectral_bank_query,
    bounded_relative_group_gain,
    materialized_bank_statistics,
    materialized_signed_pool,
)
from ember.ecp.native_factors import (
    NativeOutputBankState,
    native_output_group_count,
    rms_normalize,
)
from ember.ecp.shared_compiler_dual_basis import (
    WORKER_SCHEMA as DUAL_WORKER_SCHEMA,
    quantiles,
    task_distribution,
    tensor_prefix,
    update_geometry,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from scripts.analyze_ecp_g3_dual_basis import (
    _prepare_condition,
    _runtime_for,
    _teacher_rows,
    build_parser as build_dual_parser,
    finalize_args as finalize_dual_args,
)


WORKER_SCHEMA = "ember_ecp_g3_bank_operator_worker_v1"
REPORT_SCHEMA = "ember_ecp_g3_bank_operator_report_v1"
TARGETS = (20, 21, 36, 37)


def _dual_sources(root: Path) -> tuple[dict[str, Any], ...]:
    sources = []
    for worker in sorted((root / "workers").glob("worker_*")):
        completion = read_json(worker / "completion.json")
        tensor_path = worker / "duals.safetensors"
        if (
            completion.get("schema_version") != DUAL_WORKER_SCHEMA
            or completion.get("status") != "complete"
            or not tensor_path.is_file()
        ):
            raise ValueError("F1 lost its sealed analytic-dual authority")
        for record in completion["records"]:
            if tuple(int(row["target"]) for row in record["targets"]) != TARGETS:
                raise ValueError("F1 analytic-dual target families changed")
            sources.append(
                {"worker": worker, "completion": completion, "record": record}
            )
    tasks = {int(source["record"]["authority_id"]) for source in sources}
    if len(sources) != 98 or len(tasks) != 50:
        raise ValueError("F1 authority is not the sealed 50-task/98-condition set")
    return tuple(sources)


def _task_shards(
    sources: Sequence[Mapping[str, Any]], count: int
) -> tuple[tuple[int, ...], ...]:
    costs: dict[int, int] = defaultdict(int)
    for source in sources:
        record = source["record"]
        costs[int(record["authority_id"])] += int(record["frame_count"]) * int(
            record["member_count"]
        )
    if not 0 < count <= len(costs):
        raise ValueError("invalid F1 worker count")
    shards: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    for task, cost in sorted(costs.items(), key=lambda row: (-row[1], row[0])):
        shard = min(range(count), key=lambda index: (loads[index], index))
        shards[shard].append(task)
        loads[shard] += cost
    return tuple(tuple(sorted(shard)) for shard in shards)


def _runtime_args(
    args: argparse.Namespace, task_ids: Sequence[int]
) -> argparse.Namespace:
    selected = tuple(sorted(map(int, task_ids)))
    if not selected:
        raise ValueError("F1 runtime received no tasks")
    source_run = args.asset_root / "runs/outputs/pi05_source_base_v1_seed7_1k_e2cc238_20260722"
    argv = [
        "worker",
        "--config", str(args.config),
        "--g1-config", str(args.g1_config),
        "--asset-root", str(args.asset_root),
        "--data-root", str(args.data_root),
        "--output-dir", str(args.output_dir),
        "--shard-count", "1",
        "--shard-index", "0",
        "--source-run", str(source_run),
        "--checkpoint", str(source_run / "checkpoints/step_00001000"),
        "--tokenizer-path",
        str(args.asset_root / "models/tokenizers/openpi/paligemma_tokenizer.model"),
        "--effect-bank-root",
        str(
            args.asset_root
            / "runs/analysis/"
            "pi05_ecp_shared_compiler_g3_effect_bank_37aca5f_gpu01p123456_20260825/"
            "manifest.json"
        ),
        "--target-indices", *(str(value) for value in TARGETS),
        "--max-videos-per-task", "2",
        "--basis-dimensions", "16", "32", "64", "96", "128",
        "--score-bound", str(args.score_bound),
        "--task-ids", *(str(task_id) for task_id in selected),
    ]
    return finalize_dual_args(build_dual_parser().parse_args(argv))


def _fixed_banks(runtime: Any, condition: Any) -> dict[int, tuple[torch.Tensor, ...]]:
    video = condition.videos[0]
    frames = video.native.frame_count
    assignment = video.canonical_assignment.double().clamp_min(0)
    assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-30)
    rho = condition.program.rho.double().clamp_min(1e-12)
    rho = rho / rho.sum()
    frame_mass = (assignment @ rho) * runtime.compiler._quadrature(
        video.frame_positions
    ).double()
    frame_mass = frame_mass / frame_mass.sum().clamp_min(1e-30)
    input_mass = frame_mass[:, None, None].expand(frames, 2, 50)
    output_mass = frame_mass[:, None, None, None].expand(frames, 2, 50, 4)
    inputs: dict[int, list[torch.Tensor]] = {target: [] for target in TARGETS}
    outputs: dict[int, list[torch.Tensor]] = {target: [] for target in TARGETS}
    boundaries = {
        target: NativeOutputBankState(
            final=video.native.final_outputs[target].detach()
        )
        for target in TARGETS
    }
    next_frame = 0
    for chunk in video.native.chunks():
        if chunk.start_frame != next_frame:
            raise RuntimeError("F1 native frame chunks are not contiguous")
        for target in TARGETS:
            owner = runtime.owners[target]
            groups = native_output_group_count(owner)
            dynamic = boundaries[target].build(
                chunk.outputs[target].detach(), start_frame=next_frame
            )
            grouped = dynamic.reshape(
                *dynamic.shape[:-1], groups, owner.out_features // groups
            ).movedim(-2, 0)
            inputs[target].append(chunk.inputs[target].detach())
            outputs[target].append(grouped)
        next_frame += chunk.frame_count
    if next_frame != frames or any(
        boundary.next_frame != frames for boundary in boundaries.values()
    ):
        raise RuntimeError("F1 native stream ended early")
    return {
        target: (
            torch.cat(inputs[target], dim=0),
            torch.cat(outputs[target], dim=1),
            input_mass,
            output_mass,
        )
        for target in TARGETS
    }


def _reference_compatibility(
    reference: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    *,
    score_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1, values.shape[-1]).double()
    weights = mass.reshape(-1).double()
    weights = weights / weights.sum()
    mean = torch.einsum("n,nd->d", weights, flat)
    score = reference.reshape(-1, reference.shape[-1]).double() @ (flat - mean).T
    maximum = score.abs().amax(-1).clamp_min(1e-30)
    bounded = torch.tanh(score * (score_bound / maximum)[:, None])
    return bounded.reshape(*reference.shape[:-1], *values.shape[:-1]), maximum.reshape(
        reference.shape[:-1]
    )


def _statistics_pair(
    values: torch.Tensor,
    mass: torch.Tensor,
    reference: torch.Tensor,
    *,
    score_bound: float,
    frame_chunk: int,
) -> tuple[BankStatistics, BankStatistics, torch.Tensor]:
    compatibility, maximum = _reference_compatibility(
        reference, values, mass, score_bound=score_bound
    )
    full = materialized_bank_statistics(values, mass, compatibility)
    streamed = StreamingBankStatistics(
        width=values.shape[-1],
        query_shape=tuple(reference.shape[:-1]),
        device=values.device,
        dtype=torch.float64,
    )
    for start in range(0, values.shape[0], frame_chunk):
        stop = min(values.shape[0], start + frame_chunk)
        streamed.add(
            values[start:stop],
            mass[start:stop],
            compatibility[..., start:stop, *([slice(None)] * (mass.ndim - 1))],
        )
    return full, streamed.finalize(), maximum


def _solve_entries(
    entries: Sequence[tuple[tuple[Any, ...], BankStatistics]],
    *,
    relative_floor: float,
) -> dict[tuple[Any, ...], Any]:
    grouped: dict[
        tuple[int, tuple[int, ...]],
        list[tuple[tuple[Any, ...], BankStatistics]],
    ] = defaultdict(list)
    for key, statistics in entries:
        grouped[(statistics.mean.numel(), tuple(statistics.anchor.shape[:-1]))].append(
            (key, statistics)
        )
    solved = {}
    for rows in grouped.values():
        results = batched_spectral_bank_query(
            tuple(row[1] for row in rows),
            relative_eigenvalue_floor=relative_floor,
        )
        solved.update(
            (row[0], result) for row, result in zip(rows, results, strict=True)
        )
    return solved


def _stream_pool(
    query: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    *,
    frame_chunk: int,
) -> torch.Tensor:
    accumulator = StreamingSignedPool(query, dtype=torch.float64)
    for start in range(0, values.shape[0], frame_chunk):
        stop = min(values.shape[0], start + frame_chunk)
        accumulator.add(values[start:stop], mass[start:stop])
    return accumulator.signed_mean()


def _target_statistics(
    *,
    target_row: Mapping[str, Any],
    banks: Mapping[int, tuple[torch.Tensor, ...]],
    tensors: Mapping[str, torch.Tensor],
    record_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    target = int(target_row["target"])
    x, y, input_mass, output_mass = banks[target]
    prefix = tensor_prefix(record_index, target)
    input_reference = tensors[f"{prefix}/input/unit"].to(
        device=x.device, dtype=torch.float64
    ) * tensors[f"{prefix}/input/dual_l2_norm"].to(
        device=x.device, dtype=torch.float64
    )[..., None]
    full, streamed, input_maximum = _statistics_pair(
        x,
        input_mass,
        input_reference,
        score_bound=args.score_bound,
        frame_chunk=args.operator_frame_chunk,
    )
    full_entries = [((target, "input", 0), full)]
    stream_entries = [((target, "input", 0), streamed)]
    output_references = []
    output_maxima = []
    for group in range(int(target_row["groups"])):
        reference = tensors[f"{prefix}/output/g{group:03d}/unit"].to(
            device=y.device, dtype=torch.float64
        ) * tensors[f"{prefix}/output/g{group:03d}/dual_l2_norm"].to(
            device=y.device, dtype=torch.float64
        )[..., None]
        full, streamed, maximum = _statistics_pair(
            y[group],
            output_mass,
            reference,
            score_bound=args.score_bound,
            frame_chunk=args.operator_frame_chunk,
        )
        full_entries.append(((target, "output", group), full))
        stream_entries.append(((target, "output", group), streamed))
        output_references.append(reference)
        output_maxima.append(maximum)
    payload = {
        "target_row": target_row,
        "x": x,
        "y": y,
        "input_mass": input_mass,
        "output_mass": output_mass,
        "input_reference": input_reference,
        "input_maximum": input_maximum,
        "output_references": output_references,
        "output_maxima": torch.stack(output_maxima),
    }
    return payload, full_entries, stream_entries


def _factor_replays(
    *,
    target: int,
    payload: Mapping[str, Any],
    full_solved: Mapping[tuple[Any, ...], Any],
    stream_solved: Mapping[tuple[Any, ...], Any],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    x = payload["x"]
    y = payload["y"]
    input_mass = payload["input_mass"]
    output_mass = payload["output_mass"]
    input_scale = args.score_bound / payload["input_maximum"]
    factors = {
        "analytic_a": rms_normalize(
            materialized_signed_pool(
                payload["input_reference"] * input_scale[..., None], x, input_mass
            )
        ),
        "full_a": rms_normalize(
            materialized_signed_pool(
                full_solved[(target, "input", 0)].query, x, input_mass
            )
        ),
        "stream_a": rms_normalize(
            _stream_pool(
                stream_solved[(target, "input", 0)].query,
                x,
                input_mass,
                frame_chunk=args.operator_frame_chunk,
            )
        ),
    }
    gain = bounded_relative_group_gain(payload["output_maxima"])
    blocks = {"analytic": [], "full": [], "stream": []}
    for group, reference in enumerate(payload["output_references"]):
        group_gain = gain[group, ..., None]
        group_scale = args.score_bound / payload["output_maxima"][group]
        blocks["analytic"].append(
            materialized_signed_pool(
                reference * group_scale[..., None] * group_gain,
                y[group],
                output_mass,
            )
        )
        blocks["full"].append(
            materialized_signed_pool(
                full_solved[(target, "output", group)].query * group_gain,
                y[group],
                output_mass,
            )
        )
        blocks["stream"].append(
            _stream_pool(
                stream_solved[(target, "output", group)].query * group_gain,
                y[group],
                output_mass,
                frame_chunk=args.operator_frame_chunk,
            )
        )
    for name, rows in blocks.items():
        factors[f"{name}_b"] = rms_normalize(torch.cat(rows, dim=-1))
    return factors, gain


def _geometry_rows(
    *,
    record: Mapping[str, Any],
    target: int,
    target_row: Mapping[str, Any],
    teachers: Sequence[Any],
    factors: Mapping[str, torch.Tensor],
    gain: torch.Tensor,
    solved: Mapping[tuple[Any, ...], Any],
) -> list[dict[str, Any]]:
    teacher_a, teacher_b = _teacher_rows(teachers, target)
    rows = []
    for member, teacher in enumerate(teachers):
        scales = teacher.scales[target].to(factors["full_a"])
        comparisons = {
            "materialized_to_analytic_update_cosine": update_geometry(
                factors["full_a"][member], factors["full_b"][member],
                factors["analytic_a"][member], factors["analytic_b"][member],
                scales,
            )["update_cosine"],
            "streaming_to_analytic_update_cosine": update_geometry(
                factors["stream_a"][member], factors["stream_b"][member],
                factors["analytic_a"][member], factors["analytic_b"][member],
                scales,
            )["update_cosine"],
            "streaming_to_materialized_update_cosine": update_geometry(
                factors["stream_a"][member], factors["stream_b"][member],
                factors["full_a"][member], factors["full_b"][member], scales,
            )["update_cosine"],
            "analytic_to_teacher_update_cosine": update_geometry(
                factors["analytic_a"][member], factors["analytic_b"][member],
                teacher_a[member].to(factors["analytic_a"]),
                teacher_b[member].to(factors["analytic_b"]), scales,
            )["update_cosine"],
        }
        input_solve = solved[(target, "input", 0)]
        rows.append(
            {
                "authority_id": int(record["authority_id"]),
                "role": record["role"],
                "suite": record["suite"],
                "video_demo": int(record["video_demo"]),
                "member_name": teacher.member_name,
                "target": target,
                "family": target_row["family"],
                **comparisons,
                "input_retained_rank": input_solve.retained_rank,
                "input_solve_residual": input_solve.relative_residual_maximum,
                "group_gain_minimum": float(gain.min()),
            }
        )
    return rows


def _condition_rows(
    runtime: Any, source: Mapping[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    record = source["record"]
    task_id = int(record["authority_id"])
    video_demo = int(record["video_demo"])
    banks = _fixed_banks(runtime, _prepare_condition(runtime, task_id, video_demo))
    effect_bank = runtime.effect_banks.get(task_id)
    teachers = runtime.native_teachers.lookup_members(
        authority_id=task_id,
        k=1,
        video_demo=video_demo,
        member_names=effect_bank.member_names,
    )
    if teachers is None or tuple(teacher.member_name for teacher in teachers) != tuple(
        record["member_names"]
    ):
        raise RuntimeError("F1 exact teacher lookup changed")
    payloads = {}
    full_entries = []
    stream_entries = []
    for target_row in record["targets"]:
        target = int(target_row["target"])
        payload, full, streamed = _target_statistics(
            target_row=target_row,
            banks=banks,
            tensors=source["tensors"],
            record_index=int(record["record_index"]),
            args=args,
        )
        payloads[target] = payload
        full_entries.extend(full)
        stream_entries.extend(streamed)
    full_solved = _solve_entries(
        full_entries, relative_floor=args.relative_eigenvalue_floor
    )
    stream_solved = _solve_entries(
        stream_entries, relative_floor=args.relative_eigenvalue_floor
    )
    rows = []
    for target, payload in payloads.items():
        factors, gain = _factor_replays(
            target=target,
            payload=payload,
            full_solved=full_solved,
            stream_solved=stream_solved,
            args=args,
        )
        rows.extend(
            _geometry_rows(
                record=record,
                target=target,
                target_row=payload["target_row"],
                teachers=teachers,
                factors=factors,
                gain=gain,
                solved=full_solved,
            )
        )
    return rows


def worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    worker_dir = args.output_dir / "workers" / f"worker_{args.shard_index:03d}"
    if worker_dir.exists():
        raise ValueError("F1 worker output already exists")
    state = git_state(REPO_ROOT)
    if args.formal and (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal F1 requires a clean detached origin/main authority")
    sources = _dual_sources(args.dual_authority)
    if args.task_ids is not None:
        requested = set(map(int, args.task_ids))
        available = {int(source["record"]["authority_id"]) for source in sources}
        if not requested or not requested <= available:
            raise ValueError("F1 exploratory task subset crossed its authority")
        sources = tuple(
            source
            for source in sources
            if int(source["record"]["authority_id"]) in requested
        )
    assigned = set(_task_shards(sources, args.shard_count)[args.shard_index])
    selected = [
        source
        for source in sources
        if int(source["record"]["authority_id"]) in assigned
    ]
    tensor_cache = {}
    for source in selected:
        tensor_path = source["worker"] / "duals.safetensors"
        if tensor_path not in tensor_cache:
            tensor_cache[tensor_path] = load_file(str(tensor_path))
        source["tensors"] = tensor_cache[tensor_path]
    authority_ready = time.perf_counter()
    runtime = None
    try:
        runtime = _runtime_for(
            _runtime_args(args, sorted(assigned)), worker_dir / "runtime"
        )
        runtime_ready = time.perf_counter()
        if runtime.native_teachers is None:
            raise RuntimeError("F1 did not load its fit-only teacher authority")
        rows = []
        for source in selected:
            with torch.no_grad():
                rows.extend(_condition_rows(runtime, source, args))
        operator_ready = time.perf_counter()
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
                "operator": {
                    "anchor": "bounded_tanh_analytic_native_anchor",
                    "solve": "FP64 truncated covariance eigensystem",
                    "relative_eigenvalue_floor": args.relative_eigenvalue_floor,
                    "output_group_gain": "bounded analytic relative score maximum",
                    "score_bound": args.score_bound,
                    "frame_chunk": args.operator_frame_chunk,
                },
                "information_wall": {
                    "roles": sorted({source["record"]["role"] for source in selected}),
                    "held_authority_reads": 0,
                    "action_meta_installed": False,
                    "source_policy_trainable_parameter_count": 0,
                    "natural_program_trainable_parameter_count": 0,
                    "deployment_use": False,
                },
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                    runtime.context.device
                ),
                "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                    runtime.context.device
                ),
                "profile": {
                    "authority_load_seconds": authority_ready - started,
                    "runtime_initialization_seconds": runtime_ready - authority_ready,
                    "operator_seconds": operator_ready - runtime_ready,
                    "total_seconds": operator_ready - started,
                    "conditions_per_operator_second": len(selected)
                    / max(operator_ready - runtime_ready, 1e-12),
                },
                "git": state,
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def report(args: argparse.Namespace) -> None:
    if (args.output_dir / "report.json").exists():
        raise ValueError("F1 report already exists")
    rows = []
    workers = []
    for index in range(args.shard_count):
        result = read_json(
            args.output_dir / "workers" / f"worker_{index:03d}" / "results.json"
        )
        if (
            result.get("schema_version") != WORKER_SCHEMA
            or result.get("status") != "complete"
            or result.get("shard_count") != args.shard_count
        ):
            raise ValueError("F1 report received an invalid worker")
        rows.extend(result["rows"])
        workers.append(
            {
                "shard_index": index,
                "task_ids": result["task_ids"],
                "record_count": result["record_count"],
                "row_count": result["row_count"],
                "max_cuda_allocated_bytes": result["max_cuda_allocated_bytes"],
                "max_cuda_reserved_bytes": result["max_cuda_reserved_bytes"],
                "profile": result["profile"],
            }
        )
    tasks = {int(row["authority_id"]) for row in rows}
    records = {(int(row["authority_id"]), int(row["video_demo"])) for row in rows}
    if len(tasks) != 50 or len(records) != 98:
        raise ValueError("F1 report did not cover its 50-task/98-condition authority")
    families = {}
    passed = True
    for family in ("q", "v", "action_in", "action_out"):
        family_rows = [row for row in rows if row["family"] == family]
        materialized = task_distribution(
            family_rows, "materialized_to_analytic_update_cosine"
        )
        streaming = task_distribution(
            family_rows, "streaming_to_analytic_update_cosine"
        )
        chunk = quantiles(
            [row["streaming_to_materialized_update_cosine"] for row in family_rows]
        )
        family_pass = (
            materialized["task_mean"]["median"] >= 0.995
            and materialized["task_mean"]["minimum"] >= 0.99
            and streaming["task_mean"]["median"] >= 0.995
            and streaming["task_mean"]["minimum"] >= 0.99
            and chunk["minimum"] >= 0.99999
        )
        passed = passed and family_pass
        families[family] = {
            "materialized_to_analytic": materialized,
            "streaming_to_analytic": streaming,
            "streaming_to_materialized": chunk,
            "pass": family_pass,
        }
    write_json_atomic(
        args.output_dir / "report.json",
        {
            "schema_version": REPORT_SCHEMA,
            "status": "complete",
            "question": (
                "Can the current-bank statistics/anchor/solve/replay operator "
                "preserve the sealed analytic Native-Factor upper bound?"
            ),
            "gate": {
                "each_family_task_mean_median": 0.995,
                "each_family_task_mean_minimum": 0.99,
                "chunk_materialized_row_minimum": 0.99999,
            },
            "gate_pass": passed,
            "task_count": len(tasks),
            "condition_count": len(records),
            "row_count": len(rows),
            "families": families,
            "workers": workers,
            "claim_boundary": (
                "F1 isolates operator capacity with analytic native anchors and gains; "
                "it does not prove the shared Program-to-anchor mapping."
            ),
            "git": git_state(REPO_ROOT),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dual-authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--score-bound", type=float, default=0.1)
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--operator-frame-chunk", type=int, default=4)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--formal", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "config", "g1_config", "asset_root", "data_root", "dual_authority",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or not 0.0 < args.score_bound <= 0.25
        or not 0.0 < args.relative_eigenvalue_floor < 1.0
        or args.operator_frame_chunk <= 0
    ):
        raise ValueError("invalid F1 operator contract")
    if args.formal and args.task_ids is not None:
        raise ValueError("formal F1 cannot use a task subset")
    if args.command == "worker":
        worker(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
