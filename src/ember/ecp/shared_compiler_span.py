"""Measure whether G3 fit-task mobile effects lie in their native X/Y spans."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.g1_initialization import (
    cache_native_video_readout,
    initialize_oracle_from_reference,
)
from ember.ecp.native_factors import TaskLocalNativeFactorOracle
from ember.ecp.native_materialization import (
    residual_lora_state,
    small_core_balanced_svd,
)
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.ecp.shared_compiler_training import prepare_runtime
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "ember_ecp_g3_fit_native_span_shard_v1"


def k1_schedule_sample(runtime: Any, task_id: int) -> tuple[NaturalProgramSample, int]:
    """Return the first deterministic G3 K1 visit and its original action split."""

    for macro in range(12):
        sampled = runtime.schedule.sample(task_id, macro)
        if sampled.k == 1:
            return sampled, macro
    raise RuntimeError(f"G3 schedule did not expose K1 for fit task {task_id}")


def _k1_sample(runtime: Any, task_id: int) -> NaturalProgramSample:
    sampled, _ = k1_schedule_sample(runtime, task_id)
    return NaturalProgramSample(
        video_demos=sampled.video_demos,
        action_demos=(),
        k=1,
        robustness_view="fit_span_k1",
    )


def capture_k1_native_readout(
    runtime: Any,
    task: Any,
    sample: NaturalProgramSample,
) -> tuple[Any, Mapping[str, Any]]:
    """Capture one real K1 Pass-B bank while leaving action episodes unread."""

    if sample.k != 1 or len(sample.video_demos) != 1:
        raise ValueError("G3 native-span diagnostic requires exactly one video")
    video_only = NaturalProgramSample(
        video_demos=sample.video_demos,
        action_demos=(),
        k=1,
        robustness_view="fit_span_k1",
    )
    packed = pack_shared_compiler_videos(
        task=task,
        sample=video_only,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.context.device,
    )
    tokens, mask = runtime.language_tokens[task.authority_id]
    prepared = prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=runtime.owners,
        packed=packed,
        language_tokens=tokens,
        language_mask=mask,
        chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
    )
    if len(prepared.videos) != 1:
        raise RuntimeError("fit-span diagnostic lost its K1 identity")
    return cache_native_video_readout(prepared.videos[0].native), packed.metrics


def project_member_into_k1_native_span(
    runtime: Any,
    *,
    task_id: int,
    member_index: int,
    member_name: str,
    native: Any,
    reference: Mapping[str, torch.Tensor],
    relative_singular_threshold: float,
    probability_floor_mass: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Project one verified mobile-rank4 member into a real K1 native bank."""

    oracle = TaskLocalNativeFactorOracle(
        runtime.owners,
        frame_counts=(native.frame_count,),
        event_slots=int(runtime.config["model"]["event_slots"]),
        program_width=int(runtime.config["model"]["program_width"]),
        initialization_seed=(
            int(runtime.config["optimization"]["seed"]) + task_id * 17 + member_index
        ),
    ).to(runtime.context.device)
    initialization = initialize_oracle_from_reference(
        oracle=oracle,
        video=native,
        owners=runtime.owners,
        contract=runtime.rank4_contract,
        reference=reference,
        s_ref=runtime.ranks.s_ref,
        relative_singular_threshold=relative_singular_threshold,
        probability_floor_mass=probability_floor_mass,
        reference_member=member_name,
    )
    residual = oracle((native,), s_ref=runtime.ranks.s_ref)
    state = residual_lora_state(residual, runtime.rank4_contract, canonicalize=False)
    geometry = _low_rank_geometry(state, reference, runtime)
    del oracle, residual
    return state, initialization, geometry


def _low_rank_geometry(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    runtime: Any,
) -> dict[str, Any]:
    by_family: dict[str, list[float]] = {}
    per_target = []
    for target, owner in zip(
        runtime.rank4_contract.targets, runtime.owners, strict=True
    ):
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        candidate_a = candidate[a_name].float()
        candidate_b = candidate[b_name].float()
        reference_a = reference[a_name].float()
        reference_b = reference[b_name].float()
        inner = (
            (candidate_b.transpose(0, 1) @ reference_b)
            * (candidate_a @ reference_a.transpose(0, 1))
        ).sum()
        candidate_norm2 = (
            (candidate_b.transpose(0, 1) @ candidate_b)
            * (candidate_a @ candidate_a.transpose(0, 1))
        ).sum()
        reference_norm2 = (
            (reference_b.transpose(0, 1) @ reference_b)
            * (reference_a @ reference_a.transpose(0, 1))
        ).sum()
        values = [float(inner), float(candidate_norm2), float(reference_norm2)]
        family = owner.family.value
        aggregate = by_family.setdefault(family, [0.0, 0.0, 0.0])
        for index, value in enumerate(values):
            aggregate[index] += value
        per_target.append(
            {
                "target": target.name,
                "family": family,
                **_summarize_geometry(values),
            }
        )
    overall = [sum(values[index] for values in by_family.values()) for index in range(3)]
    return {
        "overall": _summarize_geometry(overall),
        "families": {
            family: _summarize_geometry(values)
            for family, values in sorted(by_family.items())
        },
        "targets": per_target,
    }


def _summarize_geometry(values: Sequence[float]) -> dict[str, float]:
    inner, candidate_norm2, reference_norm2 = map(float, values)
    denominator = math.sqrt(max(candidate_norm2 * reference_norm2, 1e-30))
    squared_error = max(candidate_norm2 + reference_norm2 - 2.0 * inner, 0.0)
    return {
        "update_cosine": inner / denominator,
        "candidate_to_reference_norm": math.sqrt(
            max(candidate_norm2, 0.0) / max(reference_norm2, 1e-30)
        ),
        "relative_update_error": math.sqrt(
            squared_error / max(reference_norm2, 1e-30)
        ),
    }


def _summarize_scale_ratios(values: Sequence[float]) -> dict[str, float | int]:
    ratios = tuple(map(float, values))
    if not ratios:
        raise ValueError("native-span scale summary is empty")
    return {
        "rank_components": len(ratios),
        "median_to_s_ref": statistics.median(ratios),
        "maximum_to_s_ref": max(ratios),
        "above_cap_fraction": sum(value > 1.0 + 1e-6 for value in ratios)
        / len(ratios),
    }


def _reference_scale_pressure(
    reference: Mapping[str, torch.Tensor], runtime: Any
) -> dict[str, Any]:
    all_ratios = []
    by_family: dict[str, list[float]] = {}
    for index, (target, owner) in enumerate(
        zip(runtime.rank4_contract.targets, runtime.owners, strict=True)
    ):
        canonical_a, _ = small_core_balanced_svd(
            reference[target.name + LORA_A_SUFFIX].float(),
            reference[target.name + LORA_B_SUFFIX].float(),
        )
        ratios = (
            canonical_a.square().sum(-1)
            / math.sqrt(target.in_features * target.out_features)
            / runtime.ranks.s_ref[index].float()
        ).detach().cpu().tolist()
        all_ratios.extend(ratios)
        by_family.setdefault(owner.family.value, []).extend(ratios)
    return {
        "overall": _summarize_scale_ratios(all_ratios),
        "families": {
            family: _summarize_scale_ratios(values)
            for family, values in sorted(by_family.items())
        },
    }


def _fit_span_task(
    runtime: Any,
    task: Any,
    *,
    relative_singular_threshold: float,
    probability_floor_mass: float,
) -> dict[str, Any]:
    sample = _k1_sample(runtime, task.authority_id)
    native, packed_metrics = capture_k1_native_readout(runtime, task, sample)
    bank = runtime.effect_banks.get(task.authority_id)
    members = []
    for member, (name, reference) in enumerate(
        zip(bank.member_names, bank.projections, strict=True)
    ):
        state, initialization, geometry = project_member_into_k1_native_span(
            runtime,
            task_id=task.authority_id,
            member_index=member,
            member_name=name,
            native=native,
            reference=reference,
            relative_singular_threshold=relative_singular_threshold,
            probability_floor_mass=probability_floor_mass,
        )
        projected_scale_ratios = [
            value
            for row in initialization["targets"]
            for value in row["scale_to_s_ref"]
        ]
        members.append(
            {
                "member": name,
                "reliability": float(bank.reliability[member]),
                "geometry": geometry,
                "reference_scale_pressure": _reference_scale_pressure(
                    reference, runtime
                ),
                "projected_scale_cap_fraction": sum(
                    value >= 1.0 - 2e-6 for value in projected_scale_ratios
                )
                / len(projected_scale_ratios),
                "minimum_signed_input_realization_cosine": min(
                    value
                    for row in initialization["targets"]
                    for value in row["input_direction_cosine"]
                ),
                "minimum_signed_output_realization_cosine": min(
                    value
                    for row in initialization["targets"]
                    for value in row["output_direction_cosine"]
                ),
            }
        )
        del state
        torch.cuda.empty_cache()
    best = min(
        range(len(members)),
        key=lambda index: members[index]["geometry"]["overall"][
            "relative_update_error"
        ],
    )
    return {
        "authority_id": task.authority_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "role": task.role,
        "language": task.language,
        "video_demo": sample.video_demos[0],
        "sampled_frames": packed_metrics["sampled_frames"][0],
        "raw_frame_count": packed_metrics["raw_frame_counts"][0],
        "member_count": len(members),
        "best_member": members[best]["member"],
        "best_relative_update_error": members[best]["geometry"]["overall"][
            "relative_update_error"
        ],
        "members": members,
    }


def analyze_fit_span(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    runtime = None
    try:
        runtime = prepare_runtime(args, context)
        fit = tuple(
            task
            for task in sorted(runtime.tasks, key=lambda value: value.authority_id)
            if task.role in {"meta_fit", "target_fit"}
        )
        if len(fit) != 75 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("G3 fit-span shard authority changed")
        selected = fit[args.shard_index :: args.shard_count]
        if args.task_limit is not None:
            selected = selected[: args.task_limit]
        if not selected:
            raise ValueError("G3 fit-span shard is empty")
        g1 = read_json(args.g1_config)
        if g1.get("schema_version") != "ember_ecp_native_factor_g1_v1":
            raise ValueError("G1 projection authority schema changed")
        initialization = g1["optimization"]["initialization"]
        threshold = float(initialization["relative_singular_threshold"])
        floor = float(initialization["probability_floor_mass"])
        rows = []
        with torch.no_grad():
            for task in selected:
                rows.append(
                    _fit_span_task(
                        runtime,
                        task,
                        relative_singular_threshold=threshold,
                        probability_floor_mass=floor,
                    )
                )
                write_json_atomic(
                    args.output_dir / "span_progress.json",
                    {
                        "schema_version": RESULT_SCHEMA,
                        "status": "partial",
                        "shard": {
                            "index": args.shard_index,
                            "count": args.shard_count,
                            "completed_tasks": len(rows),
                            "scheduled_tasks": len(selected),
                        },
                        "tasks": rows,
                    },
                )
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "question": (
                "whether each fit task's verified mobile-rank4 effect is "
                "representable by its real K1 native X/Y bank under the "
                "bounded rank4 signed-pooling form"
            ),
            "claim_boundary": (
                "training-only fit-task span diagnostic; it does not prove a "
                "shared Program-to-attention mapping or a closed-loop Gate"
            ),
            "shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "task_count": len(rows),
            },
            "solver": {
                "kind": "g1_fp64_stable_span_reference_projection",
                "relative_singular_threshold": threshold,
                "probability_floor_mass": floor,
                "g1_config": str(args.g1_config),
            },
            "information_wall": {
                "roles": ["meta_fit", "target_fit"],
                "held_tasks": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "shuffled_or_reversed_use": False,
            },
            "tasks": rows,
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                runtime.context.device
            ),
        }
        write_json_atomic(args.output_dir / "span_results.json", payload)
        write_json_atomic(
            args.output_dir / "span_completion.json",
            {"schema_version": RESULT_SCHEMA, "task_count": len(rows)},
        )
        return payload
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--effect-bank-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--task-limit", type=int)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "g1_config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "effect_bank_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.mode = "profile"
    args.stop_after_macro = 1
    args.resume = None
    args.log_every = 1
    if args.shard_count <= 0 or (
        args.task_limit is not None and args.task_limit <= 0
    ):
        raise ValueError("invalid G3 fit-span shard request")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    payload = analyze_fit_span(args)
    print(
        f"completed G3 fit-span shard {args.shard_index}: "
        f"{payload['shard']['task_count']} tasks",
        flush=True,
    )
    return 0
