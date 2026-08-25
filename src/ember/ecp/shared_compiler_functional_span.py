"""Locate functional loss across full, mobile-rank4, and native-span G3 states."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.native_materialization import compose_rank12_plus_rank4
from ember.ecp.policy_effects import PolicyEffectResponse
from ember.ecp.shared_compiler_effects import (
    SharedCompilerEffectBank,
    member_effect_losses,
)
from ember.ecp.shared_compiler_functional import _functional_query
from ember.ecp.shared_compiler_span import (
    _low_rank_geometry,
    capture_k1_native_readout,
    k1_schedule_sample,
    project_member_into_k1_native_span,
)
from ember.ecp.shared_compiler_train_step import _effect_responses
from ember.ecp.shared_compiler_training import prepare_runtime
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    task_logical_batch_policy_rng_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = "ember_ecp_g3_functional_native_span_shard_v1"
DEFAULT_SCREEN_TASK_IDS = (22, 62, 17, 85, 80, 93)


def _member_response(
    bank: SharedCompilerEffectBank, member_index: int
) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=bank.members.owner[member_index],
        flow=bank.members.flow[member_index],
        action=bank.members.action[member_index],
    )


def _effect_record(
    response: PolicyEffectResponse,
    bank: SharedCompilerEffectBank,
    *,
    member_index: int,
) -> dict[str, Any]:
    losses = member_effect_losses(response, bank)
    named_total = float(losses.member_totals[member_index])
    return {
        "global_set_loss": float(losses.global_effect),
        "family_functional_loss": float(losses.family_functional),
        "member_flow_loss": float(losses.member_flow_response),
        "action_response_loss": float(losses.action_response),
        "responsibilities": losses.responsibilities.float().cpu().tolist(),
        "member_totals": losses.member_totals.float().cpu().tolist(),
        "named_member_total": named_total,
        "named_member_retention": 1.0 - named_total,
    }


def _retention(
    *, carrier_loss: float, full_loss: float, candidate_loss: float
) -> float | None:
    denominator = carrier_loss - full_loss
    if abs(denominator) <= 1e-8:
        return None
    return (carrier_loss - candidate_loss) / denominator


def _difference_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    runtime: Any,
) -> dict[str, torch.Tensor]:
    """Represent the exact per-target delta left-right without dense matrices."""

    difference: dict[str, torch.Tensor] = {}
    for target in runtime.ranks.contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        difference[a_name] = torch.cat(
            (left[a_name].float(), right[a_name].float()), dim=0
        )
        difference[b_name] = torch.cat(
            (left[b_name].float(), -right[b_name].float()), dim=1
        )
    return difference


def _action_flow_value(
    runtime: Any,
    *,
    query: Any,
    policy_rng_seed: int,
    state: Mapping[str, torch.Tensor],
) -> float:
    value, details, gradients = functional_lora_loss_gradient(
        runtime.policy,
        state,
        runtime.ranks.contract,
        batch=query.batch,
        policy_rng_seed=policy_rng_seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(
            runtime.config["optimization"]["functional_policy_microbatch_size"]
        ),
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("G3 functional-span action-flow query changed")
    result = float(value)
    del gradients
    return result


def _task_member_authorities(runtime: Any, task_id: int) -> Any:
    rows = [row for row in runtime.members if row.task.authority_id == task_id]
    if len(rows) != 1:
        raise ValueError(f"G3 functional span lost member task {task_id}")
    return rows[0]


def _functional_span_task(
    runtime: Any,
    task: Any,
    *,
    relative_singular_threshold: float,
    probability_floor_mass: float,
) -> dict[str, Any]:
    sample, visit = k1_schedule_sample(runtime, task.authority_id)
    if set(sample.video_demos) & set(sample.action_demos):
        raise RuntimeError("G3 functional-span video/action split is not disjoint")
    with torch.no_grad():
        native, video_metrics = capture_k1_native_readout(runtime, task, sample)
    bank = runtime.effect_banks.get(task.authority_id)
    authorities = _task_member_authorities(runtime, task.authority_id)
    if tuple(member.name for member in authorities.members) != bank.member_names:
        raise ValueError("G3 functional-span expert/effect member order changed")

    query = _functional_query(
        dataset=runtime.query_dataset,
        processor=runtime.query_processor,
        task_id=task.authority_id,
        action_demos=sample.action_demos,
        visit=visit,
        seed=int(runtime.config["optimization"]["seed"]),
        query_count=int(runtime.config["optimization"]["functional_query_count"]),
    )
    policy_rng_seed = task_logical_batch_policy_rng_seed(
        optimization_seed=int(runtime.config["optimization"]["seed"]),
        task_id=task.authority_id,
        task_visit=visit,
        demo_indices=query.demo_indices,
        frame_indices=query.frame_indices,
    )
    carrier_flow = _action_flow_value(
        runtime,
        query=query,
        policy_rng_seed=policy_rng_seed,
        state=runtime.ranks.carrier_complete,
    )
    members = []
    for member_index, member in enumerate(authorities.members):
        if (
            not member.adapter.is_file()
            or member.adapter.stat().st_size != member.adapter_bytes
        ):
            raise ValueError("G3 functional-span full expert authority changed")
        full = load_file(str(member.adapter), device=str(runtime.context.device))
        validate_lora_state(full, runtime.ranks.contract)
        mobile_residual = {
            name: value.to(
                device=runtime.context.device,
                dtype=runtime.ranks.carrier_rank12[name].dtype,
            )
            for name, value in bank.projections[member_index].items()
        }
        mobile = compose_rank12_plus_rank4(
            carrier_state=runtime.ranks.carrier_rank12,
            residual_state=mobile_residual,
            rank16_contract=runtime.ranks.contract,
        )
        with torch.no_grad():
            native_residual, initialization, native_geometry = (
                project_member_into_k1_native_span(
                    runtime,
                    task_id=task.authority_id,
                    member_index=member_index,
                    member_name=member.name,
                    native=native,
                    reference=mobile_residual,
                    relative_singular_threshold=relative_singular_threshold,
                    probability_floor_mass=probability_floor_mass,
                )
            )
            native_complete = compose_rank12_plus_rank4(
                carrier_state=runtime.ranks.carrier_rank12,
                residual_state=native_residual,
                rank16_contract=runtime.ranks.contract,
            )
            mobile_response, native_response = _effect_responses(
                runtime,
                bank=bank,
                states=(mobile, native_complete),
            )
        responses = {
            "carrier": bank.carrier,
            "full_expert": _member_response(bank, member_index),
            "mobile_rank4": mobile_response,
            "native_projected": native_response,
        }
        effect = {
            name: _effect_record(response, bank, member_index=member_index)
            for name, response in responses.items()
        }
        carrier_set = effect["carrier"]["global_set_loss"]
        full_set = effect["full_expert"]["global_set_loss"]
        for name in ("mobile_rank4", "native_projected"):
            effect[name]["global_set_retention"] = _retention(
                carrier_loss=carrier_set,
                full_loss=full_set,
                candidate_loss=effect[name]["global_set_loss"],
            )

        full_flow = _action_flow_value(
            runtime,
            query=query,
            policy_rng_seed=policy_rng_seed,
            state=full,
        )
        mobile_flow = _action_flow_value(
            runtime,
            query=query,
            policy_rng_seed=policy_rng_seed,
            state=mobile,
        )
        native_flow = _action_flow_value(
            runtime,
            query=query,
            policy_rng_seed=policy_rng_seed,
            state=native_complete,
        )
        action_flow = {
            "carrier": carrier_flow,
            "full_expert": full_flow,
            "mobile_rank4": mobile_flow,
            "native_projected": native_flow,
            "full_benefit_over_carrier": carrier_flow - full_flow,
            "mobile_benefit_over_carrier": carrier_flow - mobile_flow,
            "native_benefit_over_carrier": carrier_flow - native_flow,
            "mobile_retention": _retention(
                carrier_loss=carrier_flow,
                full_loss=full_flow,
                candidate_loss=mobile_flow,
            ),
            "native_retention": _retention(
                carrier_loss=carrier_flow,
                full_loss=full_flow,
                candidate_loss=native_flow,
            ),
        }
        full_delta = _difference_state(full, runtime.ranks.carrier_complete, runtime)
        geometry = {
            "full_to_mobile": _low_rank_geometry(mobile_residual, full_delta, runtime),
            "mobile_to_native": native_geometry,
            "full_to_native": _low_rank_geometry(native_residual, full_delta, runtime),
        }
        projected_scale_ratios = [
            value
            for row in initialization["targets"]
            for value in row["scale_to_s_ref"]
        ]
        members.append(
            {
                "member": member.name,
                "step": member.step,
                "verified_successes": member.successes,
                "reliability": float(bank.reliability[member_index]),
                "effect": effect,
                "action_flow": action_flow,
                "geometry": geometry,
                "projected_scale_cap_fraction": sum(
                    value >= 1.0 - 2e-6 for value in projected_scale_ratios
                )
                / len(projected_scale_ratios),
            }
        )
        del (
            full,
            mobile_residual,
            mobile,
            native_residual,
            native_complete,
            mobile_response,
            native_response,
            full_delta,
        )
        torch.cuda.empty_cache()
    return {
        "authority_id": task.authority_id,
        "domain": task.domain,
        "domain_task_id": task.domain_task_id,
        "role": task.role,
        "language": task.language,
        "k1_schedule_visit": visit,
        "video_demo": sample.video_demos[0],
        "reserved_action_demos": list(sample.action_demos),
        "functional_action_demos": list(query.demo_indices),
        "functional_action_frames": list(query.frame_indices),
        "functional_policy_rng_seed": policy_rng_seed,
        "sampled_frames": video_metrics["sampled_frames"][0],
        "raw_frame_count": video_metrics["raw_frame_counts"][0],
        "member_count": len(members),
        "members": members,
    }


def analyze_functional_span(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    runtime = None
    try:
        runtime = prepare_runtime(args, context)
        fit = {
            task.authority_id: task
            for task in runtime.tasks
            if task.role in {"meta_fit", "target_fit"}
        }
        requested = tuple(args.task_ids)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(task_id not in fit for task_id in requested)
            or not 0 <= args.shard_index < args.shard_count
        ):
            raise ValueError("G3 functional-span task authority changed")
        selected = requested[args.shard_index :: args.shard_count]
        if not selected:
            raise ValueError("G3 functional-span shard is empty")
        action_meta_modules = tuple(
            name
            for name, _ in runtime.program.named_modules()
            if "action_meta" in name.lower()
        )
        if action_meta_modules:
            raise ValueError("G3 functional-span loaded Action Meta")
        g1 = read_json(args.g1_config)
        initialization = g1["optimization"]["initialization"]
        threshold = float(initialization["relative_singular_threshold"])
        floor = float(initialization["probability_floor_mass"])
        rows = []
        for task_id in selected:
            rows.append(
                _functional_span_task(
                    runtime,
                    fit[task_id],
                    relative_singular_threshold=threshold,
                    probability_floor_mass=floor,
                )
            )
            write_json_atomic(
                args.output_dir / "functional_span_progress.json",
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
                "where verified expert function is lost across full expert, "
                "carrier plus mobile rank4, and carrier plus K1-native-projected "
                "rank4 states"
            ),
            "claim_boundary": (
                "fit-task K1 successful-occupancy and paired action-query "
                "diagnostic; it is neither held-task evidence nor a shared "
                "Program-to-attention or closed-loop Gate"
            ),
            "shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "task_count": len(rows),
            },
            "information_wall": {
                "roles": sorted({row["role"] for row in rows}),
                "held_tasks": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "action_meta_named_modules": list(action_meta_modules),
                "shuffled_or_reversed_use": False,
                "video_action_cross_episode": True,
            },
            "tasks": rows,
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                runtime.context.device
            ),
        }
        write_json_atomic(args.output_dir / "functional_span_results.json", payload)
        write_json_atomic(
            args.output_dir / "functional_span_completion.json",
            {
                "schema_version": RESULT_SCHEMA,
                "task_count": len(rows),
            },
        )
        return payload
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _task_ids(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item)
    if not rows or len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("task IDs must be unique integers")
    return rows


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
    parser.add_argument(
        "--task-ids",
        type=_task_ids,
        default=DEFAULT_SCREEN_TASK_IDS,
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
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
    if args.shard_count <= 0:
        raise ValueError("invalid G3 functional-span shard request")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    payload = analyze_functional_span(args)
    print(
        f"completed G3 functional-span shard {args.shard_index}: "
        f"{payload['shard']['task_count']} tasks",
        flush=True,
    )
    return 0
