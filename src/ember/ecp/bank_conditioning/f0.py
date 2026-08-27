"""Real K1/K4 information-wall and materialization qualification."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.bank_conditioning.mapping import load_mapping_split
from ember.ecp.bank_conditioning.mapping_eval_runtime import load_mapping_tasks
from ember.ecp.contracts import build_target_owners
from ember.ecp.g1_initialization import cache_native_video_readout
from ember.ecp.native_factors import NativeTargetChunk, NativeVideoReadout
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.policy_effects import capture_policy_effect_response
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    G3_CONFIG_SCHEMA,
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.shared_compiler_authority import pure_shared_compiler_inventory
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.ecp.shared_compiler_effects import SharedEffectBankStore
from ember.ecp.shared_compiler_native_teacher import (
    NativeTeacherStore,
    native_teacher_supervision_loss,
)
from ember.ecp.stage0_training import stage0_source_authority, tokenize_stage0_languages
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class F0Runtime:
    state: dict[str, Any]
    config_path: Path
    config: dict[str, Any]
    asset_root: Path
    device: torch.device
    started: float
    policy: torch.nn.Module
    tasks: tuple[Any, ...]
    task: Any
    mapping_split: Any
    ranks: Any
    rank4_contract: Any
    owners: tuple[Any, ...]
    program: torch.nn.Module
    compiler: SharedNativeFactorCompiler
    inventory: dict[str, Any]
    tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    video_store: RawTeacherVideoStore
    teachers: NativeTeacherStore
    query_points: int

    def close(self) -> None:
        self.video_store.close()


@dataclass(frozen=True)
class F0K1:
    video: int
    output: Any
    complete_adapter: Mapping[str, torch.Tensor]
    gradient_norms: dict[str, float]
    raw_slot_error: float
    update_cosine_minimum: float
    update_cosine_median: float
    update_relative_error_maximum: float
    update_relative_error_median: float
    solve_metric_error: float
    feature_metric_error: float
    seconds: float


@dataclass(frozen=True)
class F0K4:
    videos: tuple[int, ...]
    output: Any
    permutation_error: float
    teacher_reads: int
    seconds: float


@dataclass(frozen=True)
class F0ChunkEquivalence:
    raw_slot_error: float
    update_cosine_minimum: float
    update_cosine_median: float
    update_relative_error_maximum: float
    update_relative_error_median: float
    solve_metric_error: float
    feature_metric_error: float


def _prepare_runtime(args: argparse.Namespace) -> F0Runtime:
    asset_root = args.asset_root.resolve()
    state = git_state(REPO_ROOT)
    clean = (
        git_state_is_clean_pushed_or_frozen_authority(state)
        and state.get("branch") == ""
        and state.get("upstream") is None
    )
    if not clean:
        raise RuntimeError("formal F0 requires clean detached pushed authority")
    config_path = REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v3.json"
    config = load_shared_compiler_config(config_path)
    if config.get("schema_version") != G3_CONFIG_SCHEMA:
        raise RuntimeError("F0 requires the active bank-conditioned compiler")
    device = torch.device("cuda:0")
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=asset_root
    )
    source = stage0_source_authority(
        SimpleNamespace(
            checkpoint=source_checkpoint,
            source_run=source_checkpoint.parent.parent,
        )
    )
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=asset_root)
    )
    started = time.monotonic()
    policy = load_policy(Path(source["model_path"]), source_config, device)
    policy.requires_grad_(False).eval()
    tasks = load_mapping_tasks(
        config, asset_root=asset_root, data_root=args.data_root.resolve()
    )
    task_by_id = {task.authority_id: task for task in tasks}
    if args.task not in task_by_id:
        raise RuntimeError("F0 task is outside the mapping authority")
    mapping_split = load_mapping_split(config, asset_root=asset_root)
    ranks = load_shared_rank_assets(
        config,
        asset_root=asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=device,
    )
    owners = build_target_owners(ranks.contract)
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    program = build_frozen_g2_program(
        config, asset_root=asset_root, owners=owners, device=device
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    torch.manual_seed(int(config["optimization"]["seed"]))
    torch.cuda.manual_seed_all(int(config["optimization"]["seed"]))
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(config["model"]["program_width"]),
        event_slots=int(config["model"]["event_slots"]),
        anchor_width=int(config["model"]["anchor_width"]),
        relative_eigenvalue_floor=float(config["model"]["relative_eigenvalue_floor"]),
        global_statistics=True,
    ).to(device).train()
    inventory = pure_shared_compiler_inventory(
        policy=policy, program=program, compiler=compiler, owners=owners
    )
    tokens = tokenize_stage0_languages(
        tasks,
        tokenizer_path=authority_path(config, "tokenizer", asset_root=asset_root),
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=device,
    )
    video_store = RawTeacherVideoStore(
        tuple(task.writer_authority() for task in tasks),
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=8,
    )
    teacher_path = authority_path(
        config, "native_teacher_manifest", asset_root=asset_root
    )
    teacher_root = read_json(teacher_path)
    teachers = NativeTeacherStore(
        teacher_path,
        contract=rank4_contract,
        expected_fit_task_ids=set(map(int, teacher_root["coverage"]["task_ids"])),
        expected_full_fit_task_ids=set(
            map(int, teacher_root["fit_authority_task_ids"])
        ),
        device=device,
    )
    g2 = read_json(authority_path(config, "g2_config", asset_root=asset_root))
    return F0Runtime(
        state=state,
        config_path=config_path,
        config=config,
        asset_root=asset_root,
        device=device,
        started=started,
        policy=policy,
        tasks=tasks,
        task=task_by_id[args.task],
        mapping_split=mapping_split,
        ranks=ranks,
        rank4_contract=rank4_contract,
        owners=owners,
        program=program,
        compiler=compiler,
        inventory=inventory,
        tokens=tokens,
        video_store=video_store,
        teachers=teachers,
        query_points=int(g2["data"]["query_points"]),
    )


def _condition(
    runtime: F0Runtime, video_demos: tuple[int, ...], *, chunk_size: int
) -> Any:
    sample = NaturalProgramSample(
        video_demos=video_demos,
        action_demos=(),
        k=len(video_demos),
        robustness_view="f0_bank_compiler_smoke",
    )
    packed = pack_shared_compiler_videos(
        task=runtime.task,
        sample=sample,
        video_store=runtime.video_store,
        query_points=runtime.query_points,
        device=runtime.device,
    )
    language, mask = runtime.tokens[runtime.task.authority_id]
    return prepare_shared_compiler_condition(
        policy=runtime.policy,
        program_model=runtime.program,
        owners=runtime.owners,
        packed=packed,
        language_tokens=language,
        language_mask=mask,
        chunk_size=chunk_size,
    )


def _video_panel(runtime: F0Runtime) -> tuple[int, tuple[int, ...]]:
    manifest = read_json(
        Path(str(runtime.teachers.records[runtime.task.authority_id]["manifest"]))
    )
    videos = tuple(map(int, manifest["video_demos"]))
    if len(videos) < 4:
        raise RuntimeError("F0 task has fewer than four sealed K1 videos")
    return videos[0], videos[:4]


def _single_chunk_readout(readout: NativeVideoReadout) -> NativeVideoReadout:
    """Repartition one cached native bank without recapturing X/Y values."""

    chunks = tuple(readout.chunks())
    if not chunks or sum(chunk.frame_count for chunk in chunks) != readout.frame_count:
        raise RuntimeError("F0 cached native reference is incomplete")
    targets = len(chunks[0].inputs)
    if targets != len(readout.final_outputs) or any(
        len(chunk.inputs) != targets or len(chunk.outputs) != targets
        for chunk in chunks
    ):
        raise RuntimeError("F0 cached native target topology changed")
    inputs = tuple(
        torch.cat(tuple(chunk.inputs[target] for chunk in chunks), dim=0)
        for target in range(targets)
    )
    outputs = tuple(
        torch.cat(tuple(chunk.outputs[target] for chunk in chunks), dim=0)
        for target in range(targets)
    )
    def one_chunk() -> tuple[NativeTargetChunk, ...]:
        return (NativeTargetChunk(start_frame=0, inputs=inputs, outputs=outputs),)

    return NativeVideoReadout(
        frame_count=readout.frame_count,
        process=readout.process,
        state_posterior=readout.state_posterior,
        final_outputs=readout.final_outputs,
        chunks=one_chunk,
    )


def _low_rank_update_similarity(
    left_a: torch.Tensor,
    left_b: torch.Tensor,
    right_a: torch.Tensor,
    right_b: torch.Tensor,
) -> tuple[float, float]:
    """Compare ``B.T @ A`` without materializing a target-sized update."""

    # This is a qualification metric over rank-four Gram matrices, not part of
    # the compiler's numerical path.  Accumulate it in FP64 so cancellation in
    # ||left - right|| does not dominate the small chunk-order difference.
    left_a = left_a.detach().double()
    left_b = left_b.detach().double()
    right_a = right_a.detach().double()
    right_b = right_b.detach().double()
    inner = ((left_a @ right_a.T) * (left_b @ right_b.T)).sum()
    left_squared = ((left_a @ left_a.T) * (left_b @ left_b.T)).sum()
    right_squared = ((right_a @ right_a.T) * (right_b @ right_b.T)).sum()
    left_norm = left_squared.clamp_min(0).sqrt()
    right_norm = right_squared.clamp_min(0).sqrt()
    if float(right_norm) == 0.0:
        if float(left_norm) == 0.0:
            return 1.0, 0.0
        return 0.0, float("inf")
    denominator = (left_norm * right_norm).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    cosine = (inner / denominator).clamp(-1.0, 1.0)
    difference_squared = (left_squared + right_squared - 2.0 * inner).clamp_min(0)
    relative_error = difference_squared.sqrt() / right_norm
    return float(cosine), float(relative_error)


def _chunk_equivalence(runtime: F0Runtime, prepared: Any) -> F0ChunkEquivalence:
    runtime.compiler.zero_grad(set_to_none=True)
    runtime.compiler.eval()
    cached = cache_native_video_readout(prepared.videos[0].native)
    chunked_video = replace(prepared.videos[0], native=cached)
    single_video = replace(
        prepared.videos[0], native=_single_chunk_readout(cached)
    )
    with torch.no_grad():
        chunked_output = runtime.compiler(
            prepared.program, (chunked_video,), s_ref=runtime.ranks.s_ref
        )
        reference_output = runtime.compiler(
            prepared.program, (single_video,), s_ref=runtime.ranks.s_ref
        )
    raw_slot_error = max(
        float((left.detach() - right.detach()).abs().max())
        for left, right in zip(
            (*chunked_output.residual.a, *chunked_output.residual.b),
            (*reference_output.residual.a, *reference_output.residual.b),
            strict=True,
        )
    )
    similarities = tuple(
        _low_rank_update_similarity(left_a, left_b, right_a, right_b)
        for left_a, left_b, right_a, right_b in zip(
            chunked_output.residual.a,
            chunked_output.residual.b,
            reference_output.residual.a,
            reference_output.residual.b,
            strict=True,
        )
    )
    cosines = torch.tensor(
        tuple(row[0] for row in similarities), dtype=torch.float64
    )
    relative_errors = torch.tensor(
        tuple(row[1] for row in similarities), dtype=torch.float64
    )
    solve_error = float(
        (chunked_output.solve_metrics - reference_output.solve_metrics)
        .detach()
        .abs()
        .max()
    )
    feature_error = float(
        (
            chunked_output.feature_whitening_metrics
            - reference_output.feature_whitening_metrics
        )
        .detach()
        .abs()
        .max()
    )
    return F0ChunkEquivalence(
        raw_slot_error=raw_slot_error,
        update_cosine_minimum=float(cosines.min()),
        update_cosine_median=float(cosines.median()),
        update_relative_error_maximum=float(relative_errors.max()),
        update_relative_error_median=float(relative_errors.median()),
        solve_metric_error=solve_error,
        feature_metric_error=feature_error,
    )


def _run_k1(runtime: F0Runtime, *, video: int, chunk_size: int) -> F0K1:
    tick = time.monotonic()
    prepared = _condition(runtime, (video,), chunk_size=chunk_size)
    output = runtime.compiler(
        prepared.program, prepared.videos, s_ref=runtime.ranks.s_ref
    )
    member_names = tuple(
        sorted(runtime.mapping_split.member_names[runtime.task.authority_id])
    )
    teachers = runtime.teachers.lookup_members(
        authority_id=runtime.task.authority_id,
        k=1,
        video_demo=video,
        member_names=member_names,
    )
    if not teachers or tuple(row.member_name for row in teachers) != member_names:
        raise RuntimeError("F0 K1 lost its fit-only teacher")
    loss = native_teacher_supervision_loss(
        student_a_directions=output.input_directions,
        student_b_directions=output.output_directions,
        student_scales=output.residual.scales,
        teachers=teachers,
        owners=runtime.owners,
    )
    loss.total.backward()
    gradients = {
        "input_anchor": runtime.compiler.anchor_scorer.input_anchor_query["q"][
            -1
        ].weight.grad,
        "output_anchor": runtime.compiler.anchor_scorer.output_anchor_query["q"][
            -1
        ].weight.grad,
        "input_owner_query": (
            runtime.compiler.anchor_scorer.query_owner_film.input_shift.grad
        ),
        "output_owner_query": (
            runtime.compiler.anchor_scorer.query_owner_film.output_shift[0].grad
        ),
        "input_candidate": runtime.compiler.anchor_scorer.input_candidates["q"]
        .direction_input.weight.grad,
        "output_candidate": runtime.compiler.anchor_scorer.output_candidates["q"]
        .direction_input.weight.grad,
        "input_joint_query": runtime.compiler.anchor_scorer.input_joint_compatibility[
            "q"
        ].query_projection.weight.grad,
        "input_joint_key": runtime.compiler.anchor_scorer.input_joint_compatibility[
            "q"
        ].key_projection.weight.grad,
        "input_joint_scalar": runtime.compiler.anchor_scorer.input_joint_compatibility[
            "q"
        ].scalar.weight.grad,
        "output_joint_query": runtime.compiler.anchor_scorer.output_joint_compatibility[
            "q"
        ].query_projection.weight.grad,
        "output_joint_key": runtime.compiler.anchor_scorer.output_joint_compatibility[
            "q"
        ].key_projection.weight.grad,
        "output_joint_scalar": runtime.compiler.anchor_scorer.output_joint_compatibility[
            "q"
        ].scalar.weight.grad,
        "stable_language": runtime.compiler.anchor_scorer.language_context["q"][
            1
        ].weight.grad,
        "group_gain": runtime.compiler.anchor_scorer.group_gain["q"][
            -1
        ].weight.grad,
        "scale": runtime.compiler.scale_head[-1].weight.grad,
    }
    valid_gradients = all(
        value is not None
        and bool(torch.isfinite(value).all())
        and float(value.float().norm()) > 0
        for value in gradients.values()
    )
    if not valid_gradients:
        raise RuntimeError("F0 compiler gradient path is absent or non-finite")
    gradient_norms = {
        name: float(value.float().norm()) for name, value in gradients.items()
    }
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=True
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    equivalence = _chunk_equivalence(runtime, prepared)
    return F0K1(
        video=video,
        output=output,
        complete_adapter=complete,
        gradient_norms=gradient_norms,
        raw_slot_error=equivalence.raw_slot_error,
        update_cosine_minimum=equivalence.update_cosine_minimum,
        update_cosine_median=equivalence.update_cosine_median,
        update_relative_error_maximum=equivalence.update_relative_error_maximum,
        update_relative_error_median=equivalence.update_relative_error_median,
        solve_metric_error=equivalence.solve_metric_error,
        feature_metric_error=equivalence.feature_metric_error,
        seconds=time.monotonic() - tick,
    )


def _consume_adapter(runtime: F0Runtime, state: Mapping[str, torch.Tensor]) -> tuple[Any, float]:
    store = SharedEffectBankStore(
        authority_path(
            runtime.config, "shared_effect_bank", asset_root=runtime.asset_root
        ),
        contract=runtime.ranks.contract,
        owners=runtime.owners,
        expected_task_ids={
            task.authority_id
            for task in runtime.tasks
            if task.role in {"meta_fit", "target_fit"}
        },
        device=runtime.device,
    )
    bank = store.get(runtime.task.authority_id)
    lora = BatchedLoRAInference(runtime.policy, runtime.ranks.contract)
    tick = time.monotonic()
    try:
        response = capture_policy_effect_response(
            policy=runtime.policy,
            observer=runtime.program.encoder.observer,
            lora=lora,
            state=state,
            prefix=type(bank.prefix)(
                embeddings=bank.prefix.embeddings[:1],
                padding=bank.prefix.padding[:1],
            ),
            suffix_noise=bank.suffix_noise[:1],
            denoising_steps=1,
        )
    finally:
        lora.close()
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (response.owner, response.flow, response.action)
    )
    if not finite:
        raise RuntimeError("F0 policy consumption is non-finite")
    return response, time.monotonic() - tick


def _run_k4(
    runtime: F0Runtime, *, videos: tuple[int, ...], chunk_size: int
) -> F0K4:
    before = runtime.teachers.tensor_reads
    tick = time.monotonic()
    prepared = _condition(runtime, videos, chunk_size=chunk_size)
    with torch.no_grad():
        output = runtime.compiler(
            prepared.program, prepared.videos, s_ref=runtime.ranks.s_ref
        )
        reverse = runtime.compiler(
            prepared.program,
            tuple(reversed(prepared.videos)),
            s_ref=runtime.ranks.s_ref,
        )
    error = max(
        float((left - right).abs().max())
        for left, right in zip(
            (*output.residual.a, *output.residual.b),
            (*reverse.residual.a, *reverse.residual.b),
            strict=True,
        )
    )
    reads = runtime.teachers.tensor_reads - before
    if reads != 0:
        raise RuntimeError("F0 K4 read training-only teacher tensors")
    return F0K4(
        videos=videos,
        output=output,
        permutation_error=error,
        teacher_reads=reads,
        seconds=time.monotonic() - tick,
    )


def _forbidden_checkpoint_keys(compiler: torch.nn.Module) -> list[str]:
    forbidden = (
        "task_lookup",
        "video_lookup",
        "frame_lookup",
        "teacher",
        "covariance",
        "analytic_dual",
    )
    return [
        name
        for name in compiler.state_dict()
        if any(token in name for token in forbidden)
    ]


def _qualification_checks(result: Mapping[str, Any]) -> dict[str, bool]:
    video_weights = tuple(map(float, result["video_weights"]))
    uniform = 1.0 / len(video_weights)
    return {
        "compiler_forward_information_wall": result[
            "compiler_forward_parameters"
        ]
        == ["program", "videos", "s_ref"],
        "action_meta_absent": result["action_meta_modules"] == 0
        and result["action_meta_parameters"] == 0,
        "source_and_program_frozen": result["source_trainable"] == 0
        and result["program_trainable"] == 0,
        "unique_complete_rank16": result["rank16_tensor_count"] == 76
        and result["rank16_targets"] == 38,
        "checkpoint_has_no_lookup_or_teacher_state": not result[
            "checkpoint_forbidden_keys"
        ],
        "K4_teacher_tensor_reads_zero": result["k4_teacher_tensor_reads"] == 0,
        "K4_uniform_video_measure": bool(video_weights)
        and max(abs(value - uniform) for value in video_weights) <= 1e-7,
        "K4_permutation_invariant": result["k4_permutation_maximum_error"]
        <= 2e-6,
        "chunked_matches_nonchunked_effective_update": result[
            "chunked_to_nonchunked_update_cosine_minimum"
        ]
        >= 0.99999
        and result["chunked_to_nonchunked_update_relative_error_maximum"]
        <= 5e-3
        and result["chunked_to_nonchunked_update_relative_error_median"]
        <= 1e-3,
        "chunked_matches_nonchunked_feature_whitening": result[
            "chunked_to_nonchunked_feature_metric_maximum_error"
        ]
        <= 1e-4,
    }


def _build_result(
    runtime: F0Runtime,
    *,
    k1: F0K1,
    k4: F0K4,
    response: Any,
    consume_seconds: float,
) -> dict[str, Any]:
    result = {
        "schema_version": "ember_ecp_shared_compiler_f0_v1",
        "git": runtime.state,
        "config": str(runtime.config_path),
        "task": runtime.task.authority_id,
        "k1_video": k1.video,
        "k4_videos": list(k4.videos),
        "compiler_forward_parameters": list(
            inspect.signature(runtime.compiler.forward).parameters
        ),
        "action_meta_modules": runtime.inventory["action_meta_module_count"],
        "action_meta_parameters": runtime.inventory["action_meta_parameter_count"],
        "source_trainable": runtime.inventory[
            "source_policy_trainable_parameter_count"
        ],
        "program_trainable": runtime.inventory[
            "natural_program_trainable_parameter_count"
        ],
        "gradient_norms": k1.gradient_norms,
        "chunked_to_nonchunked_raw_slot_maximum_error": k1.raw_slot_error,
        "chunked_to_nonchunked_update_cosine_minimum": (
            k1.update_cosine_minimum
        ),
        "chunked_to_nonchunked_update_cosine_median": (
            k1.update_cosine_median
        ),
        "chunked_to_nonchunked_update_relative_error_maximum": (
            k1.update_relative_error_maximum
        ),
        "chunked_to_nonchunked_update_relative_error_median": (
            k1.update_relative_error_median
        ),
        "chunked_to_nonchunked_solve_metric_maximum_error": (
            k1.solve_metric_error
        ),
        "chunked_to_nonchunked_feature_metric_maximum_error": (
            k1.feature_metric_error
        ),
        "chunked_update_cosine_minimum_threshold": 0.99999,
        "chunked_update_relative_error_maximum_threshold": 5e-3,
        "chunked_update_relative_error_median_threshold": 1e-3,
        "chunk_reference": "same cached native X/Y bank, chunk4 versus one chunk",
        "solve_metrics": k1.output.solve_metrics.detach().cpu().tolist(),
        "feature_whitening_metrics": (
            k1.output.feature_whitening_metrics.detach().cpu().tolist()
        ),
        "rank16_tensor_count": len(k1.complete_adapter),
        "rank16_targets": len(k1.complete_adapter) // 2,
        "policy_consumed_shapes": {
            "owner": list(response.owner.shape),
            "flow": list(response.flow.shape),
            "action": list(response.action.shape),
        },
        "k4_teacher_tensor_reads": k4.teacher_reads,
        "k4_permutation_maximum_error": k4.permutation_error,
        "video_weights": k4.output.video_weights.detach().cpu().tolist(),
        "checkpoint_forbidden_keys": _forbidden_checkpoint_keys(runtime.compiler),
        "k1_seconds": k1.seconds,
        "policy_consume_seconds": consume_seconds,
        "k4_seconds": k4.seconds,
        "total_seconds": time.monotonic() - runtime.started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
    }
    checks = _qualification_checks(result)
    result["qualification_checks"] = checks
    result["passed"] = all(checks.values())
    if not result["passed"]:
        failed = {name: value for name, value in checks.items() if not value}
        diagnostic = {
            "failed": failed,
            "chunked_to_nonchunked_raw_slot_maximum_error": result[
                "chunked_to_nonchunked_raw_slot_maximum_error"
            ],
            "chunked_to_nonchunked_update_cosine_minimum": result[
                "chunked_to_nonchunked_update_cosine_minimum"
            ],
            "chunked_to_nonchunked_update_relative_error_maximum": result[
                "chunked_to_nonchunked_update_relative_error_maximum"
            ],
            "chunked_to_nonchunked_update_relative_error_median": result[
                "chunked_to_nonchunked_update_relative_error_median"
            ],
            "chunked_to_nonchunked_solve_metric_maximum_error": result[
                "chunked_to_nonchunked_solve_metric_maximum_error"
            ],
            "chunked_to_nonchunked_feature_metric_maximum_error": result[
                "chunked_to_nonchunked_feature_metric_maximum_error"
            ],
            "k4_permutation_maximum_error": result[
                "k4_permutation_maximum_error"
            ],
            "video_weights": result["video_weights"],
        }
        raise RuntimeError(
            "F0 bank-conditioned compiler contract did not pass: "
            + json.dumps(diagnostic, sort_keys=True)
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", type=int, default=93)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime: F0Runtime | None = None
    try:
        runtime = _prepare_runtime(args)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        k1_video, k4_videos = _video_panel(runtime)
        k1 = _run_k1(runtime, video=k1_video, chunk_size=args.chunk)
        response, consume_seconds = _consume_adapter(
            runtime, k1.complete_adapter
        )
        k4 = _run_k4(runtime, videos=k4_videos, chunk_size=args.chunk)
        result = _build_result(
            runtime,
            k1=k1,
            k4=k4,
            response=response,
            consume_seconds=consume_seconds,
        )
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError("F0 output already exists")
        write_json_atomic(output, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
