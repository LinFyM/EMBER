"""Fixed zero-gradient functional evaluation for J2 checkpoints."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.bank_conditioning.frozen_condition_cache import (
    FrozenMappingConditionCache,
    frozen_condition_cache_authority,
)
from ember.ecp.bank_conditioning.mapping import MappingCondition
from ember.ecp.bank_conditioning.primal_capacity import (
    TaskLocalPrimalCode,
    recovery_record,
    subset_teacher,
    task_local_output,
)
from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.joint_program_primal.gate import (
    POSITIVE_CONTROL_SCHEMA,
    _functional_value,
)
from ember.ecp.joint_program_primal.runtime import (
    J2_RUN_SCHEMA,
    J2_STAGE,
    JointProgramPrimalRuntime,
    load_joint_program_primal_config,
    prepare_joint_program_primal_runtime,
)
from ember.ecp.joint_program_primal.train_step import (
    functional_panel_batch,
    prepare_joint_condition,
)
from ember.ecp.native_factors import NativeFactorResidual
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler import SharedCompilerOutput
from ember.ecp.shared_compiler_assets import authority_path
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_joint_program_primal_condition,
    prepare_shared_compiler_condition,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


J2_GATE_SCHEMA = "ember_ecp_joint_program_primal_gate_v1"
J2_EVALUATION_SCHEMA = "ember_ecp_joint_program_primal_evaluation_task_v1"
FAMILY_NAMES = ("q", "v", "action_in", "action_out")


def load_joint_program_primal_gate(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    evaluation = config.get("evaluation", {})
    wall = config.get("information_wall", {})
    if (
        config.get("schema_version") != J2_GATE_SCHEMA
        or config.get("status") != "active_12_task_functional_qualification"
        or config.get("checkpoint_optimizer_steps") != [70, 110]
        or evaluation.get("functional_panel") != "panel_b"
        or evaluation.get("panel_visits") != 16
        or evaluation.get("wrong_pairing")
        != "next_j2_authority_id_within_same_meta_or_target_role_cyclic"
        or evaluation.get("selected_family_report_targets")
        != [0, 16, 34, 1, 17, 35, 36, 37]
        or wall.get("shuffled_or_reversed_use") is not False
        or wall.get("action_meta_installed") is not False
        or wall.get("single_complete_rank16") is not True
    ):
        raise ValueError("unsupported J2 functional Gate config")
    return config


def _checkpoint_authority(
    runtime: JointProgramPrimalRuntime,
    *,
    compiler_run: Path,
    compiler_checkpoint: Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    compiler_run = compiler_run.resolve()
    compiler_checkpoint = compiler_checkpoint.resolve()
    step = checkpoint_macro(compiler_checkpoint)
    run_contract = read_json(compiler_run / "run_contract.json")
    manifest = read_json(compiler_checkpoint / "checkpoint_manifest.json")
    files = manifest.get("files", {})
    expected = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(6)),
    }
    if (
        compiler_checkpoint.parent.parent != compiler_run
        or step not in set(map(int, gate["checkpoint_optimizer_steps"]))
        or run_contract.get("schema_version") != J2_RUN_SCHEMA
        or run_contract.get("stage") != J2_STAGE
        or run_contract.get("phase") != "joint"
        or run_contract.get("mode") != "formal"
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != J2_STAGE
        or int(manifest.get("next_macro", -1)) != step
        or int(manifest.get("world_size", -1)) != 6
        or manifest.get("run_contract_schema") != J2_RUN_SCHEMA
        or set(files) != expected
    ):
        raise ValueError("J2 evaluation checkpoint authority changed")
    for name, record in files.items():
        candidate = compiler_checkpoint / name
        if not candidate.is_file() or candidate.stat().st_size != int(record["bytes"]):
            raise ValueError(f"J2 evaluation checkpoint file changed: {name}")
    runtime.writer_state.load_state_dict(
        load_file(
            str(compiler_checkpoint / "ecp.safetensors"),
            device=str(runtime.context.device),
        ),
        strict=True,
    )
    runtime.writer_state.requires_grad_(False).eval()
    runtime.program.eval()
    runtime.compiler.eval()
    if (
        any(parameter.requires_grad for parameter in runtime.writer_state.parameters())
        or any(parameter.requires_grad for parameter in runtime.policy.parameters())
    ):
        raise RuntimeError("J2 evaluator did not freeze checkpoint and policy")
    return {
        "optimizer_step": step,
        "path": str(compiler_checkpoint),
        "training_commit": str(run_contract["git"]["commit"]),
        "world_size": int(manifest["world_size"]),
        "tensor_bytes": int(files["ecp.safetensors"]["bytes"]),
    }


def _task_conditions(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> tuple[MappingCondition, MappingCondition, MappingCondition]:
    fit = runtime.mapping_split.fit_by_task[task_id]
    held = runtime.mapping_split.video_held_by_task[task_id]
    if len(fit) < 2 or len(held) != 1:
        raise ValueError("J2 evaluation task video panel changed")
    return fit[0], fit[1], held[0]


def _wrong_task(runtime: JointProgramPrimalRuntime, task_id: int) -> int:
    role = runtime.panels[task_id].role
    tasks = tuple(
        sorted(task for task, panel in runtime.panels.items() if panel.role == role)
    )
    if len(tasks) != 6 or task_id not in tasks:
        raise ValueError("J2 wrong-task role panel changed")
    return tasks[(tasks.index(task_id) + 1) % len(tasks)]


def balanced_task_assignments(
    runtime: JointProgramPrimalRuntime, worker_count: int
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= worker_count <= 6:
        raise ValueError("J2 evaluator worker count changed")
    tasks = tuple(sorted(runtime.panels))
    costs = {}
    for task in tasks:
        first, second, held = _task_conditions(runtime, task)
        wrong = _task_conditions(runtime, _wrong_task(runtime, task))[0]
        costs[task] = (
            2 * first.sampled_frames
            + second.sampled_frames
            + held.sampled_frames
            + wrong.sampled_frames
        )
    rows: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for task in sorted(tasks, key=lambda value: (-costs[value], value)):
        worker = min(range(worker_count), key=lambda value: (loads[value], value))
        rows[worker].append(task)
        loads[worker] += costs[task]
    return tuple(tuple(sorted(row)) for row in rows)


def _compile_program(runtime: JointProgramPrimalRuntime, condition: Any) -> NaturalProgram:
    times = torch.linspace(
        0.0,
        1.0,
        runtime.query_points,
        dtype=torch.float32,
        device=runtime.context.device,
    )[None]
    program, _ = prepare_joint_program_primal_condition(
        program_model=runtime.program,
        condition=condition,
        query_times=times,
    )
    return program


def _language_program(program: NaturalProgram) -> NaturalProgram:
    events = program.rho.shape[0]
    boundaries = torch.linspace(
        0.0, 1.0, events + 1, device=program.rho.device, dtype=program.rho.dtype
    )
    return NaturalProgram(
        p_lang=program.p_lang,
        p_scene=torch.zeros_like(program.p_scene),
        p_process=torch.zeros_like(program.p_process),
        rho=torch.full_like(program.rho, 1.0 / events),
        tau=torch.stack((boundaries[:-1], boundaries[1:]), dim=-1),
        sigma=torch.zeros_like(program.sigma),
    )


def _complete_state(
    runtime: JointProgramPrimalRuntime,
    *,
    program: NaturalProgram,
    bank: Any,
) -> tuple[dict[str, torch.Tensor], SharedCompilerOutput]:
    teacher_reads = runtime.native_teachers.tensor_reads
    output = runtime.compiler.forward_compact(
        program, bank.videos, s_ref=runtime.ranks.s_ref
    )
    if (
        runtime.native_teachers.tensor_reads != teacher_reads
        or output.video_weights.shape != (1,)
        or float(output.video_weights[0]) != 1.0
    ):
        raise RuntimeError("J2 evaluation deployment forward changed")
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=True
    )
    complete = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )
    return complete, output


def _endpoint_cache(
    runtime: JointProgramPrimalRuntime, root: Path
) -> FrozenMappingConditionCache:
    base = runtime.base_config
    authority = frozen_condition_cache_authority(
        config_schema=f"{runtime.config['schema_version']}:endpoints",
        config_bytes=runtime.args.config.stat().st_size,
        source_checkpoint=runtime.args.checkpoint,
        g2_program_checkpoint=authority_path(
            base, "g2_program_checkpoint", asset_root=runtime.args.asset_root
        ),
        native_observer_checkpoint=authority_path(
            base, "native_observer_checkpoint", asset_root=runtime.args.asset_root
        ),
        frame_stride=int(base["data"]["frame_stride"]),
        owners=runtime.owners,
    )
    return FrozenMappingConditionCache(
        root,
        owners=runtime.owners,
        operator=runtime.compiler.bank_operator,
        authority=authority,
        cache_program=False,
    )


def _endpoint_condition(
    runtime: JointProgramPrimalRuntime,
    cache: FrozenMappingConditionCache,
    condition: MappingCondition,
) -> Any:
    def builder():
        sample = NaturalProgramSample(
            video_demos=(condition.video_demo,),
            action_demos=(),
            k=1,
            robustness_view="j2_gate_endpoints_k1",
        )
        packed = pack_shared_compiler_videos(
            task=runtime.task_by_id[condition.authority_id],
            sample=sample,
            video_store=runtime.video_store,
            query_points=runtime.query_points,
            device=runtime.context.device,
            view="endpoints",
        )
        tokens, mask = runtime.language_tokens[condition.authority_id]
        return prepare_shared_compiler_condition(
            policy=runtime.policy,
            program_model=runtime.program,
            owners=runtime.owners,
            packed=packed,
            language_tokens=tokens,
            language_mask=mask,
            chunk_size=int(runtime.base_config["model"]["frame_chunk_size"]),
        )

    return cache.get_or_build(
        authority_id=condition.authority_id,
        video_demo=condition.video_demo,
        device=runtime.context.device,
        builder=builder,
        retain=True,
    ).condition


def _panel_value(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    rows = []
    for visit in range(16):
        batch, panel = functional_panel_batch(
            runtime, task_id=task_id, panel_name="b", visit_index=visit
        )
        loss = _functional_value(
            runtime, state=state, batch=batch, seed=panel.policy_rng_seed
        )
        rows.append(
            {
                "visit": visit,
                "carrier_loss": panel.flow_loss,
                "generated_loss": loss,
                "benefit_over_carrier": panel.flow_loss - loss,
            }
        )
    return {
        "visits": rows,
        "carrier_loss": statistics.fmean(row["carrier_loss"] for row in rows),
        "generated_loss": statistics.fmean(row["generated_loss"] for row in rows),
        "benefit_over_carrier": statistics.fmean(
            row["benefit_over_carrier"] for row in rows
        ),
    }


def _normalized(value: Mapping[str, Any], free_loss: float) -> dict[str, Any]:
    denominator = float(value["carrier_loss"]) - float(free_loss)
    recovery = (
        float(value["benefit_over_carrier"]) / denominator
        if denominator > 0
        else None
    )
    return {
        **dict(value),
        "free_primal_loss": float(free_loss),
        "free_primal_benefit": denominator,
        "functional_recovery": recovery,
    }


def _mean_recovery(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [row.get("functional_recovery") for row in rows]
    if any(value is None for value in values):
        return None
    return statistics.fmean(map(float, values))


def _positive_control_losses(
    root: Path, task_id: int
) -> tuple[dict[int, float], dict[str, Any]]:
    path = root / f"task_{task_id:03d}" / "result.json"
    result = read_json(path)
    evaluation = result.get("evaluation", {})
    rows = (*evaluation.get("fit_videos", ()), evaluation.get("held_video", {}))
    losses = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    if (
        result.get("schema_version") != POSITIVE_CONTROL_SCHEMA
        or result.get("status") != "complete"
        or int(result.get("task", -1)) != task_id
        or len(losses) != 3
        or result.get("panel_b_backward_calls") != 0
        or result.get("action_meta_installed") is not False
        or result.get("single_complete_rank16") is not True
    ):
        raise ValueError("J2 positive-control denominator authority changed")
    return losses, {"path": str(path), "bytes": path.stat().st_size}


def _task_local_code(
    runtime: JointProgramPrimalRuntime,
    task_id: int,
    fit: Sequence[MappingCondition],
) -> TaskLocalPrimalCode:
    names = tuple(sorted(runtime.mapping_split.member_names[task_id]))
    teachers = []
    for condition in fit:
        rows = runtime.native_teachers.lookup_members(
            authority_id=task_id,
            k=1,
            video_demo=condition.video_demo,
            member_names=names,
        )
        if rows is None or tuple(row.member_name for row in rows) != names:
            raise RuntimeError("J2 task-held free-primal authority changed")
        teachers.extend(rows)
    code = TaskLocalPrimalCode(
        runtime.owners, teachers, s_ref=runtime.ranks.s_ref
    ).to(runtime.context.device)
    return code.requires_grad_(False).eval()


def _task_local_state(
    runtime: JointProgramPrimalRuntime, code: TaskLocalPrimalCode, bank: Any
) -> dict[str, torch.Tensor]:
    output = task_local_output(
        operator=runtime.compiler.bank_operator,
        prepared=bank.videos[0],
        code=code,
        s_ref=runtime.ranks.s_ref,
    )
    residual = residual_lora_state(
        output.residual, runtime.rank4_contract, canonicalize=True
    )
    return compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual,
        rank16_contract=runtime.ranks.contract,
    )


def _family_record(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    condition: MappingCondition,
    output: SharedCompilerOutput,
    indices: Sequence[int],
) -> dict[str, Any]:
    names = tuple(sorted(runtime.mapping_split.member_names[task_id]))
    teachers = runtime.native_teachers.lookup_members(
        authority_id=task_id,
        k=1,
        video_demo=condition.video_demo,
        member_names=names,
    )
    if teachers is None:
        raise RuntimeError("J2 family diagnostic teacher authority changed")
    selected = tuple(map(int, indices))
    subset = SharedCompilerOutput(
        residual=NativeFactorResidual(
            a=tuple(output.residual.a[index] for index in selected),
            b=tuple(output.residual.b[index] for index in selected),
            scales=output.residual.scales[list(selected)],
        ),
        input_directions=tuple(output.input_directions[index] for index in selected),
        output_directions=tuple(output.output_directions[index] for index in selected),
        video_weights=output.video_weights,
        frame_measures=output.frame_measures,
        output_group_gains=output.output_group_gains,
        solve_metrics=output.solve_metrics,
        conditioning_metrics=output.conditioning_metrics,
    )
    _, record = recovery_record(
        subset,
        tuple(subset_teacher(row, selected) for row in teachers),
        tuple(runtime.owners[index] for index in selected),
        temperature=float(runtime.base_config["optimization"]["mapping"]["temperature"]),
    )
    return record


def _evaluate_task(
    runtime: JointProgramPrimalRuntime,
    *,
    task_id: int,
    gate: Mapping[str, Any],
    positive_control_root: Path,
    endpoint_cache: FrozenMappingConditionCache,
) -> dict[str, Any]:
    started = time.monotonic()
    gradient = task_id in {
        *map(int, runtime.config["task_split"]["gradient_meta"]),
        *map(int, runtime.config["task_split"]["gradient_target"]),
    }
    first, second, held = _task_conditions(runtime, task_id)
    correct_conditions = (first, second, held)
    free_reference: dict[int, float] = {}
    free_authority: dict[str, Any]
    code: TaskLocalPrimalCode | None = None
    if gradient:
        free_reference, free_authority = _positive_control_losses(
            positive_control_root, task_id
        )
    else:
        code = _task_local_code(runtime, task_id, (first, second))
        free_authority = {
            "method": "frozen_fit_teacher_initialized_code_without_action_gradient",
            "panel_b_backward_calls": 0,
        }

    correct_programs: dict[int, NaturalProgram] = {}
    outputs: dict[int, SharedCompilerOutput] = {}
    correct_rows: dict[int, dict[str, Any]] = {}
    active_events: dict[int, int] = {}
    for condition in correct_conditions:
        prepared, _ = prepare_joint_condition(runtime, condition)
        with torch.inference_mode():
            program = _compile_program(runtime, prepared)
            state, output = _complete_state(runtime, program=program, bank=prepared)
            if code is not None:
                free_state = _task_local_state(runtime, code, prepared)
        if code is not None:
            free_value = _panel_value(runtime, task_id=task_id, state=free_state)
            free_reference[condition.video_demo] = float(free_value["generated_loss"])
            del free_state
        value = _panel_value(runtime, task_id=task_id, state=state)
        correct_rows[condition.video_demo] = _normalized(
            value, free_reference[condition.video_demo]
        )
        correct_programs[condition.video_demo] = program
        outputs[condition.video_demo] = output
        active_events[condition.video_demo] = int((program.rho > 0.2).sum())
        del state, prepared

    primary_program = correct_programs[first.video_demo]
    primary_free = free_reference[first.video_demo]
    primary_correct = correct_rows[first.video_demo]

    prepared, _ = prepare_joint_condition(runtime, first)
    with torch.inference_mode():
        language_state, _ = _complete_state(
            runtime, program=_language_program(primary_program), bank=prepared
        )
    language = _normalized(
        _panel_value(runtime, task_id=task_id, state=language_state), primary_free
    )
    del language_state, prepared

    endpoints_prepared = _endpoint_condition(runtime, endpoint_cache, first)
    with torch.inference_mode():
        endpoints_program = _compile_program(runtime, endpoints_prepared)
        endpoints_state, _ = _complete_state(
            runtime, program=endpoints_program, bank=endpoints_prepared
        )
    endpoints = _normalized(
        _panel_value(runtime, task_id=task_id, state=endpoints_state), primary_free
    )
    del endpoints_state, endpoints_prepared

    wrong_task = _wrong_task(runtime, task_id)
    wrong_condition = _task_conditions(runtime, wrong_task)[0]
    wrong_prepared, _ = prepare_joint_condition(runtime, wrong_condition)
    with torch.inference_mode():
        wrong_program = _compile_program(runtime, wrong_prepared)
        correct_wrong_bank_state, _ = _complete_state(
            runtime, program=primary_program, bank=wrong_prepared
        )
        wrong_wrong_state, _ = _complete_state(
            runtime, program=wrong_program, bank=wrong_prepared
        )
    correct_wrong_bank = _normalized(
        _panel_value(runtime, task_id=task_id, state=correct_wrong_bank_state),
        primary_free,
    )
    wrong_wrong = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_wrong_state),
        primary_free,
    )
    del correct_wrong_bank_state, wrong_wrong_state, wrong_prepared

    primary_prepared, _ = prepare_joint_condition(runtime, first)
    with torch.inference_mode():
        wrong_correct_bank_state, _ = _complete_state(
            runtime, program=wrong_program, bank=primary_prepared
        )
    wrong_correct_bank = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_correct_bank_state),
        primary_free,
    )
    del wrong_correct_bank_state, primary_prepared

    indices = tuple(map(int, gate["evaluation"]["selected_family_report_targets"]))
    family = {
        str(condition.video_demo): _family_record(
            runtime,
            task_id=task_id,
            condition=condition,
            output=outputs[condition.video_demo],
            indices=indices,
        )
        for condition in correct_conditions
    }
    fit_recovery = _mean_recovery(
        [correct_rows[row.video_demo] for row in (first, second)]
    )
    held_value = correct_rows[held.video_demo]["functional_recovery"]
    held_recovery = None if held_value is None else float(held_value)
    task_held_recovery = _mean_recovery(
        [correct_rows[row.video_demo] for row in correct_conditions]
    )
    return {
        "schema_version": J2_EVALUATION_SCHEMA,
        "task": task_id,
        "role": runtime.panels[task_id].role,
        "split": "gradient" if gradient else "true_task_held",
        "fit_videos": [first.video_demo, second.video_demo],
        "held_video": held.video_demo,
        "correct": {str(key): value for key, value in correct_rows.items()},
        "functional_summary": {
            "fit_recovery": fit_recovery,
            "held_video_recovery": held_recovery,
            "true_task_held_recovery": task_held_recovery if not gradient else None,
            "same_task_raw_benefit_retention": (
                float(correct_rows[held.video_demo]["benefit_over_carrier"])
                / max(
                    statistics.fmean(
                        float(correct_rows[row.video_demo]["benefit_over_carrier"])
                        for row in (first, second)
                    ),
                    1e-12,
                )
            ),
        },
        "controls": {
            "primary_correct": primary_correct,
            "language_only": language,
            "endpoints": endpoints,
            "wrong_program_correct_bank": wrong_correct_bank,
            "correct_program_wrong_bank": correct_wrong_bank,
            "wrong_program_wrong_bank": wrong_wrong,
            "wrong_task": wrong_task,
            "wrong_video_demo": wrong_condition.video_demo,
        },
        "family_diagnostic": family,
        "active_events": active_events,
        "free_primal_authority": free_authority,
        "information_wall": {
            "deployment_native_teacher_tensor_reads": 0,
            "panel_b_backward_calls": 0,
            "same_task_held_backward_calls": 0,
            "true_task_held_backward_calls": 0,
            "action_meta_installed": False,
            "single_complete_rank16": True,
            "K1_identity": True,
            "shuffled_or_reversed_use": False,
        },
        "task_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
    }


def evaluate_worker(args: argparse.Namespace) -> None:
    state = git_state(Path(__file__).resolve().parents[4])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal J2 evaluation requires clean detached authority")
    gate = load_joint_program_primal_gate(args.gate_config)
    if args.config != (args.asset_root / gate["training_config"]).resolve():
        raise ValueError("J2 evaluator training config authority changed")
    config = load_joint_program_primal_config(args.config)
    positive_root = (
        args.asset_root / gate["authorities"]["positive_control_root"]
    ).resolve()
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("J2 evaluator worker index changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("J2 evaluation workers are independent single-GPU processes")
    runtime_args = argparse.Namespace(
        config=args.config,
        base_config=args.base_config,
        mode="profile",
        phase="joint",
        task=None,
        asset_root=args.asset_root,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_root=args.data_root,
        output_dir=args.output_dir / f"worker_{args.worker_index:02d}_runtime",
        condition_cache_root=args.condition_cache_root,
        resume=None,
        stop_after_step=1,
        log_every=1,
    )
    runtime: JointProgramPrimalRuntime | None = None
    started = time.monotonic()
    try:
        runtime = prepare_joint_program_primal_runtime(runtime_args, context)
        checkpoint = _checkpoint_authority(
            runtime,
            compiler_run=args.compiler_run,
            compiler_checkpoint=args.compiler_checkpoint,
            gate=gate,
        )
        assignments = balanced_task_assignments(runtime, args.worker_count)
        endpoint_cache = _endpoint_cache(runtime, args.endpoint_cache_root)
        worker_dir = args.output_dir / f"worker_{args.worker_index:02d}"
        if worker_dir.exists():
            raise ValueError("J2 evaluator worker output already exists")
        worker_dir.mkdir(parents=True)
        rows = []
        for task_id in assignments[args.worker_index]:
            rows.append(
                _evaluate_task(
                    runtime,
                    task_id=task_id,
                    gate=gate,
                    positive_control_root=positive_root,
                    endpoint_cache=endpoint_cache,
                )
            )
            runtime.panel_batch_cache.clear()
            torch.cuda.empty_cache()
        payload = {
            "schema_version": J2_EVALUATION_SCHEMA,
            "status": "complete",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "assignments": [list(row) for row in assignments],
            "checkpoint": checkpoint,
            "tasks": rows,
            "elapsed_seconds": time.monotonic() - started,
            "physical_visible_device": __import__("os").environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "git": {"commit": state["commit"], "branch": state["branch"]},
        }
        write_json_atomic(worker_dir / "result.json", payload)
        write_json_atomic(
            worker_dir / "completion.json",
            {
                "schema_version": J2_EVALUATION_SCHEMA,
                "worker_index": args.worker_index,
                "task_count": len(rows),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
