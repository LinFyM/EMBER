"""Runtime, exact-resume, and evaluation bridge for shared EBSRI S2."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import checkpoint_macro, load_ecp_checkpoint
from ember.ecp.joint_program_primal.bank_set_shared_training import (
    ARM_SCHEDULE,
    GRADIENT_TASKS,
    SharedArmSpec,
    SharedTaskTargets,
    _apply_task_profile,
    _arm_spec,
    _asset_config,
    _clear_panel_cache,
    _contract_module,
    _device_target,
    _prepare_arm,
    _shared_interaction_output,
    _validate_shared_config,
    _validate_task_cursors,
)
from ember.ecp.joint_program_primal.bank_set_tasklocal import (
    _complete,
    _panel_b,
)
from ember.ecp.joint_program_primal.bank_set_tasklocal_evaluation import (
    CorrectionCollector,
    effective_rank4_diagnostics,
)
from ember.ecp.joint_program_primal.runtime import (
    REPO_ROOT,
    JointProgramPrimalRuntime,
    _authority_assets,
    _data_assets,
    _inventory,
    _model_assets,
    _optimizer,
    _scheduler,
    _topology,
)
from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group


def _run_contract(runtime: Any, contract: Any) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    inventory = _inventory(
        runtime.policy, runtime.program, runtime.compiler, runtime.owners
    )
    inventory.update(contract.writer_trainable_inventory(runtime.writer_state))
    return {
        "schema_version": contract.BANK_SET_SHARED_RUN_SCHEMA,
        "stage": contract.BANK_SET_SHARED_STAGE,
        "phase": "shared_loto",
        "mode": runtime.args.mode,
        "git": {
            "branch": state["branch"],
            "commit": state["commit"],
            "authority_commit": state["commit"]
            if runtime.args.mode == "formal"
            else state["authority_commit"],
        },
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "base_g3_config": {
            "path": str(runtime.args.base_config),
            "bytes": runtime.args.base_config.stat().st_size,
        },
        "source_checkpoint": str(runtime.args.checkpoint),
        "tokenizer": str(runtime.args.tokenizer_path),
        "data_root": str(runtime.args.data_root),
        "condition_cache": {
            "root": str(runtime.args.condition_cache_root),
            "program_bank_root": str(runtime.args.program_bank_condition_cache_root),
            "resident_real_banks_per_rank": 1,
            "training_target_cache_builds": 0,
            "evaluation_target_cache": "lazy_per_task_cpu_diagnostics_gate_only",
        },
        "task_split": dict(runtime.config["task_split"]),
        "shared_training": dict(runtime.config["shared_training"]),
        "task_profiles": dict(runtime.config["shared_training"]["task_profiles"]),
        "schedule": {
            "global_tasks_per_optimizer_step": 6,
            "tasks_per_role": 3,
            "role_task_pool": 4,
            "task_weight": "one_sixth_for_each_active_task",
            "role_weight": "one_half_each",
            "arm_cursor": "independent_per_task_appearance_count",
            "cursor_reconstruction": "deterministic_from_global_optimizer_step",
            "arm_schedule": list(ARM_SCHEDULE),
        },
        "positive_control_denominators": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in runtime.positive_control_files
        ],
        "primal_scorer_initialization": dict(runtime.primal_scorer_initialization),
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]),
        "throughput_gate": dict(runtime.config.get("throughput_gate", {})),
        "information_wall": dict(runtime.config["information_wall"]),
        "inventory": inventory,
        "world_topology": _topology(runtime.context),
    }


def _optimizer_cursor(
    args: Any, config: Mapping[str, Any], trainable: tuple[Any, ...]
) -> tuple[Any, Any, tuple[int, ...], int]:
    optimizer = _optimizer(trainable, config)
    scheduler = _scheduler(optimizer, config)
    joint = config["optimization"]["joint"]
    warmup = int(joint["warmup_optimizer_steps"])
    checkpoints = tuple(
        warmup + int(value) for value in joint["checkpoint_effective_steps"]
    )
    stop = int(
        getattr(args, "stop_after_step", None)
        or (
            1
            if args.mode == "profile"
            else warmup + int(joint["effective_optimizer_steps"])
        )
    )
    if stop not in ({1, 2} if args.mode == "profile" else set(checkpoints)):
        raise ValueError("S2 stop step is not pre-registered")
    return optimizer, scheduler, checkpoints, stop


def _build_runtime(
    args: Any, context: Any, *, training: bool
) -> JointProgramPrimalRuntime:
    contract = _contract_module()
    config = contract.load_bank_set_shared_config(args.config)
    _validate_shared_config(config)
    base_path = (args.asset_root / config["authorities"]["base_g3_config"]).resolve()
    if args.base_config != base_path:
        raise ValueError("S2 base G3 config authority changed")
    base = load_shared_compiler_config(base_path)
    assets_config = _asset_config(config)
    authority = _authority_assets(args, context, assets_config, base)
    model = _model_assets(args, context, assets_config, base, authority)
    data = _data_assets(args, assets_config, base, context, authority, model)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    writer, trainable, frozen = contract.bank_set_shared_parameter_ownership(
        model.program, model.compiler
    )
    if context.world_size > 1:
        for value in writer.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer, scheduler, checkpoints, stop = _optimizer_cursor(
        args, config, trainable
    )
    runtime = JointProgramPrimalRuntime(
        args=args,
        config=config,
        base_config=base,
        context=context,
        tasks=authority.selected_tasks,
        task_by_id=authority.task_by_id,
        mapping_split=authority.mapping_split,
        task_conditions=authority.task_conditions,
        panels=authority.panels,
        video_store=data.video_store,
        query_dataset=data.query_dataset,
        query_processor=data.query_processor,
        panel_batch_cache={},
        counterfactual_margin_scales=authority.counterfactual_margin_scales,
        positive_control_files=authority.positive_control_files,
        language_tokens=data.language_tokens,
        policy=model.policy,
        program=model.program,
        compiler=model.compiler,
        writer_state=writer,
        owners=model.owners,
        ranks=model.ranks,
        rank4_contract=model.rank4_contract,
        native_teachers=model.native_teachers,
        consensus_teachers=model.consensus_teachers,
        condition_cache=data.condition_cache,
        query_points=data.query_points,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_presence=None,
        optimizer_steps=0,
        stop_after_step=stop,
        checkpoint_steps=checkpoints,
        metrics_rows=0,
        primal_scorer_initialization=dict(model.primal_scorer_initialization),
        functional_code_targets={},
        functional_code_authority={},
        run_contract={},
    )
    runtime.run_contract = _run_contract(runtime, contract)
    if training:
        _seal_or_validate_run_contract(runtime)
        _restore_training_checkpoint(runtime, contract)
    if runtime.context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.context.device)
    return runtime


def _seal_or_validate_run_contract(runtime: Any) -> None:
    error = None
    if runtime.context.is_main:
        try:
            path = runtime.args.output_dir / "run_contract.json"
            resume = getattr(runtime.args, "resume", None)
            if resume is None:
                if runtime.args.output_dir.exists() and any(
                    runtime.args.output_dir.iterdir()
                ):
                    raise ValueError("fresh S2 output root is not empty")
                runtime.args.output_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, runtime.run_contract)
            elif (
                resume.resolve().parent.parent != runtime.args.output_dir.resolve()
                or not path.is_file()
                or read_json(path) != runtime.run_contract
            ):
                raise ValueError("S2 exact-resume run contract changed")
        except Exception as exc:  # keep all ranks on one filesystem path
            error = f"{type(exc).__name__}: {exc}"
    if runtime.context.world_size > 1:
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise ValueError(error)
    if runtime.context.world_size > 1:
        dist.barrier()


def _restore_training_checkpoint(runtime: Any, contract: Any) -> None:
    resume = getattr(runtime.args, "resume", None)
    if resume is not None:
        macro, expected_rows = load_ecp_checkpoint(
            checkpoint=resume,
            stage=contract.BANK_SET_SHARED_STAGE,
            context=runtime.context,
            model=runtime.writer_state,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            run_contract_schema=contract.BANK_SET_SHARED_RUN_SCHEMA,
        )
        if macro >= runtime.stop_after_step:
            raise ValueError("S2 resume checkpoint is not before the stop step")
        rows, error = 0, None
        if runtime.context.is_main:
            try:
                rows = reconcile_metrics(
                    runtime.args.output_dir / "metrics.jsonl",
                    macro,
                    expected_rows,
                    cursor_key="optimizer_step",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if runtime.context.world_size > 1:
            payload = [rows, error]
            dist.broadcast_object_list(payload, src=0)
            rows, error = payload
        if error is not None:
            raise ValueError(error)
        runtime.optimizer_steps = macro
        runtime.metrics_rows = int(rows)
    _validate_task_cursors(runtime)


def prepare_shared_training_runtime(
    args: Any, context: Any
) -> JointProgramPrimalRuntime:
    return _build_runtime(args, context, training=True)


def prepare_shared_evaluation_runtime(
    args: Any, context: Any
) -> JointProgramPrimalRuntime:
    return _build_runtime(args, context, training=False)


def load_shared_checkpoint(runtime: Any, checkpoint: Path) -> Mapping[str, Any]:
    contract = _contract_module()
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    macro = checkpoint_macro(checkpoint)
    if (
        manifest.get("stage") != contract.BANK_SET_SHARED_STAGE
        or int(manifest.get("next_macro", -1)) != macro
        or manifest.get("run_contract_schema") != contract.BANK_SET_SHARED_RUN_SCHEMA
    ):
        raise ValueError("S2 evaluation checkpoint authority changed")
    runtime.writer_state.load_state_dict(
        load_file(
            str(checkpoint / "ecp.safetensors"),
            device=str(runtime.context.device),
        ),
        strict=True,
    )
    runtime.optimizer_steps = macro
    return {
        "optimizer_step": macro,
        "task_arm_cursors": {
            str(task): value for task, value in _validate_task_cursors(runtime).items()
        },
        "training_world_size": int(manifest["world_size"]),
    }


def _job_arm_spec(runtime: Any, job: Mapping[str, Any]) -> SharedArmSpec:
    task = int(job["task"])
    spec = _arm_spec(runtime, task, str(job["arm"]))
    if not all(
        (
            int(job["program_task"]) == task,
            int(job["bank_task"]) == spec.bank_task,
            int(job["video_demo"]) == int(spec.condition.video_demo),
            job.get("receives_gradient") is False,
        )
    ):
        raise ValueError("S2 evaluation job escaped its sealed arm")
    return SharedArmSpec(**{**spec.__dict__, "receives_gradient": False})


def _free_loss_for_job(runtime: Any, spec: SharedArmSpec) -> float:
    root = (
        runtime.args.asset_root
        / runtime.config["authorities"]["positive_control_root"]
        / f"task_{spec.task:03d}"
        / "result.json"
    ).resolve()
    source = read_json(root)
    rows = (*source["evaluation"]["fit_videos"], source["evaluation"]["held_video"])
    values = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    if spec.name.startswith("wrong"):
        demos = runtime.task_conditions[spec.task].fit_views
        return statistics.fmean(values[int(condition.video_demo)] for condition in demos)
    return values[int(spec.condition.video_demo)]


def evaluate_shared_job(
    runtime: Any,
    job: Mapping[str, Any],
    *,
    target_cache: Mapping[int, SharedTaskTargets],
) -> Mapping[str, Any]:
    if getattr(runtime, "_shared_live_job", None) is not None:
        raise RuntimeError("S2 evaluator already has a resident real bank")
    spec = _job_arm_spec(runtime, job)
    profile = _apply_task_profile(runtime, spec.task)
    if spec.task not in target_cache:
        raise ValueError("S2 evaluator target cache missed its task")
    cached = target_cache[spec.task]
    target_name = spec.name if spec.name in cached.targets else "wrong_fit0"
    target = _device_target(cached.targets[target_name], runtime.context.device)
    denominators = {
        family: value.to(runtime.context.device)
        for family, value in cached.denominators.items()
    }
    if runtime.context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.context.device)
    arm = _prepare_arm(runtime, spec)
    collector = CorrectionCollector(
        float(runtime.config["model"]["interaction_correction_bound"])
    )
    with torch.inference_mode():
        output = _shared_interaction_output(
            runtime, arm, correction_observer=collector.observe
        )
        state = _complete(runtime, output)
        effective = effective_rank4_diagnostics(runtime, output, target, denominators)
        panel = _panel_b(
            runtime,
            task=spec.task,
            state=state,
            free_loss=_free_loss_for_job(runtime, spec),
            visits=int(runtime.config["data"]["panel_visits"]),
        )
    runtime._shared_live_job = {"task": spec.task, "arm": spec.name, "bank": arm}
    return {
        "task": spec.task,
        "role": spec.role,
        "split": str(job["split"]),
        "arm": spec.name,
        "program_task": spec.task,
        "bank_task": spec.bank_task,
        "video_demo": int(spec.condition.video_demo),
        "checkpoint_optimizer_step": int(job["checkpoint_optimizer_step"]),
        "panel_b": {
            "rows": panel["visits"],
            "carrier_loss": panel["carrier_loss"],
            "generated_loss": panel["generated_loss"],
            "free_primal_loss": panel["free_primal_loss"],
            "free_primal_benefit": panel["free_primal_benefit"],
            "functional_recovery": panel["functional_recovery"],
        },
        "functional_recovery": panel["functional_recovery"],
        "free_primal_benefit": panel["free_primal_benefit"],
        "family_recovery": {
            family: 1.0 - float(row["normalized_squared_error"])
            for family, row in effective["families"].items()
        },
        "effective_rank4": effective,
        "correction": collector.finalize(),
        "target_authority": {
            "effective_target": cached.authority[
                "wrong" if spec.name.startswith("wrong") else "correct"
            ],
            "family_denominator": cached.authority["denominator"],
            "cached_on_cpu": True,
            "real_bank_cached": False,
            "source": dict(cached.authority),
        },
        "condition_metrics": dict(arm.bank.condition_metrics),
        "information_wall": {
            "receives_gradient": False,
            "panel_b_backward_calls": 0,
            "held_interaction_backward_calls": 0,
            "result_or_action_gradient_calls": 0,
            "forbidden_task_reads": 0,
            "action_meta_installed": False,
            "shuffled_or_reversed_use": False,
            "single_complete_rank16": True,
            "adapter_rank": 16,
            "adapter_target_count": 38,
            "adapter_tensor_count": 76,
        },
        "resource": {
            "resident_real_bank_count": 1,
            "sampled_frames": int(spec.condition.sampled_frames),
            "task_profile": profile,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if runtime.context.device.type == "cuda"
            else 0,
        },
    }


def release_shared_job(runtime: Any, job: Mapping[str, Any]) -> Mapping[str, int]:
    live = getattr(runtime, "_shared_live_job", None)
    if (
        live is None
        or int(live["task"]) != int(job["task"])
        or live["arm"] != job["arm"]
    ):
        raise RuntimeError("S2 evaluator release did not match its resident bank")
    runtime._shared_live_job = None
    _clear_panel_cache(runtime, int(job["task"]))
    return {"resident_real_bank_count_after_release": 0}
