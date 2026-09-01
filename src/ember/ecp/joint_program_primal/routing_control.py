"""Training-only orthogonal routing-token boundary control for G3.

This module is deliberately not a deployment Writer.  It replaces Natural
Program content with fixed, non-trainable task routing tokens so the existing
ProgramNativePrimalScorer can be tested independently of Program geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import checkpoint_macro, load_ecp_checkpoint
from ember.ecp.joint_program_primal.bank_set_tasklocal_contract import (
    BANK_SET_S0_STAGE,
    BANK_SET_S1_STAGE,
    BANK_SET_TASKLOCAL_RUN_SCHEMA,
    bank_set_config_valid,
    bank_set_parameter_ownership,
    is_bank_set_tasklocal_config,
    required_s0_gate_authority,
    writer_trainable_inventory,
)
from ember.ecp.joint_program_primal.runtime import (
    REPO_ROOT,
    SCORER_INTERACTION_ONLY,
    JointProgramPrimalRuntime,
    _authority_assets,
    _data_assets,
    _inventory,
    _model_assets,
    _optimizer,
    _scheduler,
    _topology,
)
from ember.ecp.joint_program_primal.routing_initialization import (
    FUNCTIONAL_CODE_INITIALIZATION,
    initialize_functional_code_heads,
)
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group


ROUTING_CONTROL_SCHEMA = "ember_ecp_routing_token_control_r1_v1"
ROUTING_CONTROL_RUN_SCHEMA = "ember_ecp_routing_token_control_run_v2"
ROUTING_CONTROL_STAGE = "g3_training_only_routing_token_grouped_decoder_control"
PROGRAM_BANK_INTERACTION_SCHEMA = "ember_ecp_program_bank_candidate_interaction_v4"
PROGRAM_BANK_INTERACTION_RUN_SCHEMA = (
    "ember_ecp_program_bank_candidate_interaction_run_v4"
)
PROGRAM_BANK_INTERACTION_STAGE = (
    "g3_program_bank_candidate_interaction_base_score_qualification"
)
INTERACTION_BASE_SCORE_FEATURE = (
    "detached_q0_dot_value_minus_global_b0_mean_div_replay_score_rms"
)
ROUTING_TASK_IDS = (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
ROUTING_WIDTH = 128
OUTPUT_PRIMAL_DECODER = "owner_group_specific_linear_heads"
SCORER_ALL_PARAMETERS = "all"
SCORER_NATIVE_HEADS_ONLY = "native_heads_only"


class RoutingControlWriterState(torch.nn.Module):
    """Checkpoint only the scorer exercised by this boundary control."""

    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self.primal_scorer = scorer


def is_program_bank_interaction_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == PROGRAM_BANK_INTERACTION_SCHEMA


def routing_run_schema(config: Mapping[str, Any]) -> str:
    if is_bank_set_tasklocal_config(config):
        return BANK_SET_TASKLOCAL_RUN_SCHEMA
    return ROUTING_CONTROL_RUN_SCHEMA


def routing_stage(config: Mapping[str, Any]) -> str:
    if is_bank_set_tasklocal_config(config):
        return str(config["stage"])
    return ROUTING_CONTROL_STAGE


def _hadamard(order: int) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("routing-token Hadamard order must be a power of two")
    value = torch.ones(1, 1, dtype=torch.float32)
    while value.shape[0] < order:
        value = torch.cat(
            (
                torch.cat((value, value), dim=1),
                torch.cat((value, -value), dim=1),
            ),
            dim=0,
        )
    return value


_ROUTING_TOKEN_TABLE = _hadamard(ROUTING_WIDTH)[1 : len(ROUTING_TASK_IDS) + 1]


def fixed_routing_token(
    task_id: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return one mean-zero, unit-RMS orthogonal token for a gradient task."""

    try:
        index = ROUTING_TASK_IDS.index(int(task_id))
    except ValueError as error:
        raise ValueError(f"task {task_id} has no routing-control token") from error
    return _ROUTING_TOKEN_TABLE[index].to(device=device)


def fixed_routing_program(
    runtime: JointProgramPrimalRuntime, task_id: int
) -> NaturalProgram:
    """Construct the fixed Program-schema carrier for one training-only token."""

    targets = len(runtime.owners)
    events = int(runtime.compiler.event_slots)
    width = int(runtime.compiler.program_width)
    if width != ROUTING_WIDTH:
        raise ValueError("routing-control Program width changed")
    token = fixed_routing_token(task_id, device=runtime.context.device)
    owner = token[None].expand(targets, -1)
    event_owner = token[None, None].expand(events, targets, -1)
    boundaries = torch.linspace(
        0.0, 1.0, events + 1, device=token.device, dtype=token.dtype
    )
    return NaturalProgram(
        p_lang=owner,
        p_scene=owner,
        p_process=event_owner,
        rho=torch.full(
            (events,), 1.0 / events, device=token.device, dtype=token.dtype
        ),
        tau=torch.stack((boundaries[:-1], boundaries[1:]), dim=-1),
        sigma=event_owner,
    )


def load_routing_control_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    if is_program_bank_interaction_config(config):
        raise ValueError(
            "retired pointwise Program-bank interaction config is not executable"
        )
    split = config.get("task_split", {})
    data = config.get("data", {})
    joint = config.get("optimization", {}).get("joint", {})
    model = config.get("model", {})
    wall = config.get("information_wall", {})
    critic = config.get("optimization", {}).get("privileged_critic")
    initialization = model.get("primal_scorer_initialization")
    scorer_partition = model.get(
        "primal_scorer_trainable_partition", SCORER_ALL_PARAMETERS
    )
    tasks = tuple(
        map(int, (*split.get("gradient_meta", ()), *split.get("gradient_target", ())))
    )
    all_tasks = tuple(
        map(
            int,
            (
                *tasks,
                *split.get("true_task_held_meta", ()),
                *split.get("true_task_held_target", ()),
            ),
        )
    )
    common_valid = all(
        (
            tasks == ROUTING_TASK_IDS,
            len(all_tasks) == len(set(all_tasks)) == 12,
            split.get("true_task_held_meta") == [2],
            split.get("true_task_held_target") == [74],
            set(map(int, config.get("authorities", {}).get("functional_panel_records", {})))
            == set(all_tasks),
            data.get("K") == 1,
            data.get("fit_video_views_per_task") == 2,
            data.get("panel_visits") == 16,
            data.get("rows_per_visit") == 16,
            joint.get("warmup_optimizer_steps") == 10,
            joint.get("effective_optimizer_steps") == 100,
            joint.get("checkpoint_effective_steps") == [60, 100],
            joint.get("global_tasks_per_optimizer_step")
            == (1 if is_bank_set_tasklocal_config(config) else 6),
            joint.get("video_views_per_task")
            == (3 if is_bank_set_tasklocal_config(config) else 2),
            wall.get("action_meta_installed") is False,
            wall.get("shuffled_or_reversed_use") is False,
        )
    )
    routing_valid = all(
        (
            config.get("schema_version") == ROUTING_CONTROL_SCHEMA,
            config.get("status") == "training_only_routing_boundary_control",
            model.get("program_source")
            == "fixed_nontrainable_128d_orthogonal_task_token",
            model.get("output_primal_decoder") == OUTPUT_PRIMAL_DECODER,
            (
                scorer_partition == SCORER_ALL_PARAMETERS
                and model.get("trainable") == ["ProgramNativePrimalScorer"]
            )
            or (
                scorer_partition == SCORER_NATIVE_HEADS_ONLY
                and initialization == FUNCTIONAL_CODE_INITIALIZATION
                and wall.get("primal_scorer_feature_chart_frozen") is True
                and model.get("trainable")
                == [
                    "ProgramNativePrimalScorer.input_primal_heads",
                    "ProgramNativePrimalScorer.output_primal_heads",
                ]
            ),
            model.get("deployment_candidate") is False,
            wall.get("fixed_routing_token_training_only") is True,
            (
                initialization == "fresh"
                or all(
                    (
                        initialization == FUNCTIONAL_CODE_INITIALIZATION,
                        critic is None,
                        model.get("training_signal")
                        == (
                            "correct_functional_after_fit_only_functional_"
                            "positive_control_minimum_norm_head_initialization"
                        ),
                        config.get("optimization", {}).get("loss")
                        == "two_correct_fit_video_functional_only",
                        wall.get(
                            "functional_positive_control_initialization_training_only"
                        )
                        is True,
                    )
                )
            ),
            critic is None
            or all(
                (
                    critic.get("kind")
                    == "fit_only_set_valued_paired_update_direction",
                    float(critic.get("weight", 0.0)) > 0.0,
                    float(critic.get("temperature", 0.0)) > 0.0,
                    critic.get("deployment_input") is False,
                    critic.get("held_or_validation_reads") is False,
                )
            ),
        )
    )
    bank_set_valid = bank_set_config_valid(config)
    if not common_valid or not (routing_valid or bank_set_valid):
        raise ValueError("unsupported G3 routing-token boundary-control config")
    return config


def _scorer_parameter_ownership(
    program: torch.nn.Module,
    compiler: torch.nn.Module,
    *,
    partition: str,
    stage: str = ROUTING_CONTROL_STAGE,
) -> tuple[torch.nn.Module, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    program.requires_grad_(False).eval()
    compiler.requires_grad_(False).eval()
    scorer = compiler.primal_scorer
    if partition == SCORER_ALL_PARAMETERS:
        scorer.requires_grad_(True).train()
    elif partition == SCORER_NATIVE_HEADS_ONLY:
        scorer.input_primal_heads.requires_grad_(True).train()
        scorer.output_primal_heads.requires_grad_(True).train()
    elif partition == SCORER_INTERACTION_ONLY:
        return bank_set_parameter_ownership(program, compiler, stage=stage)
    else:
        raise ValueError("unsupported routing-control scorer partition")
    writer = RoutingControlWriterState(compiler.primal_scorer)
    trainable = tuple(
        parameter for parameter in writer.parameters() if parameter.requires_grad
    )
    frozen = tuple(
        parameter
        for root in (program, compiler)
        for parameter in root.parameters()
        if not parameter.requires_grad
    )
    if not trainable or len(set(map(id, trainable))) != len(trainable):
        raise ValueError("routing-control scorer ownership changed")
    return writer, trainable, frozen


def _optimizer_cursor(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: Any,
    writer_state: torch.nn.Module,
    trainable: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.optim.Optimizer, Any, tuple[int, ...], int, int, int]:
    optimizer = _optimizer(trainable, config)
    scheduler = _scheduler(optimizer, config)
    joint = config["optimization"]["joint"]
    warmup = int(joint["warmup_optimizer_steps"])
    checkpoints = tuple(
        warmup + int(value) for value in joint["checkpoint_effective_steps"]
    )
    stop = int(
        args.stop_after_step
        or (
            1
            if args.mode == "profile"
            else warmup + int(joint["effective_optimizer_steps"])
        )
    )
    if stop not in ({1} if args.mode == "profile" else set(checkpoints)):
        raise ValueError("routing-control stop step is not pre-registered")
    optimizer_steps = 0
    metrics_rows = 0
    return optimizer, scheduler, checkpoints, stop, optimizer_steps, metrics_rows


def _run_contract(
    runtime: JointProgramPrimalRuntime,
    *,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    inventory = _inventory(
        runtime.policy, runtime.program, runtime.compiler, runtime.owners
    )
    inventory.update(writer_trainable_inventory(runtime.writer_state))
    task_local_qualification = None
    if is_bank_set_tasklocal_config(runtime.config):
        task = int(runtime.args.task)
        wrong_task = int(
            runtime.config["task_local"]["wrong_task_by_task"][str(task)]
        )
        correct = runtime.task_conditions[task]
        wrong = runtime.task_conditions[wrong_task]
        task_local_qualification = {
            "program_language_authority": task,
            "wrong_bank_task": wrong_task,
            "arms": [
                {
                    "name": name,
                    "bank_task": bank_task,
                    "video_demo": int(condition.video_demo),
                    "receives_gradient": receives_gradient,
                    "summary_token": (
                        ("free_correct" if name.startswith("correct") else "free_wrong")
                        if routing_stage(runtime.config) == BANK_SET_S0_STAGE
                        else "real_bank_set_summary"
                    ),
                }
                for name, bank_task, condition, receives_gradient in (
                    ("correct_fit0", task, correct.fit_views[0], True),
                    ("correct_fit1", task, correct.fit_views[1], True),
                    ("correct_held", task, correct.held_video, False),
                    ("wrong_fit0", wrong_task, wrong.fit_views[0], True),
                    ("wrong_fit1", wrong_task, wrong.fit_views[1], False),
                )
            ],
            "panel_b_receives_gradient": False,
        }
    return {
        "schema_version": routing_run_schema(runtime.config),
        "stage": routing_stage(runtime.config),
        "phase": runtime.args.phase,
        "mode": runtime.args.mode,
        "git": {
            "branch": state["branch"],
            "commit": state["commit"],
            "authority_commit": (
                state["commit"]
                if runtime.args.mode == "formal"
                else state["authority_commit"]
            ),
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
            "program_bank_root": (
                str(runtime.args.program_bank_condition_cache_root)
                if is_bank_set_tasklocal_config(runtime.config)
                else None
            ),
            "program_output_cached": False,
            "checkpoint_payload": False,
        },
        "task_split": dict(runtime.config["task_split"]),
        "task_local_qualification": task_local_qualification,
        "required_s0_gate": required_s0_gate_authority(
            runtime.config, asset_root=runtime.args.asset_root
        ),
        "functional_panels": {
            str(task): {"path": str(panel.path), "bytes": panel.path.stat().st_size}
            for task, panel in runtime.panels.items()
        },
        "positive_control_denominators": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in runtime.positive_control_files
        ],
        "diagnostic": {
            "deployment_candidate": False,
            "fixed_routing_token_training_only": True,
            "token_width": ROUTING_WIDTH,
            "token_construction": "mean_zero_unit_rms_sylvester_hadamard_rows",
            "task_to_row": {
                str(task): index + 1
                for index, task in enumerate(ROUTING_TASK_IDS)
            },
            "removal_trigger": (
                "retire executable control after Gate interpretation and before "
                "the next canonical deployment architecture"
            ),
            "event_bank_set_qualification": is_bank_set_tasklocal_config(
                runtime.config
            ),
            "task": runtime.args.task,
        },
        "primal_scorer_initialization": dict(initialization),
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]),
        "throughput_gate": dict(runtime.config["throughput_gate"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "inventory": inventory,
        "world_topology": _topology(runtime.context),
    }


def _resolve_routing_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_routing_control_config(args.config)
    required_s0_gate_authority(config, asset_root=args.asset_root)
    if is_bank_set_tasklocal_config(config):
        config["optimization"]["functional_policy_microbatch_size"] = int(
            config["optimization"]["functional_policy_microbatch_size_by_task"][
                str(args.task)
            ]
        )
        config["model"]["replay_frame_chunk_size"] = int(
            config["model"]["replay_frame_chunk_size_by_task"][str(args.task)]
        )
        config["model"]["interaction_group_batch_size"] = int(
            config["model"]["interaction_group_batch_size_by_task"][str(args.task)]
        )
    program_bank_root = getattr(args, "program_bank_condition_cache_root", None)
    if is_bank_set_tasklocal_config(config):
        if (
            program_bank_root is None
            or not program_bank_root.is_absolute()
            or program_bank_root == args.condition_cache_root
        ):
            raise ValueError("interaction cross-language cache root changed")
    elif program_bank_root is not None:
        raise ValueError("cross-language cache is only valid for interaction")
    base_path = (args.asset_root / config["authorities"]["base_g3_config"]).resolve()
    if args.base_config != base_path:
        raise ValueError("routing-control base G3 config authority changed")
    return config, load_shared_compiler_config(base_path)


def _writer_initialization(
    args: argparse.Namespace,
    context: Any,
    config: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    initialization_kind = config["model"]["primal_scorer_initialization"]
    if is_bank_set_tasklocal_config(config):
        return dict(model.primal_scorer_initialization)
    if args.resume is not None:
        return {
            "kind": initialization_kind,
            "state": "restored_from_checkpoint",
        }
    if bool(getattr(args, "skip_routing_initialization", False)):
        return {
            "kind": initialization_kind,
            "state": "skipped_before_immediate_checkpoint_load",
        }
    if initialization_kind == FUNCTIONAL_CODE_INITIALIZATION:
        initialization_view = SimpleNamespace(
            owners=model.owners,
            compiler=model.compiler,
            context=context,
        )
        initialization = (
            initialize_functional_code_heads(
                config=config,
                asset_root=args.asset_root,
                compiler=model.compiler,
                owners=model.owners,
                task_ids=ROUTING_TASK_IDS,
                program_for_task=lambda task: fixed_routing_program(
                    initialization_view, task
                ),
            )
            if context.is_main
            else {}
        )
        if context.world_size > 1:
            rows: list[Any] = [initialization]
            dist.broadcast_object_list(rows, src=0)
            initialization = dict(rows[0])
        return initialization
    return {"kind": "fresh", "state": "seeded_random"}


def _new_routing_runtime(
    *,
    args: argparse.Namespace,
    context: Any,
    config: dict[str, Any],
    base: dict[str, Any],
    authority: Any,
    model: Any,
    data: Any,
    writer_state: torch.nn.Module,
    trainable: tuple[torch.nn.Parameter, ...],
    frozen: tuple[torch.nn.Parameter, ...],
    initialization: Mapping[str, Any],
) -> JointProgramPrimalRuntime:
    if context.world_size > 1:
        for value in writer_state.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer, scheduler, checkpoints, stop, optimizer_steps, metrics_rows = (
        _optimizer_cursor(args, config, context, writer_state, trainable)
    )
    return JointProgramPrimalRuntime(
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
        writer_state=writer_state,
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
        optimizer_steps=optimizer_steps,
        stop_after_step=stop,
        checkpoint_steps=checkpoints,
        metrics_rows=metrics_rows,
        primal_scorer_initialization=dict(initialization),
        functional_code_targets={},
        functional_code_authority={},
        run_contract={},
    )


def _formal_topology_valid(runtime: JointProgramPrimalRuntime) -> bool:
    topology = runtime.run_contract["world_topology"]
    return all(
        (
            1 <= len(topology) <= 6,
            len({str(row.get("hostname", "")) for row in topology}) == 1,
            sorted(int(row.get("rank", -1)) for row in topology)
            == list(range(len(topology))),
            sorted(int(row.get("local_rank", -1)) for row in topology)
            == list(range(len(topology))),
            len({str(row.get("device", "")) for row in topology}) == len(topology),
        )
    )


def _write_bank_set_contract(runtime: JointProgramPrimalRuntime, path: Path) -> str | None:
    args = runtime.args
    try:
        if args.resume is not None:
            valid = all(
                (
                    args.mode == "formal",
                    args.resume.resolve().parent.parent == args.output_dir.resolve(),
                    checkpoint_macro(args.resume) == 70,
                    runtime.stop_after_step == 110,
                    path.is_file(),
                    path.is_file() and read_json(path) == runtime.run_contract,
                )
            )
            if not valid:
                raise ValueError("interaction resume authority changed")
        else:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("fresh interaction output root is not empty")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, runtime.run_contract)
    except Exception as error:  # propagate rank-zero filesystem failures
        return f"{type(error).__name__}: {error}"
    return None


def _seal_routing_contract(
    runtime: JointProgramPrimalRuntime, *, initialization: Mapping[str, Any]
) -> None:
    args = runtime.args
    context = runtime.context
    runtime.run_contract = _run_contract(runtime, initialization=initialization)
    contract_path = args.output_dir / "run_contract.json"
    interaction = is_bank_set_tasklocal_config(runtime.config)
    if interaction and args.mode == "formal" and not _formal_topology_valid(runtime):
        raise ValueError("interaction formal topology changed")
    contract_error = (
        _write_bank_set_contract(runtime, contract_path)
        if interaction and context.is_main
        else None
    )
    if interaction:
        if context.world_size > 1:
            payload = [contract_error]
            dist.broadcast_object_list(payload, src=0)
            contract_error = payload[0]
        if contract_error is not None:
            raise ValueError(contract_error)
    elif context.is_main and args.resume is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(contract_path, runtime.run_contract)
    if context.world_size > 1:
        dist.barrier()


def _restore_routing_cursor(runtime: JointProgramPrimalRuntime) -> None:
    args = runtime.args
    context = runtime.context
    if args.resume is not None:
        optimizer_steps, expected_rows = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=routing_stage(runtime.config),
            context=context,
            model=runtime.writer_state,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            run_contract_schema=routing_run_schema(runtime.config),
        )
        metrics_rows = 0
        reconcile_error = None
        if context.is_main:
            try:
                metrics_rows = reconcile_metrics(
                    args.output_dir / "metrics.jsonl",
                    optimizer_steps,
                    expected_rows,
                    cursor_key="optimizer_step",
                )
            except Exception as error:  # keep all ranks on the same resume path
                reconcile_error = f"{type(error).__name__}: {error}"
        if context.world_size > 1:
            payload = [metrics_rows, reconcile_error]
            dist.broadcast_object_list(payload, src=0)
            metrics_rows, reconcile_error = payload
        if reconcile_error is not None:
            raise ValueError(reconcile_error)
        runtime.optimizer_steps = optimizer_steps
        runtime.metrics_rows = int(metrics_rows)


def prepare_routing_control_runtime(
    args: argparse.Namespace, context: Any
) -> JointProgramPrimalRuntime:
    config, base = _resolve_routing_inputs(args)
    authority = _authority_assets(args, context, config, base)
    model = _model_assets(args, context, config, base, authority)
    writer_state, trainable, frozen = _scorer_parameter_ownership(
        model.program,
        model.compiler,
        partition=config["model"].get(
            "primal_scorer_trainable_partition", SCORER_ALL_PARAMETERS
        ),
        stage=routing_stage(config),
    )
    data = _data_assets(args, config, base, context, authority, model)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    initialization = _writer_initialization(args, context, config, model)
    runtime = _new_routing_runtime(
        args=args,
        context=context,
        config=config,
        base=base,
        authority=authority,
        model=model,
        data=data,
        writer_state=writer_state,
        trainable=trainable,
        frozen=frozen,
        initialization=initialization,
    )
    _seal_routing_contract(runtime, initialization=initialization)
    _restore_routing_cursor(runtime)
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime
