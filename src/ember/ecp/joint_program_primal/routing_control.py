"""Training-only orthogonal routing-token boundary control for G3.

This module is deliberately not a deployment Writer.  It replaces Natural
Program content with fixed, non-trainable task routing tokens so the existing
ProgramNativePrimalScorer can be tested independently of Program geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint
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
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.shared_compiler_assets import load_shared_compiler_config
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import initialize_deferred_process_group


ROUTING_CONTROL_SCHEMA = "ember_ecp_routing_token_control_r1_v1"
ROUTING_CONTROL_RUN_SCHEMA = "ember_ecp_routing_token_control_run_v1"
ROUTING_CONTROL_STAGE = "g3_training_only_routing_token_control"
ROUTING_TASK_IDS = (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
ROUTING_WIDTH = 128


class RoutingControlWriterState(torch.nn.Module):
    """Checkpoint only the scorer exercised by this boundary control."""

    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self.primal_scorer = scorer


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
    split = config.get("task_split", {})
    data = config.get("data", {})
    joint = config.get("optimization", {}).get("joint", {})
    model = config.get("model", {})
    wall = config.get("information_wall", {})
    critic = config.get("optimization", {}).get("privileged_critic")
    tasks = tuple(
        map(
            int,
            (*split.get("gradient_meta", ()), *split.get("gradient_target", ())),
        )
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
    valid = all(
        (
            config.get("schema_version") == ROUTING_CONTROL_SCHEMA,
            config.get("status") == "training_only_routing_boundary_control",
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
            joint.get("global_tasks_per_optimizer_step") == 6,
            joint.get("video_views_per_task") == 2,
            model.get("program_source")
            == "fixed_nontrainable_128d_orthogonal_task_token",
            model.get("trainable") == ["ProgramNativePrimalScorer"],
            model.get("deployment_candidate") is False,
            wall.get("fixed_routing_token_training_only") is True,
            wall.get("action_meta_installed") is False,
            wall.get("shuffled_or_reversed_use") is False,
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
    if not valid:
        raise ValueError("unsupported G3 routing-token boundary-control config")
    return config


def _scorer_parameter_ownership(
    program: torch.nn.Module, compiler: torch.nn.Module
) -> tuple[RoutingControlWriterState, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    program.requires_grad_(False).eval()
    compiler.requires_grad_(False).eval()
    compiler.primal_scorer.requires_grad_(True).train()
    writer = RoutingControlWriterState(compiler.primal_scorer)
    trainable = tuple(writer.parameters())
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
    writer_state: RoutingControlWriterState,
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
    if args.resume is not None:
        optimizer_steps, expected_rows = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=ROUTING_CONTROL_STAGE,
            context=context,
            model=writer_state,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=ROUTING_CONTROL_RUN_SCHEMA,
        )
        if context.is_main:
            metrics_rows = reconcile_metrics(
                args.output_dir / "metrics.jsonl",
                optimizer_steps,
                expected_rows,
                cursor_key="optimizer_step",
            )
    return optimizer, scheduler, checkpoints, stop, optimizer_steps, metrics_rows


def _run_contract(runtime: JointProgramPrimalRuntime) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": ROUTING_CONTROL_RUN_SCHEMA,
        "stage": ROUTING_CONTROL_STAGE,
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
            "program_output_cached": False,
            "checkpoint_payload": False,
        },
        "task_split": dict(runtime.config["task_split"]),
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
            "removal_trigger": "retire executable control after Gate interpretation and before the next canonical deployment architecture",
        },
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]),
        "throughput_gate": dict(runtime.config["throughput_gate"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "inventory": _inventory(
            runtime.policy, runtime.program, runtime.compiler, runtime.owners
        ),
        "world_topology": _topology(runtime.context),
    }


def prepare_routing_control_runtime(
    args: argparse.Namespace, context: Any
) -> JointProgramPrimalRuntime:
    config = load_routing_control_config(args.config)
    base_path = (args.asset_root / config["authorities"]["base_g3_config"]).resolve()
    if args.base_config != base_path:
        raise ValueError("routing-control base G3 config authority changed")
    base = load_shared_compiler_config(base_path)
    authority = _authority_assets(args, context, config, base)
    model = _model_assets(args, context, base, authority)
    writer_state, trainable, frozen = _scorer_parameter_ownership(
        model.program, model.compiler
    )
    data = _data_assets(args, config, base, context, authority, model)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in writer_state.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer, scheduler, checkpoints, stop, optimizer_steps, metrics_rows = (
        _optimizer_cursor(args, config, context, writer_state, trainable)
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
        run_contract={},
    )
    runtime.run_contract = _run_contract(runtime)
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "run_contract.json", runtime.run_contract)
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime
