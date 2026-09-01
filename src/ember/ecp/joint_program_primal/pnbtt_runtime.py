"""Minimal E1 runtime authority for the canonical PNBTT compiler."""

from __future__ import annotations

import argparse
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.bank_conditioning.frozen_condition_cache import (
    FROZEN_CONDITION_CACHE_SCHEMA,
    FrozenMappingConditionCache,
    frozen_condition_cache_authority,
)
from ember.ecp.bank_conditioning.mapping import MappingCondition, load_mapping_split
from ember.ecp.bank_conditioning.tangent_transport import TaskLocalFreeTangentQuery
from ember.ecp.checkpoint import load_ecp_checkpoint
from ember.ecp.contracts import build_target_owners
from ember.ecp.joint_program_primal.runtime import (
    REPO_ROOT,
    FunctionalPanelAuthority,
    _load_panels,
    _optimizer,
    _scheduler,
    _tasks,
)
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
    load_shared_scale_prior,
)
from ember.ecp.stage0_training import stage0_source_authority, tokenize_stage0_languages
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


PNBTT_TASKLOCAL_SCHEMA = "ember_ecp_pnbtt_tasklocal_v1"
PNBTT_TASKLOCAL_RUN_SCHEMA = "ember_ecp_pnbtt_tasklocal_run_v1"
PNBTT_E1_STAGE = "g3_pnbtt_e1_free_query_transport"


@dataclass(frozen=True)
class PNBTTTaskConditions:
    fit_views: tuple[MappingCondition, MappingCondition]
    held_video: MappingCondition


class PNBTTTaskLocalWriterState(torch.nn.Module):
    """Checkpoint only E1 free queries and the shared bank key transport."""

    def __init__(
        self,
        compiler: SharedNativeFactorCompiler,
        free_query: TaskLocalFreeTangentQuery | None,
    ) -> None:
        super().__init__()
        self.tangent_transport = compiler.tangent_transport
        if free_query is not None:
            self.free_query = free_query


@dataclass
class PNBTTTaskLocalRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    base_config: dict[str, Any]
    context: DistributedContext
    task_by_id: dict[int, Any]
    task_conditions: dict[int, PNBTTTaskConditions]
    panels: dict[int, FunctionalPanelAuthority]
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset
    query_processor: Pi05LiberoProcessor
    panel_batch_cache: dict[tuple[int, str, int], dict[str, Any]]
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: torch.nn.Module
    compiler: SharedNativeFactorCompiler
    free_query: TaskLocalFreeTangentQuery | None
    writer_state: PNBTTTaskLocalWriterState
    owners: tuple[Any, ...]
    ranks: Any
    rank4_contract: Any
    condition_cache: FrozenMappingConditionCache
    query_points: int
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer_steps: int
    stop_after_step: int
    checkpoint_steps: tuple[int, ...]
    metrics_rows: int
    margin_scales: dict[int, float]
    run_contract: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()
        self.query_dataset.close()


def is_pnbtt_tasklocal_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == PNBTT_TASKLOCAL_SCHEMA


def load_pnbtt_tasklocal_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    model = config.get("model", {})
    task_local = config.get("task_local", {})
    optimization = config.get("optimization", {})
    valid = all(
        (
            is_pnbtt_tasklocal_config(config),
            config.get("stage") == PNBTT_E1_STAGE,
            tuple(task_local.get("task_ids", ())) == (1, 93),
            task_local.get("wrong_task_by_task") == {"1": 8, "93": 94},
            task_local.get("preservation_only_task_by_task")
            == {"1": 8, "93": 94},
            tuple(config.get("task_split", {}).get("preservation_only", ()))
            == (8, 94),
            model.get("query_source") == "tasklocal_free_query",
            int(model.get("key_width", 0)) > 0,
            int(model.get("event_slots", 0)) == 8,
            0.0 < float(config.get("gate", {}).get("near_bound_weight_threshold", 0)) < 1.0,
            config.get("gate", {}).get("adjacent_checkpoint_conclusion_consistent")
            is True,
            optimization.get("loss")
            == "correct_functional_plus_wrong_necessity_plus_wrong_carrier_bound_and_unrelated_policy_distance",
            config.get("information_wall", {}).get("action_meta_installed") is False,
            config.get("information_wall", {}).get("single_complete_rank16") is True,
        )
    )
    if not valid:
        raise ValueError("invalid PNBTT task-local config")
    return config


def _condition_rows(
    split: Any, task_ids: tuple[int, ...]
) -> dict[int, PNBTTTaskConditions]:
    output = {}
    for task in task_ids:
        fit = split.fit_by_task[task]
        held = split.video_held_by_task[task]
        if len(fit) < 2 or len(held) != 1:
            raise ValueError("PNBTT task-local video split changed")
        output[task] = PNBTTTaskConditions(
            fit_views=(fit[0], fit[1]), held_video=held[0]
        )
    return output


def _margin_scales(
    config: Mapping[str, Any], *, asset_root: Path, task_ids: tuple[int, ...]
) -> dict[int, float]:
    root = (asset_root / config["authorities"]["positive_control_root"]).resolve()
    result = {}
    for task in task_ids:
        row = read_json(root / f"task_{task:03d}" / "result.json")
        benefits = [
            float(value["panel_a"]["benefit_over_carrier"])
            for value in row["evaluation"]["fit_videos"]
        ]
        if len(benefits) != 2 or not all(math.isfinite(x) and x > 0 for x in benefits):
            raise ValueError("PNBTT positive-control margin authority changed")
        result[task] = sum(benefits) / len(benefits)
    return result


def _model_inventory(runtime: PNBTTTaskLocalRuntime) -> dict[str, Any]:
    action_meta_modules = tuple(
        module
        for root in (runtime.policy, runtime.program)
        for module in root.modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    )
    action_meta_parameters = {
        id(parameter): parameter.numel()
        for module in action_meta_modules
        for parameter in module.parameters()
    }
    trainable = [
        name
        for name, parameter in runtime.writer_state.named_parameters()
        if parameter.requires_grad
    ]
    forbidden = [
        name
        for name in trainable
        if any(token in name for token in ("native_value", "task_lookup", "video_lookup"))
    ]
    if (
        action_meta_modules
        or forbidden
        or any(parameter.requires_grad for parameter in runtime.policy.parameters())
        or any(parameter.requires_grad for parameter in runtime.program.parameters())
    ):
        raise ValueError("PNBTT E1 information wall changed")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in runtime.trainable_parameters
        ),
        "source_policy_trainable_parameter_count": 0,
        "natural_program_trainable_parameter_count": 0,
        "action_meta_module_count": len(action_meta_modules),
        "action_meta_parameter_count": sum(action_meta_parameters.values()),
        "tasklocal_free_query_training_only": runtime.free_query is not None,
        "program_to_native_value_parameter_count": 0,
    }


def _world_topology(context: DistributedContext) -> list[dict[str, Any]]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "hostname": socket.gethostname(),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return rows


def _run_contract(runtime: PNBTTTaskLocalRuntime) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": PNBTT_TASKLOCAL_RUN_SCHEMA,
        "stage": runtime.config["stage"],
        "mode": runtime.args.mode,
        "git": state,
        "config": {
            "path": str(runtime.args.config),
            "bytes": runtime.args.config.stat().st_size,
        },
        "base_config": {
            "path": str(runtime.args.base_config),
            "bytes": runtime.args.base_config.stat().st_size,
        },
        "source_checkpoint": str(runtime.args.checkpoint),
        "tokenizer": str(runtime.args.tokenizer_path),
        "data_root": str(runtime.args.data_root),
        "condition_cache": {
            "root": str(runtime.args.condition_cache_root),
            "schema_version": FROZEN_CONDITION_CACHE_SCHEMA,
            "program_output_cached": False,
        },
        "program_bank_condition_cache_root": str(
            runtime.args.program_bank_condition_cache_root
        ),
        "task_local": dict(runtime.config["task_local"]),
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "inventory": _model_inventory(runtime),
        "world_topology": _world_topology(runtime.context),
    }


def _optimizer_cursor(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    writer: PNBTTTaskLocalWriterState,
    trainable: tuple[torch.nn.Parameter, ...],
) -> tuple[Any, Any, tuple[int, ...], int, int, int]:
    optimizer = _optimizer(trainable, config)
    scheduler = _scheduler(optimizer, config)
    joint = config["optimization"]["joint"]
    warmup = int(joint["warmup_optimizer_steps"])
    checkpoints = tuple(
        warmup + int(step) for step in joint["checkpoint_effective_steps"]
    )
    stop = int(args.stop_after_step or (1 if args.mode == "profile" else checkpoints[-1]))
    allowed = {1, 2} if args.mode == "profile" else set(checkpoints)
    if stop not in allowed:
        raise ValueError("PNBTT stop step is outside its direct contract")
    optimizer_steps = metrics_rows = 0
    if args.resume is not None:
        optimizer_steps, expected_rows = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=str(config["stage"]),
            context=context,
            model=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=PNBTT_TASKLOCAL_RUN_SCHEMA,
        )
        if context.is_main:
            metrics_rows = reconcile_metrics(
                args.output_dir / "metrics.jsonl",
                optimizer_steps,
                expected_rows,
                cursor_key="optimizer_step",
            )
    return optimizer, scheduler, checkpoints, stop, optimizer_steps, metrics_rows


def prepare_pnbtt_tasklocal_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> PNBTTTaskLocalRuntime:
    config = load_pnbtt_tasklocal_config(args.config)
    if args.program_bank_condition_cache_root is None:
        raise ValueError("PNBTT E1 requires the exact-language wrong-bank cache")
    task_ids = tuple(map(int, config["task_local"]["task_ids"]))
    allowed_world_sizes = tuple(map(int, config["profile"]["allowed_world_sizes"]))
    if context.world_size not in allowed_world_sizes:
        raise ValueError("PNBTT world size is outside its launch contract")
    if args.mode == "formal":
        state = git_state(REPO_ROOT)
        if (
            not git_state_is_clean_pushed_or_frozen_authority(state)
            or state.get("branch") != ""
            or state.get("upstream") is not None
        ):
            raise ValueError("formal PNBTT requires detached pushed authority")
    seed_everything(int(config["optimization"]["seed"]), context)
    base_path = (args.asset_root / config["authorities"]["base_g3_config"]).resolve()
    if args.base_config != base_path:
        raise ValueError("PNBTT base config authority changed")
    base = load_shared_compiler_config(base_path)
    source = stage0_source_authority(args)
    source_config = load_config(
        authority_path(base, "source_base_config", asset_root=args.asset_root)
    )
    expected_checkpoint = authority_path(base, "source_checkpoint", asset_root=args.asset_root)
    expected_tokenizer = authority_path(base, "tokenizer", asset_root=args.asset_root)
    if (
        args.checkpoint != expected_checkpoint
        or args.source_run != expected_checkpoint.parent.parent
        or args.tokenizer_path != expected_tokenizer
    ):
        raise ValueError("PNBTT source/tokenizer authority changed")

    tasks = _tasks(base, args.data_root, args.asset_root)
    task_by_id = {task.authority_id: task for task in tasks}
    wrong_ids = tuple(config["task_local"]["wrong_task_by_task"].values())
    selected_ids = tuple(dict.fromkeys((*task_ids, *map(int, wrong_ids))))
    selected_tasks = tuple(task_by_id[task] for task in selected_ids)
    panels = _load_panels(config, asset_root=args.asset_root)
    split = load_mapping_split(base, asset_root=args.asset_root)
    conditions = _condition_rows(split, selected_ids)

    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    policy.requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        base,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, base["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    program = build_frozen_g2_program(
        base, asset_root=args.asset_root, owners=owners, device=context.device
    )
    program.requires_grad_(False).eval()
    prepare_frozen_writer_policy(policy, ranks.contract)
    model = config["model"]
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(model["program_width"]),
        event_slots=int(model["event_slots"]),
        key_width=int(model["key_width"]),
        query_hidden_width=int(model["query_hidden_width"]),
        covariance_ridge=float(model["covariance_ridge"]),
        native_rms_epsilon=float(model["native_rms_epsilon"]),
        direction_epsilon=float(model["direction_epsilon"]),
        query_epsilon=float(model["query_epsilon"]),
        score_epsilon=float(model["score_epsilon"]),
        replay_chunk_size=int(model["replay_chunk_size"]),
        temperature_by_side=tuple(model["temperature_by_side"]),
        type_balance=torch.tensor(model["fixed_type_balance"], dtype=torch.float32),
        scale_prior_ratio=load_shared_scale_prior(
            base, asset_root=args.asset_root, device=context.device
        ),
        relative_eigenvalue_floor=float(base["model"]["relative_eigenvalue_floor"]),
        replay_score_rms=float(base["model"]["replay_score_rms"]),
        covariance_frame_chunk=int(base["model"]["frame_chunk_size"]),
    ).to(context.device)
    compiler.requires_grad_(False).eval()
    compiler.tangent_transport.key_encoder.requires_grad_(True).train()
    free_query = TaskLocalFreeTangentQuery(
        task_ids,
        owners,
        event_slots=int(model["event_slots"]),
        key_width=int(model["key_width"]),
        query_epsilon=float(model["query_epsilon"]),
    ).to(context.device)
    free_query.requires_grad_(True).train()
    writer = PNBTTTaskLocalWriterState(compiler, free_query)
    trainable = tuple(parameter for parameter in writer.parameters() if parameter.requires_grad)
    frozen = tuple(
        parameter
        for root in (policy, program, compiler)
        for parameter in root.parameters()
        if not parameter.requires_grad
    )

    task_authorities = tuple(task.writer_authority() for task in selected_tasks)
    video_store = RawTeacherVideoStore(
        task_authorities,
        frame_stride=int(base["data"]["frame_stride"]),
        max_open_files=8,
    )
    query_dataset = FunctionalQueryDataset(
        task_authorities,
        demo_indices=range(50),
        action_chunk_size=int(source_config["features"]["chunk_size"]),
        max_open_files_per_worker=8,
    )
    query_processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language_tokens = tokenize_stage0_languages(
        selected_tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    cache_authority = frozen_condition_cache_authority(
        config_schema=str(config["frozen_condition_cache_authority"]["config_schema"]),
        config_bytes=int(config["frozen_condition_cache_authority"]["config_bytes"]),
        source_checkpoint=expected_checkpoint,
        g2_program_checkpoint=authority_path(
            base, "g2_program_checkpoint", asset_root=args.asset_root
        ),
        native_observer_checkpoint=authority_path(
            base, "native_observer_checkpoint", asset_root=args.asset_root
        ),
        frame_stride=int(base["data"]["frame_stride"]),
        owners=owners,
    )
    condition_cache = FrozenMappingConditionCache(
        args.condition_cache_root,
        owners=owners,
        operator=compiler.bank_operator,
        authority=cache_authority,
        cache_program=False,
    )
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in writer.state_dict().values():
            dist.broadcast(value, src=0)
    optimizer, scheduler, checkpoints, stop, steps, rows = _optimizer_cursor(
        args, config, context, writer, trainable
    )
    runtime = PNBTTTaskLocalRuntime(
        args=args,
        config=config,
        base_config=base,
        context=context,
        task_by_id=task_by_id,
        task_conditions=conditions,
        panels=panels,
        video_store=video_store,
        query_dataset=query_dataset,
        query_processor=query_processor,
        panel_batch_cache={},
        language_tokens=language_tokens,
        policy=policy,
        program=program,
        compiler=compiler,
        free_query=free_query,
        writer_state=writer,
        owners=owners,
        ranks=ranks,
        rank4_contract=rank4_contract,
        condition_cache=condition_cache,
        query_points=int(
            read_json(authority_path(base, "g2_config", asset_root=args.asset_root))[
                "data"
            ]["query_points"]
        ),
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_steps=steps,
        stop_after_step=stop,
        checkpoint_steps=checkpoints,
        metrics_rows=rows,
        margin_scales=_margin_scales(config, asset_root=args.asset_root, task_ids=task_ids),
        run_contract={},
    )
    runtime.run_contract = _run_contract(runtime)
    if context.is_main:
        if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("fresh PNBTT output root is not empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.resume is None:
            write_json_atomic(args.output_dir / "run_contract.json", runtime.run_contract)
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    return runtime
