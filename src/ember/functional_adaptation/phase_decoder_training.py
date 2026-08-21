"""Distributed fixed-decoder fitting on successful closed-loop phase panels."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.functional_adaptation.decoder import FunctionalAdapterDecoder
from ember.functional_adaptation.functional_response import (
    functional_response_distillation_loss,
)
from ember.functional_adaptation.phase_decoder_codes import (
    PhaseDecoderCodeAuthority,
    load_phase_decoder_code_authority,
)
from ember.functional_adaptation.phase_decoder_panels import (
    CachedPhasePanel,
    PhaseMemberSource,
    ProjectedOccupancySource,
    cache_phase_member_panels,
    cache_projected_occupancy_panels,
    load_phase_member_sources,
    load_projected_occupancy_sources,
)
from ember.functional_adaptation.phase_decoder_projection import (
    materialize_phase_decoder_projections,
    phase_decoder_asset,
    save_phase_decoder,
    save_stable_shared_prior,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    initialize_distributed,
    load_policy,
    seed_everything,
)
from ember.writer.as_step import (
    accumulate_flat_gradient,
    assign_flat_gradient,
    parameter_layout,
)
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "ember_phase_aligned_functional_decoder_run_v1"
RESULT_SCHEMA = "ember_phase_aligned_functional_decoder_result_v1"
CHECKPOINT_SCHEMA = "ember_phase_aligned_functional_decoder_checkpoint_v1"


@dataclass
class Runtime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    repository: dict[str, Any]
    source: dict[str, Any]
    code_authority: PhaseDecoderCodeAuthority
    contract: LoRAContract
    policy: torch.nn.Module
    identity_state: Mapping[str, torch.Tensor]
    decoder: FunctionalAdapterDecoder
    decoder_mode: str
    shared_prior_asset: Mapping[str, Any] | None
    optimizer: torch.optim.Optimizer
    member_sources: tuple[PhaseMemberSource, ...]
    fit_panels: dict[int, tuple[CachedPhasePanel, ...]]
    occupancy_sources: dict[int, ProjectedOccupancySource]
    onpolicy_panels: dict[int, tuple[CachedPhasePanel, ...]]
    schedule: tuple[tuple[int, int], ...]
    topology: tuple[dict[str, Any], ...]
    started: float


def _authority_path(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_base_config(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version")
        != "ember_pi05_train24_phase_aligned_decoder_v1"
        or value.get("status") != "preregistered_before_decoder_optimization"
        or value.get("decoder", {}).get("class") != "FunctionalAdapterDecoder"
        or value.get("decoder", {}).get("fully_fixed_after_fit") is not True
        or value.get("representation", {}).get("held_code_optimization_steps") != 0
    ):
        raise ValueError("unsupported phase-aligned decoder config")


def _overlay_path(value: Mapping[str, Any]) -> Path:
    base_path = Path(str(value.get("base_config", "")))
    return base_path if base_path.is_absolute() else REPO_ROOT / base_path


def _load_config(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError("phase decoder config overlay cycle")
    value = read_json(resolved)
    schema = value.get("schema_version")
    if schema == "ember_pi05_train24_phase_aligned_decoder_v1":
        _validate_base_config(value)
        value["_base_config_path"] = str(resolved)
        value["_decoder_mode"] = "phase_decoder"
        return value
    if schema == "ember_pi05_train24_phase_decoder_onpolicy_state_aggregation_v1":
        base_path = _overlay_path(value)
        base = _load_config(base_path, seen | {resolved})
        _validate_base_config(base)
        aggregation = value.get("state_aggregation", {})
        if (
            value.get("status") != "preregistered_before_state_aggregation"
            or aggregation.get("learner_occupancy")
            != "fit19_projected_policy_trajectories"
            or aggregation.get("panel_mixture")
            != "one_successful_expert_panel_to_one_projected_occupancy_panel"
            or int(aggregation.get("fit_task_count", -1)) != 19
            or int(aggregation.get("held_task_gradient_count", -1)) != 0
            or int(aggregation.get("projected_trajectory_count", -1)) != 30
            or int(aggregation.get("privileged_expert_member_targets", -1)) != 37
            or int(aggregation.get("phase_points_per_member", -1)) != 8
            or int(aggregation.get("panels_per_member", -1)) != 4
        ):
            raise ValueError("unsupported phase decoder state-aggregation config")
        merged = dict(base)
        merged.update(
            schema_version=value["schema_version"],
            status=value["status"],
            purpose=value["purpose"],
            base_config=value["base_config"],
            state_aggregation=dict(aggregation),
            training=dict(value["training"]),
            claim_boundary=value["claim_boundary"],
            _base_config_path=str(base_path.resolve()),
            _decoder_mode="state_aggregation",
        )
        merged["authorities"] = {
            **base["authorities"],
            **value.get("authorities", {}),
        }
        return merged
    if schema == "ember_pi05_train24_stable_shared_prior_v1":
        base_path = _overlay_path(value)
        base = _load_config(base_path, seen | {resolved})
        prior = value.get("stable_shared_prior", {})
        if (
            value.get("status") != "preregistered_before_shared_prior_optimization"
            or base.get("state_aggregation") is None
            or prior.get("code_condition") != "fixed_zero_code"
            or prior.get("template") != "complete_rank16_identity_lora"
            or prior.get("panel_mixture")
            != "one_successful_expert_panel_to_one_projected_occupancy_panel"
            or int(prior.get("fit_task_count", -1)) != 19
            or int(prior.get("held_task_gradient_count", -1)) != 0
            or int(prior.get("shared_rank", -1)) != 12
            or int(prior.get("reserved_task_residual_rank", -1)) != 4
            or prior.get("output") != "one_task_independent_complete_rank16_lora"
            or int(value.get("decoder", {}).get("active_rank_start", -1)) != 0
            or int(value.get("decoder", {}).get("active_rank_end", -1)) != 12
        ):
            raise ValueError("unsupported stable shared-prior config")
        merged = dict(base)
        merged.update(
            schema_version=schema,
            status=value["status"],
            purpose=value["purpose"],
            base_config=value["base_config"],
            stable_shared_prior=dict(prior),
            training=dict(value["training"]),
            claim_boundary=value["claim_boundary"],
            _decoder_mode="stable_shared_prior",
        )
        merged["decoder"] = {**base["decoder"], **value.get("decoder", {})}
        merged["authorities"] = {**base["authorities"], **value.get("authorities", {})}
        return merged
    if schema == "ember_pi05_train24_shared_prior_residual_decoder_v1":
        base_path = _overlay_path(value)
        base = _load_config(base_path, seen | {resolved})
        residual = value.get("shared_prior_residual", {})
        if (
            value.get("status") != "preregistered_before_residual_optimization"
            or base.get("state_aggregation") is None
            or residual.get("shared_prior_authority") != "formal_run_cli"
            or residual.get("shared_prior_frozen") is not True
            or residual.get("residual_centered_at_zero_code") is not True
            or residual.get("panel_mixture")
            != "one_successful_expert_panel_to_one_projected_occupancy_panel"
            or int(residual.get("fit_task_count", -1)) != 19
            or int(residual.get("held_task_gradient_count", -1)) != 0
            or int(residual.get("shared_rank", -1)) != 12
            or int(residual.get("task_residual_rank", -1)) != 4
            or residual.get("output")
            != "one_complete_rank16_shared_prior_plus_task_residual_lora"
            or residual.get("merge_rule")
            != "exact_disjoint_rank_concatenation_delta_shared_rank12_plus_delta_task_rank4"
            or int(value.get("decoder", {}).get("active_rank_start", -1)) != 12
            or int(value.get("decoder", {}).get("active_rank_end", -1)) != 16
        ):
            raise ValueError("unsupported shared-prior residual config")
        merged = dict(base)
        merged.update(
            schema_version=schema,
            status=value["status"],
            purpose=value["purpose"],
            base_config=value["base_config"],
            shared_prior_residual=dict(residual),
            training=dict(value["training"]),
            claim_boundary=value["claim_boundary"],
            _decoder_mode="shared_prior_residual",
        )
        merged["decoder"] = {**base["decoder"], **value.get("decoder", {})}
        merged["authorities"] = {**base["authorities"], **value.get("authorities", {})}
        return merged
    raise ValueError("unsupported phase-aligned decoder config")


def load_config(path: Path) -> dict[str, Any]:
    return _load_config(path, frozenset())


def _task_schedule(config: Mapping[str, Any], world_size: int) -> tuple[tuple[int, int], ...]:
    fit_count = int(config["roles"]["fit_task_count"])
    visits = int(config["training"]["visits_per_fit_task"])
    generator = torch.Generator(device="cpu")
    rows: list[int] = []
    for epoch in range(visits):
        generator.manual_seed(int(config["training"]["schedule_seed"]) + epoch)
        rows.extend(torch.randperm(fit_count, generator=generator).tolist())
    if (
        len(rows) != int(config["training"]["total_task_visits"])
        or len(rows) % world_size
        or any(rows.count(index) != visits for index in range(fit_count))
    ):
        raise ValueError("phase decoder task-equal schedule changed")
    seen = [0] * fit_count
    schedule = []
    for task_index in rows:
        schedule.append((task_index, seen[task_index]))
        seen[task_index] += 1
    return tuple(schedule)


def _load_stable_shared_prior(
    root: Path,
    contract: LoRAContract,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    resolved = root.resolve()
    result_path = resolved / "result.json"
    result = read_json(result_path)
    record = result.get("shared_prior", {})
    path = Path(str(record.get("path", ""))).resolve()
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("formal_authority") is not True
        or result.get("decoder_mode") != "stable_shared_prior"
        or result.get("repository", {}).get("dirty_paths") != []
        or path.parent != resolved
        or path.name != "shared_prior.safetensors"
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
    ):
        raise ValueError("stable shared-prior authority changed")
    state = load_file(str(path), device=str(device))
    validate_lora_state(state, contract)
    return state, {
        "root": str(resolved),
        "adapter": phase_decoder_asset(path),
        "result": phase_decoder_asset(result_path),
        "repository": result["repository"],
    }


def _prepare(args: argparse.Namespace) -> Runtime:
    if args.resume is None and args.output_dir.exists():
        raise ValueError("fresh phase decoder output directory already exists")
    if args.resume is not None and not args.output_dir.is_dir():
        raise ValueError("phase decoder resume output directory is missing")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    config_path = args.config.resolve()
    config = load_config(config_path)
    decoder_mode = str(config["_decoder_mode"])
    repository = git_state(REPO_ROOT)
    expected_world = int(config["training"]["world_size"])
    if (
        args.mode != "formal"
        or context.world_size != expected_world
        or os.environ.get("NCCL_P2P_DISABLE") != "1"
        or not git_state_is_clean_pushed_or_frozen_authority(repository)
    ):
        raise ValueError("formal phase decoder launch authority changed")
    seed_everything(int(config["decoder"]["initialization_seed"]), context)
    codes = load_phase_decoder_code_authority(
        args.code_artifact,
        config=config,
        config_path=Path(config["_base_config_path"]),
        device=context.device,
    )
    authorities = load_evaluation_authorities(
        _authority_path(config, "evaluation"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    policy = load_policy(
        Path(str(source["model_path"])), authorities.source_base_config, context.device
    )
    contract = load_pi05_lora_contract(_authority_path(config, "lora_contract"))
    identity = prepare_frozen_writer_policy(policy, contract)
    policy.requires_grad_(False).eval()
    shared_prior_asset = None
    template_state = identity
    if decoder_mode == "shared_prior_residual":
        if args.shared_prior_root is None:
            raise ValueError("residual decoder requires a formal shared-prior root")
        template_state, shared_prior_asset = _load_stable_shared_prior(
            args.shared_prior_root,
            contract,
            context.device,
        )
    elif args.shared_prior_root is not None:
        raise ValueError("only the residual decoder accepts a shared-prior root")
    decoder = FunctionalAdapterDecoder(
        contract,
        template_state,
        code_width=int(config["decoder"]["code_width"]),
        address_width=int(config["decoder"]["address_width"]),
        hidden_width=int(config["decoder"]["hidden_width"]),
        initialization_seed=int(config["decoder"]["initialization_seed"]),
        center_residual_at_zero_code=bool(
            config["decoder"].get("center_residual_at_zero_code", False)
        ),
        active_rank_start=int(config["decoder"].get("active_rank_start", 0)),
        active_rank_end=int(
            config["decoder"].get("active_rank_end", contract.rank)
        ),
    ).to(context.device)
    aggregation = config.get("state_aggregation")
    if aggregation is not None:
        if args.state_bank_root is None:
            raise ValueError("state-aggregation training requires its occupancy root")
        if decoder_mode == "state_aggregation":
            initial_decoder = _authority_path(config, "initial_decoder")
            initial_result = read_json(_authority_path(config, "initial_training_result"))
            recorded_decoder = initial_result.get("decoder", {})
            if (
                initial_result.get("schema_version") != RESULT_SCHEMA
                or initial_result.get("formal_authority") is not True
                or initial_result.get("repository", {}).get("dirty_paths") != []
                or Path(str(recorded_decoder.get("path", ""))).resolve()
                != initial_decoder.resolve()
                or initial_decoder.stat().st_size
                != int(recorded_decoder.get("bytes", -1))
            ):
                raise ValueError("state-aggregation initial decoder authority changed")
            decoder.load_state_dict(
                load_file(str(initial_decoder), device=str(context.device)), strict=True
            )
    elif args.state_bank_root is not None:
        raise ValueError("base phase decoder cannot consume a state bank")
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    member_sources = load_phase_member_sources(
        analysis_path=_authority_path(config, "phase_analysis"),
        codes=codes,
        repo_root=REPO_ROOT,
    )
    runtime = Runtime(
        args=args,
        context=context,
        config=config,
        repository=repository,
        source=source,
        code_authority=codes,
        contract=contract,
        policy=policy,
        identity_state=identity,
        decoder=decoder,
        decoder_mode=decoder_mode,
        shared_prior_asset=shared_prior_asset,
        optimizer=optimizer,
        member_sources=member_sources,
        fit_panels={},
        occupancy_sources={},
        onpolicy_panels={},
        schedule=_task_schedule(config, context.world_size),
        topology=(),
        started=time.monotonic(),
    )
    fit_member_indices = [
        index
        for index, member in enumerate(codes.members)
        if member.fold_role == "fit"
    ]
    train_seed = int(config["functional_supervision"]["train_policy_seed"])
    runtime.fit_panels = {
        index: cache_phase_member_panels(
            policy=runtime.policy,
            identity_state=runtime.identity_state,
            contract=runtime.contract,
            sources=runtime.member_sources,
            member_index=index,
            device=runtime.context.device,
            policy_seed=train_seed,
        )
        for index in fit_member_indices
    }
    if aggregation is not None:
        runtime.occupancy_sources = load_projected_occupancy_sources(
            root=args.state_bank_root,
            codes=codes,
        )
        aggregation_seed = int(aggregation["expert_query_policy_seed"])
        runtime.onpolicy_panels = {
            index: cache_projected_occupancy_panels(
                policy=runtime.policy,
                identity_state=runtime.identity_state,
                contract=runtime.contract,
                member_sources=runtime.member_sources,
                occupancy_sources=runtime.occupancy_sources,
                member_index=index,
                device=runtime.context.device,
                policy_seed=aggregation_seed,
            )
            for index in fit_member_indices
        }
    initialize_deferred_process_group(
        context, rendezvous_root=args.output_dir.resolve().parent
    )
    for value in decoder.state_dict().values():
        dist.broadcast(value, src=0)
    topology: list[Any] = [None] * context.world_size
    dist.all_gather_object(
        topology,
        {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "device": str(context.device),
            "device_name": torch.cuda.get_device_name(context.device),
            "numa_node": context.numa_node,
            "cpu_affinity": list(context.cpu_affinity or ()),
        },
    )
    runtime.topology = tuple(dict(value) for value in topology)
    return runtime


def _run_contract(runtime: Runtime) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "mode": runtime.args.mode,
        "decoder_mode": runtime.decoder_mode,
        "repository": runtime.repository,
        "host": socket.gethostname(),
        "runtime": {
            "world_size": runtime.context.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "deferred_nccl": True,
            "topology": list(runtime.topology),
        },
        "inputs": {
            "config": str(runtime.args.config.resolve()),
            "config_bytes": runtime.args.config.resolve().stat().st_size,
            "code_artifact": str(runtime.code_authority.root),
            "source_run": str(runtime.args.source_run.resolve()),
            "checkpoint": str(runtime.args.checkpoint.resolve()),
            "expert_bank_root": str(runtime.args.expert_bank_root.resolve()),
            "state_bank_root": (
                str(runtime.args.state_bank_root.resolve())
                if runtime.args.state_bank_root is not None
                else None
            ),
            "shared_prior_root": (
                str(runtime.args.shared_prior_root.resolve())
                if runtime.args.shared_prior_root is not None
                else None
            ),
            "shared_prior_authority": runtime.shared_prior_asset,
            "initial_decoder": (
                phase_decoder_asset(_authority_path(runtime.config, "initial_decoder"))
                if runtime.decoder_mode == "state_aggregation"
                else None
            ),
            "initial_training_result": (
                phase_decoder_asset(
                    _authority_path(runtime.config, "initial_training_result")
                )
                if runtime.decoder_mode == "state_aggregation"
                else None
            ),
        },
        "roles": runtime.config["roles"],
        "representation": runtime.config["representation"],
        "functional_supervision": runtime.config["functional_supervision"],
        "decoder": runtime.config["decoder"],
        "training": runtime.config["training"],
        "state_aggregation": runtime.config.get("state_aggregation"),
        "stable_shared_prior": runtime.config.get("stable_shared_prior"),
        "shared_prior_residual": runtime.config.get("shared_prior_residual"),
        "decision_gates": runtime.config["decision_gates"],
        "information_wall": {
            "privileged_actions": "development_train_successful_experts_only",
            "held_code_gradients": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "deployment_task_id_route": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_contract(runtime: Runtime) -> None:
    root = runtime.args.output_dir.resolve()
    contract = _run_contract(runtime)
    if runtime.context.is_main:
        if runtime.args.resume is None:
            root.mkdir(parents=True, exist_ok=False)
            write_json_atomic(root / "run_contract.json", contract)
        elif read_json(root / "run_contract.json") != contract:
            raise ValueError("phase decoder resume contract changed")
        append_jsonl(
            root / "invocations.jsonl",
            {
                "argv": list(os.sys.argv),
                "started_unix": time.time(),
                "resume": str(runtime.args.resume) if runtime.args.resume else None,
            },
        )
    dist.barrier(device_ids=[runtime.context.local_rank])


def _checkpoint_path(root: Path, task_visits: int) -> Path:
    return root / "checkpoints" / f"task_visits_{task_visits:08d}" / "state.pt"


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state(value["torch_cuda"], context.device)


def _save_checkpoint(runtime: Runtime, *, task_visits: int, metrics_rows: int) -> None:
    states: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(states, _rng_state(runtime.context))
    if runtime.context.is_main:
        path = _checkpoint_path(runtime.args.output_dir, task_visits)
        path.parent.mkdir(parents=True, exist_ok=False)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "task_visits": task_visits,
                "optimizer_updates": task_visits // runtime.context.world_size,
                "metrics_rows": metrics_rows,
                "world_size": runtime.context.world_size,
                "decoder": runtime.decoder.state_dict(),
                "optimizer": runtime.optimizer.state_dict(),
                "rng_by_rank": states,
            },
            temporary,
        )
        os.replace(temporary, path)
    dist.barrier(device_ids=[runtime.context.local_rank])


def _rewind_metrics(path: Path, rows: int) -> None:
    values = path.read_bytes().splitlines(keepends=True) if path.is_file() else []
    if len(values) < rows:
        raise ValueError("phase decoder metrics precede resume checkpoint")
    if len(values) > rows:
        temporary = path.with_name(f".{path.name}.resume.{os.getpid()}")
        temporary.write_bytes(b"".join(values[:rows]))
        os.replace(temporary, path)


def _resume(runtime: Runtime) -> tuple[int, int]:
    if runtime.args.resume is None:
        return 0, 0
    if runtime.args.resume.parents[2] != runtime.args.output_dir:
        raise ValueError("phase decoder resume checkpoint is outside its run root")
    value = torch.load(
        runtime.args.resume.resolve(),
        map_location=runtime.context.device,
        weights_only=False,
    )
    task_visits = int(value.get("task_visits", -1))
    metrics_rows = int(value.get("metrics_rows", -1))
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or int(value.get("world_size", -1)) != runtime.context.world_size
        or task_visits not in runtime.config["training"]["checkpoint_task_visits"]
        or task_visits % runtime.context.world_size
        or len(value.get("rng_by_rank", ())) != runtime.context.world_size
    ):
        raise ValueError("phase decoder exact-resume checkpoint changed")
    runtime.decoder.load_state_dict(value["decoder"], strict=True)
    runtime.optimizer.load_state_dict(value["optimizer"])
    _restore_rng(value["rng_by_rank"][runtime.context.rank], runtime.context)
    if runtime.context.is_main:
        _rewind_metrics(runtime.args.output_dir / "metrics.jsonl", metrics_rows)
    dist.barrier(device_ids=[runtime.context.local_rank])
    return task_visits, metrics_rows


def _fit_code(runtime: Runtime, task_index: int) -> torch.Tensor:
    if runtime.decoder_mode == "stable_shared_prior":
        return torch.zeros(
            runtime.decoder.code_width,
            device=runtime.context.device,
        )
    return runtime.code_authority.fit_task_codes[task_index]


def _member_for_visit(
    runtime: Runtime, task_index: int, visit: int
) -> tuple[int, int, str]:
    ordinal = runtime.code_authority.fit_ordinals[task_index]
    members = tuple(
        index
        for index, row in enumerate(runtime.code_authority.members)
        if row.ordinal == ordinal
    )
    if not members:
        raise ValueError("phase decoder fit task lost successful members")
    if runtime.onpolicy_panels:
        paired_visit = visit // 2
        member_index = members[paired_visit % len(members)]
        panel_index = (paired_visit // len(members)) % 4
        panel_source = (
            "successful_expert" if visit % 2 == 0 else "projected_occupancy"
        )
    else:
        member_index = members[visit % len(members)]
        cycle = visit // len(members)
        panel_source = "successful_expert"
        panel_index = cycle % 4
    return member_index, panel_index, panel_source


def _fit(runtime: Runtime, *, start_visits: int, metrics_rows: int) -> int:
    context = runtime.context
    layout = parameter_layout(runtime.decoder)
    gradient = torch.zeros(layout[-1].stop, device=context.device, dtype=torch.float32)
    world = context.world_size
    total = len(runtime.schedule)
    checkpoints = set(
        int(value)
        for value in runtime.config["training"]["checkpoint_task_visits"]
    )
    clip = float(runtime.config["training"]["optimizer"]["gradient_clip_norm"])
    metrics_path = runtime.args.output_dir / "metrics.jsonl"
    for cursor in range(start_visits, total, world):
        task_index, visit = runtime.schedule[cursor + context.rank]
        member_index, panel_index, panel_source = _member_for_visit(
            runtime, task_index, visit
        )
        panels = (
            runtime.fit_panels
            if panel_source == "successful_expert"
            else runtime.onpolicy_panels
        )
        panel = panels[member_index][panel_index]
        runtime.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = functional_response_distillation_loss(
                runtime.policy,
                runtime.decoder(_fit_code(runtime, task_index)),
                runtime.contract,
                panel.batch,
                panel.target,
                policy_seed=panel.policy_seed,
            )
        loss.backward()
        gradient.zero_()
        gradients = tuple(item.parameter.grad for item in layout)
        accumulate_flat_gradient(gradient, gradients, layout)
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world)
        gradient_norm = torch.linalg.vector_norm(gradient)
        if float(gradient_norm) > clip:
            gradient.mul_(clip / float(gradient_norm))
        assign_flat_gradient(gradient, layout)
        runtime.optimizer.step()
        local = {
            "rank": context.rank,
            "task_index": task_index,
            "task_ordinal": runtime.code_authority.fit_ordinals[task_index],
            "task_visit": visit,
            "member_index": member_index,
            "member": runtime.code_authority.members[member_index].member,
            "panel_index": panel_index,
            "panel_source": panel_source,
            "loss": float(loss.detach()),
        }
        records: list[Any] = [None] * world
        dist.all_gather_object(records, local)
        task_visits = cursor + world
        metrics_rows += 1
        if context.is_main:
            append_jsonl(
                metrics_path,
                {
                    "optimizer_update": task_visits // world,
                    "task_visits": task_visits,
                    "mean_functional_loss": sum(float(row["loss"]) for row in records)
                    / world,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "records": records,
                    "elapsed_seconds": time.monotonic() - runtime.started,
                    "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                },
            )
        if task_visits in checkpoints:
            _save_checkpoint(runtime, task_visits=task_visits, metrics_rows=metrics_rows)
    return metrics_rows


def _member_code(runtime: Runtime, member_index: int) -> torch.Tensor:
    if runtime.decoder_mode == "stable_shared_prior":
        return torch.zeros(
            runtime.decoder.code_width,
            device=runtime.context.device,
        )
    member = runtime.code_authority.members[member_index]
    if member.fold_role == "held_transform_only":
        return runtime.code_authority.member_codes[member_index]
    task_index = runtime.code_authority.fit_ordinals.index(member.ordinal)
    return runtime.code_authority.fit_task_codes[task_index]


def _evaluate_members(runtime: Runtime) -> list[dict[str, Any]]:
    local = []
    seed = int(runtime.config["functional_supervision"]["evaluation_policy_seed"])
    for member_index in range(
        runtime.context.rank,
        len(runtime.code_authority.members),
        runtime.context.world_size,
    ):
        panels = cache_phase_member_panels(
            policy=runtime.policy,
            identity_state=runtime.identity_state,
            contract=runtime.contract,
            sources=runtime.member_sources,
            member_index=member_index,
            device=runtime.context.device,
            policy_seed=seed,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = [
                float(
                    functional_response_distillation_loss(
                        runtime.policy,
                        runtime.decoder(_member_code(runtime, member_index)),
                        runtime.contract,
                        panel.batch,
                        panel.target,
                        policy_seed=panel.policy_seed,
                    )
                )
                for panel in panels
            ]
        member = runtime.code_authority.members[member_index]
        local.append(
            {
                "member_index": member_index,
                "suite": member.suite,
                "task_id": member.task_id,
                "global_task_id": member.global_task_id,
                "ordinal": member.ordinal,
                "fold_role": member.fold_role,
                "member": member.member,
                "expert_step": member.expert_step,
                "panel_losses": losses,
                "mean_loss": sum(losses) / len(losses),
            }
        )
    shards: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(shards, local)
    return sorted(
        (row for shard in shards for row in shard),
        key=lambda row: int(row["member_index"]),
    )


def _evaluate_onpolicy_members(runtime: Runtime) -> list[dict[str, Any]]:
    if not runtime.onpolicy_panels:
        return []
    local = []
    fit_indices = sorted(runtime.onpolicy_panels)
    for member_index in fit_indices[runtime.context.rank :: runtime.context.world_size]:
        panels = runtime.onpolicy_panels[member_index]
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            losses = [
                float(
                    functional_response_distillation_loss(
                        runtime.policy,
                        runtime.decoder(_member_code(runtime, member_index)),
                        runtime.contract,
                        panel.batch,
                        panel.target,
                        policy_seed=panel.policy_seed,
                    )
                )
                for panel in panels
            ]
        member = runtime.code_authority.members[member_index]
        local.append(
            {
                "member_index": member_index,
                "ordinal": member.ordinal,
                "global_task_id": member.global_task_id,
                "member": member.member,
                "projected_rollout_success": runtime.occupancy_sources[
                    member_index
                ].success,
                "panel_losses": losses,
                "mean_loss": sum(losses) / len(losses),
            }
        )
    shards: list[Any] = [None] * runtime.context.world_size
    dist.all_gather_object(shards, local)
    return sorted(
        (row for shard in shards for row in shard),
        key=lambda row: int(row["member_index"]),
    )


def _functional_gate(runtime: Runtime, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held = [row for row in rows if row["fold_role"] == "held_transform_only"]
    families = {}
    threshold = float(
        runtime.config["decision_gates"]["functional"][
            "both_held_member_families_mean_loss_below"
        ]
    )
    minimum_tasks = int(
        runtime.config["decision_gates"]["functional"][
            "minimum_tasks_below_identity_per_member_family"
        ]
    )
    for family in ("earliest", "latest"):
        selected = [row for row in held if row["member"] == family]
        mean = sum(float(row["mean_loss"]) for row in selected) / len(selected)
        below = sum(float(row["mean_loss"]) < 1.0 for row in selected)
        families[family] = {
            "mean_loss": mean,
            "tasks_below_identity": below,
            "passes": mean < threshold and below >= minimum_tasks,
        }
    return {
        "identity_relative_loss": 1.0,
        "families": families,
        "passes": all(row["passes"] for row in families.values()),
        "closed_loop_still_required": True,
    }


def _result(
    runtime: Runtime,
    *,
    rows: Sequence[Mapping[str, Any]],
    functional_gate: Mapping[str, Any],
    metrics_rows: int,
    decoder_path: Path,
    shared_prior_path: Path | None,
    initial_onpolicy_rows: Sequence[Mapping[str, Any]],
    final_onpolicy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_visits = int(runtime.config["training"]["total_task_visits"])
    checkpoint = _checkpoint_path(runtime.args.output_dir, final_visits)
    return {
        "schema_version": RESULT_SCHEMA,
        "formal_authority": True,
        "decoder_mode": runtime.decoder_mode,
        "repository": runtime.repository,
        "config": phase_decoder_asset(runtime.args.config),
        "code_artifact": {
            "root": str(runtime.code_authority.root),
            "result": phase_decoder_asset(runtime.code_authority.root / "result.json"),
            "codes": phase_decoder_asset(runtime.code_authority.root / "phase_codes.safetensors"),
        },
        "decoder": phase_decoder_asset(decoder_path),
        "shared_prior": (
            phase_decoder_asset(shared_prior_path)
            if shared_prior_path is not None
            else dict(runtime.shared_prior_asset["adapter"])
            if runtime.shared_prior_asset is not None
            else None
        ),
        "shared_prior_authority": runtime.shared_prior_asset,
        "parameterization": {
            "template": runtime.config["decoder"].get("template"),
            "active_rank": [
                runtime.decoder.active_rank_start,
                runtime.decoder.active_rank_end,
            ],
            "rank_composition": (
                "exact_disjoint_rank_concatenation_shared12_task4"
                if runtime.decoder_mode
                in {"stable_shared_prior", "shared_prior_residual"}
                else None
            ),
            "residual_centered_at_zero_code": bool(
                runtime.config["decoder"].get(
                    "center_residual_at_zero_code", False
                )
            ),
            "single_complete_lora": True,
            "second_adapter_deployed": False,
        },
        "final_exact_resume_checkpoint": phase_decoder_asset(checkpoint),
        "training": {
            "task_visits": final_visits,
            "optimizer_updates": final_visits // runtime.context.world_size,
            "world_size": runtime.context.world_size,
            "metrics_rows": metrics_rows,
            "elapsed_seconds": time.monotonic() - runtime.started,
        },
        "functional_evaluation": {
            "member_rows": list(rows),
            "fit_mean": sum(
                float(row["mean_loss"]) for row in rows if row["fold_role"] == "fit"
            )
            / sum(row["fold_role"] == "fit" for row in rows),
            "held_mean": sum(
                float(row["mean_loss"])
                for row in rows
                if row["fold_role"] == "held_transform_only"
            )
            / sum(row["fold_role"] == "held_transform_only" for row in rows),
            "gate": dict(functional_gate),
        },
        "state_aggregation_evaluation": (
            {
                "member_rows": list(final_onpolicy_rows),
                "initial_mean_loss": sum(
                    float(row["mean_loss"]) for row in initial_onpolicy_rows
                )
                / len(initial_onpolicy_rows),
                "final_mean_loss": sum(
                    float(row["mean_loss"]) for row in final_onpolicy_rows
                )
                / len(final_onpolicy_rows),
                "captured_trajectories": len(
                    {
                        source.trajectory_path
                        for source in runtime.occupancy_sources.values()
                    }
                ),
                "member_targets": len(final_onpolicy_rows),
            }
            if final_onpolicy_rows
            else None
        ),
        "information_wall": {
            "fit_task_gradients": 19,
            "held_task_gradients": 0,
            "validation_reads": 0,
            "test_reads": 0,
            "final_lora_averaging": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    args.code_artifact = args.code_artifact.resolve()
    args.source_run = args.source_run.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.expert_bank_root = args.expert_bank_root.resolve()
    if args.state_bank_root is not None:
        args.state_bank_root = args.state_bank_root.resolve()
    if args.shared_prior_root is not None:
        args.shared_prior_root = args.shared_prior_root.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    runtime = _prepare(args)
    _publish_contract(runtime)
    initial_onpolicy_rows = _evaluate_onpolicy_members(runtime)
    start_visits, metrics_rows = _resume(runtime)
    metrics_rows = _fit(
        runtime, start_visits=start_visits, metrics_rows=metrics_rows
    )
    rows = _evaluate_members(runtime)
    final_onpolicy_rows = _evaluate_onpolicy_members(runtime)
    gate = _functional_gate(runtime, rows)
    if runtime.context.is_main:
        decoder_path = save_phase_decoder(runtime.decoder, runtime.args.output_dir)
        shared_prior_path = (
            save_stable_shared_prior(
                runtime.decoder,
                runtime.contract,
                runtime.args.output_dir,
            )
            if runtime.decoder_mode == "stable_shared_prior"
            else None
        )
        result = _result(
            runtime,
            rows=rows,
            functional_gate=gate,
            metrics_rows=metrics_rows,
            decoder_path=decoder_path,
            shared_prior_path=shared_prior_path,
            initial_onpolicy_rows=initial_onpolicy_rows,
            final_onpolicy_rows=final_onpolicy_rows,
        )
        write_json_atomic(runtime.args.output_dir / "result.json", result)
        materialize_phase_decoder_projections(
            config_path=runtime.args.config,
            task_expert_config_path=_authority_path(runtime.config, "task_experts"),
            codes=runtime.code_authority,
            member_sources=runtime.member_sources,
            decoder=runtime.decoder,
            contract=runtime.contract,
            repository=runtime.repository,
            source=runtime.source,
            expert_bank_root=runtime.args.expert_bank_root,
            output_dir=runtime.args.output_dir,
            functional_rows=rows,
            decoder_path=decoder_path,
            decoder_mode=runtime.decoder_mode,
            shared_adapter_path=shared_prior_path,
            shared_prior_authority=runtime.shared_prior_asset,
        )
        print(
            json.dumps(
                {
                    "event": "complete",
                    "functional_gate": gate,
                    "output_dir": str(runtime.args.output_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(device_ids=[runtime.context.local_rank])
    dist.destroy_process_group()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--code-artifact", type=Path, required=True)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("formal",), required=True)
    result.add_argument("--state-bank-root", type=Path)
    result.add_argument("--shared-prior-root", type=Path)
    result.add_argument("--resume", type=Path)
    return result
