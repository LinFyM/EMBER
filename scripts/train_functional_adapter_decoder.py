#!/usr/bin/env python3
"""Train the canonical fixed decoder on complete PI0.5 Action flow responses."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import (
    build_dataset,
    load_task_expert_config,
    load_train_tasks,
)
from ember.functional_adaptation.decoder_training import (
    FunctionalDecoderSystem,
    authority_path,
    balanced_task_order,
    decoder_task_split,
    expert_records,
    inspect_nonheld_meta_expert_bank,
    inspect_train24_expert_bank,
    load_expert_states,
    load_functional_adapter_config,
    meta_decoder_task_split,
)
from ember.functional_adaptation.decoder_flow_checkpoint import (
    RUN_SCHEMA,
    DecoderFlowCursor,
    inspect_decoder_flow_checkpoint,
    load_decoder_flow_checkpoint,
    save_decoder_flow_checkpoint,
)
from ember.functional_adaptation.fingerprint_codes import (
    FunctionalFingerprintCodeTargets,
    load_functional_fingerprint_code_targets,
)
from ember.functional_adaptation.probe_panels import (
    FunctionalProbePanel,
    build_probe_panels,
    mean_functional_probe_loss,
    panel_for_visit,
    selected_probe_rows,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import load_policy, load_stats
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FlowProfile:
    args: argparse.Namespace
    config: dict[str, Any]
    mechanism: Mapping[str, Any]
    flow: Mapping[str, Any]
    schedule: Mapping[str, Any]
    bank: Mapping[str, Any]
    active_fit: Sequence[Any]
    active_held: Sequence[Any]
    contract: Any
    system: FunctionalDecoderSystem
    held_codes: torch.nn.Parameter
    dataset: Any
    fit_states: tuple[dict[str, torch.Tensor], ...]
    held_states: tuple[dict[str, torch.Tensor], ...]
    policy: torch.nn.Module
    identity_state: Mapping[str, torch.Tensor]
    processor: Pi05LiberoProcessor
    started: float
    metrics_rows: int
    fixed_code_authority: FunctionalFingerprintCodeTargets | None


@dataclass(frozen=True)
class FlowPanels:
    fit_train: tuple[tuple[FunctionalProbePanel, ...], ...]
    fit_eval: tuple[tuple[FunctionalProbePanel, ...], ...]
    held_train: tuple[tuple[FunctionalProbePanel, ...], ...]
    held_eval: tuple[tuple[FunctionalProbePanel, ...], ...]


def _demo_range(values: Sequence[int]) -> tuple[int, ...]:
    first, last = map(int, values)
    return tuple(range(first, last + 1))


def _checkpoint_steps(schedule: Mapping[str, Any], phase: str) -> tuple[int, ...]:
    total = int(
        schedule["decoder_steps" if phase == "decoder" else "held_code_steps"]
    )
    values = tuple(int(value) for value in schedule["checkpoint_steps"][phase])
    if total == 0 and not values:
        return ()
    if (
        not values
        or values != tuple(sorted(set(values)))
        or values[-1] != total
        or any(not 0 < value <= total for value in values)
    ):
        raise ValueError("functional-decoder checkpoint schedule changed")
    return values


def _run_contract(runtime: FlowProfile) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
    return {
        "schema_version": RUN_SCHEMA,
        "mode": runtime.args.mode,
        "surface": runtime.args.surface,
        "repository": {
            "branch": repository["branch"],
            "commit": repository["commit"],
            "authority_ref": repository["authority_ref"],
            "authority_contains_commit": repository["authority_contains_commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "host": socket.gethostname(),
        "runtime": {
            "device": str(runtime.args.device),
            "world_size": 1,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "exact_resume_topology_locked": True,
        },
        "inputs": {
            "config": str(runtime.args.config.resolve()),
            "source_run": str(runtime.args.source_run.resolve()),
            "checkpoint": str(runtime.args.checkpoint.resolve()),
            "expert_bank_root": str(runtime.args.expert_bank_root.resolve()),
            "effective_decoder": (
                None
                if runtime.args.effective_decoder is None
                else str(runtime.args.effective_decoder.resolve())
            ),
            "functional_code_artifact": (
                None
                if runtime.fixed_code_authority is None
                else str(runtime.fixed_code_authority.root)
            ),
            "tokenizer": str(runtime.args.tokenizer_path.resolve()),
            "data_root": str(runtime.args.data_root.resolve()),
        },
        "tasks": {
            "fit_global_task_ids": [row.global_task_id for row in runtime.active_fit],
            "held_global_task_ids": [row.global_task_id for row in runtime.active_held],
            "fit_count": len(runtime.active_fit),
            "held_count": len(runtime.active_held),
            "task_equal_decoder_order": True,
            "task_equal_held_code_order": runtime.fixed_code_authority is None,
        },
        "schedule": dict(runtime.schedule),
        "information_wall": {
            "privileged_task_codes": "train/meta decoder fitting only",
            "target40_action_or_reward_reads": 0,
            "decoder_frozen_after_fit": True,
            "code_source": (
                "learned_codebook_plus_free_held_codes"
                if runtime.fixed_code_authority is None
                else "unified_policy_functional_fingerprint"
            ),
            "deployment_task_id_route": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _publish_run_contract(runtime: FlowProfile) -> dict[str, Any]:
    contract = _run_contract(runtime)
    root = runtime.args.output_dir.resolve()
    if runtime.args.resume is None:
        if root.exists() and any(root.iterdir()):
            raise ValueError("fresh functional-decoder output directory is not empty")
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(root / "run_contract.json", contract)
    elif (
        runtime.args.resume.resolve().parent.parent != root
        or not (root / "run_contract.json").is_file()
        or read_json(root / "run_contract.json") != contract
    ):
        raise ValueError("functional-decoder exact-resume contract changed")
    append_jsonl(
        root / "invocations.jsonl",
        {
            "argv": sys.argv,
            "started_unix": time.time(),
            "resume": str(runtime.args.resume) if runtime.args.resume else None,
        },
    )
    return contract


def _cache_task_panels(
    *,
    policy: torch.nn.Module,
    processor: Pi05LiberoProcessor,
    dataset: Any,
    records: Sequence[Any],
    expert_states: Sequence[Mapping[str, torch.Tensor]],
    identity_state: Mapping[str, torch.Tensor],
    contract: Any,
    demos: Sequence[int],
    panel_count: int,
    panel_batch_size: int,
    base_seed: int,
) -> tuple[tuple[FunctionalProbePanel, ...], ...]:
    result = []
    for record, expert_state in zip(records, expert_states, strict=True):
        rows = selected_probe_rows(
            dataset.task_episode_rows[record.global_task_id],
            demo_indices=demos,
            panel_count=panel_count,
            batch_size=panel_batch_size,
            seed=base_seed + record.global_task_id * 1009,
        )
        result.append(
            build_probe_panels(
                policy=policy,
                processor=processor,
                dataset=dataset,
                rows=rows,
                identity_state=identity_state,
                expert_state=expert_state,
                contract=contract,
                policy_seed=base_seed + record.global_task_id * 9173,
            )
        )
    return tuple(result)


def _panel_losses(
    *,
    policy: torch.nn.Module,
    decoder: torch.nn.Module,
    codes: torch.Tensor,
    contract: Any,
    panels: Sequence[Sequence[FunctionalProbePanel]],
) -> list[float]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return [
            float(
                mean_functional_probe_loss(
                    policy,
                    decoder(codes[index]),
                    contract,
                    task_panels,
                ).item()
            )
            for index, task_panels in enumerate(panels)
        ]


def _prepare(args: argparse.Namespace) -> FlowProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("PI0.5 flow-response profile requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    mechanism = config[
        "production_meta" if args.surface == "nonheld_meta" else "train24_mechanism"
    ]
    flow = mechanism["flow_response"]
    fixed_code_requested = args.functional_code_artifact is not None
    if args.surface == "nonheld_meta" and not fixed_code_requested:
        raise ValueError(
            "non-held meta decoder requires unified functional fingerprint codes"
        )
    if args.surface != "nonheld_meta" and fixed_code_requested:
        raise ValueError("functional fingerprints only cover non-held meta tasks")
    schedule = flow[args.mode]
    _checkpoint_steps(schedule, "decoder")
    _checkpoint_steps(schedule, "held_code")
    repository = git_state(REPO_ROOT)
    if args.mode == "formal" and not git_state_is_clean_pushed_or_frozen_authority(
        repository
    ):
        raise ValueError(
            "formal functional-decoder training requires a clean pushed authority"
        )
    if args.surface == "nonheld_meta":
        bank = inspect_nonheld_meta_expert_bank(
            config,
            REPO_ROOT,
            source_run=args.source_run,
            checkpoint=args.checkpoint,
            bank_root=args.expert_bank_root,
        )
        split = meta_decoder_task_split(expert_records(bank))
        expert_config_name = "meta_experts"
        code_width = int(config["decoder"]["production_code_width"])
    else:
        bank = inspect_train24_expert_bank(
            config,
            REPO_ROOT,
            source_run=args.source_run,
            checkpoint=args.checkpoint,
            bank_root=args.expert_bank_root,
        )
        split = decoder_task_split(
            expert_records(bank),
            fold_count=int(mechanism["fold_count"]),
            held_out_fold=int(mechanism["held_out_fold"]),
        )
        expert_config_name = "train24_experts"
        code_width = int(config["decoder"]["train24_smoke_code_width"])
    fit_count = int(schedule["active_fit_tasks"])
    held_count = int(schedule["active_held_tasks"])
    active_fit = split.fit[:fit_count]
    active_held = split.held[:held_count]
    if args.mode == "formal" and (
        len(active_fit) != len(split.fit) or len(active_held) != len(split.held)
    ):
        raise ValueError("formal functional-decoder training requires the full split")
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    decoder_config = config["decoder"]
    system = FunctionalDecoderSystem(
        contract,
        identity_lora_state(contract, device=device),
        task_count=len(split.fit),
        code_width=code_width,
        address_width=int(decoder_config["address_width"]),
        hidden_width=int(decoder_config["hidden_width"]),
        seed=int(decoder_config["initialization_seed"]),
    ).to(device)
    fixed_code_authority: FunctionalFingerprintCodeTargets | None = None
    if fixed_code_requested:
        fixed_code_authority = load_functional_fingerprint_code_targets(
            args.functional_code_artifact,
            expected_train_task_ids=tuple(row.global_task_id for row in split.fit),
            expected_held_task_ids=tuple(row.global_task_id for row in split.held),
            code_width=code_width,
            device=device,
        )
        with torch.no_grad():
            system.codebook.weight.copy_(fixed_code_authority.train_codes)
        system.codebook.requires_grad_(False)
    if args.surface == "train24":
        if args.effective_decoder is None:
            raise ValueError("train24 flow fitting requires its effective decoder")
        warmstart = load_file(str(args.effective_decoder), device=str(device))
        system.load_state_dict(warmstart, strict=True)
    system.requires_grad_(True)
    if fixed_code_authority is not None:
        system.codebook.requires_grad_(False)
    if fixed_code_authority is not None:
        held_codes = torch.nn.Parameter(
            fixed_code_authority.held_codes.clone(), requires_grad=False
        )
    elif args.surface == "train24":
        effective_held_codes = load_file(
            str(args.effective_decoder.parent / "held_codes.safetensors"),
            device=str(device),
        )["held_codes"]
        held_codes = torch.nn.Parameter(effective_held_codes.clone())
    else:
        held_codes = torch.nn.Parameter(
            torch.zeros(len(split.held), code_width, device=device)
        )

    expert_config = load_task_expert_config(
        authority_path(config, expert_config_name, REPO_ROOT)
    )
    all_tasks = load_train_tasks(expert_config, args.data_root)
    active_ids = {row.global_task_id for row in (*active_fit, *active_held)}
    selected_tasks = tuple(
        task for task in all_tasks if task.global_task_id in active_ids
    )
    dataset = build_dataset(expert_config, selected_tasks)
    fit_states = load_expert_states(active_fit, contract, device)
    held_states = load_expert_states(active_held, contract, device)
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", REPO_ROOT), REPO_ROOT
    )
    policy = load_policy(
        Path(str(bank["source"]["model_path"])),
        authorities.source_base_config,
        device,
    )
    identity_state = prepare_frozen_writer_policy(policy, contract)
    stats = load_stats(
        authorities.source_base_config,
        authorities.source_base_config["data"]["active_task_ids"],
    )
    processor = Pi05LiberoProcessor(
        stats,
        args.tokenizer_path,
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    return FlowProfile(
        args=args,
        config=config,
        mechanism=mechanism,
        flow=flow,
        schedule=schedule,
        bank=bank,
        active_fit=active_fit,
        active_held=active_held,
        contract=contract,
        system=system,
        held_codes=held_codes,
        dataset=dataset,
        fit_states=fit_states,
        held_states=held_states,
        policy=policy,
        identity_state=identity_state,
        processor=processor,
        started=time.monotonic(),
        metrics_rows=0,
        fixed_code_authority=fixed_code_authority,
    )


def _cache_all_panels(runtime: FlowProfile) -> FlowPanels:
    count = int(runtime.schedule["panel_count"])
    batch_size = int(runtime.schedule["panel_batch_size"])
    seed = int(runtime.flow["policy_seed"])

    def cache(records, states, demos, offset):
        return _cache_task_panels(
            policy=runtime.policy,
            processor=runtime.processor,
            dataset=runtime.dataset,
            records=records,
            expert_states=states,
            identity_state=runtime.identity_state,
            contract=runtime.contract,
            demos=_demo_range(demos),
            panel_count=count,
            panel_batch_size=batch_size,
            base_seed=seed + offset,
        )

    return FlowPanels(
        fit_train=cache(
            runtime.active_fit,
            runtime.fit_states,
            runtime.flow["fit_demo_indices"],
            0,
        ),
        fit_eval=cache(
            runtime.active_fit,
            runtime.fit_states,
            runtime.flow["evaluation_demo_indices"],
            1_000_000,
        ),
        held_train=cache(
            runtime.active_held,
            runtime.held_states,
            runtime.flow["fit_demo_indices"],
            2_000_000,
        ),
        held_eval=cache(
            runtime.active_held,
            runtime.held_states,
            runtime.flow["evaluation_demo_indices"],
            3_000_000,
        ),
    )


def _initial_losses(
    runtime: FlowProfile, panels: FlowPanels
) -> tuple[list[float], list[float]]:
    fit = _panel_losses(
        policy=runtime.policy,
        decoder=runtime.system.decoder,
        codes=runtime.system.codebook.weight[: len(runtime.active_fit)],
        contract=runtime.contract,
        panels=panels.fit_eval,
    )
    held = _panel_losses(
        policy=runtime.policy,
        decoder=runtime.system.decoder,
        codes=runtime.held_codes[: len(runtime.active_held)],
        contract=runtime.contract,
        panels=panels.held_eval,
    )
    return fit, held


def _decoder_optimizer(runtime: FlowProfile) -> torch.optim.Optimizer:
    optimizer_config = runtime.flow["optimizer"]
    runtime.system.requires_grad_(True)
    if runtime.fixed_code_authority is not None:
        runtime.system.codebook.requires_grad_(False)
    return torch.optim.AdamW(
        (value for value in runtime.system.parameters() if value.requires_grad),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def _held_optimizer(runtime: FlowProfile) -> torch.optim.Optimizer:
    runtime.system.requires_grad_(False)
    runtime.held_codes.requires_grad_(True)
    return torch.optim.Adam(
        (runtime.held_codes,),
        lr=float(runtime.flow["held_code_optimizer"]["learning_rate"]),
    )


def _fit_decoder(
    runtime: FlowProfile,
    panels: FlowPanels,
    metrics_path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    start_step: int,
    visits: list[int],
) -> None:
    optimizer_config = runtime.flow["optimizer"]
    order = balanced_task_order(
        len(runtime.active_fit),
        int(runtime.schedule["decoder_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]),
    )
    if len(visits) != len(runtime.active_fit) or sum(visits) != start_step:
        raise ValueError("functional-decoder fit cursor changed")
    checkpoints = set(_checkpoint_steps(runtime.schedule, "decoder"))
    for step, task_index in enumerate(order[start_step:], start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            response_loss = mean_functional_probe_loss(
                runtime.policy,
                runtime.system(task_index),
                runtime.contract,
                panel_for_visit(
                    panels.fit_train[task_index], visits[task_index]
                ),
            )
            gauge_loss = runtime.system.codebook.gauge_loss()
            loss = response_loss + float(
                runtime.mechanism["objective"]["codebook_gauge_weight"]
            ) * gauge_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runtime.system.parameters(), float(optimizer_config["gradient_clip_norm"])
        )
        optimizer.step()
        visits[task_index] += 1
        append_jsonl(
            metrics_path,
            {
                "phase": "flow_decoder_fit",
                "step": step,
                "task_index": task_index,
                "flow_response_loss": float(response_loss.detach()),
                "gauge_loss": float(gauge_loss.detach()),
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - runtime.started,
                "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            },
        )
        runtime.metrics_rows += 1
        if step in checkpoints:
            save_decoder_flow_checkpoint(
                output_dir=runtime.args.output_dir,
                phase="decoder",
                step=step,
                metrics_rows=runtime.metrics_rows,
                visits=visits,
                system=runtime.system,
                held_codes=runtime.held_codes,
                optimizer=optimizer,
            )


def _fit_held_codes(
    runtime: FlowProfile,
    panels: FlowPanels,
    metrics_path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    start_step: int,
    visits: list[int],
) -> None:
    held_optimizer_config = runtime.flow["held_code_optimizer"]
    held_order = balanced_task_order(
        len(runtime.active_held),
        int(runtime.schedule["held_code_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]) + 1,
    )
    if len(visits) != len(runtime.active_held) or sum(visits) != start_step:
        raise ValueError("functional-decoder held-code cursor changed")
    checkpoints = set(_checkpoint_steps(runtime.schedule, "held_code"))
    for step, task_index in enumerate(
        held_order[start_step:], start=start_step + 1
    ):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            response_loss = mean_functional_probe_loss(
                runtime.policy,
                runtime.system.decoder(runtime.held_codes[task_index]),
                runtime.contract,
                panel_for_visit(
                    panels.held_train[task_index], visits[task_index]
                ),
            )
            code_l2 = runtime.held_codes[task_index].square().mean()
            loss = response_loss + float(
                held_optimizer_config["code_l2_weight"]
            ) * code_l2
        loss.backward()
        optimizer.step()
        visits[task_index] += 1
        append_jsonl(
            metrics_path,
            {
                "phase": "flow_held_code_fit",
                "step": step,
                "task_index": task_index,
                "flow_response_loss": float(response_loss.detach()),
                "code_l2": float(code_l2.detach()),
                "elapsed_seconds": time.monotonic() - runtime.started,
                "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            },
        )
        runtime.metrics_rows += 1
        if step in checkpoints:
            save_decoder_flow_checkpoint(
                output_dir=runtime.args.output_dir,
                phase="held_code",
                step=step,
                metrics_rows=runtime.metrics_rows,
                visits=visits,
                system=runtime.system,
                held_codes=runtime.held_codes,
                optimizer=optimizer,
            )



def _write_result(
    runtime: FlowProfile,
    panels: FlowPanels,
    initial_fit: list[float],
    initial_held: list[float],
) -> None:
    final_fit = _panel_losses(
        policy=runtime.policy,
        decoder=runtime.system.decoder,
        codes=runtime.system.codebook.weight[: len(runtime.active_fit)],
        contract=runtime.contract,
        panels=panels.fit_eval,
    )
    final_held = _panel_losses(
        policy=runtime.policy,
        decoder=runtime.system.decoder,
        codes=runtime.held_codes[: len(runtime.active_held)],
        contract=runtime.contract,
        panels=panels.held_eval,
    )
    if (runtime.args.output_dir / "result.json").exists():
        raise ValueError("functional-decoder result already exists")
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in runtime.system.state_dict().items()
        },
        str(runtime.args.output_dir / "decoder.safetensors"),
    )
    save_file(
        {"held_codes": runtime.held_codes.detach().cpu().contiguous()},
        str(runtime.args.output_dir / "held_codes.safetensors"),
    )
    repository = git_state(REPO_ROOT)
    held_steps = int(runtime.schedule["held_code_steps"])
    final_phase = "held_code" if held_steps else "decoder"
    final_step = held_steps or int(runtime.schedule["decoder_steps"])
    final_checkpoint = (
        runtime.args.output_dir
        / "checkpoints"
        / f"{final_phase}_step_{final_step:08d}"
    )
    run_contract_path = runtime.args.output_dir / "run_contract.json"
    if not final_checkpoint.is_dir() or not run_contract_path.is_file():
        raise ValueError("functional-decoder final authority is incomplete")
    write_json_atomic(
        runtime.args.output_dir / "result.json",
        {
            "schema_version": "ember_pi05_functional_flow_profile_v1",
            "mode": runtime.args.mode,
            "surface": runtime.args.surface,
            "formal_authority": runtime.args.mode == "formal",
            "repository": {
                "branch": repository["branch"],
                "commit": repository["commit"],
                "dirty_paths": repository["dirty_paths"],
            },
            "active_fit_ordinals": [row.ordinal for row in runtime.active_fit],
            "active_held_ordinals": [row.ordinal for row in runtime.active_held],
            "active_fit_global_task_ids": [
                row.global_task_id for row in runtime.active_fit
            ],
            "active_held_global_task_ids": [
                row.global_task_id for row in runtime.active_held
            ],
            "initial_fit_mean": sum(initial_fit) / len(initial_fit),
            "initial_fit_per_task": initial_fit,
            "final_fit_mean": sum(final_fit) / len(final_fit),
            "final_fit_per_task": final_fit,
            "initial_held_mean": sum(initial_held) / len(initial_held),
            "initial_held_per_task": initial_held,
            "final_held_mean": sum(final_held) / len(final_held),
            "final_held_per_task": final_held,
            "steps": {
                "decoder": int(runtime.schedule["decoder_steps"]),
                "held_code": int(runtime.schedule["held_code_steps"]),
            },
            "code_source": (
                "learned_codebook_plus_free_held_codes"
                if runtime.fixed_code_authority is None
                else "unified_policy_functional_fingerprint"
            ),
            "functional_code_artifact": (
                None
                if runtime.fixed_code_authority is None
                else {
                    "root": str(runtime.fixed_code_authority.root),
                    "result_bytes": (
                        runtime.fixed_code_authority.root / "result.json"
                    ).stat().st_size,
                    "codes_bytes": (
                        runtime.fixed_code_authority.root
                        / "fingerprint_codes.safetensors"
                    ).stat().st_size,
                }
            ),
            "metrics_rows": runtime.metrics_rows,
            "run_contract": {
                "path": str(run_contract_path.resolve()),
                "bytes": run_contract_path.stat().st_size,
                "schema_version": RUN_SCHEMA,
            },
            "final_exact_resume_checkpoint": str(final_checkpoint.resolve()),
            "probe_panel": {
                "fit_demos": list(_demo_range(runtime.flow["fit_demo_indices"])),
                "evaluation_demos": list(
                    _demo_range(runtime.flow["evaluation_demo_indices"])
                ),
                "panel_count": int(runtime.schedule["panel_count"]),
                "batch_size": int(runtime.schedule["panel_batch_size"]),
                "full_action_tokens": 50,
            },
            "elapsed_seconds": time.monotonic() - runtime.started,
            "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "content_hash_policy": "disabled_by_owner",
        },
    )
    runtime.dataset.close()


def _load_or_create_initial_losses(
    runtime: FlowProfile, panels: FlowPanels
) -> tuple[list[float], list[float]]:
    path = runtime.args.output_dir / "initial_losses.json"
    if runtime.args.resume is None:
        fit, held = _initial_losses(runtime, panels)
        write_json_atomic(path, {"fit": fit, "held": held})
        return fit, held
    observed = read_json(path)
    fit = [float(value) for value in observed.get("fit", ())]
    held = [float(value) for value in observed.get("held", ())]
    if len(fit) != len(runtime.active_fit) or len(held) != len(runtime.active_held):
        raise ValueError("functional-decoder initial-loss evidence changed")
    return fit, held


def _rewind_metrics(path: Path, rows: int) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < rows:
        raise ValueError("functional-decoder metrics precede the resume cursor")
    if len(lines) > rows:
        temporary = path.with_name(f".{path.name}.resume-{os.getpid()}")
        temporary.write_bytes(b"".join(lines[:rows]))
        os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if args.functional_code_artifact is not None:
        args.functional_code_artifact = args.functional_code_artifact.resolve()
    runtime = _prepare(args)
    _publish_run_contract(runtime)
    panels = _cache_all_panels(runtime)
    initial_fit, initial_held = _load_or_create_initial_losses(runtime, panels)
    metrics_path = args.output_dir / "metrics.jsonl"
    resume = (
        None
        if args.resume is None
        else inspect_decoder_flow_checkpoint(args.resume)
    )
    if resume is not None:
        _rewind_metrics(metrics_path, resume.metrics_rows)

    if resume is None or resume.phase == "decoder":
        decoder_optimizer = _decoder_optimizer(runtime)
        decoder_cursor = (
            DecoderFlowCursor(
                phase="decoder",
                step=0,
                metrics_rows=0,
                visits=tuple(0 for _ in runtime.active_fit),
            )
            if resume is None
            else load_decoder_flow_checkpoint(
                checkpoint=args.resume,
                expected_phase="decoder",
                system=runtime.system,
                held_codes=runtime.held_codes,
                optimizer=decoder_optimizer,
            )
        )
        runtime.metrics_rows = decoder_cursor.metrics_rows
        _fit_decoder(
            runtime,
            panels,
            metrics_path,
            optimizer=decoder_optimizer,
            start_step=decoder_cursor.step,
            visits=list(decoder_cursor.visits),
        )

    if int(runtime.schedule["held_code_steps"]):
        held_optimizer = _held_optimizer(runtime)
        held_cursor = (
            load_decoder_flow_checkpoint(
                checkpoint=args.resume,
                expected_phase="held_code",
                system=runtime.system,
                held_codes=runtime.held_codes,
                optimizer=held_optimizer,
            )
            if resume is not None and resume.phase == "held_code"
            else DecoderFlowCursor(
                phase="held_code",
                step=0,
                metrics_rows=runtime.metrics_rows,
                visits=tuple(0 for _ in runtime.active_held),
            )
        )
        runtime.metrics_rows = held_cursor.metrics_rows
        _fit_held_codes(
            runtime,
            panels,
            metrics_path,
            optimizer=held_optimizer,
            start_step=held_cursor.step,
            visits=list(held_cursor.visits),
        )
    _write_result(runtime, panels, initial_fit, initial_held)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    result.add_argument(
        "--mode", choices=("smoke", "profile", "formal"), required=True
    )
    result.add_argument(
        "--surface", choices=("train24", "nonheld_meta"), default="train24"
    )
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--effective-decoder", type=Path)
    result.add_argument("--functional-code-artifact", type=Path)
    result.add_argument("--tokenizer-path", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--resume", type=Path)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
