#!/usr/bin/env python3
"""Profile fixed-decoder fitting on complete PI0.5 Action flow responses."""

from __future__ import annotations

import argparse
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
    inspect_train24_expert_bank,
    load_expert_states,
    load_functional_adapter_config,
)
from ember.functional_adaptation.probe_panels import (
    FunctionalProbePanel,
    build_probe_panels,
    mean_functional_probe_loss,
    panel_for_visit,
    selected_probe_rows,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import git_state, load_evaluation_authorities
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import write_json_atomic
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


@dataclass(frozen=True)
class FlowPanels:
    fit_train: tuple[tuple[FunctionalProbePanel, ...], ...]
    fit_eval: tuple[tuple[FunctionalProbePanel, ...], ...]
    held_train: tuple[tuple[FunctionalProbePanel, ...], ...]
    held_eval: tuple[tuple[FunctionalProbePanel, ...], ...]


def _demo_range(values: Sequence[int]) -> tuple[int, ...]:
    first, last = map(int, values)
    return tuple(range(first, last + 1))


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
    mechanism = config["train24_mechanism"]
    flow = mechanism["flow_response"]
    schedule = flow[args.mode]
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
    fit_count = int(schedule["active_fit_tasks"])
    held_count = int(schedule["active_held_tasks"])
    active_fit = split.fit[:fit_count]
    active_held = split.held[:held_count]
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    decoder_config = config["decoder"]
    system = FunctionalDecoderSystem(
        contract,
        identity_lora_state(contract, device=device),
        task_count=len(split.fit),
        code_width=int(decoder_config["train24_smoke_code_width"]),
        address_width=int(decoder_config["address_width"]),
        hidden_width=int(decoder_config["hidden_width"]),
        seed=int(decoder_config["initialization_seed"]),
    ).to(device)
    warmstart = load_file(str(args.effective_decoder), device=str(device))
    system.load_state_dict(warmstart, strict=True)
    system.requires_grad_(True)
    effective_held_codes = load_file(
        str(args.effective_decoder.parent / "held_codes.safetensors"),
        device=str(device),
    )["held_codes"]
    held_codes = torch.nn.Parameter(effective_held_codes.clone())

    expert_config = load_task_expert_config(
        authority_path(config, "train24_experts", REPO_ROOT)
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


def _fit_decoder(runtime: FlowProfile, panels: FlowPanels, metrics_path: Path) -> None:
    optimizer_config = runtime.flow["optimizer"]
    optimizer = torch.optim.AdamW(
        runtime.system.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    order = balanced_task_order(
        len(runtime.active_fit),
        int(runtime.schedule["decoder_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]),
    )
    visits = [0] * len(runtime.active_fit)
    for step, task_index in enumerate(order, start=1):
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



def _fit_held_codes(
    runtime: FlowProfile, panels: FlowPanels, metrics_path: Path
) -> None:
    runtime.system.requires_grad_(False)
    runtime.held_codes.requires_grad_(True)
    held_optimizer_config = runtime.flow["held_code_optimizer"]
    held_optimizer = torch.optim.Adam(
        (runtime.held_codes,), lr=float(held_optimizer_config["learning_rate"])
    )
    held_order = balanced_task_order(
        len(runtime.active_held),
        int(runtime.schedule["held_code_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]) + 1,
    )
    visits = [0] * len(runtime.active_held)
    for step, task_index in enumerate(held_order, start=1):
        held_optimizer.zero_grad(set_to_none=True)
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
        held_optimizer.step()
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
    write_json_atomic(
        runtime.args.output_dir / "result.json",
        {
            "schema_version": "ember_pi05_functional_flow_profile_v1",
            "mode": runtime.args.mode,
            "repository": {
                "branch": repository["branch"],
                "commit": repository["commit"],
                "dirty_paths": repository["dirty_paths"],
            },
            "active_fit_ordinals": [row.ordinal for row in runtime.active_fit],
            "active_held_ordinals": [row.ordinal for row in runtime.active_held],
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


def run(args: argparse.Namespace) -> None:
    runtime = _prepare(args)
    panels = _cache_all_panels(runtime)
    initial_fit, initial_held = _initial_losses(runtime, panels)
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    _fit_decoder(runtime, panels, metrics_path)
    _fit_held_codes(runtime, panels, metrics_path)
    _write_result(runtime, panels, initial_fit, initial_held)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    result.add_argument(
        "--mode", choices=("smoke", "profile", "informative"), required=True
    )
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--effective-decoder", type=Path, required=True)
    result.add_argument("--tokenizer-path", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
