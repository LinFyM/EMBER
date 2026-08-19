#!/usr/bin/env python3
"""Profile gauge-invariant fixed-decoder fitting on the sealed train24 bank."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from ember.functional_adaptation.decoder_training import (
    FunctionalDecoderSystem,
    authority_path,
    balanced_task_order,
    decoder_task_split,
    expert_records,
    load_expert_states,
    load_functional_adapter_config,
    inspect_train24_expert_bank,
)
from ember.functional_adaptation.objectives import (
    effective_update_probe_loss,
    effective_update_probes,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_contract import append_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EffectiveProfile:
    args: argparse.Namespace
    config: dict[str, Any]
    settings: Mapping[str, Any]
    schedule: Mapping[str, Any]
    bank: Mapping[str, Any]
    split: Any
    contract: Any
    fit_states: tuple[dict[str, torch.Tensor], ...]
    held_states: tuple[dict[str, torch.Tensor], ...]
    system: FunctionalDecoderSystem
    probes: dict[str, torch.Tensor]
    metrics_path: Path
    started: float


def _authority(config: dict[str, Any], name: str) -> Path:
    return authority_path(config, name, REPO_ROOT)


def _bank_evidence(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return inspect_train24_expert_bank(
        config,
        REPO_ROOT,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        bank_root=args.expert_bank_root,
    )


def _mean_losses(
    decoder: torch.nn.Module,
    codes: torch.Tensor,
    states: tuple[dict[str, torch.Tensor], ...],
    contract: Any,
    probes: dict[str, torch.Tensor],
) -> list[float]:
    with torch.no_grad():
        return [
            float(
                effective_update_probe_loss(
                    decoder(codes[index]), target, contract, probes
                ).item()
            )
            for index, target in enumerate(states)
        ]


def _prepare(args: argparse.Namespace) -> EffectiveProfile:
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    settings = config["train24_mechanism"]
    schedule = settings[args.mode]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = _bank_evidence(args, config)
    records = expert_records(bank)
    split = decoder_task_split(
        records,
        fold_count=int(settings["fold_count"]),
        held_out_fold=int(settings["held_out_fold"]),
    )
    if (
        len(split.fit) != int(settings["fit_task_count"])
        or len(split.held) != int(settings["held_task_count"])
    ):
        raise ValueError("train24 decoder split differs from its profile contract")
    contract = load_pi05_lora_contract(_authority(config, "lora_contract"))
    fit_states = load_expert_states(split.fit, contract, device)
    held_states = load_expert_states(split.held, contract, device)
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
    probes = effective_update_probes(
        contract,
        probe_count=int(settings["objective"]["effective_update_probe_count"]),
        seed=int(settings["objective"]["effective_update_probe_seed"]),
        device=device,
    )
    args.output_dir.mkdir(parents=True)
    return EffectiveProfile(
        args=args,
        config=config,
        settings=settings,
        schedule=schedule,
        bank=bank,
        split=split,
        contract=contract,
        fit_states=fit_states,
        held_states=held_states,
        system=system,
        probes=probes,
        metrics_path=args.output_dir / "metrics.jsonl",
        started=time.monotonic(),
    )


def _fit_decoder(runtime: EffectiveProfile) -> list[float]:
    settings = runtime.settings
    optimizer_config = settings["optimizer"]
    optimizer = torch.optim.AdamW(
        runtime.system.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    initial_fit = _mean_losses(
        runtime.system.decoder,
        runtime.system.codebook.weight,
        runtime.fit_states,
        runtime.contract,
        runtime.probes,
    )
    order = balanced_task_order(
        len(runtime.split.fit),
        int(runtime.schedule["decoder_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]),
    )
    for step, task_index in enumerate(order, start=1):
        optimizer.zero_grad(set_to_none=True)
        candidate = runtime.system(task_index)
        response_loss = effective_update_probe_loss(
            candidate,
            runtime.fit_states[task_index],
            runtime.contract,
            runtime.probes,
        )
        gauge_loss = runtime.system.codebook.gauge_loss()
        loss = response_loss + float(
            settings["objective"]["codebook_gauge_weight"]
        ) * gauge_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runtime.system.parameters(), float(optimizer_config["gradient_clip_norm"])
        )
        optimizer.step()
        append_jsonl(
            runtime.metrics_path,
            {
                "phase": "decoder_fit",
                "step": step,
                "task_index": task_index,
                "effective_update_loss": float(response_loss.detach()),
                "gauge_loss": float(gauge_loss.detach()),
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - runtime.started,
            },
        )
    return initial_fit


def _fit_held_codes(runtime: EffectiveProfile) -> torch.nn.Parameter:
    runtime.system.decoder.requires_grad_(False)
    held_codes = torch.nn.Parameter(
        torch.zeros(
            len(runtime.split.held),
            int(runtime.config["decoder"]["train24_smoke_code_width"]),
            device=runtime.system.codebook.weight.device,
        )
    )
    held_optimizer_config = runtime.settings["held_code_optimizer"]
    held_optimizer = torch.optim.Adam(
        (held_codes,), lr=float(held_optimizer_config["learning_rate"])
    )
    held_order = balanced_task_order(
        len(runtime.split.held),
        int(runtime.schedule["held_code_steps"]),
        seed=int(runtime.config["decoder"]["initialization_seed"]) + 1,
    )
    for step, task_index in enumerate(held_order, start=1):
        held_optimizer.zero_grad(set_to_none=True)
        response_loss = effective_update_probe_loss(
            runtime.system.decoder(held_codes[task_index]),
            runtime.held_states[task_index],
            runtime.contract,
            runtime.probes,
        )
        code_l2 = held_codes[task_index].square().mean()
        loss = response_loss + float(
            held_optimizer_config["code_l2_weight"]
        ) * code_l2
        loss.backward()
        held_optimizer.step()
        append_jsonl(
            runtime.metrics_path,
            {
                "phase": "held_code_fit",
                "step": step,
                "task_index": task_index,
                "effective_update_loss": float(response_loss.detach()),
                "code_l2": float(code_l2.detach()),
                "elapsed_seconds": time.monotonic() - runtime.started,
            },
        )
    return held_codes


def _write_result(
    runtime: EffectiveProfile,
    held_codes: torch.Tensor,
    initial_fit: list[float],
) -> None:
    final_fit = _mean_losses(
        runtime.system.decoder,
        runtime.system.codebook.weight,
        runtime.fit_states,
        runtime.contract,
        runtime.probes,
    )
    final_held = _mean_losses(
        runtime.system.decoder,
        held_codes,
        runtime.held_states,
        runtime.contract,
        runtime.probes,
    )
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in runtime.system.state_dict().items()
    }
    save_file(state, str(runtime.args.output_dir / "decoder.safetensors"))
    save_file(
        {"held_codes": held_codes.detach().cpu().contiguous()},
        str(runtime.args.output_dir / "held_codes.safetensors"),
    )
    repository = git_state(REPO_ROOT)
    write_json_atomic(
        runtime.args.output_dir / "result.json",
        {
            "schema_version": "ember_pi05_functional_adapter_profile_v1",
            "mode": runtime.args.mode,
            "repository": {
                "branch": repository["branch"],
                "commit": repository["commit"],
                "dirty_paths": repository["dirty_paths"],
            },
            "expert_bank": {
                "root": str(runtime.args.expert_bank_root),
                "step": int(runtime.bank["step"]),
                "training_commit": runtime.bank["training_commit"],
            },
            "split": {
                "fit_ordinals": [row.ordinal for row in runtime.split.fit],
                "held_ordinals": [row.ordinal for row in runtime.split.held],
            },
            "steps": dict(runtime.schedule),
            "initial_fit_mean": sum(initial_fit) / len(initial_fit),
            "final_fit_mean": sum(final_fit) / len(final_fit),
            "final_fit_per_task": final_fit,
            "final_held_mean": sum(final_held) / len(final_held),
            "final_held_per_task": final_held,
            "elapsed_seconds": time.monotonic() - runtime.started,
            "content_hash_policy": "disabled_by_owner",
        },
    )


def run(args: argparse.Namespace) -> None:
    runtime = _prepare(args)
    initial_fit = _fit_decoder(runtime)
    held_codes = _fit_held_codes(runtime)
    _write_result(runtime, held_codes, initial_fit)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    result.add_argument("--mode", choices=("smoke", "profile"), required=True)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
