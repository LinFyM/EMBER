#!/usr/bin/env python3
"""Measure fixed Writer-head reachability with free train-task Programs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.writer.adapter_analysis_metrics import effective_metrics
from ember.writer.as_config import REPO_ROOT, authority_path, load_writer_config
from ember.writer.checkpoint import checkpoint_macro
from ember.writer.model import PARAMETER_GROUPS, PROGRAM_WIDTH, PUBLIC_RANK
from ember.writer.training import build_writer


SCHEMA = "ember_writer_fixed_head_reachability_oracle_v1"
FAMILIES = (
    "q_a",
    "q_b",
    "v_a",
    "v_b",
    "action_in_a",
    "action_in_b",
    "action_out_a",
    "action_out_b",
)


def _pairs(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for name in state:
        for suffix, factor in (
            (".lora_A.default.weight", "a"),
            (".lora_B.default.weight", "b"),
        ):
            if name.endswith(suffix):
                pairs.setdefault(name.removesuffix(suffix), {})[factor] = name
                break
    if not pairs or any(set(value) != {"a", "b"} for value in pairs.values()):
        raise ValueError("reachability target is not a complete LoRA")
    return pairs


def _effective_family(name: str) -> str:
    if name.endswith(".self_attn.q_proj"):
        return "q"
    if name.endswith(".self_attn.v_proj"):
        return "v"
    if name.endswith(".action_in_proj"):
        return "action_in"
    if name.endswith(".action_out_proj"):
        return "action_out"
    raise ValueError(f"unknown LoRA target family: {name}")


def _relative_raw_by_family(
    generated: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    decoding: Mapping[str, tuple[str, int | None]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    numerator = {
        family: torch.zeros(
            generated[next(iter(generated))].shape[0],
            device=next(iter(generated.values())).device,
        )
        for family in FAMILIES
    }
    denominator = {
        family: torch.zeros_like(value) for family, value in numerator.items()
    }
    for name, value in generated.items():
        family = decoding[name][0]
        difference = value.float() - targets[name]
        dimensions = tuple(range(1, difference.ndim))
        numerator[family] = numerator[family] + difference.square().sum(dim=dimensions)
        denominator[family] = denominator[family] + targets[name].square().sum(
            dim=dimensions
        )
    relative = {
        family: numerator[family] / denominator[family].clamp_min(1e-20)
        for family in FAMILIES
    }
    return torch.stack(tuple(relative.values())).mean(), relative


def _load_writer(
    *,
    config: Mapping[str, Any],
    source_checkpoint: Path,
    writer_checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    if checkpoint_macro(writer_checkpoint) != 25:
        raise ValueError(
            "reachability oracle requires the supported macro25 checkpoint"
        )
    source_config = read_json(authority_path(config, "source_base_config"))
    policy = load_policy(source_checkpoint / "policy", source_config, device)
    writer, _ = build_writer(config, policy, asset_root=REPO_ROOT)
    writer.to(device)
    writer.load_state_dict(
        load_file(str(writer_checkpoint / "writer.safetensors"), device=str(device)),
        strict=True,
    )
    writer.requires_grad_(False).eval()
    del policy
    torch.cuda.empty_cache()
    return writer


def _expert_records(
    *,
    expert_config: Path,
    expert_bank_root: Path,
    expert_step: int,
    source_run: Path,
    source_checkpoint: Path,
    writer_config: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    manifest = read_json(authority_path(writer_config, "target_data_manifest"))
    rows = tuple(
        sorted(
            (dict(row) for row in manifest["tasks"] if row["split_role"] == "train"),
            key=lambda row: int(row["global_task_id"]),
        )
    )
    source = {
        "source_run": str(source_run.resolve()),
        "checkpoint": str(source_checkpoint.resolve()),
        "model_path": str((source_checkpoint / "policy").resolve()),
    }
    bank = inspect_task_expert_bank(
        config_path=expert_config,
        bank_root=expert_bank_root,
        step=expert_step,
        source=source,
        task_keys=tuple((str(row["suite"]), int(row["task_id"])) for row in rows),
        evaluation_role="development_train",
        require_formal=True,
    )
    return bank, rows


def _task_metrics(
    target: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    pairs = _pairs(target)
    result = {"all": effective_metrics(pairs, target, candidate)}
    for family in ("q", "v", "action_in", "action_out"):
        selected = {
            name: names
            for name, names in pairs.items()
            if _effective_family(name) == family
        }
        result[family] = effective_metrics(selected, target, candidate)
    raw_target = math.fsum(
        float(value.double().square().sum()) for value in target.values()
    )
    raw_error = math.fsum(
        float((candidate[name].double() - value.double()).square().sum())
        for name, value in target.items()
    )
    result["raw_factor_relative_l2"] = math.sqrt(raw_error / max(raw_target, 1e-24))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or args.steps != 3000
        or args.learning_rate != 0.03
        or args.output_dir.exists()
    ):
        raise ValueError("reachability oracle optimization contract changed")
    config = load_writer_config(args.writer_config)
    bank, task_rows = _expert_records(
        expert_config=args.expert_config,
        expert_bank_root=args.expert_bank_root,
        expert_step=args.expert_step,
        source_run=args.source_run,
        source_checkpoint=args.source_checkpoint,
        writer_config=config,
    )
    writer = _load_writer(
        config=config,
        source_checkpoint=args.source_checkpoint,
        writer_checkpoint=args.writer_checkpoint,
        device=device,
    )
    bank_by_key = {
        (str(record["suite"]), int(record["task_id"])): record
        for record in bank["tasks"]
    }
    bank_records = tuple(
        bank_by_key[(str(task["suite"]), int(task["task_id"]))] for task in task_rows
    )
    target_states = tuple(
        load_file(str(Path(record["checkpoint"]) / "adapter.safetensors"), device="cpu")
        for record in bank_records
    )
    if len(target_states) != 24 or any(
        set(state) != set(target_states[0]) for state in target_states
    ):
        raise ValueError(
            "reachability oracle did not resolve one complete train24 bank"
        )
    targets = {
        name: torch.stack([state[name].float() for state in target_states]).to(device)
        for name in target_states[0]
    }
    generator = torch.Generator(device=device).manual_seed(7)
    program = torch.nn.Parameter(
        torch.randn(
            len(target_states),
            PARAMETER_GROUPS,
            PUBLIC_RANK,
            PROGRAM_WIDTH,
            generator=generator,
            device=device,
        )
    )
    optimizer = torch.optim.Adam([program], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.001
    )
    curve = []
    decoding = dict(writer._decoding)
    for step in range(args.steps + 1):
        generated = writer.decode_program(program)
        loss, relative = _relative_raw_by_family(generated, targets, decoding)
        if step == 0 or step % 100 == 0 or step == args.steps:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "family_relative_l2_mean": {
                        family: float(value.detach().sqrt().mean())
                        for family, value in relative.items()
                    },
                }
            )
            print(json.dumps(curve[-1], sort_keys=True), flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((program,), 10.0)
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        final = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in writer.decode_program(program).items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_file(
        {"program": program.detach().to(device="cpu").contiguous()},
        str(args.output_dir / "program.safetensors"),
    )
    task_records = []
    for index, (task, expert, bank_record) in enumerate(
        zip(task_rows, target_states, bank_records, strict=True)
    ):
        candidate = {name: value[index] for name, value in final.items()}
        path = (
            args.output_dir
            / f"task_{index:02d}_global_{int(task['global_task_id']):02d}.safetensors"
        )
        save_file(candidate, str(path))
        task_records.append(
            {
                "suite": task["suite"],
                "task_id": int(task["task_id"]),
                "ordinal": index,
                "global_task_id": int(task["global_task_id"]),
                "expert_checkpoint": bank_record["checkpoint"],
                "projected_adapter": str(path.resolve()),
                "metrics": _task_metrics(expert, candidate),
            }
        )
    repository = git_state(REPO_ROOT)
    result = {
        "schema_version": SCHEMA,
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "writer_config": str(args.writer_config.resolve()),
        "writer_checkpoint": str(args.writer_checkpoint.resolve()),
        "expert_config": str(args.expert_config.resolve()),
        "expert_bank_root": str(args.expert_bank_root.resolve()),
        "expert_step": args.expert_step,
        "optimization": {
            "program_shape": [24, PARAMETER_GROUPS, PUBLIC_RANK, PROGRAM_WIDTH],
            "steps": args.steps,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "scheduler": "cosine_to_0.001",
            "gradient_clip_norm": 10.0,
            "seed": 7,
            "factor_heads_frozen": True,
            "all_other_writer_parameters_unused_and_frozen": True,
        },
        "curve": curve,
        "tasks": task_records,
        "elapsed_seconds": time.monotonic() - started,
        "information_wall": {
            "role": "development_train_oracle_only",
            "validation_experts": 0,
            "test_experts": 0,
            "deployment_carrier": False,
        },
    }
    write_json_atomic(args.output_dir / "projection_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer-config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--writer-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-config", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--expert-step", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "writer_config",
        "source_run",
        "source_checkpoint",
        "writer_checkpoint",
        "expert_config",
        "expert_bank_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(
        json.dumps(
            {
                "event": "complete",
                "tasks": len(result["tasks"]),
                "output": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
