"""Canonical EMBER-ECP Stage 1 authority and training entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.observer_authority import (
    FrozenObserverAuthority,
    load_frozen_observer_authority,
)
from ember.ecp.stage0_training import load_stage0_config, stage0_source_authority
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_config import (
    REPO_ROOT,
    load_stage1_config,
    stage1_asset_authority,
    stage1_repo_authority,
)
from ember.ecp.stage1_data import gauge_canonicalize_lora_state
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


@dataclass(frozen=True)
class ECPStage1Authorities:
    source: dict[str, Any]
    source_config: dict[str, Any]
    policy: torch.nn.Module
    contract: LoRAContract
    identity_state: Mapping[str, torch.Tensor]
    observer: FrozenObserverAuthority
    prior_state: Mapping[str, torch.Tensor]
    model: ECPStage1Model


def load_stage1_authorities(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> ECPStage1Authorities:
    """Load the sole frozen policy/observer and the shared Stage 1 model."""

    source = stage0_source_authority(args)
    source_config = load_config(stage1_repo_authority(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    contract = load_pi05_lora_contract(stage1_repo_authority(config, "lora_contract"))
    identity = prepare_frozen_writer_policy(policy, contract)
    owners = build_target_owners(contract)
    stage0_config = load_stage0_config(
        stage1_repo_authority(config, "stage0_config")
    )
    observer = load_frozen_observer_authority(
        stage0_config=stage0_config,
        owners=owners,
        policy=policy,
        native_checkpoint=stage1_asset_authority(
            config, "native_observer_checkpoint", args.asset_root
        ),
        action_meta_checkpoint=stage1_asset_authority(
            config, "action_meta_checkpoint", args.asset_root
        ),
        device=context.device,
        max_frames_per_call=args.max_frames_per_call,
    )
    prior = load_file(
        str(stage1_asset_authority(config, "stable_shared_prior", args.asset_root)),
        device=str(context.device),
    )
    validate_lora_state(prior, contract)
    prior = gauge_canonicalize_lora_state(prior, contract)
    model = ECPStage1Model(
        owners,
        contract,
        prior,
        program_width=int(config["model"]["program_width"]),
        compiler_width=int(config["model"]["compiler_width"]),
        event_slots=int(config["model"]["event_slots"]),
        phase_width=int(config["model"]["phase_response_width"]),
        support_channels=len(config["policy_support"]["channels"]),
        support_horizon_basis=int(config["policy_support"]["horizon_basis"]),
        factor_head_init=config["model"]["factor_head_init_std"],
        replacement_head_init_multiplier=float(
            config["model"]["replacement_head_init_multiplier"]
        ),
        selector_max_angle_radians=float(
            config["model"]["selector_max_angle_radians"]
        ),
    ).to(context.device)
    return ECPStage1Authorities(
        source=source,
        source_config=source_config,
        policy=policy,
        contract=contract,
        identity_state=identity,
        observer=observer,
        prior_state=prior,
        model=model,
    )


def train(args: argparse.Namespace) -> None:
    """Run the one active Stage 1 lifecycle through the canonical entrypoint."""

    load_stage1_config(args.config)
    from ember.ecp.stage1_outcome_training import (
        train_fixed_compiler_program_outcome,
    )

    train_fixed_compiler_program_outcome(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_stage1_fixed_compiler_program_binding_v19.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--max-frames-per-call", type=int)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("ECP Stage 1 log interval must be positive")
    return args
