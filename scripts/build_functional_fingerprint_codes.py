#!/usr/bin/env python3
"""Build one train-only-whitened code space from unified policy responses."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file
from torch.utils.data import default_collate

from ember.expert_manifold.contract import (
    build_dataset,
    load_task_expert_config,
    load_train_tasks,
)
from ember.functional_adaptation.decoder_training import (
    authority_path,
    decoder_task_split,
    expert_records,
    inspect_nonheld_meta_expert_bank,
    inspect_train24_expert_bank,
    load_expert_states,
    load_functional_adapter_config,
    meta_decoder_task_split,
)
from ember.functional_adaptation.fingerprint_codes import (
    FINGERPRINT_CODE_SCHEMA,
    uniformly_spaced_task_ids,
    whiten_functional_fingerprints,
)
from ember.functional_adaptation.functional_response import pi05_flow_response
from ember.functional_adaptation.probe_panels import selected_probe_rows
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_source_setup import load_policy, load_stats
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    parser.add_argument(
        "--surface",
        choices=("train24", "nonheld_meta"),
        default="nonheld_meta",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _resolved(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "expert_bank_root",
        "tokenizer_path",
        "data_root",
    ):
        value = getattr(args, name).resolve()
        if not value.exists():
            raise ValueError(f"missing functional-fingerprint input: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("functional-fingerprint output already exists")
    return args


def _response(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: Any,
    batch: Mapping[str, Any],
    *,
    policy_seed: int,
) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return (
            pi05_flow_response(
                policy,
                state,
                contract,
                batch,
                policy_seed=policy_seed,
            )
            .detach()
            .float()
            .cpu()
        )


def _code_stats(codes: torch.Tensor) -> dict[str, float]:
    norms = codes.norm(dim=1)
    return {
        "coordinate_mean_absolute": float(codes.mean(dim=0).abs().mean()),
        "coordinate_std_mean": float(codes.std(dim=0).mean()),
        "task_norm_mean": float(norms.mean()),
        "task_norm_min": float(norms.min()),
        "task_norm_max": float(norms.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _resolved(args)
    if not torch.cuda.is_available():
        raise RuntimeError("functional-fingerprint collection requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal functional fingerprints require clean pushed code")
    started = time.monotonic()
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    if args.surface == "nonheld_meta":
        mechanism = config["production_meta"]
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
        fit_surface = "meta_train_only_pca_whitening"
    else:
        mechanism = config["train24_mechanism"]
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
        fit_surface = "train24_fit_only_pca_whitening"
    settings = mechanism["functional_fingerprint"]
    train_ids = tuple(row.global_task_id for row in split.fit)
    held_ids = tuple(row.global_task_id for row in split.held)
    anchor_ids = uniformly_spaced_task_ids(
        train_ids, int(settings["anchor_task_count"])
    )
    expert_config = load_task_expert_config(
        authority_path(config, expert_config_name, REPO_ROOT)
    )
    all_tasks = load_train_tasks(expert_config, args.data_root)
    dataset = build_dataset(expert_config, all_tasks)
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config", REPO_ROOT), REPO_ROOT
    )
    policy = load_policy(
        Path(str(bank["source"]["model_path"])),
        authorities.source_base_config,
        device,
    )
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    identity = prepare_frozen_writer_policy(policy, contract)
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
    panels: list[tuple[int, int, Mapping[str, Any], int]] = []
    demos = tuple(int(value) for value in settings["anchor_demo_indices"])
    panel_count = int(settings["panels_per_anchor"])
    batch_size = int(settings["anchor_batch_size"])
    seed = int(settings["policy_seed"])
    for anchor_ordinal, task_id in enumerate(anchor_ids):
        rows = selected_probe_rows(
            dataset.task_episode_rows[task_id],
            demo_indices=demos,
            panel_count=panel_count,
            batch_size=batch_size,
            seed=seed + task_id * 1009,
        )
        for panel_ordinal, selected in enumerate(rows):
            batch = processor.training_batch(
                default_collate([dataset[index] for index in selected])
            )
            panels.append(
                (
                    task_id,
                    panel_ordinal,
                    batch,
                    seed + anchor_ordinal * 9173 + panel_ordinal,
                )
            )
    identity_responses = [
        _response(policy, identity, contract, batch, policy_seed=panel_seed)
        for _, _, batch, panel_seed in panels
    ]
    expert_states = load_expert_states(records, contract, device)
    fingerprints: dict[int, torch.Tensor] = {}
    for record, state in zip(records, expert_states, strict=True):
        blocks = []
        for (_, _, batch, panel_seed), baseline in zip(
            panels, identity_responses, strict=True
        ):
            expert = _response(
                policy, state, contract, batch, policy_seed=panel_seed
            )
            blocks.append((expert - baseline).flatten())
        fingerprints[record.global_task_id] = torch.cat(blocks).contiguous()
    train_fingerprints = torch.stack([fingerprints[value] for value in train_ids])
    held_fingerprints = torch.stack([fingerprints[value] for value in held_ids])
    code_space = whiten_functional_fingerprints(
        train_fingerprints,
        held_fingerprints,
        code_width=code_width,
    )
    args.output_dir.mkdir(parents=True)
    codes_path = args.output_dir / "fingerprint_codes.safetensors"
    save_file(
        {
            "train_fingerprints": train_fingerprints,
            "held_fingerprints": held_fingerprints,
            "train_codes": code_space.train_codes,
            "held_codes": code_space.held_codes,
            "pca_mean": code_space.mean,
            "pca_components": code_space.components,
            "pca_scales": code_space.scales,
        },
        str(codes_path),
    )
    result = {
        "schema_version": FINGERPRINT_CODE_SCHEMA,
        "formal_authority": True,
        "surface": args.surface,
        "fit_surface": fit_surface,
        "repository": {
            "branch": repository["branch"],
            "commit": repository["commit"],
            "authority_ref": repository["authority_ref"],
            "authority_contains_commit": repository["authority_contains_commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "host": socket.gethostname(),
        "device": str(device),
        "source_run": str(args.source_run),
        "checkpoint": str(args.checkpoint),
        "expert_bank_root": str(args.expert_bank_root),
        "train_global_task_ids": list(train_ids),
        "held_global_task_ids": list(held_ids),
        "anchors": {
            "selection": "uniform_positions_in_fixed_meta_train_order",
            "global_task_ids": list(anchor_ids),
            "demo_indices": list(demos),
            "panels_per_anchor": panel_count,
            "batch_size": batch_size,
            "policy_seed": seed,
            "response": "expert_minus_frozen_source_full_50x32_action_flow",
            "feature_count": int(train_fingerprints.shape[1]),
        },
        "code_width": int(code_space.train_codes.shape[1]),
        "explained_variance_fraction": code_space.explained_variance_fraction,
        "train_code_stats": _code_stats(code_space.train_codes),
        "held_code_stats": _code_stats(code_space.held_codes),
        "information_wall": {
            "pca_fit_roles": [
                "meta_train" if args.surface == "nonheld_meta" else "target_train"
            ],
            "held_fingerprint_transform_only": True,
            "unified_anchor_role": (
                "meta_train" if args.surface == "nonheld_meta" else "target_train_fit"
            ),
            "target_train_action_state_used": args.surface == "train24",
            "target_train_use": (
                None
                if args.surface == "nonheld_meta"
                else "fixed_fit-task_fingerprint_anchors"
            ),
            "held_task_action_state_reward_reads": 0,
            "target_validation_action_state_reward_reads": 0,
            "target_test_action_state_reward_reads": 0,
            "deployment_task_id_route": False,
        },
        "role_separation": {
            "source_skill_and_adaptation_meta_task_identities_disjoint": (
                args.surface == "train24"
            ),
            "interpretation": (
                "role_disjoint_development_diagnostic"
                if args.surface == "train24"
                else "source_meta_overlap_control"
            ),
        },
        "files": {"fingerprint_codes.safetensors": codes_path.stat().st_size},
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(args.output_dir / "result.json", result)
    dataset.close()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run(_args())
