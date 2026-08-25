"""Fit and seal the G3 learned language-only held5 control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
)
from ember.ecp.shared_compiler_assets import (
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.shared_compiler_evaluation_runtime import (
    REPO_ROOT,
    load_g3_gate_config,
    load_g3_tasks,
)
from ember.ecp.shared_compiler_language_mapping import (
    kernel_ridge_weights,
    language_feature,
    load_language_effect_records,
    load_task_rank4_target,
    mix_rank4_states,
)
from ember.ecp.stage0_training import (
    stage0_source_authority,
    tokenize_stage0_languages,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import (
    initialize_distributed,
    load_config,
    load_policy,
    seed_everything,
)
from ember.static_task_lora import STATIC_TASK_LORA_MANIFEST_SCHEMA
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


LANGUAGE_BASELINE_RUN_SCHEMA = "ember_ecp_g3_language_only_baseline_v1"
LANGUAGE_BASELINE_ADAPTER_SCHEMA = "ember_ecp_g3_language_only_adapter_v1"


def materialize_language_baseline(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(require_numa=True)
    state = git_state(REPO_ROOT)
    if (
        context.world_size != 1
        or not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("language baseline requires one GPU and detached authority")
    config = load_shared_compiler_config(args.config)
    gate = load_g3_gate_config(args.gate_config)
    if (
        args.config != (REPO_ROOT / gate["training_config"]).resolve()
        or args.effect_bank_root
        != authority_path(config, "shared_effect_bank", asset_root=args.asset_root)
    ):
        raise ValueError("language baseline config or effect authority changed")
    expected_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    if (
        args.checkpoint != expected_checkpoint
        or args.source_run != expected_checkpoint.parent.parent
        or args.tokenizer_path
        != authority_path(config, "tokenizer", asset_root=args.asset_root)
    ):
        raise ValueError("language baseline source authority changed")
    seed_everything(int(config["optimization"]["seed"]), context)
    tasks = load_g3_tasks(
        config, asset_root=args.asset_root, data_root=args.data_root
    )
    fit = tuple(task for task in tasks if task.role in {"meta_fit", "target_fit"})
    held = tuple(task for task in tasks if task.role == "target_held")
    if len(fit) != 75 or len(held) != 5:
        raise ValueError("language baseline fit75/held5 split changed")
    source = stage0_source_authority(args)
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    policy = load_policy(
        Path(source["model_path"]), source_config, context.device
    ).requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        config,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    program = build_frozen_g2_program(
        config, asset_root=args.asset_root, owners=owners, device=context.device
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    action_meta = [
        name
        for root in (policy, program)
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    if action_meta or any(
        parameter.requires_grad
        for root in (policy, program)
        for parameter in root.parameters()
    ):
        raise ValueError("language baseline information wall changed")
    tokens = tokenize_stage0_languages(
        fit + held,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = {
            task.authority_id: language_feature(
                policy=policy,
                program=program,
                tokens=tokens[task.authority_id][0],
                mask=tokens[task.authority_id][1],
            ).cpu()
            for task in fit + held
        }
    fit_features = torch.stack([features[task.authority_id] for task in fit])
    held_features = torch.stack([features[task.authority_id] for task in held])
    ridge = float(gate["language_baseline"]["relative_ridge"])
    weights, feature_mean, centered_fit = kernel_ridge_weights(
        fit_features, held_features, relative_ridge=ridge
    )
    records = load_language_effect_records(
        args.effect_bank_root, {task.authority_id for task in fit}
    )
    rank4 = derive_pi05_lora_rank(ranks.contract, rank=4)
    fit_states = [
        load_task_rank4_target(
            record=records[task.authority_id], contract=rank4, device=context.device
        )
        for task in fit
    ]

    final_root = args.output_dir
    partial_root = final_root.parent / f".{final_root.name}.partial-{os.getpid()}"
    if final_root.exists() or partial_root.exists():
        raise ValueError("language baseline output already exists")
    partial_root.mkdir(parents=True)
    save_file(
        {
            "feature_mean": feature_mean.contiguous(),
            "fit_features": centered_fit.contiguous(),
            "fit_authority_ids": torch.tensor(
                [task.authority_id for task in fit], dtype=torch.int64
            ),
            "held_weights": weights.contiguous(),
        },
        str(partial_root / "language_fit.safetensors"),
    )
    target_manifest = read_json(
        authority_path(config, "target_manifest", asset_root=args.asset_root)
    )
    target_keys = {
        int(row["global_task_id"]): (str(row["suite"]), int(row["task_id"]))
        for row in target_manifest["tasks"]
        if row["split_role"] == "train"
    }
    task_rows = []
    for held_index, task in enumerate(held):
        residual = mix_rank4_states(
            fit_states,
            weights[held_index],
            contract=rank4,
            device=context.device,
        )
        complete = compose_rank12_plus_rank4(
            carrier_state=ranks.carrier_rank12,
            residual_state=residual,
            rank16_contract=ranks.contract,
        )
        relative = Path("adapters") / f"task_{task.domain_task_id:02d}"
        write_root = partial_root / relative
        final_checkpoint = final_root / relative
        write_root.mkdir(parents=True)
        adapter_path = write_root / "adapter.safetensors"
        save_file(
            {
                name: value.detach().float().cpu().contiguous()
                for name, value in complete.items()
            },
            str(adapter_path),
        )
        top = torch.topk(weights[held_index].abs(), k=5).indices.tolist()
        adapter_manifest = {
            "schema_version": LANGUAGE_BASELINE_ADAPTER_SCHEMA,
            "condition": "learned_language_only",
            "authority_id": task.authority_id,
            "global_task_id": task.domain_task_id,
            "suite": task.suite,
            "task_id": target_keys[task.domain_task_id][1],
            "language": task.language,
            "top_fit_authority_ids": [fit[index].authority_id for index in top],
            "top_fit_weights": [float(weights[held_index, index]) for index in top],
            "weight_sum": float(weights[held_index].sum()),
            "weight_l1": float(weights[held_index].abs().sum()),
            "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
            "single_complete_rank16": True,
            "files": {"adapter.safetensors": adapter_path.stat().st_size},
        }
        manifest_path = write_root / "manifest.json"
        write_json_atomic(manifest_path, adapter_manifest)
        task_rows.append(
            {
                "suite": task.suite,
                "task_id": target_keys[task.domain_task_id][1],
                "natural_program_authority_id": task.authority_id,
                "global_task_id": task.domain_task_id,
                "language": task.language,
                "condition": "learned_language_only",
                "checkpoint": str(final_checkpoint),
                "checkpoint_manifest_bytes": manifest_path.stat().st_size,
                "adapter_path": str(final_checkpoint / "adapter.safetensors"),
                "adapter_bytes": adapter_path.stat().st_size,
                "single_complete_rank16": True,
            }
        )
        del residual, complete
        torch.cuda.empty_cache()

    lora_path = authority_path(config, "lora_contract", asset_root=args.asset_root)
    run_contract = {
        "schema_version": LANGUAGE_BASELINE_RUN_SCHEMA,
        "stage": "g3_learned_language_only",
        "mode": "formal",
        "method": dict(gate["language_baseline"]),
        "git": {"commit": str(state["commit"]), "branch": ""},
        "frozen_program": str(
            authority_path(config, "g2_program_checkpoint", asset_root=args.asset_root)
        ),
        "fit_task_count": 75,
        "fit_member_count": 93,
        "held_task_count": 5,
        "effect_bank": str(args.effect_bank_root),
        "held_video_reads": 0,
        "held_action_or_reward_reads": 0,
    }
    write_json_atomic(partial_root / "run_contract.json", run_contract)
    payload = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": "ecp_shared_compiler_g3_learned_language_only",
        "source": {
            "source_run": str(args.source_run),
            "checkpoint": str(args.checkpoint),
            "model_path": str(args.checkpoint / "policy"),
        },
        "lora_contract": {"path": str(lora_path), "bytes": lora_path.stat().st_size},
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": str(state["commit"]),
        "materialization_commit": str(state["commit"]),
        "shared_run_contract": run_contract,
        "condition": {
            "name": "learned_language_only",
            "view": None,
            "video_demos": [],
            "K": 0,
        },
        "tasks": task_rows,
        "information_wall": {
            "deployment_inputs": ["exact language"],
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
            "materialization_teacher_video_count": 0,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "shuffled_or_reversed_use": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(partial_root / "manifest.json", payload)
    write_json_atomic(
        partial_root / "completion.json",
        {
            "schema_version": "ember_ecp_g3_language_only_completion_v1",
            "tasks": len(task_rows),
        },
    )
    partial_root.rename(final_root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v1.json",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_gate_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--effect-bank-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "gate_config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "effect_bank_root",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    payload = materialize_language_baseline(args)
    print(f"sealed {len(payload['tasks'])} language-only rank16 adapters", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
