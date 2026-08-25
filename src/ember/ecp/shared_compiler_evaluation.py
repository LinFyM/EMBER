"""Materialize one frozen G3 checkpoint into sealed held5 task LoRAs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.natural_program_data import (
    NaturalProgramSample,
)
from ember.ecp.shared_compiler_assets import authority_path
from ember.ecp.shared_compiler_data import (
    pack_shared_compiler_videos,
    prepare_shared_compiler_condition,
)
from ember.ecp.shared_compiler_evaluation_runtime import (
    CONDITIONS,
    REPO_ROOT,
    prepare_g3_materialization_runtime,
)
from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.static_task_lora import STATIC_TASK_LORA_MANIFEST_SCHEMA


G3_MATERIALIZED_ADAPTER_SCHEMA = (
    "ember_ecp_shared_compiler_g3_materialized_adapter_v1"
)


def materialize_g3_evaluation_bank(args: argparse.Namespace) -> dict[str, Any]:
    runtime = prepare_g3_materialization_runtime(args)
    final_root = args.output_dir
    partial_root = final_root.parent / f".{final_root.name}.partial-{os.getpid()}"
    if final_root.exists() or partial_root.exists():
        raise ValueError("G3 materialization output already exists")
    partial_root.mkdir(parents=True)
    records = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for task in runtime.held:
                sample = NaturalProgramSample(
                    video_demos=runtime.demos,
                    action_demos=(),
                    k=len(runtime.demos),
                    robustness_view="formal_fixed",
                )
                packed = pack_shared_compiler_videos(
                    task=task,
                    sample=sample,
                    video_store=runtime.video_store,
                    query_points=runtime.query_points,
                    device=runtime.device,
                    view=runtime.view,
                )
                language_tokens, language_mask = runtime.tokens[task.authority_id]
                prepared = prepare_shared_compiler_condition(
                    policy=runtime.policy,
                    program_model=runtime.program,
                    owners=runtime.owners,
                    packed=packed,
                    language_tokens=language_tokens,
                    language_mask=language_mask,
                    chunk_size=int(runtime.config["model"]["frame_chunk_size"]),
                )
                output = runtime.compiler(
                    prepared.program, prepared.videos, s_ref=runtime.ranks.s_ref
                )
                residual = residual_lora_state(
                    output.residual,
                    runtime.rank4_contract,
                    canonicalize=True,
                )
                complete = compose_rank12_plus_rank4(
                    carrier_state=runtime.ranks.carrier_rank12,
                    residual_state=residual,
                    rank16_contract=runtime.ranks.contract,
                )
                validate_lora_state(complete, runtime.ranks.contract)
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
                adapter_manifest = {
                    "schema_version": G3_MATERIALIZED_ADAPTER_SCHEMA,
                    "condition": args.condition,
                    "compiler_macro": runtime.compiler_macro,
                    "compiler_checkpoint": str(args.compiler_checkpoint),
                    "authority_id": task.authority_id,
                    "global_task_id": task.domain_task_id,
                    "suite": task.suite,
                    "task_id": runtime.target_keys[task.domain_task_id][1],
                    "language": task.language,
                    "video_demos": list(runtime.demos),
                    "view": runtime.view,
                    "sampled_frames": packed.metrics["sampled_frames"],
                    "raw_frame_counts": packed.metrics["raw_frame_counts"],
                    "video_weights": output.video_weights.float().cpu().tolist(),
                    "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
                    "single_complete_rank16": True,
                    "files": {"adapter.safetensors": adapter_path.stat().st_size},
                }
                write_json_atomic(write_root / "manifest.json", adapter_manifest)
                records.append(
                    {
                        "suite": task.suite,
                        "task_id": runtime.target_keys[task.domain_task_id][1],
                        "natural_program_authority_id": task.authority_id,
                        "global_task_id": task.domain_task_id,
                        "language": task.language,
                        "condition": args.condition,
                        "compiler_macro": runtime.compiler_macro,
                        "checkpoint": str(final_checkpoint),
                        "checkpoint_manifest_bytes": (
                            write_root / "manifest.json"
                        ).stat().st_size,
                        "adapter_path": str(final_checkpoint / "adapter.safetensors"),
                        "adapter_bytes": adapter_path.stat().st_size,
                        "single_complete_rank16": True,
                    }
                )
                del packed, prepared, output, residual, complete
                torch.cuda.empty_cache()
    finally:
        runtime.close()

    lora_path = authority_path(
        runtime.config, "lora_contract", asset_root=args.asset_root
    )
    training_commit = str(runtime.shared_contract["git"]["commit"])
    payload = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": f"ecp_shared_compiler_g3_{args.condition}",
        "source": {
            "source_run": str(args.source_run),
            "checkpoint": str(args.checkpoint),
            "model_path": str(args.checkpoint / "policy"),
        },
        "lora_contract": {"path": str(lora_path), "bytes": lora_path.stat().st_size},
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": training_commit,
        "materialization_commit": str(runtime.state["commit"]),
        "shared_run_contract": runtime.shared_contract,
        "compiler_checkpoint": {
            "path": str(args.compiler_checkpoint),
            "macro": runtime.compiler_macro,
        },
        "condition": {
            "name": args.condition,
            "view": runtime.view,
            "video_demos": list(runtime.demos),
            "K": len(runtime.demos),
        },
        "tasks": records,
        "information_wall": {
            "deployment_inputs": [
                "exact language",
                "same-task action-hidden internally ordered videos",
            ],
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
            "writer_invocations_per_condition": 1,
            "materialization_teacher_video_count": len(records)
            * len(runtime.demos),
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "shuffled_or_reversed_use": False,
            **runtime.wall,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(partial_root / "manifest.json", payload)
    write_json_atomic(
        partial_root / "completion.json",
        {
            "schema_version": (
                "ember_ecp_shared_compiler_g3_materialization_completion_v1"
            ),
            "condition": args.condition,
            "tasks": len(records),
            "compiler_macro": runtime.compiler_macro,
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
    parser.add_argument("--compiler-run", type=Path, required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
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
        "compiler_run",
        "compiler_checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = finalize_args(build_parser().parse_args(argv))
    payload = materialize_g3_evaluation_bank(args)
    print(
        f"sealed {len(payload['tasks'])} G3 {args.condition} rank16 adapters",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
