"""Single-A40 launcher for compiler-only cache population."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.launcher import gpu_preflight
from ember.pi05_eval.rank_reserved_cache_transform import (
    transform_compiler_only_cache,
)
from ember.pi05_eval.rank_reserved_compiler_diagnostic import (
    COMPILER_DIAGNOSTIC_PREFLIGHT,
    REPO_ROOT,
    compiler_diagnostic_lineage_matches,
    load_compiler_diagnostic_authority,
    validate_completed_compiler_transform,
    validate_compiler_diagnostic_contract,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    load_run_contract,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.evaluation_cache import writer_cache_manifest_is_ready
from ember.writer.topology import bind_current_process_to_cuda_numa, cuda_numa_node


EVALUATION_SCRIPT = REPO_ROOT / "scripts/evaluate_pi05.py"


def _gpu_identity(preflight: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    result = []
    for row in preflight.get("gpus", ()):
        fields = tuple(value.strip() for value in str(row).split(",")[:3])
        if len(fields) != 3:
            raise Pi05EvaluationError("compiler-only GPU identity is invalid")
        result.append(fields)
    return tuple(result)


def _live_launch_contract(
    contract: Mapping[str, Any],
    *,
    physical_gpu_index: int,
    preflight: Mapping[str, Any],
) -> None:
    live_git = git_state(REPO_ROOT)
    if (
        physical_gpu_index not in contract["parallel"]["physical_gpu_ids"]
        or preflight.get("physical_gpu_ids") != [physical_gpu_index]
        or preflight.get("compute_applications") != []
        or preflight.get("device_names") != ["NVIDIA A40"]
        or not git_state_is_clean_pushed_or_frozen_authority(live_git)
        or live_git.get("commit") != contract.get("git", {}).get("commit")
        or not git_state_is_clean_pushed_or_frozen_authority(contract.get("git", {}))
    ):
        raise Pi05EvaluationError("compiler-only launcher preflight changed")


def compiler_cache_run(args: Any) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    validate_compiler_diagnostic_contract(
        output_dir, contract, require_cache_ready=False
    )
    if writer_cache_manifest_is_ready(contract):
        return validate_completed_compiler_transform(
            load_compiler_diagnostic_authority(), output_dir, contract
        )
    physical = int(args.gpu_index)
    preflight = gpu_preflight((physical,))
    _live_launch_contract(
        contract, physical_gpu_index=physical, preflight=preflight
    )
    write_json_atomic(output_dir / COMPILER_DIAGNOSTIC_PREFLIGHT, preflight)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(REPO_ROOT / "src"),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(physical),
        OMP_NUM_THREADS="1",
    )
    subprocess.run(
        (
            sys.executable,
            str(EVALUATION_SCRIPT),
            "rank-reserved-compiler-cache-worker",
            "--output-dir",
            str(output_dir),
            "--gpu-index",
            str(physical),
        ),
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    validate_compiler_diagnostic_contract(
        output_dir, contract, require_cache_ready=True
    )
    return validate_completed_compiler_transform(
        load_compiler_diagnostic_authority(), output_dir, contract
    )


def compiler_cache_worker_run(args: Any) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    physical = int(args.gpu_index)
    contract = load_run_contract(output_dir / "run_contract.json")
    authority = validate_compiler_diagnostic_contract(
        output_dir, contract, require_cache_ready=False
    )
    launcher_preflight = read_json(output_dir / COMPILER_DIAGNOSTIC_PREFLIGHT)
    live_preflight = gpu_preflight((physical,))
    _live_launch_contract(
        contract, physical_gpu_index=physical, preflight=live_preflight
    )
    if (
        _gpu_identity(launcher_preflight) != _gpu_identity(live_preflight)
        or launcher_preflight.get("compute_applications") != []
        or launcher_preflight.get("device_names") != ["NVIDIA A40"]
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA A40"
        or not compiler_diagnostic_lineage_matches(
            authority, str(contract["git"]["commit"])
        )
    ):
        raise Pi05EvaluationError("compiler-only worker preflight changed")
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    numa_node = cuda_numa_node(0)
    if affinity is None or numa_node is None:
        raise Pi05EvaluationError(
            "compiler-only worker requires GPU-local NUMA affinity"
        )
    worker_preflight = {
        **live_preflight,
        "visible_cuda_device": 0,
        "numa_node": numa_node,
        "cpu_affinity": list(affinity),
    }
    torch.backends.cuda.matmul.allow_tf32 = True
    result = transform_compiler_only_cache(
        output_dir,
        physical_gpu_index=physical,
        preflight=worker_preflight,
    )
    print(
        json.dumps(
            {"event": "compiler_only_cache_complete", **result}, sort_keys=True
        )
    )
    return result
