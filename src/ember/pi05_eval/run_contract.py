"""Construction of one immutable PI05 evaluation run without content hashes."""

from __future__ import annotations

import hashlib
import json
import socket
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    RUNTIME_OMP_THREADS,
    RUNTIME_REPLICA_PROFILES,
    RUN_CONTRACT_SCHEMA,
    SEEN_PANEL_RELATIVE_PATH,
    EvaluationAuthorities,
    TargetTaskContract,
    _read_object,
    git_state,
)


def _resolve_gpu_ids(
    authorities: EvaluationAuthorities,
    physical_gpu_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    configured = int(authorities.config["parallel"]["physical_gpu_count"])
    values = (
        tuple(range(configured))
        if physical_gpu_ids is None
        else tuple(int(value) for value in physical_gpu_ids)
    )
    if (
        not values
        or len(set(values)) != len(values)
        or any(index < 0 or index >= configured for index in values)
    ):
        raise Pi05EvaluationError(
            "physical GPU subset is invalid or exceeds the configured node topology"
        )
    return values


def _parallel_contract(
    authorities: EvaluationAuthorities,
    *,
    physical_gpu_ids: Sequence[int],
    replicas_per_gpu: int,
) -> dict[str, Any]:
    configured = int(authorities.config["parallel"]["physical_gpu_count"])
    physical_count = len(physical_gpu_ids)
    return {
        **authorities.config["parallel"],
        "authority_allowed_replicas_per_gpu": authorities.config["parallel"][
            "allowed_replicas_per_gpu"
        ],
        "allowed_replicas_per_gpu": list(RUNTIME_REPLICA_PROFILES),
        "omp_threads_per_worker": RUNTIME_OMP_THREADS,
        "configured_physical_gpu_count": configured,
        "physical_gpu_ids": list(physical_gpu_ids),
        "physical_gpu_count": physical_count,
        "replicas_per_gpu": replicas_per_gpu,
        "worker_count": physical_count * replicas_per_gpu,
        "one_policy_per_worker": True,
        "cpu_only_launcher": True,
        "sharding_algorithm": (
            "max-horizon task states balanced across physical_gpu_count times "
            "replicas_per_gpu worker slots with preferred-GPU affinity, then "
            "ordinary cost-balanced dynamic queue with at least two worker "
            "waves when enough states remain"
        ),
    }


def _validate_build_request(
    authorities: EvaluationAuthorities,
    *,
    model: Mapping[str, Any],
    tasks: Sequence[TargetTaskContract],
    mode: str,
    replicas_per_gpu: int,
) -> dict[str, Any]:
    if mode not in {"smoke", "screen", "formal"}:
        raise Pi05EvaluationError(f"unsupported PI05 evaluation mode: {mode}")
    git = git_state(authorities.repo_root)
    if mode != "smoke" and git["dirty_paths"]:
        raise Pi05EvaluationError(
            "screen/formal PI05 evaluation requires a clean worktree"
        )
    if replicas_per_gpu not in RUNTIME_REPLICA_PROFILES or not tasks:
        raise Pi05EvaluationError(
            "PI05 evaluation runtime profile or task panel is invalid"
        )
    return git


def build_run_contract(
    *,
    authorities: EvaluationAuthorities,
    tasks: Sequence[TargetTaskContract],
    libero_paths: Mapping[str, str],
    model: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    output_dir: Path,
    role: str,
    mode: str,
    replicas_per_gpu: int,
    command: Sequence[str],
    adapter: Mapping[str, Any] | None = None,
    physical_gpu_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    git = _validate_build_request(
        authorities,
        model=model,
        tasks=tasks,
        mode=mode,
        replicas_per_gpu=replicas_per_gpu,
    )
    gpu_ids = _resolve_gpu_ids(authorities, physical_gpu_ids)
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "mode": mode,
        "arm": str(adapter["arm"]) if adapter else authorities.config["policy"]["arm"],
        "adapter": dict(adapter) if adapter else None,
        "role": role,
        "output_dir": str(output_dir.resolve()),
        "prepared_unix": time.time(),
        "host": socket.gethostname(),
        "command": list(command),
        "git": git,
        "content_hash_policy": "disabled_by_owner",
        "authorities": {
            "config_path": str(authorities.config_path),
            "paths": authorities.paths,
        },
        "role_authority": (
            {
                "path": str(authorities.repo_root / SEEN_PANEL_RELATIVE_PATH),
                "bytes": (authorities.repo_root / SEEN_PANEL_RELATIVE_PATH)
                .stat()
                .st_size,
                "schema_version": authorities.seen_panel.get("schema_version"),
            }
            if role == "seen_panel"
            else (
                {
                    "path": authorities.paths["meta_protocol"],
                    "bytes": Path(authorities.paths["meta_protocol"]).stat().st_size,
                    "schema_version": authorities.meta_protocol.get("schema_version"),
                }
                if role.startswith("nonheld_meta")
                else None
            )
        ),
        "model": dict(model),
        "tokenizer": dict(tokenizer),
        "normalization": {
            "path": str(
                authorities.repo_root
                / authorities.config["authorities"]["normalization"]["path"]
            ),
            "bytes": (
                authorities.repo_root
                / authorities.config["authorities"]["normalization"]["path"]
            )
            .stat()
            .st_size,
            "source_only_numeric_reads": True,
            "validation_or_test_numeric_reads": 0,
        },
        "tasks": [asdict(task) for task in tasks],
        "environment": authorities.config["environment"],
        "policy": authorities.config["policy"],
        "rng": authorities.config["rng"],
        "parallel": _parallel_contract(
            authorities,
            physical_gpu_ids=gpu_ids,
            replicas_per_gpu=replicas_per_gpu,
        ),
        "artifacts": authorities.config["artifacts"],
        "libero_paths": dict(libero_paths),
    }
    contract["contract_reference"] = f"{RUN_CONTRACT_SCHEMA}:{uuid.uuid4().hex}"
    return contract


def load_run_contract(path: Path) -> dict[str, Any]:
    contract = _read_object(path)
    reference = contract.get("contract_reference")
    if (
        contract.get("schema_version") != RUN_CONTRACT_SCHEMA
        or contract.get("content_hash_policy") != "disabled_by_owner"
        or not isinstance(reference, str)
        or not reference.startswith(f"{RUN_CONTRACT_SCHEMA}:")
        or Path(str(contract.get("output_dir", ""))).resolve() != path.resolve().parent
    ):
        raise Pi05EvaluationError("PI05 evaluation run contract changed")
    return contract


def policy_noise_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    replan_index: int,
) -> int:
    encoded = json.dumps(
        [root_seed, suite, task_id, init_state_id, replan_index],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
