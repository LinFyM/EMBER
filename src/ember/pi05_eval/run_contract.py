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

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    EXPERT_MANIFOLD_WRITER_KIND,
    WRITER_ADAPTER_KINDS,
    paired_writer_identity,
)
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


_BATCH_THROUGHPUT_POLICY = (
    "highest_measured_batch_throughput_with_device_memory_headroom"
)
_WRITER_THROUGHPUT_BY_SCHEMA = {
    "ember_pi05_v6_prior_eval_adapter_v5": (
        "highest_measured_throughput_with_device_memory_headroom"
    ),
    "ember_pi05_v6_ecp_eval_adapter_v6": (
        "highest_measured_throughput_with_device_memory_headroom"
    ),
    "ember_pi05_v6_tangent_tube_eval_adapter_v7": (
        "highest_measured_throughput_with_device_memory_headroom"
    ),
    "ember_pi05_v6_condition_program_residual_eval_adapter_v8": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_v6_condition_program_residual_eval_adapter_v9": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_v6_causal_goal_interaction_joint_credit_eval_adapter_v10": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_v6_magnitude_gated_causal_interaction_joint_credit_eval_adapter_v11": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_dynamic_k_backbone_memory_rank8_eval_adapter_v1": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_eval_adapter_v1": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_dynamic_k_task_grounded_visual_value_rank8_eval_adapter_v1": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_v6_shared_core_procedure_common_value_bridge_eval_adapter_v1": (
        _BATCH_THROUGHPUT_POLICY
    ),
    "ember_pi05_v6_layerwise_probe_conditioned_procedure_eval_adapter_v1": (
        _BATCH_THROUGHPUT_POLICY
    ),
}


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
    writer_adapter: bool,
    writer_generators_per_gpu: int,
    writer_generation_batch_size: int,
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
        "writer_generators_per_gpu": (
            writer_generators_per_gpu if writer_adapter else 0
        ),
        "writer_generation_worker_count": (
            physical_count * writer_generators_per_gpu if writer_adapter else 0
        ),
        "writer_generation_batch_size": (
            writer_generation_batch_size if writer_adapter else 0
        ),
        "writer_and_rollout_parallelism_decoupled": writer_adapter,
        "generator_source_policy_processes_reused_for_rollout": (
            writer_adapter
        ),
        "one_policy_per_worker": True,
        "cpu_only_launcher": True,
        "sharding_algorithm": (
            "max-horizon task states balanced across physical_gpu_count times "
            "replicas_per_gpu worker slots with preferred-GPU affinity, then "
            "ordinary cost-balanced dynamic queue with at least two worker "
            "waves when enough states remain"
        ),
    }


def _writer_lora_contract(
    authorities: EvaluationAuthorities,
    adapter: Mapping[str, Any],
) -> Any:
    from ember.expert_manifold.v6_prior_contract import (
        authority_path as expert_authority_path,
        load_v6_prior_config,
    )
    from ember.pi05_lora import load_pi05_lora_contract

    config_path = Path(adapter["config"]["path"])
    if adapter["kind"] == EXPERT_MANIFOLD_WRITER_KIND:
        config = load_v6_prior_config(config_path)
        path = expert_authority_path(config, "lora_contract")
    elif adapter["kind"] == DYNAMIC_K_WRITER_KIND:
        from ember.writer.as_config import authority_path, load_writer_config

        config = load_writer_config(config_path)
        path = authority_path(config, "lora_contract")
    else:
        raise Pi05EvaluationError("unknown Writer LoRA authority")
    result = load_pi05_lora_contract(path)
    expected_reference = (
        f"{path.relative_to(authorities.repo_root)}:"
        f"{result.state_tensor_count}tensors:{result.parameter_count}parameters"
    )
    if adapter.get("lora_contract", {}).get("reference") != expected_reference:
        raise Pi05EvaluationError("Writer cache LoRA authority changed")
    return result


def _attach_writer_cache(
    contract: dict[str, Any],
    *,
    authorities: EvaluationAuthorities,
    adapter: Mapping[str, Any] | None,
    output_dir: Path,
    writer_cache_root: Path | None,
    writer_generators_per_gpu: int,
    writer_generation_batch_size: int,
) -> None:
    contract["writer_lora_cache"] = None
    if adapter is None or adapter.get("kind") not in WRITER_ADAPTER_KINDS:
        return
    from ember.writer.evaluation_cache import build_writer_lora_cache_descriptor

    lora = _writer_lora_contract(authorities, adapter)
    root = (
        writer_cache_root.resolve()
        if writer_cache_root is not None
        else output_dir.resolve() / "writer_lora_cache"
    )
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=root,
        generators_per_gpu=writer_generators_per_gpu,
        generation_batch_size=writer_generation_batch_size,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
        lora_storage_per_entry=adapter["writer_asset"]["writer_state"][
            "template_lora_storage"
        ],
    )


def _validate_build_request(
    authorities: EvaluationAuthorities,
    *,
    model: Mapping[str, Any],
    tasks: Sequence[TargetTaskContract],
    mode: str,
    replicas_per_gpu: int,
    adapter: Mapping[str, Any] | None,
    writer_generators_per_gpu: int,
    writer_generation_batch_size: int,
    writer_cache_root: Path | None,
) -> tuple[dict[str, Any], bool]:
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
    writer_adapter = adapter is not None and adapter.get("kind") in WRITER_ADAPTER_KINDS
    valid_writer_topology = (
        0 < writer_generators_per_gpu <= replicas_per_gpu
        and writer_generation_batch_size > 0
    )
    if writer_adapter and not valid_writer_topology:
        raise Pi05EvaluationError(
            "Writer generation and rollout topology are incompatible"
        )
    if writer_adapter:
        evaluation = adapter.get("evaluation_authority", {})
        try:
            minimum_batch_size = int(
                evaluation["minimum_smoke_writer_model_batch_size"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Pi05EvaluationError(
                "Writer generation lacks its throughput authority"
            ) from error
        expected_throughput_policy = _WRITER_THROUGHPUT_BY_SCHEMA.get(
            str(adapter.get("schema_version"))
        )
        if (
            expected_throughput_policy is None
            or evaluation.get("throughput_policy") != expected_throughput_policy
            or writer_generation_batch_size < minimum_batch_size
        ):
            raise Pi05EvaluationError(
                "Writer generation batch violates its throughput authority"
            )
        smoke = evaluation.get("online_smoke_evidence")
        sealed_dynamic_k = (
            adapter.get("kind") == DYNAMIC_K_WRITER_KIND
            and evaluation.get("formal_status") == "sealed"
        )
        sealed_expert_manifold = (
            adapter.get("kind") == EXPERT_MANIFOLD_WRITER_KIND
            and evaluation.get("formal_status") in {
                "sealed",
                "sealed_from_unchanged_v6_deployment_graph",
                "sealed_from_live_pick_gc_deployment_profile",
                "sealed_from_live_residual_deployment_profile",
                "sealed_from_unchanged_v6_residual_deployment_graph",
                "sealed_from_live_osg_pc_deployment_smoke",
                "sealed_from_live_sknc_deployment_smoke",
                "sealed_from_live_srtp_deployment_smoke",
                "sealed_from_live_cveg_deployment_smoke",
                "sealed_from_live_cgik_full96_profile",
                "sealed_from_live_mgci_full96_profile",
            }
        )
        if (sealed_dynamic_k or sealed_expert_manifold) and (
            not isinstance(smoke, Mapping)
            or writer_generation_batch_size
            != int(
                smoke.get(
                    "selected_writer_model_batch_size",
                    smoke.get("writer_model_batch_size", -1),
                )
            )
        ):
            raise Pi05EvaluationError(
                "sealed Writer evaluation requires its selected Writer batch"
            )
    if not writer_adapter and writer_cache_root is not None:
        raise Pi05EvaluationError("a Writer LoRA cache was supplied without a Writer")
    return git, writer_adapter


def _paired_control_contract(
    contract: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    paired_keys = (
        "mode",
        "role",
        "git",
        "model",
        "tokenizer",
        "normalization",
        "tasks",
        "environment",
        "policy",
        "rng",
        "parallel",
    )
    return {
        "schema_version": "ember_pi05_writer_paired_control_v2",
        **{key: contract[key] for key in paired_keys},
        "writer": paired_writer_identity(adapter),
    }


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
    writer_generators_per_gpu: int = 1,
    writer_generation_batch_size: int = 8,
    writer_cache_root: Path | None = None,
) -> dict[str, Any]:
    git, writer_adapter = _validate_build_request(
        authorities,
        model=model,
        tasks=tasks,
        mode=mode,
        replicas_per_gpu=replicas_per_gpu,
        adapter=adapter,
        writer_generators_per_gpu=writer_generators_per_gpu,
        writer_generation_batch_size=writer_generation_batch_size,
        writer_cache_root=writer_cache_root,
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
            else None
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
            writer_adapter=writer_adapter,
            writer_generators_per_gpu=writer_generators_per_gpu,
            writer_generation_batch_size=writer_generation_batch_size,
        ),
        "artifacts": authorities.config["artifacts"],
        "libero_paths": dict(libero_paths),
    }
    _attach_writer_cache(
        contract,
        authorities=authorities,
        adapter=adapter,
        output_dir=output_dir,
        writer_cache_root=writer_cache_root,
        writer_generators_per_gpu=writer_generators_per_gpu,
        writer_generation_batch_size=writer_generation_batch_size,
    )
    contract["paired_control"] = None
    if writer_adapter:
        contract["paired_control"] = _paired_control_contract(contract, adapter)
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
