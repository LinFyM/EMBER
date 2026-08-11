"""One-time old-v8-cache to rank14 compiler-only closed-loop diagnostic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_ADAPTER_SCHEMA,
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_EPISODE_SCHEMA,
    load_rank_reserved_config,
    rank_reserved_asset,
    rank_reserved_output_path,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json
from ember.writer.evaluation_cache import (
    WRITER_LORA_CACHE_MANIFEST_SCHEMA,
    validate_writer_cache_manifest,
    writer_cache_manifest_is_ready,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_DIAGNOSTIC_AUTHORITY_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_compiler_only_diagnostic_authority_v1"
)
COMPILER_DIAGNOSTIC_POPULATION_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_compiler_only_population_recipe_v1"
)
COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_compiler_only_cache_transform_v1"
)
COMPILER_DIAGNOSTIC_ENTRY_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_compiler_only_cache_entry_v1"
)
COMPILER_DIAGNOSTIC_AUTHORITY = (
    REPO_ROOT
    / "configs/pi05_v6_qv_rank_reserved_compiler_only_diagnostic_v1.json"
)
COMPILER_DIAGNOSTIC_PREFLIGHT = "compiler_only_transform_preflight.json"
COMPILER_DIAGNOSTIC_TRANSFORM = "compiler_only_transform_manifest.json"
COMPILER_DIAGNOSTIC_TARGET_ROOT = (
    "runs/outputs/pi05_v6_qv_rank_reserved_compiler_only_old134_to_rank14_"
    "correct400_20260811"
)
COMPILER_DIAGNOSTIC_SOURCE_ROOT = (
    "runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_"
    "noreplacement_seed7_method_macro0000_6b5f7a6_20260810"
)
COMPILER_DIAGNOSTIC_ONLINE_ROOT = (
    "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_correct400_"
    "macro0000_20260811"
)
COMPILER_DIAGNOSTIC_SOURCE_COMMIT = (
    "6b5f7a6ad6ef1a778205071f38faec9f936cf54e"
)
COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE = (
    "ember_pi05_target_eval_launch_v2:151f9d5f9dc4499497e41306ab99363a"
)
COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v2:"
    "historical_v6_macro400_load_only:m0:base45818648bytes:residual0bytes:"
    "rank16:400episodes:seed7:batch8:native2641920bytes"
)
COMPILER_DIAGNOSTIC_ONLINE_COMMIT = (
    "0fd823f8cb5ab45164b185c0a42cb358044b095d"
)
_LATER_DOC_PATHS = {
    "AGENTS.md",
    "README.md",
    "docs/active_session_handoff.md",
    "docs/execution_brief.md",
    "docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md",
    "task_plan.md",
    "findings.md",
    "progress.md",
}


def compiler_diagnostic_population_recipe() -> dict[str, Any]:
    """Return the exact cache-identity recipe for this no-Writer population."""

    return {
        "schema_version": COMPILER_DIAGNOSTIC_POPULATION_SCHEMA,
        "mode": "prefilled",
        "reference_suffix": "old134-rank14-compiler-only-v1",
        "source_contract_reference": COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE,
        "source_cache_reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
        "batch_size": 8,
        "qv_leading_shape": [8, 18],
        "qv_helper": "compile_rank_reserved_qv_factors",
        "action_copy": "cpu_native_direct_tensor_copy_without_arithmetic",
        "action_bit_exact_validation": (
            "target_safetensors_vs_source_tensor_torch_equal"
        ),
        "ambient_bf16_autocast": True,
        "fp32_matmul_allow_tf32": True,
        "generator_processes": 0,
        "writer_forwards": 0,
        "video_reads": 0,
        "source_policy_loads": 0,
        "cuda_local_device": 0,
        "gpu_local_numa_binding": "required",
    }


def validate_compiler_diagnostic_prepare_args(args: Any) -> None:
    """Reject a malformed one-time prepare before it creates staging state."""

    try:
        config_path = args.expert_manifold_config.resolve()
        config = load_rank_reserved_config(config_path)
        asset = rank_reserved_asset(
            config, args.expert_manifold_checkpoint.resolve()
        )
        output_dir = args.output_dir.resolve()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise Pi05EvaluationError(
            "compiler-only diagnostic prepare arguments are incomplete"
        ) from error
    if not all(
        (
            output_dir == (REPO_ROOT / COMPILER_DIAGNOSTIC_TARGET_ROOT).resolve(),
            args.mode == "formal",
            args.role == "validation",
            int(args.state_count) == 50,
            int(args.writer_generation_batch_size) == 8,
            args.writer_lora_cache_root is None,
            config_path == RANK_RESERVED_CANONICAL_CONFIG.resolve(),
            asset.get("kind") == "v6_qv_rank14_zero_program_load_only",
            int(asset.get("method_macro", -1)) == 0,
            asset.get("enable_program_residual") is False,
            args.expert_manifold_video_condition == "correct",
            args.expert_manifold_video_sampling == "without_replacement",
            args.expert_manifold_video_data_root is not None,
            isinstance(args.gpu_indices, str) and bool(args.gpu_indices.strip()),
            args.source_sft_config is None,
            args.source_sft_checkpoint is None,
            args.task_expert_config is None,
            args.task_expert_bank_root is None,
            args.task_expert_step is None,
        )
    ):
        raise Pi05EvaluationError(
            "compiler-only diagnostic prepare arguments changed"
        )


def _active_config_authority() -> dict[str, Any]:
    return {
        "path": "configs/pi05_v6_qv_rank_reserved_native_reward_v1.json",
        "bytes": 5_634,
        "schema": RANK_RESERVED_CONFIG_SCHEMA,
    }


def compiler_diagnostic_cli_run(args: Any, prepare_run_fn: Any) -> Any:
    """Dispatch the four tightly scoped one-time diagnostic commands."""

    if args.command == "rank-reserved-compiler-prepare":
        validate_compiler_diagnostic_prepare_args(args)
        args.writer_cache_population_recipe = compiler_diagnostic_population_recipe()
        return prepare_run_fn(args)
    if args.command == "rank-reserved-compiler-cache":
        from ember.pi05_eval.rank_reserved_cache_launch import compiler_cache_run

        return compiler_cache_run(args)
    if args.command == "rank-reserved-compiler-cache-worker":
        from ember.pi05_eval.rank_reserved_cache_launch import (
            compiler_cache_worker_run,
        )

        return compiler_cache_worker_run(args)
    if args.command == "rank-reserved-compiler-evidence":
        from ember.pi05_eval.rank_reserved_compiler_evidence import (
            compiler_evidence_run,
        )

        return compiler_evidence_run(args)
    raise Pi05EvaluationError("unknown compiler-only diagnostic command")


def compiler_diagnostic_authority_payload(implementation_commit: str) -> dict[str, Any]:
    """Return the exact one-time authority payload for the authority-only commit."""

    return {
        "schema_version": COMPILER_DIAGNOSTIC_AUTHORITY_SCHEMA,
        "status": "authorized_one_time_after_original_gate_b_nonpass",
        "implementation_commit": implementation_commit,
        "active_config": _active_config_authority(),
        "source_old134": {
            "root": COMPILER_DIAGNOSTIC_SOURCE_ROOT,
            "commit": COMPILER_DIAGNOSTIC_SOURCE_COMMIT,
            "correct": 134,
            "breadth": 6,
            "run_contract": {
                "path": f"{COMPILER_DIAGNOSTIC_SOURCE_ROOT}/run_contract.json",
                "bytes": 128_166,
                "schema": "ember_pi05_target_eval_launch_v2",
                "reference": COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE,
            },
            "results": {
                "path": f"{COMPILER_DIAGNOSTIC_SOURCE_ROOT}/results.json",
                "bytes": 1_869_489,
                "schema": "ember_pi05_target_eval_results_v2",
            },
            "cache": {
                "root": f"{COMPILER_DIAGNOSTIC_SOURCE_ROOT}/writer_lora_cache",
                "manifest": {
                    "path": (
                        f"{COMPILER_DIAGNOSTIC_SOURCE_ROOT}/writer_lora_cache/"
                        "cache_manifest.json"
                    ),
                    "bytes": 110_652,
                    "schema": WRITER_LORA_CACHE_MANIFEST_SCHEMA,
                },
                "reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
                "entry_count": 400,
                "episode_schema": (
                    "ember_pi05_v6_condition_program_residual_episode_v8"
                ),
            },
        },
        "original_gate_b_nonpass": {
            "root": COMPILER_DIAGNOSTIC_ONLINE_ROOT,
            "commit": COMPILER_DIAGNOSTIC_ONLINE_COMMIT,
            "correct": 128,
            "breadth": 7,
            "gained_from_old134": 15,
            "lost_from_old134": 21,
            "passed": False,
            "run_contract": {
                "path": f"{COMPILER_DIAGNOSTIC_ONLINE_ROOT}/run_contract.json",
                "bytes": 130_222,
                "schema": "ember_pi05_target_eval_launch_v2",
            },
            "results": {
                "path": f"{COMPILER_DIAGNOSTIC_ONLINE_ROOT}/results.json",
                "bytes": 1_909_406,
                "schema": "ember_pi05_target_eval_results_v2",
            },
        },
        "target": {
            "root": COMPILER_DIAGNOSTIC_TARGET_ROOT,
            "mode": "formal",
            "role": "validation",
            "condition": "correct",
            "method_macro": 0,
            "task_count": 8,
            "states_per_task": 50,
            "entry_count": 400,
            "cache_root_owned_by_output": True,
            "same_rollout_identity_as_old134": True,
            "rollout_identity_fields": [
                "mode",
                "role",
                "model",
                "tokenizer",
                "normalization",
                "policy",
                "environment",
                "rng",
                "libero_paths",
                "tasks",
                "adapter_source",
            ],
        },
        "transform": {
            "population": "prefilled_without_writer_generator_marker",
            "batch_size": 8,
            "batch_count": 50,
            "qv_leading_shape": [8, 18],
            "qv_target_count": 36,
            "qv_public_topology": "pivot_rank14_plus_two_physical_zero_a_b_slots",
            "action_target_count": 2,
            "action_copy": "cpu_native_direct_tensor_copy_without_arithmetic",
            "action_bit_exact_validation": (
                "target_safetensors_vs_source_tensor_torch_equal"
            ),
            "action_tensor_equal_checks": 1_600,
            "ambient_bf16_autocast": True,
            "fp32_matmul_allow_tf32": True,
            "cuda_local_device": 0,
            "gpu_local_numa_binding": "required",
            "source_cache_read_only": True,
            "teacher_video_or_action_reads": 0,
            "policy_forwards": 0,
            "rollouts": 0,
            "training_updates": 0,
        },
        "decision": {
            "scientific_role": (
                "deconfound_rank14_compression_from_upstream_writer_regeneration"
            ),
            "counterfactual_gate_uses_original_gate_b_thresholds": True,
            "retroactively_changes_original_gate_b": False,
            "authorizes_cycle1": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def load_compiler_diagnostic_authority(
    path: Path = COMPILER_DIAGNOSTIC_AUTHORITY,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise Pi05EvaluationError("compiler-only diagnostic authority is missing")
    try:
        authority = json.loads(path.read_text(encoding="utf-8"))
        implementation_commit = str(authority["implementation_commit"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(
            "compiler-only diagnostic authority is invalid"
        ) from error
    if (
        len(implementation_commit) != 40
        or any(value not in "0123456789abcdef" for value in implementation_commit)
        or authority != compiler_diagnostic_authority_payload(implementation_commit)
    ):
        raise Pi05EvaluationError("compiler-only diagnostic authority changed")
    return authority


def compiler_diagnostic_output_path(
    relative: str, *, label: str, require_file: bool = False
) -> Path:
    try:
        return rank_reserved_output_path(
            relative,
            label=label,
            require_file=require_file,
            require_directory=not require_file,
        )
    except Exception as error:
        raise Pi05EvaluationError(f"{label} escaped canonical outputs") from error


def compiler_diagnostic_target_root() -> Path:
    return compiler_diagnostic_output_path(
        COMPILER_DIAGNOSTIC_TARGET_ROOT,
        label="compiler-only diagnostic target",
    )


def load_compiler_diagnostic_source_contract(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the immutable old134 contract using its non-hash authority."""

    source = authority["source_old134"]
    record = source["run_contract"]
    path = compiler_diagnostic_output_path(
        str(record["path"]),
        label="compiler-only source run contract",
        require_file=True,
    )
    contract = read_json(path)
    if (
        path.stat().st_size != int(record["bytes"])
        or contract.get("schema_version") != record["schema"]
        or contract.get("contract_reference") != record["reference"]
        or contract.get("git", {}).get("commit") != source["commit"]
        or contract.get("mode") != "formal"
        or contract.get("role") != "validation"
        or contract.get("adapter", {}).get("schema_version")
        != "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
    ):
        raise Pi05EvaluationError("compiler-only source run contract changed")
    return contract


def _configs_suffix(value: object) -> str:
    parts = Path(str(value)).parts
    try:
        index = len(parts) - 1 - tuple(reversed(parts)).index("configs")
    except ValueError as error:
        raise Pi05EvaluationError(
            "compiler-only repository asset path changed"
        ) from error
    return str(Path(*parts[index:]))


def compiler_diagnostic_rollout_projection(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project stable rollout identity while removing frozen-worktree prefixes."""

    try:
        tokenizer = contract["tokenizer"]
        normalization = contract["normalization"]
        adapter_source = contract["adapter"]["source"]
        return {
            "mode": contract["mode"],
            "role": contract["role"],
            "model": contract["model"],
            "tokenizer": {
                "path": tokenizer["path"],
                "bytes": tokenizer["bytes"],
                "manifest": _configs_suffix(tokenizer["manifest_path"]),
            },
            "normalization": {
                **{
                    key: value
                    for key, value in normalization.items()
                    if key != "path"
                },
                "path": _configs_suffix(normalization["path"]),
            },
            "policy": contract["policy"],
            "environment": contract["environment"],
            "rng": contract["rng"],
            "libero_paths": contract["libero_paths"],
            "tasks": [
                {
                    **task,
                    "init_state_ids": list(task["init_state_ids"]),
                }
                for task in contract["tasks"]
            ],
            "adapter_source": adapter_source,
        }
    except (KeyError, TypeError) as error:
        raise Pi05EvaluationError(
            "compiler-only rollout identity is incomplete"
        ) from error


def compiler_diagnostic_rollout_contract_matches(
    authority: Mapping[str, Any],
    target_contract: Mapping[str, Any],
) -> bool:
    try:
        source_contract = load_compiler_diagnostic_source_contract(authority)
        paired = target_contract["paired_control"]
        paired_fields = (
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
        return (
            compiler_diagnostic_rollout_projection(target_contract)
            == compiler_diagnostic_rollout_projection(source_contract)
            and all(
                paired.get(name) == target_contract.get(name)
                for name in paired_fields
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def is_compiler_diagnostic_output(output_dir: Path) -> bool:
    try:
        expected = (REPO_ROOT / COMPILER_DIAGNOSTIC_TARGET_ROOT).resolve()
        return output_dir.resolve() == expected
    except OSError:
        return False


def compiler_diagnostic_lineage_matches(
    authority: Mapping[str, Any],
    evaluation_commit: str,
) -> bool:
    implementation_commit = str(authority.get("implementation_commit", ""))
    if len(implementation_commit) != 40 or len(evaluation_commit) != 40:
        return False
    authority_relative = str(COMPILER_DIAGNOSTIC_AUTHORITY.relative_to(REPO_ROOT))
    try:
        ancestry = subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                evaluation_commit,
            ),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        changed = subprocess.run(
            ("git", "diff", "--name-only", implementation_commit, evaluation_commit),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    paths = {row for row in changed.stdout.splitlines() if row}
    return (
        ancestry.returncode == 0
        and authority_relative in paths
        and paths <= ({authority_relative} | _LATER_DOC_PATHS)
    )


def _target_contract_matches(
    authority: Mapping[str, Any],
    output_dir: Path,
    contract: Mapping[str, Any],
) -> bool:
    target = authority["target"]
    adapter = contract.get("adapter", {})
    asset = adapter.get("writer_asset", {})
    descriptor = contract.get("writer_lora_cache", {})
    recipe = descriptor.get("generation_recipe", {})
    parallel = contract.get("parallel", {})
    tasks = contract.get("tasks", ())
    commit = str(contract.get("git", {}).get("commit", ""))
    try:
        active = load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG)
        selected = rank_reserved_asset(
            active,
            Path(str(asset.get("checkpoint", ""))),
        )
        config_record = adapter["config"]
        cache_root = Path(str(descriptor["root"])).resolve()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return all(
        (
            output_dir.resolve()
            == (REPO_ROOT / str(target["root"])).resolve(),
            contract.get("output_dir") == str(output_dir.resolve()),
            contract.get("mode") == target["mode"],
            contract.get("role") == target["role"],
            adapter.get("schema_version") == RANK_RESERVED_ADAPTER_SCHEMA,
            adapter.get("video_condition") == target["condition"],
            adapter.get("video_schedule", {}).get("sampling_mode")
            == "without_replacement",
            int(asset.get("method_macro", -1)) == int(target["method_macro"]),
            asset.get("kind") == selected.get("kind"),
            asset.get("enable_program_residual") is False,
            config_record.get("schema") == RANK_RESERVED_CONFIG_SCHEMA,
            int(config_record.get("bytes", -1))
            == int(authority["active_config"]["bytes"]),
            Path(str(config_record.get("path", ""))).name
            == RANK_RESERVED_CANONICAL_CONFIG.name,
            RANK_RESERVED_CANONICAL_CONFIG.stat().st_size
            == int(authority["active_config"]["bytes"]),
            cache_root == output_dir.resolve() / "writer_lora_cache",
            not cache_root.is_symlink(),
            int(descriptor.get("entry_count", -1)) == int(target["entry_count"]),
            int(recipe.get("generation_batch_size", -1))
            == int(authority["transform"]["batch_size"]),
            recipe.get("population") == compiler_diagnostic_population_recipe(),
            int(recipe.get("generators_per_gpu", -1)) == 0,
            int(recipe.get("generator_worker_count", -1)) == 0,
            recipe.get("episode_evidence_schema") == RANK_RESERVED_EPISODE_SCHEMA,
            descriptor.get("population_mode") == "prefilled",
            descriptor.get("persistent_source_policy_handoff") is False,
            descriptor.get("writer_modules_released_before_rollout_scale_out")
            is False,
            descriptor.get("writer_or_video_forward_required") is False,
            descriptor.get("source_policy_load_required_for_population") is False,
            int(parallel.get("writer_generators_per_gpu", -1)) == 0,
            int(parallel.get("writer_generation_worker_count", -1)) == 0,
            int(parallel.get("writer_generation_batch_size", -1)) == 8,
            parallel.get("writer_and_rollout_parallelism_decoupled") is True,
            parallel.get("generator_source_policy_processes_reused_for_rollout")
            is False,
            int(parallel.get("worker_count", -1))
            == int(parallel.get("physical_gpu_count", -2))
            * int(parallel.get("replicas_per_gpu", -3)),
            str(descriptor.get("reference", "")).endswith(
                ":population-old134-rank14-compiler-only-v1"
            ),
            descriptor.get("identity", {}).get("implementation_commit") == commit,
            len(tasks) == int(target["task_count"]),
            target.get("same_rollout_identity_as_old134") is True,
            all(
                row.get("split_role") == "validation"
                and tuple(row.get("init_state_ids", ()))
                == tuple(range(int(target["states_per_task"])))
                for row in tasks
            ),
            git_state_is_clean_pushed_or_frozen_authority(contract.get("git", {})),
            compiler_diagnostic_lineage_matches(authority, commit),
            compiler_diagnostic_rollout_contract_matches(authority, contract),
        )
    )


def validate_compiler_diagnostic_contract(
    output_dir: Path,
    contract: Mapping[str, Any],
    *,
    require_cache_ready: bool,
) -> dict[str, Any]:
    authority = load_compiler_diagnostic_authority()
    if not _target_contract_matches(authority, output_dir, contract):
        raise Pi05EvaluationError("compiler-only diagnostic contract changed")
    if require_cache_ready:
        if not writer_cache_manifest_is_ready(contract):
            raise Pi05EvaluationError(
                "compiler-only diagnostic cache is missing; "
                "Writer fallback is forbidden"
            )
        validate_completed_compiler_transform(authority, output_dir, contract)
    return authority


def validate_completed_compiler_transform(
    authority: Mapping[str, Any],
    output_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    transform_path = output_dir / COMPILER_DIAGNOSTIC_TRANSFORM
    if not transform_path.is_file() or transform_path.is_symlink():
        raise Pi05EvaluationError("compiler-only transform manifest is missing")
    transform = read_json(transform_path)
    manifest = validate_writer_cache_manifest(contract, verify_entry_files=False)
    preflight = transform.get("preflight", {})
    expected_population = {
        "schema_version": COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA,
        "path": COMPILER_DIAGNOSTIC_TRANSFORM,
        "bytes": transform_path.stat().st_size,
        "implementation_commit": authority["implementation_commit"],
        "evaluation_commit": contract["git"]["commit"],
        "source_cache_reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
        "authorizes_cycle1": False,
    }
    if (
        transform.get("schema_version") != COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA
        or transform.get("status") != "complete"
        or transform.get("root") != str(output_dir.resolve())
        or transform.get("implementation_commit")
        != authority["implementation_commit"]
        or transform.get("evaluation_commit") != contract["git"]["commit"]
        or transform.get("target", {}).get("contract_reference")
        != contract["contract_reference"]
        or int(transform.get("batch_size", -1)) != 8
        or int(transform.get("batch_count", -1)) != 50
        or transform.get("qv_leading_shape") != [8, 18]
        or transform.get("ambient_bf16_autocast") is not True
        or transform.get("fp32_matmul_allow_tf32") is not True
        or int(transform.get("source_target_action_tensor_equal_checks", -1))
        != 1_600
        or transform.get("source_target_action_tensors_bit_exact") is not True
        or int(preflight.get("visible_cuda_device", -1)) != 0
        or int(preflight.get("numa_node", -1)) < 0
        or not isinstance(preflight.get("cpu_affinity"), list)
        or not preflight.get("cpu_affinity")
        or transform.get("authorizes_cycle1") is not False
        or manifest.get("population", {}).get("evidence") != expected_population
    ):
        raise Pi05EvaluationError("compiler-only completed transform changed")
    return transform
