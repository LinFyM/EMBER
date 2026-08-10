"""Live profile and vertical evidence for rank-reserved deployment sealing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.rank_reserved_authority import (
    RANK_RESERVED_ADAPTER_SCHEMA,
    RANK_RESERVED_EPISODE_SCHEMA,
    _REGISTERED_ROOTS,
    rank_reserved_output_path,
)
from ember.expert_manifold.v6_prior_contract import git_commit_in_active_authority_lineage


def _deployment_evidence_matches(evidence: object) -> bool:
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema",
        "run_commit",
        "writer_model_batch_size",
        "profile",
        "vertical",
    }:
        return False
    try:
        batch_size = int(evidence.get("writer_model_batch_size", -1))
    except (TypeError, ValueError):
        return False
    if (
        evidence.get("schema") != "ember_pi05_v6_qv_rank_reserved_deployment_seal_v1"
        or batch_size not in (8, 16, 32)
        or not git_commit_in_active_authority_lineage(
            str(evidence.get("run_commit", ""))
        )
    ):
        return False
    try:
        profile = _validated_profile_record(
            evidence["profile"],
            run_commit=str(evidence["run_commit"]),
            selected_batch=batch_size,
        )
        _validated_vertical_record(
            evidence["vertical"],
            run_commit=str(evidence["run_commit"]),
            selected_batch=batch_size,
            profile=profile,
        )
    except (ExpertManifoldError, KeyError, TypeError, ValueError):
        return False
    return True


def _evidence_file(
    record: object,
    *,
    root_name: str,
    filename: str,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes"}:
        raise ExpertManifoldError("rank-reserved deployment evidence record changed")
    expected_relative = Path(_REGISTERED_ROOTS[root_name]) / filename
    if str(record.get("path")) != expected_relative.as_posix():
        raise ExpertManifoldError("rank-reserved deployment evidence path changed")
    try:
        expected_bytes = int(record["bytes"])
    except (TypeError, ValueError) as error:
        raise ExpertManifoldError(
            "rank-reserved deployment evidence byte count changed"
        ) from error
    return rank_reserved_output_path(
        expected_relative.as_posix(),
        label=f"rank-reserved {root_name} evidence",
        expected_bytes=expected_bytes,
        require_file=True,
    )


def _read_json_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ExpertManifoldError(f"invalid {label} JSON")
    return value


def _valid_idle_a40_preflight(value: object, *, physical_count: int = 1) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("compute_applications") == []
        and value.get("device_names") == ["NVIDIA A40"] * physical_count
        and len(value.get("physical_gpu_ids", ())) == physical_count
        and len(value.get("gpus", ())) == physical_count
    )


def _validated_profile_record(
    record: object,
    *,
    run_commit: str,
    selected_batch: int,
) -> dict[str, Any]:
    path = _evidence_file(
        record,
        root_name="profile",
        filename="writer_generation_profile.json",
    )
    result = _read_json_artifact(path, label="rank-reserved profile")
    root = path.parent.resolve()
    contract = _read_json_artifact(
        root / "run_contract.json",
        label="rank-reserved profile run contract",
    )
    measurements = result.get("writer_generation_measurements")
    if not isinstance(measurements, list) or len(measurements) != 3:
        raise ExpertManifoldError("rank-reserved profile measurements changed")
    by_batch = {int(row.get("batch_size", -1)): row for row in measurements}
    eligible = [
        row
        for row in measurements
        if row.get("stable") is True
        and int(row.get("memory_headroom_bytes", -1))
        >= int(row.get("required_memory_headroom_bytes", 0))
    ]
    chosen = max(
        eligible,
        key=lambda row: (float(row["loras_per_second"]), int(row["batch_size"])),
        default=None,
    )
    adapter = contract.get("adapter", {})
    asset = adapter.get("writer_asset", {})
    panel_ids = [tuple(row.get("entry_ids", ())) for row in measurements]
    panel_counts = [tuple(row.get("sampled_frame_counts", ())) for row in measurements]

    def measurement_valid(row: Mapping[str, Any]) -> bool:
        oom = int(row.get("oom_count", -1))
        completed = int(row.get("completed_measured_runs", -1))
        if oom not in (0, 1) or completed not in (0, 1, 2):
            return False
        if row.get("stable") is not True:
            return True
        return (
            oom == 0
            and completed == 2
            and int(row.get("generated_entries", -1)) == 64
        )

    valid = all(
        (
            result.get("schema_version")
            == "ember_pi05_writer_generation_profile_v2",
            Path(str(result.get("root", ""))).resolve() == root,
            result.get("device") == "NVIDIA A40",
            result.get("git", {}).get("commit") == run_commit,
            result.get("contract_reference") == contract.get("contract_reference"),
            _valid_idle_a40_preflight(result.get("preflight")),
            result.get("profiled_writer_model_batch_sizes") == [8, 16, 32],
            set(by_batch) == {8, 16, 32},
            chosen is not None,
            chosen is not None and int(chosen["batch_size"]) == selected_batch,
            int(result.get("selected_writer_model_batch_size", -1))
            == selected_batch,
            len(set(panel_ids)) == 1,
            len(set(panel_counts)) == 1,
            all(row.get("longest_video_included") is True for row in measurements),
            all(measurement_valid(row) for row in measurements),
            result.get("writer_modules_released") is True,
            result.get("source_policy_reused") is True,
            int(result.get("teacher_action_reads", -1)) == 0,
            int(result.get("teacher_state_reads", -1)) == 0,
            int(result.get("reward_reads", -1)) == 0,
            int(result.get("terminal_reads", -1)) == 0,
            int(result.get("oom_count", -1))
            == sum(int(row.get("oom_count", -1)) for row in measurements),
            int(result.get("nonfinite_count", -1)) == 0,
            contract.get("mode") == "smoke",
            contract.get("role") == "validation",
            Path(str(contract.get("output_dir", ""))).resolve() == root,
            contract.get("git", {}).get("commit") == run_commit,
            adapter.get("schema_version") == RANK_RESERVED_ADAPTER_SCHEMA,
            adapter.get("video_condition") == "correct",
            asset.get("kind") == "v6_qv_rank14_plus2_reward_program_load_only",
            int(asset.get("method_macro", -1)) == 1,
            asset.get("enable_program_residual") is True,
            int(contract.get("parallel", {}).get("physical_gpu_count", -1)) == 1,
            int(contract.get("parallel", {}).get("replicas_per_gpu", -1)) == 1,
            int(
                contract.get("parallel", {}).get(
                    "writer_generators_per_gpu",
                    -1,
                )
            )
            == 1,
            int(
                contract.get("parallel", {}).get(
                    "writer_generation_batch_size",
                    -1,
                )
            )
            == 32,
        )
    )
    if not valid:
        raise ExpertManifoldError("rank-reserved profile evidence changed")
    return result


def load_rank_reserved_profile_evidence(
    config: Mapping[str, Any],
    *,
    require_run_commit: str | None = None,
) -> dict[str, Any]:
    """Load the registered throughput winner before launching the vertical smoke."""

    root = rank_reserved_output_path(
        config["evaluation"]["registered_roots"]["profile"],
        label="rank-reserved profile root",
    )
    path = root / "writer_generation_profile.json"
    if not path.is_file():
        raise ExpertManifoldError("rank-reserved profile evidence is missing")
    result = _read_json_artifact(path, label="rank-reserved profile")
    commit = str(result.get("git", {}).get("commit", ""))
    selected = int(result.get("selected_writer_model_batch_size", -1))
    record = {
        "path": (
            Path(config["evaluation"]["registered_roots"]["profile"])
            / "writer_generation_profile.json"
        ).as_posix(),
        "bytes": path.stat().st_size,
    }
    validated = _validated_profile_record(
        record,
        run_commit=commit,
        selected_batch=selected,
    )
    if require_run_commit is not None and commit != require_run_commit:
        raise ExpertManifoldError(
            "rank-reserved vertical and profile implementation commits differ"
        )
    return validated


def _load_vertical_artifacts(record: object) -> dict[str, Any]:
    path = _evidence_file(
        record,
        root_name="vertical",
        filename="rank_reserved_vertical.json",
    )
    root = path.parent.resolve()
    return {
        "root": root,
        "mechanism": _read_json_artifact(path, label="rank-reserved vertical"),
        "contract": _read_json_artifact(
            root / "run_contract.json",
            label="rank-reserved vertical run contract",
        ),
        "results": _read_json_artifact(
            root / "results.json",
            label="rank-reserved vertical results",
        ),
        "cache": _read_json_artifact(
            root / "writer_lora_cache/cache_manifest.json",
            label="rank-reserved vertical cache manifest",
        ),
        "preflight": _read_json_artifact(
            root / "rank_reserved_vertical_preflight.json",
            label="rank-reserved vertical preflight",
        ),
    }


def _vertical_mechanism_matches(
    artifacts: Mapping[str, Any],
    *,
    run_commit: str,
    selected_batch: int,
) -> bool:
    mechanism = artifacts["mechanism"]
    contract = artifacts["contract"]
    preflight = artifacts["preflight"]
    storage = mechanism.get("native_lora_storage")
    qv = mechanism.get("native_qv_effective", {})
    return all(
        (
            mechanism.get("schema_version")
            == "ember_pi05_v6_qv_rank_reserved_native_vertical_v1",
            mechanism.get("passed") is True,
            Path(str(mechanism.get("root", ""))).resolve() == artifacts["root"],
            mechanism.get("git", {}).get("commit") == run_commit,
            mechanism.get("contract_reference")
            == contract.get("contract_reference"),
            mechanism.get("preflight") == preflight,
            _valid_idle_a40_preflight(preflight),
            mechanism.get("native_storage_valid") is True,
            mechanism.get("cache_storage_valid") is True,
            mechanism.get("cache_video_identity_exact") is True,
            mechanism.get("canonical_cache_used_for_full_reward_action") is True,
            mechanism.get("canonical_cache_used_for_qv_only_action") is True,
            mechanism.get("cached_reward_paired_base_zeroed_from_same_state")
            is True,
            isinstance(storage, Mapping),
            isinstance(storage, Mapping)
            and storage.get("dtype_tensor_counts") == {"BF16": 72, "F32": 4},
            isinstance(storage, Mapping)
            and int(storage.get("tensor_count", -1)) == 76,
            mechanism.get("writer_modules_released_before_actions") is True,
            mechanism.get("source_policy_reused") is True,
            mechanism.get("source_identity_action_exact") is True,
            mechanism.get("macro0_qv_residual_slots", {}).get("exact_zero") is True,
            int(
                mechanism.get("macro0_qv_residual_slots", {}).get(
                    "nonzero_values",
                    -1,
                )
            )
            == 0,
            mechanism.get("cached_paired_base_qv_residual_slots", {}).get(
                "exact_zero"
            )
            is True,
            int(
                mechanism.get("cached_paired_base_qv_residual_slots", {}).get(
                    "nonzero_values",
                    -1,
                )
            )
            == 0,
            int(qv.get("new_nonzero_targets", -1)) == 144,
            int(qv.get("target_comparisons", -2)) == 144,
            all(
                int(mechanism.get(name, -1)) == 0
                for name in (
                    "teacher_action_reads",
                    "teacher_state_reads",
                    "reward_reads",
                    "terminal_reads",
                )
            ),
            int(mechanism.get("configured_writer_generation_batch_size", -1))
            == selected_batch,
            int(
                mechanism.get(
                    "expected_actual_cache_generation_batch_size",
                    -1,
                )
            )
            == min(selected_batch, 8),
            int(mechanism.get("diagnostic_video_batch_size", -1)) == 4,
            int(mechanism.get("diagnostic_video_encoder_forwards", -1)) == 1,
            int(mechanism.get("diagnostic_policy_action_forwards", -1)) == 12,
            mechanism.get("diagnostic_policy_action_batch_sizes")
            == [[5, 1, 1]] * 4,
            int(mechanism.get("diagnostic_policy_action_samples", -1)) == 28,
            float(
                mechanism.get(
                    "diagnostic_generation_plus_stage_seconds",
                    0.0,
                )
            )
            > 0.0,
            float(mechanism.get("diagnostic_action_seconds", 0.0)) > 0.0,
            int(mechanism.get("diagnostic_action_peak_allocated_bytes", -1)) > 0,
            int(mechanism.get("diagnostic_action_peak_reserved_bytes", -1)) > 0,
        )
    )


def _vertical_row_matches(row: Mapping[str, Any]) -> bool:
    return all(
        (
            int(
                row.get("native_qv_effective", {}).get(
                    "new_nonzero_targets",
                    -1,
                )
            )
            == 36,
            float(
                row.get("qv_only_vs_full_policy_action", {}).get(
                    "old_delta_l2_rms_across_panel",
                    0.0,
                )
            )
            > 0.0,
            float(
                row.get("qv_only_vs_full_policy_action", {}).get(
                    "new_delta_l2_rms_across_panel",
                    0.0,
                )
            )
            > 0.0,
            row.get("source_identity_action_exact") is True,
        )
    )


def _vertical_video_matches(row: Mapping[str, Any]) -> bool:
    return all(
        (
            len(row.get("teacher_demo_indices", ())) == 1,
            int(row.get("sampled_frames", 0)) > 0,
            int(row.get("raw_frames", 0))
            >= int(row.get("sampled_frames", 0)),
        )
    )


def _vertical_panel_matches(artifacts: Mapping[str, Any]) -> bool:
    mechanism = artifacts["mechanism"]
    rows = mechanism.get("rows")
    video = mechanism.get("teacher_video_evidence")
    expected_suites = {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }
    return all(
        (
            isinstance(rows, list),
            isinstance(rows, list) and len(rows) == 4,
            isinstance(rows, list)
            and {str(row.get("suite")) for row in rows} == expected_suites,
            isinstance(rows, list) and all(_vertical_row_matches(row) for row in rows),
            isinstance(video, list),
            isinstance(video, list) and len(video) == 4,
            isinstance(video, list)
            and {str(row.get("suite")) for row in video} == expected_suites,
            isinstance(video, list)
            and all(_vertical_video_matches(row) for row in video),
        )
    )


def _vertical_contract_cache_matches(
    artifacts: Mapping[str, Any],
    *,
    run_commit: str,
    selected_batch: int,
) -> bool:
    contract = artifacts["contract"]
    cache = artifacts["cache"]
    mechanism = artifacts["mechanism"]
    adapter = contract.get("adapter", {})
    asset = adapter.get("writer_asset", {})
    descriptor = cache.get("descriptor", {})
    identity = descriptor.get("identity", {})
    recipe = descriptor.get("generation_recipe", {})
    storage = mechanism.get("native_lora_storage")
    return all(
        (
            contract.get("mode") == "smoke",
            contract.get("role") == "validation",
            Path(str(contract.get("output_dir", ""))).resolve()
            == artifacts["root"],
            contract.get("git", {}).get("commit") == run_commit,
            adapter.get("schema_version") == RANK_RESERVED_ADAPTER_SCHEMA,
            adapter.get("video_condition") == "correct",
            asset.get("kind") == "v6_qv_rank14_plus2_reward_program_load_only",
            int(asset.get("method_macro", -1)) == 1,
            asset.get("enable_program_residual") is True,
            int(contract.get("parallel", {}).get("physical_gpu_count", -1)) == 1,
            int(contract.get("parallel", {}).get("replicas_per_gpu", -1)) == 1,
            int(
                contract.get("parallel", {}).get(
                    "writer_generators_per_gpu",
                    -1,
                )
            )
            == 1,
            int(
                contract.get("parallel", {}).get(
                    "writer_generation_batch_size",
                    -1,
                )
            )
            == selected_batch,
            len(contract.get("tasks", ())) == 8,
            all(
                tuple(task.get("init_state_ids", ())) == (0,)
                for task in contract.get("tasks", ())
            ),
            cache.get("schema_version")
            == "ember_pi05_expert_manifold_writer_lora_cache_manifest_v3",
            len(cache.get("entry_ids", ())) == 8,
            identity.get("implementation_commit") == run_commit,
            identity.get("adapter", {}).get("schema_version")
            == RANK_RESERVED_ADAPTER_SCHEMA,
            int(recipe.get("generation_batch_size", -1)) == selected_batch,
            recipe.get("episode_evidence_schema") == RANK_RESERVED_EPISODE_SCHEMA,
            recipe.get("storage_per_entry") == storage,
        )
    )


def _vertical_results_match(
    artifacts: Mapping[str, Any],
    *,
    run_commit: str,
    selected_batch: int,
    profile: Mapping[str, Any],
) -> bool:
    results = artifacts["results"]
    contract = artifacts["contract"]
    generation = results.get("writer_generation", {})
    attempts = results.get("launcher_attempts", {})
    return all(
        (
            results.get("schema_version") == "ember_pi05_target_eval_results_v2",
            results.get("contract_reference") == contract.get("contract_reference"),
            results.get("adapter", {}).get("schema_version")
            == RANK_RESERVED_ADAPTER_SCHEMA,
            results.get("paired_control", {}).get("git", {}).get("commit")
            == run_commit,
            int(results.get("overall", {}).get("episodes", -1)) == 8,
            len(results.get("rows", ())) == 8,
            all(
                int(shard.get("attempt", -1)) == 1
                for shard in results.get("shards", ())
            ),
            len(attempts.get("attempts", ())) == 1,
            int(attempts.get("completed_before_final_attempt", -1)) == 0,
            generation.get("all_source_policies_not_reloaded") is True,
            generation.get("all_source_policy_processes_reused_for_rollout")
            is True,
            generation.get("all_writer_modules_released") is True,
            int(generation.get("assigned_entries", -1)) == 8,
            int(generation.get("generated_entries", -1)) == 8,
            int(generation.get("reused_entries", -1)) == 0,
            int(generation.get("redundant_writer_forwards", -1)) == 0,
            generation.get("generation_batch_size") == [selected_batch],
            int(generation.get("max_observed_forward_batch_size", -1))
            == min(selected_batch, 8),
            int(generation.get("generated_batches", -1)) == 1,
            len(generation.get("batches", ())) == 1,
            int(generation.get("batches", [{}])[0].get("batch_size", -1))
            == min(selected_batch, 8),
            results.get("launcher", {}).get("preflight", {}).get(
                "compute_applications"
            )
            == [],
            results.get("launcher", {}).get("preflight", {}).get("device_names")
            == ["NVIDIA A40"],
            profile.get("git", {}).get("commit") == run_commit,
            int(profile.get("selected_writer_model_batch_size", -1))
            == selected_batch,
        )
    )


def _validated_vertical_record(
    record: object,
    *,
    run_commit: str,
    selected_batch: int,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _load_vertical_artifacts(record)
    valid = all(
        (
            _vertical_mechanism_matches(
                artifacts,
                run_commit=run_commit,
                selected_batch=selected_batch,
            ),
            _vertical_panel_matches(artifacts),
            _vertical_contract_cache_matches(
                artifacts,
                run_commit=run_commit,
                selected_batch=selected_batch,
            ),
            _vertical_results_match(
                artifacts,
                run_commit=run_commit,
                selected_batch=selected_batch,
                profile=profile,
            ),
        )
    )
    if not valid:
        raise ExpertManifoldError("rank-reserved vertical evidence changed")
    return artifacts["mechanism"]


def build_rank_reserved_deployment_evidence(
    config: Mapping[str, Any],
    *,
    require_run_commit: str | None = None,
) -> dict[str, Any]:
    """Rebuild the one deployment seal exclusively from registered live artifacts."""

    profile_relative = (
        Path(config["evaluation"]["registered_roots"]["profile"])
        / "writer_generation_profile.json"
    ).as_posix()
    vertical_relative = (
        Path(config["evaluation"]["registered_roots"]["vertical"])
        / "rank_reserved_vertical.json"
    ).as_posix()
    profile_path = rank_reserved_output_path(
        profile_relative,
        label="rank-reserved profile evidence",
        require_file=True,
    )
    vertical_path = rank_reserved_output_path(
        vertical_relative,
        label="rank-reserved vertical evidence",
        require_file=True,
    )
    profile_raw = _read_json_artifact(profile_path, label="rank-reserved profile")
    run_commit = str(profile_raw.get("git", {}).get("commit", ""))
    selected_batch = int(profile_raw.get("selected_writer_model_batch_size", -1))
    if require_run_commit is not None and run_commit != require_run_commit:
        raise ExpertManifoldError(
            "rank-reserved deployment artifacts use another implementation commit"
        )
    profile_record = {
        "path": profile_relative,
        "bytes": profile_path.stat().st_size,
    }
    vertical_record = {
        "path": vertical_relative,
        "bytes": vertical_path.stat().st_size,
    }
    profile = _validated_profile_record(
        profile_record,
        run_commit=run_commit,
        selected_batch=selected_batch,
    )
    _validated_vertical_record(
        vertical_record,
        run_commit=run_commit,
        selected_batch=selected_batch,
        profile=profile,
    )
    evidence = {
        "schema": "ember_pi05_v6_qv_rank_reserved_deployment_seal_v1",
        "run_commit": run_commit,
        "writer_model_batch_size": selected_batch,
        "profile": profile_record,
        "vertical": vertical_record,
    }
    if not _deployment_evidence_matches(evidence):
        raise ExpertManifoldError("rank-reserved deployment seal did not close")
    return evidence
