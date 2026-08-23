"""Validate and bind projected task-expert adapter surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_source_checkpoint import read_json


PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_writer_fixed_head_projected_task_expert_eval_adapter_v1"
)
PROJECTED_TASK_EXPERT_MANIFEST_SCHEMA = "ember_writer_fixed_head_reachability_oracle_v1"
FUNCTIONAL_DECODER_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_functional_decoder_projected_task_expert_eval_adapter_v1"
)
FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_functional_decoder_train24_projection_v1"
)
FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_functional_decoder_nonheld_meta_projection_v1"
)
PHASE_ALIGNED_DECODER_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_phase_aligned_functional_decoder_train24_projection_v1"
)
ECP_STAGE1_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_ecp_stage1_process_value_selector_task_expert_eval_adapter_v10"
)
ECP_STAGE1_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_ecp_stage1_process_value_selector_projection_v10"
)
ECP_STAGE1_OUTCOME_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_ecp_stage1_outcome_binding_task_expert_eval_adapter_v11"
)
ECP_STAGE1_OUTCOME_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_ecp_stage1_outcome_binding_projection_v11"
)
ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_ecp_stage1_program_locked_compiler_task_expert_eval_adapter_v20"
)
ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_ecp_stage1_program_locked_compiler_projection_v20"
)
ECP_STAGE1_MDCO_TASK_EXPERT_ADAPTER_SCHEMA = (
    "ember_pi05_ecp_stage1_mapping_diverse_compiler_oracle_eval_adapter_v1"
)
ECP_STAGE1_MDCO_TASK_EXPERT_MANIFEST_SCHEMA = (
    "ember_ecp_stage1_mapping_diverse_compiler_oracle_projection_v1"
)
ECP_STAGE1_STATIC_LORA_ADAPTER_SCHEMA = (
    "ember_pi05_ecp_stage1_privileged_static_lora_eval_adapter_v1"
)
ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA = (
    "ember_ecp_stage1_privileged_static_lora_projection_v1"
)

ECP_STAGE1_STATIC_LORA_PURPOSES = {
    "stage1a_independent_particle_step2000",
    "stage1a_candidate_pecs_trajectory",
    "stage1b_occupancy_complete_oracle",
}


def _projection_file(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    record = manifest.get(name, {})
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ExpertManifoldError("projected adapter asset changed")
    return {"path": str(path), "bytes": path.stat().st_size}


def _require_values(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if any(observed.get(name) != value for name, value in expected.items()):
        raise ExpertManifoldError(label)


def _ecp_projection_spec(schema: Any) -> dict[str, Any]:
    if schema == ECP_STAGE1_MDCO_TASK_EXPERT_MANIFEST_SCHEMA:
        return {
            "projection_kind": "ecp_stage1_mapping_diverse_compiler_oracle",
            "objective_phase": (
                "task_equal_mapping_diverse_q_pi_compiler_identification"
            ),
            "adapter_schema": ECP_STAGE1_MDCO_TASK_EXPERT_ADAPTER_SCHEMA,
            "cursor_name": "task_visits",
            "arm_prefix": "ecp_stage1_mdco_tv",
            "parameterization": (
                "one layer-resolved direct-absolute A/B surface with continuous "
                "static/process fusion"
            ),
        }
    if schema == ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_MANIFEST_SCHEMA:
        return {
            "projection_kind": (
                "ecp_stage1_privileged_program_locked_compiler_identification"
            ),
            "objective_phase": ("task_balanced_program_locked_compiler_identification"),
            "adapter_schema": ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_ADAPTER_SCHEMA,
            "cursor_name": "task_visits",
            "arm_prefix": "ecp_stage1_q_pi_program_locked_compiler_tv",
            "parameterization": (
                "prior-only exact template; full-process process-value-only "
                "bounded rank-one retraction"
            ),
        }
    if schema == ECP_STAGE1_OUTCOME_TASK_EXPERT_MANIFEST_SCHEMA:
        return {
            "projection_kind": "ecp_stage1_privileged_outcome_binding_compiler",
            "objective_phase": "outcome_calibrated_policy_support",
            "adapter_schema": ECP_STAGE1_OUTCOME_TASK_EXPERT_ADAPTER_SCHEMA,
            "cursor_name": "outcome_macro",
            "arm_prefix": "ecp_stage1_q_pi_outcome_binding_m",
            "parameterization": (
                "prior-only exact template; full-process process-value-only "
                "bounded rank-one retraction"
            ),
        }
    if schema == ECP_STAGE1_TASK_EXPERT_MANIFEST_SCHEMA:
        return {
            "projection_kind": (
                "ecp_stage1_privileged_process_value_selector_compiler"
            ),
            "objective_phase": "policy_support",
            "adapter_schema": ECP_STAGE1_TASK_EXPERT_ADAPTER_SCHEMA,
            "cursor_name": "task_visits",
            "arm_prefix": "ecp_stage1_q_pi_process_value_selector_tv",
            "parameterization": (
                "prior-only exact template; full-process process-value-only "
                "bounded rank-one retraction"
            ),
        }
    raise ExpertManifoldError("ECP Stage 1 projection schema changed")


def _ecp_projection_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    optimization = manifest.get("optimization", {})
    information_wall = manifest.get("information_wall", {})
    schema = manifest.get("schema_version")
    spec = _ecp_projection_spec(schema)
    expected = {
        "held_shared_gradient_steps": 0,
        "compiler_frozen_for_materialization": True,
        "single_complete_lora": True,
        "final_lora_averaging": False,
        "rank": 16,
        "all_ranks_writable": True,
        "parameterization": spec["parameterization"],
        "content_address_separated": True,
        "query_content_modulated": True,
        "policy_support_teacher": True,
        "fixed_rank_partition": False,
        "second_adapter_deployed": False,
        "objective_phase": spec["objective_phase"],
    }
    if schema == ECP_STAGE1_MDCO_TASK_EXPERT_MANIFEST_SCHEMA:
        expected.update(
            fit_task_count=90,
            held_task_count=5,
            compiler_trainable_during_training=True,
            visible_program_frozen_during_training=True,
            policy_teacher_frozen_during_training=False,
            raw_factor_amplitude_retained=True,
        )
    else:
        expected["raw_factor_addition"] = False
    if schema == ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_MANIFEST_SCHEMA:
        expected.update(
            compiler_trainable_during_training=True,
            visible_program_frozen_during_training=True,
            policy_teacher_frozen_during_training=True,
        )
    if manifest.get("projection_kind") != spec["projection_kind"]:
        raise ExpertManifoldError("ECP Stage 1 projection manifest changed")
    _require_values(optimization, expected, "ECP Stage 1 projection manifest changed")
    if (
        information_wall.get("privileged_q_pi")
        not in (True, "fit90 shared training and frozen held5 inference only")
        or information_wall.get("second_adapter_deployed") is not False
    ):
        raise ExpertManifoldError("ECP Stage 1 projection manifest changed")
    cursor = int(optimization.get(spec["cursor_name"], -1))
    return {
        "adapter_schema": spec["adapter_schema"],
        "arm": f"{spec['arm_prefix']}{cursor}",
        "asset": {
            "stage1_config": _projection_file(manifest, "stage1_config"),
            "stage1_checkpoint": _projection_file(manifest, "stage1_checkpoint"),
            "base_projection_manifest": _projection_file(
                manifest, "base_projection_manifest"
            ),
            "policy_support_bank": _projection_file(manifest, "policy_support_bank"),
            "privileged_q_pi": True,
            "held_shared_gradient_steps": 0,
            "single_complete_lora": True,
            "content_address_separated": True,
            "policy_support_teacher": True,
        },
    }


def _phase_aligned_projection_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    optimization = manifest.get("optimization", {})
    member = optimization.get("code_member")
    projection_kind = manifest.get("projection_kind")
    expected_kind = (
        "stable_shared_prior_baseline"
        if member == "shared"
        else (
            "stable_shared_prior_task_residual_decoder"
            if projection_kind == "stable_shared_prior_task_residual_decoder"
            else "phase_aligned_success_equivalence_decoder"
        )
    )
    if (
        projection_kind != expected_kind
        or optimization.get("decoder_frozen") is not True
        or int(optimization.get("held_code_gradient_steps", -1)) != 0
        or optimization.get("final_lora_averaging") is not False
        or member not in {"shared", "earliest", "latest"}
        or (
            projection_kind
            in {
                "stable_shared_prior_baseline",
                "stable_shared_prior_task_residual_decoder",
            }
            and (
                optimization.get("single_complete_lora") is not True
                or optimization.get("second_adapter_deployed") is not False
                or optimization.get("rank_partition")
                != {
                    "shared": [0, 12],
                    "task_residual": [12, 16],
                    "merge": "exact_effective_delta_sum",
                }
            )
        )
    ):
        raise ExpertManifoldError("phase-aligned decoder projection changed")
    return {
        "adapter_schema": FUNCTIONAL_DECODER_TASK_EXPERT_ADAPTER_SCHEMA,
        "arm": (
            "stable_shared_prior_baseline"
            if member == "shared"
            else (
                f"stable_shared_prior_residual_{member}_projection"
                if projection_kind == "stable_shared_prior_task_residual_decoder"
                else f"phase_aligned_functional_decoder_{member}_projection"
            )
        ),
        "asset": {
            "decoder_checkpoint": _projection_file(manifest, "decoder_checkpoint"),
            "code_artifact": _projection_file(manifest, "code_artifact"),
            "training_result": _projection_file(manifest, "training_result"),
            "decoder_frozen": True,
            "held_code_gradient_steps": 0,
            "code_member": member,
            "shared_prior_adapter": (
                _projection_file(manifest, "shared_prior_adapter")
                if projection_kind
                in {
                    "stable_shared_prior_baseline",
                    "stable_shared_prior_task_residual_decoder",
                }
                else None
            ),
        },
    }


def _ecp_static_lora_projection_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    optimization = manifest.get("optimization", {})
    information_wall = manifest.get("information_wall", {})
    purpose = str(manifest.get("purpose"))
    task_panel = str(manifest.get("task_panel"))
    if (
        manifest.get("projection_kind") != "ecp_stage1_privileged_static_lora"
        or purpose not in ECP_STAGE1_STATIC_LORA_PURPOSES
        or task_panel not in {"held5", "profile_fit"}
        or optimization
        != {
            "held_shared_gradient_steps": 0,
            "single_complete_lora": True,
            "final_lora_averaging": False,
            "rank": 16,
            "second_adapter_deployed": False,
            "parameterization": "one complete rank16 static LoRA",
        }
        or information_wall.get("role")
        != "development_train_leave_task_out_oracle_only"
        or information_wall.get("deployment_carrier") is not False
        or information_wall.get("validation_action_or_reward_reads") != 0
        or information_wall.get("test_action_or_reward_reads") != 0
        or information_wall.get("second_adapter_deployed") is not False
    ):
        raise ExpertManifoldError("ECP Stage 1 static-LoRA projection changed")
    return {
        "adapter_schema": ECP_STAGE1_STATIC_LORA_ADAPTER_SCHEMA,
        "arm": purpose,
        "asset": {
            "purpose": purpose,
            "task_panel": task_panel,
            "base_projection_manifest": _projection_file(
                manifest, "base_projection_manifest"
            ),
            "held_shared_gradient_steps": 0,
            "single_complete_lora": True,
        },
    }


def _projection_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema = manifest.get("schema_version")
    if schema == PROJECTED_TASK_EXPERT_MANIFEST_SCHEMA:
        if manifest.get("optimization", {}).get("factor_heads_frozen") is not True:
            raise ExpertManifoldError("fixed-head projection manifest changed")
        return {
            "adapter_schema": PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA,
            "arm": "macro25_fixed_factor_heads_free_program_projection",
            "asset": {
                "writer_checkpoint": manifest.get("writer_checkpoint"),
                "factor_heads_frozen": True,
            },
        }
    if schema in {
        FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA,
        FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA,
    }:
        optimization = manifest.get("optimization", {})
        code_condition = optimization.get("code_condition", "task_fingerprint")
        if (
            manifest.get("projection_kind")
            != "fixed_functional_decoder_code_projection"
            or optimization.get("decoder_frozen_for_held_code_fit") is not True
            or code_condition not in {"task_fingerprint", "shared_zero"}
        ):
            raise ExpertManifoldError("functional-decoder projection manifest changed")
        meta_surface = schema == FUNCTIONAL_DECODER_META_TASK_EXPERT_MANIFEST_SCHEMA
        return {
            "adapter_schema": FUNCTIONAL_DECODER_TASK_EXPERT_ADAPTER_SCHEMA,
            "arm": (
                "functional_decoder_nonheld_meta_shared_zero_carrier"
                if meta_surface and code_condition == "shared_zero"
                else (
                    "functional_decoder_nonheld_meta_projection"
                    if meta_surface
                    else "functional_decoder_train24_projection"
                )
            ),
            "asset": {
                "decoder_checkpoint": _projection_file(manifest, "decoder_checkpoint"),
                "held_codes": _projection_file(manifest, "held_codes"),
                "profile_result": _projection_file(manifest, "profile_result"),
                "decoder_frozen_for_held_code_fit": True,
                "code_condition": code_condition,
            },
        }
    if schema == PHASE_ALIGNED_DECODER_TASK_EXPERT_MANIFEST_SCHEMA:
        return _phase_aligned_projection_contract(manifest)
    if schema == ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA:
        return _ecp_static_lora_projection_contract(manifest)
    if schema in {
        ECP_STAGE1_TASK_EXPERT_MANIFEST_SCHEMA,
        ECP_STAGE1_OUTCOME_TASK_EXPERT_MANIFEST_SCHEMA,
        ECP_STAGE1_PROGRAM_LOCKED_TASK_EXPERT_MANIFEST_SCHEMA,
        ECP_STAGE1_MDCO_TASK_EXPERT_MANIFEST_SCHEMA,
    }:
        return _ecp_projection_contract(manifest)
    raise ExpertManifoldError("projected task-expert manifest schema changed")


def inspect_projected_task_expert_bank(
    base: Mapping[str, Any], projection_manifest: Path
) -> dict[str, Any]:
    """Bind a complete or declared leave-task-out projection to its bank."""

    projection_manifest = projection_manifest.resolve()
    manifest = read_json(projection_manifest)
    projection_contract = _projection_contract(manifest)
    projected = {
        (str(row.get("suite")), int(row.get("task_id", -1))): dict(row)
        for row in manifest.get("tasks", ())
    }
    base_records = {
        (str(row["suite"]), int(row["task_id"])): dict(row) for row in base["tasks"]
    }
    evaluation_role = str(base.get("information_wall", {}).get("evaluation_role"))
    static_lora = (
        manifest.get("schema_version") == ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA
    )
    leave_task_out = manifest.get("schema_version") in {
        ECP_STAGE1_MDCO_TASK_EXPERT_MANIFEST_SCHEMA,
        ECP_STAGE1_STATIC_LORA_MANIFEST_SCHEMA,
    }
    expected_oracle_role = (
        "nonheld_meta_oracle_only"
        if evaluation_role == "nonheld_meta"
        else (
            "development_train_leave_task_out_oracle_only"
            if leave_task_out
            else "development_train_oracle_only"
        )
    )
    if (
        manifest.get("repository", {}).get("dirty_paths") != []
        or manifest.get("information_wall", {}).get("role") != expected_oracle_role
        or manifest.get("information_wall", {}).get("deployment_carrier") is not False
        or not set(projected).issubset(base_records)
        or (not leave_task_out and set(projected) != set(base_records))
        or (
            static_lora
            and len(projected)
            != (1 if manifest.get("task_panel") == "profile_fit" else 5)
        )
        or (leave_task_out and not static_lora and len(projected) != 5)
    ):
        raise ExpertManifoldError("fixed-head projection manifest changed")
    tasks = []
    for key in sorted(projected):
        source = base_records[key]
        row = projected[key]
        path = Path(str(row.get("projected_adapter", ""))).resolve()
        if (
            int(row.get("ordinal", -1)) != int(source["ordinal"])
            or int(row.get("global_task_id", -1)) != int(source["global_task_id"])
            or Path(str(row.get("expert_checkpoint", ""))).resolve()
            != Path(str(source["checkpoint"])).resolve()
            or not path.is_file()
            or path.stat().st_size != int(row.get("projected_adapter_bytes", -1))
        ):
            raise ExpertManifoldError("fixed-head projected task adapter changed")
        tasks.append(
            {
                **source,
                "projected_adapter": str(path),
                "projected_adapter_bytes": path.stat().st_size,
            }
        )
    return {
        **dict(base),
        "schema_version": projection_contract["adapter_schema"],
        "arm": projection_contract["arm"],
        "tasks": tasks,
        "projection": {
            "manifest_path": str(projection_manifest),
            "manifest_bytes": projection_manifest.stat().st_size,
            "schema": manifest["schema_version"],
            **projection_contract["asset"],
            "deployment_carrier": False,
        },
    }
