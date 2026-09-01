"""Stable S0/S1 configuration and parameter-ownership contracts for EBSRI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
    OutputProgramBankSetConditions,
    ProgramBankSetConditions,
)
from ember.ecp.joint_program_primal.runtime import (
    R5_SHARED_FUNCTIONAL_CHART,
    SCORER_INTERACTION_ONLY,
)
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.pi05_source_checkpoint import read_json


BANK_SET_TASKLOCAL_SCHEMA = "ember_ecp_program_through_bank_tasklocal_v1"
BANK_SET_TASKLOCAL_RUN_SCHEMA = "ember_ecp_program_through_bank_tasklocal_run_v1"
BANK_SET_TASKLOCAL_RESULT_SCHEMA = (
    "ember_ecp_program_through_bank_tasklocal_result_v1"
)
BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA = (
    "ember_ecp_program_through_bank_tasklocal_aggregate_v1"
)
BANK_SET_S0_STAGE = "g3_program_through_bank_s0_free_summary"
BANK_SET_S1_STAGE = "g3_program_through_bank_s1_real_summary"
BANK_CONDITIONED_PRIMAL_STAGE = "g3_bank_conditioned_primal_tasklocal"
PROGRAM_CONDITIONED_B0_QUERY = "program_shared_inducing"
TASKLOCAL_FREE_B0_QUERY = (
    "tasklocal_free_target_rank_event_shared_across_banks_videos_and_scopes"
)
TASKLOCAL_QUERY_STEP_CALIBRATIONS = frozenset(
    {
        "task93_qfree_query_space_step_calibration_after_undertravel_non_pass",
        "task93_afree_nested_native_anchor_after_calibrated_qfree_tradeoff_non_pass",
    }
)


class FreeProgramBankSetConditionTree(torch.nn.Module):
    """Train-only S0 parameters with exactly the real B0 scope topology."""

    def __init__(self, interaction: torch.nn.Module) -> None:
        super().__init__()
        reference = next(interaction.parameters())
        shape = (
            G1_RESIDUAL_RANK,
            interaction.event_slots,
            interaction.summary_width,
        )

        def parameter() -> torch.nn.Parameter:
            value = torch.nn.Parameter(
                torch.empty(shape, device=reference.device, dtype=reference.dtype)
            )
            torch.nn.init.normal_(value, std=0.02)
            return value

        self.inputs = torch.nn.ParameterList(
            [parameter() for _ in interaction.owners]
        )
        self.outputs_all = torch.nn.ModuleList()
        self.outputs_by_type = torch.nn.ModuleList()
        for owner in interaction.owners:
            groups = native_output_group_count(owner)
            self.outputs_all.append(
                torch.nn.ParameterList([parameter() for _ in range(groups)])
            )
            self.outputs_by_type.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.ParameterList(
                            [parameter() for _ in OUTPUT_BANK_TYPES]
                        )
                        for _ in range(groups)
                    ]
                )
            )

    def conditions(self) -> ProgramBankSetConditions:
        return ProgramBankSetConditions(
            inputs=tuple(self.inputs),
            outputs=tuple(
                tuple(
                    OutputProgramBankSetConditions(
                        all_types=all_types,
                        by_type=tuple(by_type),
                    )
                    for all_types, by_type in zip(
                        target_all, target_by_type, strict=True
                    )
                )
                for target_all, target_by_type in zip(
                    self.outputs_all, self.outputs_by_type, strict=True
                )
            ),
        )


class InteractionControlWriterState(torch.nn.Module):
    """Checkpoint EBSRI and, in S0 only, two training-only free summaries."""

    def __init__(
        self,
        bank_set_interaction: torch.nn.Module,
        *,
        structured_free_summary: bool = False,
    ) -> None:
        super().__init__()
        self.bank_set_interaction = bank_set_interaction
        if structured_free_summary:
            self.free_correct = FreeProgramBankSetConditionTree(
                bank_set_interaction
            )
            self.free_wrong = FreeProgramBankSetConditionTree(
                bank_set_interaction
            )


def is_bank_set_tasklocal_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == BANK_SET_TASKLOCAL_SCHEMA


def required_s0_gate_authority(
    config: Mapping[str, Any], *, asset_root: Path
) -> dict[str, Any] | None:
    """Validate that S1 only consumes the passed S0 aggregate, never its state."""

    if config.get("stage") != BANK_SET_S1_STAGE:
        return None
    specification = config.get("authorities", {}).get("required_s0_gate", {})
    relative = specification.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Program-through-bank S0 gate path changed")
    path = (asset_root / relative).resolve()
    if not path.is_file() or path.stat().st_size != specification.get("bytes"):
        raise ValueError("Program-through-bank S0 gate artifact changed")
    aggregate = read_json(path)
    valid = all(
        (
            specification.get("aggregate_schema")
            == BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
            specification.get("stage") == BANK_SET_S0_STAGE,
            specification.get("required_gate") == "pass",
            aggregate.get("schema_version")
            == BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
            aggregate.get("status") == "complete",
            aggregate.get("stage") == BANK_SET_S0_STAGE,
            aggregate.get("gate") == "pass",
            aggregate.get("authority_commit")
            == specification.get("authority_commit"),
            set(aggregate.get("tasks", {})) == {"1", "93"},
            all(
                row.get("gate") == "pass"
                and bool(row.get("checks"))
                and all(row["checks"].values())
                for row in aggregate.get("tasks", {}).values()
            ),
        )
    )
    if not valid:
        raise ValueError("Program-through-bank S0 gate did not pass")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "aggregate_schema": aggregate["schema_version"],
        "stage": aggregate["stage"],
        "gate": aggregate["gate"],
        "authority_commit": aggregate["authority_commit"],
    }


def required_s1_non_pass_authority(
    config: Mapping[str, Any], *, asset_root: Path
) -> dict[str, Any] | None:
    """Bind the primal branch to the formal S1 failure that triggered it."""

    if config.get("stage") != BANK_CONDITIONED_PRIMAL_STAGE:
        return None
    specification = config.get("authorities", {}).get("required_s1_non_pass", {})
    relative = specification.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("bank-conditioned-primal S1 authority path changed")
    path = (asset_root / relative).resolve()
    if not path.is_file() or path.stat().st_size != specification.get("bytes"):
        raise ValueError("bank-conditioned-primal S1 authority artifact changed")
    aggregate = read_json(path)
    expected_passes = {
        "wrong_each",
        "margin",
        "all_pairs",
        "correction_not_broadly_saturated",
    }
    valid = all(
        (
            specification.get("aggregate_schema")
            == BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
            specification.get("stage") == BANK_SET_S1_STAGE,
            specification.get("required_gate") == "non_pass",
            aggregate.get("schema_version")
            == BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
            aggregate.get("status") == "complete",
            aggregate.get("stage") == BANK_SET_S1_STAGE,
            aggregate.get("gate") == "non_pass",
            aggregate.get("authority_commit")
            == specification.get("authority_commit"),
            set(aggregate.get("tasks", {})) == {"1", "93"},
            all(
                row.get("gate") == "non_pass"
                and set(row.get("checks", {}))
                == expected_passes | {"correct_fit_each", "correct_held"}
                and all(row["checks"][name] for name in expected_passes)
                and not row["checks"]["correct_fit_each"]
                and not row["checks"]["correct_held"]
                for row in aggregate.get("tasks", {}).values()
            ),
        )
    )
    if not valid:
        raise ValueError("required Program-through-bank S1 result changed")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "aggregate_schema": aggregate["schema_version"],
        "stage": aggregate["stage"],
        "gate": aggregate["gate"],
        "authority_commit": aggregate["authority_commit"],
    }


def required_primal_task93_non_pass_authority(
    config: Mapping[str, Any], *, asset_root: Path
) -> dict[str, Any] | None:
    """Bind Q_free to the task93 specificity failure that requested it."""

    model = config.get("model", {})
    if model.get("b0_query_source") != TASKLOCAL_FREE_B0_QUERY:
        return None
    specification = config.get("authorities", {}).get(
        "required_primal_task93_non_pass", {}
    )
    relative = specification.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Q_free predecessor path changed")
    path = (asset_root / relative).resolve()
    contract_path = path.parent / "run_contract.json"
    if (
        not path.is_file()
        or path.stat().st_size != specification.get("bytes")
        or not contract_path.is_file()
        or contract_path.stat().st_size != specification.get("run_contract_bytes")
    ):
        raise ValueError("Q_free predecessor artifact changed")
    result = read_json(path)
    contract = read_json(contract_path)
    checks = result.get("evaluation", {}).get("checks", {})
    valid = all(
        (
            result.get("schema_version") == BANK_SET_TASKLOCAL_RESULT_SCHEMA,
            result.get("status") == "complete",
            result.get("stage") == BANK_CONDITIONED_PRIMAL_STAGE,
            result.get("task") == 93,
            result.get("completed_optimizer_steps") == 110,
            result.get("evaluation", {}).get("gate") == "non_pass",
            checks
            == {
                "correct_fit_each": True,
                "correct_held": True,
                "wrong_each": False,
                "margin": False,
                "all_pairs": True,
            },
            contract.get("schema_version") == BANK_SET_TASKLOCAL_RUN_SCHEMA,
            contract.get("stage") == BANK_CONDITIONED_PRIMAL_STAGE,
            contract.get("git", {}).get("authority_commit")
            == specification.get("authority_commit"),
            contract.get("diagnostic", {}).get(
                "b0_query_source", PROGRAM_CONDITIONED_B0_QUERY
            )
            == PROGRAM_CONDITIONED_B0_QUERY,
        )
    )
    if not valid:
        raise ValueError("Q_free predecessor did not isolate task93 specificity")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "run_contract": str(contract_path),
        "run_contract_bytes": contract_path.stat().st_size,
        "stage": result["stage"],
        "task": result["task"],
        "gate": result["evaluation"]["gate"],
        "authority_commit": contract["git"]["authority_commit"],
    }


def bank_set_config_valid(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    task_local = config.get("task_local", {})
    authorities = config.get("authorities", {})
    wall = config.get("information_wall", {})
    gate = config.get("gate", {})
    stage = config.get("stage")
    is_s0 = stage == BANK_SET_S0_STAGE
    is_s1 = stage == BANK_SET_S1_STAGE
    is_primal = stage == BANK_CONDITIONED_PRIMAL_STAGE
    query_source = model.get("b0_query_source", PROGRAM_CONDITIONED_B0_QUERY)
    free_query = is_primal and query_source == TASKLOCAL_FREE_B0_QUERY
    calibration_kind = config.get("mechanism_calibration", {}).get("kind")
    query_lr_multiplier = model.get("tasklocal_free_b0_query_lr_multiplier")
    required_s0 = authorities.get("required_s0_gate", {})
    required_s1 = authorities.get("required_s1_non_pass", {})
    required_primal = authorities.get("required_primal_task93_non_pass", {})
    required_predecessor = required_s1 if is_primal else required_s0
    required_predecessor_stage = (
        BANK_SET_S1_STAGE if is_primal else BANK_SET_S0_STAGE
    )
    required_predecessor_gate = "non_pass" if is_primal else "pass"
    trainable = (
        [
            "EventConditionedBankSetInteraction.candidate_trunk/condition_generated_heads",
            "EventConditionedBankSetInteraction.task_independent_owner/rank/event_structure",
            "training_only_scope_matched_free_correct/free_wrong",
        ]
        if is_s0
        else (
            ["EventConditionedBankSetInteraction"]
            if is_s1
            else (
                [
                    "EventConditionedBankSetInteraction.tasklocal_free_b0_query",
                    "EventConditionedBankSetInteraction.set_encoder_value_networks",
                    "EventConditionedBankSetInteraction.family_shared_primal_gates",
                ]
                if free_query
                else [
                    "EventConditionedBankSetInteraction.set_encoder",
                    "EventConditionedBankSetInteraction.task_independent_owner/rank/event_slots",
                    "EventConditionedBankSetInteraction.family_shared_primal_gates",
                ]
            )
        )
    )
    expected_summary_source = (
        "scope_matched_training_only_free_correct_and_free_wrong_tree_per_task"
        if is_s0
        else (
            "real_b0_program_relative_event_bank_set_encoder"
            if is_s1
            else (
                "real_b0_tasklocal_free_query_native_anchor"
                if free_query
                else "real_b0_program_relative_native_anchor"
            )
        )
    )
    return all(
        (
            is_bank_set_tasklocal_config(config),
            config.get("status")
            == "active_program_through_bank_tasklocal_qualification",
            is_s0 or is_s1 or is_primal,
            query_source
            in {PROGRAM_CONDITIONED_B0_QUERY, TASKLOCAL_FREE_B0_QUERY},
            is_primal or query_source == PROGRAM_CONDITIONED_B0_QUERY,
            model.get("program_source")
            == "fixed_nontrainable_128d_orthogonal_task_token",
            model.get("primal_scorer_initialization") == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_INTERACTION_ONLY,
            model.get("inverse_covariance_power") == 1.0,
            model.get("interaction_summary_value_width") == 16,
            model.get("interaction_hidden_width") == 64,
            (
                query_lr_multiplier == G1_RESIDUAL_RANK * 8
                if calibration_kind in TASKLOCAL_QUERY_STEP_CALIBRATIONS
                else "tasklocal_free_b0_query_lr_multiplier" not in model
            ),
            (
                model.get("interaction_correction_bound") == 0.1
                if not is_primal
                else "interaction_correction_bound" not in model
            ),
            model.get("interaction_context_basis")
            == (
                "program_through_bank_summary_only_b1_with_fixed_owner_rank_event_structure"
                if not is_primal
                else "real_b0_native_anchor_additive_primal_before_full_inverse_and_exact_replay"
            ),
            model.get("replay_frame_chunk_size_by_task") == {"1": 4, "93": 32},
            (
                model.get("interaction_group_batch_size_by_task")
                == {"1": 16, "93": 1}
                if not is_primal
                else "interaction_group_batch_size_by_task" not in model
            ),
            model.get("trainable") == trainable,
            model.get("deployment_candidate") is False,
            config.get("optimization", {}).get("loss")
            == "family_equal_effective_rank4_diagnostic_then_functional_panel_b_gate",
            config.get("optimization", {}).get(
                "functional_policy_microbatch_size_by_task"
            )
            == {"1": 8, "93": 2},
            task_local.get("task_ids") == [1, 93],
            task_local.get("correct_gradient_arms") == ["fit0", "fit1"],
            task_local.get("wrong_gradient_arms") == ["fit0"],
            task_local.get("zero_gradient_arms")
            == ["correct_held", "wrong_fit1", "panel_b"],
            task_local.get("wrong_task_by_task") == {"1": 8, "93": 94},
            task_local.get("wrong_conditioning_language")
            == "correct_task_exact_language",
            task_local.get("summary_source") == expected_summary_source,
            task_local.get("effective_target_is_gate") is False,
            task_local.get("functional_gate_authority") is True,
            gate.get("correct_fit_each_minimum") == 0.85,
            gate.get("correct_held_minimum") == 0.80,
            gate.get("wrong_each_maximum") == 0.25,
            gate.get("minimum_correct_minus_maximum_wrong") == 0.50,
            gate.get("all_correct_better_than_all_wrong") is True,
            gate.get("effective_target_is_gate") is False,
            gate.get("panel_b_backward_calls") == 0,
            (
                gate.get("maximum_near_bound_fraction") == 0.5
                if not is_primal
                else "maximum_near_bound_fraction" not in gate
            ),
            isinstance(authorities.get("r5_primal_scorer_checkpoint"), str),
            isinstance(authorities.get("r5_gate_aggregate"), str),
            isinstance(authorities.get("positive_control_root"), str),
            (
                True
                if is_s0
                else all(
                    (
                        isinstance(required_predecessor.get("path"), str),
                        isinstance(required_predecessor.get("bytes"), int),
                        required_predecessor.get("aggregate_schema")
                        == BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
                        required_predecessor.get("stage")
                        == required_predecessor_stage,
                        required_predecessor.get("required_gate")
                        == required_predecessor_gate,
                        isinstance(
                            required_predecessor.get("authority_commit"),
                            str,
                        ),
                    )
                )
            ),
            (
                all(
                    (
                        isinstance(required_primal.get("path"), str),
                        isinstance(required_primal.get("bytes"), int),
                        isinstance(required_primal.get("run_contract_bytes"), int),
                        isinstance(required_primal.get("authority_commit"), str),
                    )
                )
                if free_query
                else "required_primal_task93_non_pass" not in authorities
            ),
            wall.get("fixed_routing_token_training_only") is True,
            wall.get("free_summary_tokens_training_only_not_component_candidate")
            is is_s0,
            wall.get("wrong_bank_exact_language_fixed") is True,
            wall.get("single_complete_rank16") is True,
            config.get("throughput_gate", {}).get("cross_language_bank_cache")
            == "explicit_separate_operational_root",
            config.get("throughput_gate", {}).get("qualification_gate") is False,
            config.get("privileged_critic") is None,
        )
    )


def _enable_installed_free_native_anchors(interaction: torch.nn.Module) -> set[str]:
    inputs = interaction.tasklocal_free_input_anchor
    outputs = interaction.tasklocal_free_output_anchor
    if inputs is None and outputs is None:
        return set()
    if inputs is None or outputs is None:
        raise ValueError("task-local free native anchor ownership changed")
    inputs.requires_grad_(True)
    outputs.requires_grad_(True)
    return {
        "bank_set_interaction.tasklocal_free_input_anchor",
        "bank_set_interaction.tasklocal_free_output_anchor",
    }


def bank_set_parameter_ownership(
    program: torch.nn.Module,
    compiler: torch.nn.Module,
    *,
    stage: str,
    free_b0_query_initial: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    interaction = compiler.bank_set_interaction
    interaction.requires_grad_(False).eval()
    free_anchor_roots: set[str] = set()
    if stage == BANK_SET_S0_STAGE:
        interaction.requires_grad_(True).train()
        interaction.set_encoder.requires_grad_(False).eval()
    elif stage == BANK_SET_S1_STAGE:
        interaction.requires_grad_(True).train()
    elif stage == BANK_CONDITIONED_PRIMAL_STAGE:
        if free_b0_query_initial is None:
            interaction.set_encoder.requires_grad_(True).train()
            interaction.owner_slot_context.requires_grad_(True)
            interaction.rank_slot_context.requires_grad_(True)
            interaction.event_slot_context.requires_grad_(True)
        else:
            interaction.install_tasklocal_free_b0_query(free_b0_query_initial)
            for encoder in interaction.set_encoder.values():
                encoder.input_value.requires_grad_(True).train()
                encoder.output_value.requires_grad_(True).train()
            interaction.tasklocal_free_b0_query.requires_grad_(True)
            free_anchor_roots = _enable_installed_free_native_anchors(interaction)
        interaction.input_primal_gate.requires_grad_(True).train()
        interaction.output_primal_gate.requires_grad_(True).train()
    else:
        raise ValueError("bank-set interaction stage changed")
    writer = InteractionControlWriterState(
        interaction,
        structured_free_summary=stage == BANK_SET_S0_STAGE,
    )
    named_trainable = {
        name: parameter
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    }
    allowed_roots = {
        "bank_set_interaction.input_candidate",
        "bank_set_interaction.output_candidate",
        "bank_set_interaction.input_condition",
        "bank_set_interaction.output_condition",
        "bank_set_interaction.structural_gate",
    }
    allowed_parameters = {
        "bank_set_interaction.rank_slot_context",
        "bank_set_interaction.event_slot_context",
        "bank_set_interaction.owner_slot_context",
    }
    if stage == BANK_SET_S1_STAGE:
        allowed_roots.add("bank_set_interaction.set_encoder")
    elif stage == BANK_CONDITIONED_PRIMAL_STAGE:
        allowed_roots = {
            "bank_set_interaction.set_encoder",
            "bank_set_interaction.input_primal_gate",
            "bank_set_interaction.output_primal_gate",
        }
        if free_b0_query_initial is not None:
            allowed_parameters = {"bank_set_interaction.tasklocal_free_b0_query"}
            allowed_roots.update(free_anchor_roots)
    unexpected = sorted(
        name
        for name in named_trainable
        if not name.startswith(("free_correct.", "free_wrong."))
        and name not in allowed_parameters
        and not any(name.startswith(f"{root}.") for root in allowed_roots)
    )
    free = {
        name.split(".", 1)[0]
        for name in named_trainable
        if name.startswith("free_")
    }
    expected_free = {"free_correct", "free_wrong"} if stage == BANK_SET_S0_STAGE else set()
    if unexpected or free != expected_free or not named_trainable:
        raise ValueError(
            f"bank-set trainable inventory changed: unexpected={unexpected}, free={sorted(free)}"
        )
    trainable = tuple(named_trainable.values())
    frozen = tuple(
        parameter
        for root in (program, compiler)
        for parameter in root.parameters()
        if not parameter.requires_grad
    )
    return writer, trainable, frozen


def writer_trainable_inventory(writer: torch.nn.Module) -> dict[str, Any]:
    named = [
        (name, parameter)
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "writer_trainable_parameter_names": [name for name, _ in named],
        "writer_trainable_parameter_count": sum(value.numel() for _, value in named),
        "descriptor_authority": (
            "frozen_program_native_query_kappa_base_score_metadata_event_assignment_"
            "plus_program_through_bank_responses_and_task_independent_owner_rank_event_"
            "structure"
        ),
    }
