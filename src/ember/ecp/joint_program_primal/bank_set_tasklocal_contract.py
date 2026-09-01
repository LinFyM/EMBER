"""Stable S0/S1 configuration and parameter-ownership contracts for EBSRI."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.ecp.joint_program_primal.runtime import (
    R5_SHARED_FUNCTIONAL_CHART,
    SCORER_INTERACTION_ONLY,
)


BANK_SET_TASKLOCAL_SCHEMA = "ember_ecp_event_bank_set_tasklocal_v1"
BANK_SET_TASKLOCAL_RUN_SCHEMA = "ember_ecp_event_bank_set_tasklocal_run_v1"
BANK_SET_S0_STAGE = "g3_event_bank_set_s0_free_summary"
BANK_SET_S1_STAGE = "g3_event_bank_set_s1_real_summary"


class InteractionControlWriterState(torch.nn.Module):
    """Checkpoint EBSRI and, in S0 only, two training-only free summaries."""

    def __init__(
        self,
        bank_set_interaction: torch.nn.Module,
        *,
        free_summary_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.bank_set_interaction = bank_set_interaction
        if free_summary_shape is not None:
            reference = next(bank_set_interaction.parameters())
            self.free_correct = torch.nn.Parameter(
                torch.empty(free_summary_shape, device=reference.device, dtype=reference.dtype)
            )
            self.free_wrong = torch.nn.Parameter(
                torch.empty(free_summary_shape, device=reference.device, dtype=reference.dtype)
            )
            torch.nn.init.normal_(self.free_correct, std=0.02)
            torch.nn.init.normal_(self.free_wrong, std=0.02)


def is_bank_set_tasklocal_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == BANK_SET_TASKLOCAL_SCHEMA


def bank_set_config_valid(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    task_local = config.get("task_local", {})
    authorities = config.get("authorities", {})
    wall = config.get("information_wall", {})
    gate = config.get("gate", {})
    stage = config.get("stage")
    is_s0 = stage == BANK_SET_S0_STAGE
    is_s1 = stage == BANK_SET_S1_STAGE
    trainable = (
        [
            "EventConditionedBankSetInteraction.candidate_trunk/condition_generated_heads",
            "EventConditionedBankSetInteraction.task_independent_owner/rank/event_slots",
            "training_only_free_correct/free_wrong",
        ]
        if is_s0
        else ["EventConditionedBankSetInteraction"]
    )
    expected_summary_source = (
        "one_training_only_free_correct_and_free_wrong_token_per_task"
        if is_s0
        else "real_b0_program_relative_event_bank_set_encoder"
    )
    return all(
        (
            is_bank_set_tasklocal_config(config),
            config.get("status") == "active_event_bank_set_tasklocal_qualification",
            is_s0 or is_s1,
            model.get("program_source")
            == "fixed_nontrainable_128d_orthogonal_task_token",
            model.get("primal_scorer_initialization") == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_INTERACTION_ONLY,
            model.get("inverse_covariance_power") == 1.0,
            model.get("interaction_summary_value_width") == 16,
            model.get("interaction_hidden_width") == 64,
            model.get("interaction_correction_bound") == 0.1,
            model.get("interaction_context_basis")
            == "task_independent_trainable_owner_rank_event_slots_no_absolute_program_code",
            model.get("replay_frame_chunk_size_by_task") == {"1": 4, "93": 32},
            model.get("interaction_group_batch_size_by_task")
            == {"1": 16, "93": 4},
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
            gate.get("maximum_near_bound_fraction") == 0.5,
            isinstance(authorities.get("r5_primal_scorer_checkpoint"), str),
            isinstance(authorities.get("r5_gate_aggregate"), str),
            isinstance(authorities.get("positive_control_root"), str),
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


def bank_set_parameter_ownership(
    program: torch.nn.Module,
    compiler: torch.nn.Module,
    *,
    stage: str,
) -> tuple[torch.nn.Module, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    interaction = compiler.bank_set_interaction
    interaction.requires_grad_(True).train()
    if stage == BANK_SET_S0_STAGE:
        interaction.set_encoder.requires_grad_(False).eval()
    elif stage != BANK_SET_S1_STAGE:
        raise ValueError("bank-set interaction stage changed")
    writer = InteractionControlWriterState(
        interaction,
        free_summary_shape=(compiler.event_slots, interaction.summary_width)
        if stage == BANK_SET_S0_STAGE
        else None,
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
    }
    allowed_parameters = {
        "bank_set_interaction.owner_slot_context",
        "bank_set_interaction.rank_slot_context",
        "bank_set_interaction.event_slot_context",
    }
    if stage == BANK_SET_S1_STAGE:
        allowed_roots.add("bank_set_interaction.set_encoder")
    unexpected = sorted(
        name
        for name in named_trainable
        if name not in {"free_correct", "free_wrong"}
        and name not in allowed_parameters
        and not any(name.startswith(f"{root}.") for root in allowed_roots)
    )
    free = {name for name in named_trainable if name.startswith("free_")}
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
            "plus_task_independent_owner_rank_event_slots"
        ),
    }
