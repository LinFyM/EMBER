from __future__ import annotations

import inspect

import pytest
import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
    EventConditionedBankSetInteraction,
    OutputProgramBankSetConditions,
    ProgramBankContext,
    ProgramBankSetConditions,
    ProgramBankSetSummaries,
)
from ember.ecp.bank_conditioning.set_summary import (
    EventBankSetSummary,
    OutputEventBankSetSummary,
)
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.joint_program_primal.bank_set_tasklocal_contract import (
    BANK_SET_S0_STAGE,
    BANK_SET_S1_STAGE,
    BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
    required_s0_gate_authority,
)
from ember.pi05_source_checkpoint import write_json_atomic


def _fixture():
    torch.manual_seed(20260831)
    owner = TargetOwner(0, "q", TargetFamily.Q, 0, 4, 8)
    module = EventConditionedBankSetInteraction(
        (owner,),
        program_width=6,
        event_slots=2,
        summary_value_width=4,
        hidden_width=12,
        replay_score_rms=0.02,
    )
    assignment = torch.rand(3, 2).softmax(-1)
    context = ProgramBankContext(
        canonical_assignment=assignment,
        frame_positions=torch.linspace(0.0, 1.0, 3),
        local_scene=torch.randn(1, 6),
        local_process=torch.randn(2, 1, 6),
        local_presence=torch.rand(2),
        local_tau=torch.rand(2, 2),
        local_sigma=torch.randn(2, 1, 6),
    )
    state = torch.randn(4, 2, 6)
    weights = torch.randn(4, 2).softmax(-1)
    frame = torch.rand(3)
    frame = frame / frame.sum()
    return module, context, state, weights, frame


def test_bank_set_interaction_is_zero_initialized_with_real_summary() -> None:
    module, context, state, weights, frame = _fixture()
    values = torch.randn(3, 2, 50, 4)
    native_query = torch.randn(4, 2, 4)
    mean = values.reshape(-1, 4).mean(0)
    summary = module.summarize_input(
        target=0,
        program_event_state=state,
        native_event_query=native_query,
        values=values,
        native_mean=mean,
        frame_measure=frame,
        context=context,
    )
    correction = module.input_logit_corrections(
        target=0,
        native_event_query=native_query,
        event_weights=weights,
        base_query=torch.randn(4, 4),
        values=values,
        native_mean=mean,
        context=context,
        summary=summary,
    )
    assert correction.shape == (4, 2, 3, 2, 50)
    assert torch.equal(correction, torch.zeros_like(correction))
    correction[:, 0].sum().backward()
    head = module.input_condition[TargetFamily.Q.value][-1]
    assert head.weight.grad is not None
    assert bool(head.weight.grad.abs().sum() > 0)


def test_program_content_only_enters_b0_query_and_not_b1_signature() -> None:
    module, _, state, _, _ = _fixture()
    changed = state.clone()
    changed[0, 0] = changed[0, 0] + torch.randn_like(changed[0, 0])

    query = module._b0_query_context(target=0, program_event_state=state)
    other = module._b0_query_context(target=0, program_event_state=changed)

    assert not torch.equal(query, other)
    assert "program_event_state" not in inspect.signature(
        module.input_logit_corrections
    ).parameters
    assert "program_event_state" not in inspect.signature(
        module.output_logit_corrections
    ).parameters
    query.square().mean().backward()
    assert module.owner_slot_context.grad is not None
    assert module.rank_slot_context.grad is not None
    assert module.event_slot_context.grad is not None


def test_real_b0_summary_is_rank_specific() -> None:
    module, context, state, _, frame = _fixture()
    values = torch.randn(3, 2, 50, 4)
    native_query = torch.randn(4, 2, 4)
    summary = module.summarize_input(
        target=0,
        program_event_state=state,
        native_event_query=native_query,
        values=values,
        native_mean=values.reshape(-1, 4).mean(0),
        frame_measure=frame,
        context=context,
    )
    assert summary.induced_positive.shape == (4, 2, 4)
    assert summary.condition.shape == (4, 2, module.summary_width)
    changed_state = state.clone()
    changed_state[0, 0] = changed_state[0, 0] + torch.randn_like(
        changed_state[0, 0]
    )
    changed_program = module.summarize_input(
        target=0,
        program_event_state=changed_state,
        native_event_query=native_query,
        values=values,
        native_mean=values.reshape(-1, 4).mean(0),
        frame_measure=frame,
        context=context,
    )
    changed_values = values.clone()
    changed_values[0, 0, 0] = changed_values[0, 0, 0] + 1.0
    changed_bank = module.summarize_input(
        target=0,
        program_event_state=state,
        native_event_query=native_query,
        values=changed_values,
        native_mean=values.reshape(-1, 4).mean(0),
        frame_measure=frame,
        context=context,
    )
    assert not torch.equal(summary.condition, changed_program.condition)
    assert not torch.equal(summary.condition, changed_bank.condition)


def test_free_summary_generates_a_distinct_candidate_head_with_gradient() -> None:
    module, context, state, weights, frame = _fixture()
    values = torch.randn(3, 2, 50, 4)
    native_query = torch.randn(4, 2, 4)
    mean = values.reshape(-1, 4).mean(0)
    summary = module.summarize_input(
        target=0,
        program_event_state=state,
        native_event_query=native_query,
        values=values,
        native_mean=mean,
        frame_measure=frame,
        context=context,
    )
    with torch.no_grad():
        module.input_condition[TargetFamily.Q.value][-1].weight.normal_(std=0.05)
    free_a = torch.nn.Parameter(torch.zeros_like(summary.condition))
    free_b = torch.nn.Parameter(torch.randn_like(summary.condition))
    common = dict(
        target=0,
        native_event_query=native_query,
        event_weights=weights,
        base_query=torch.randn(4, 4),
        values=values,
        native_mean=mean,
        context=context,
    )
    left = module.input_logit_corrections(
        **common, summary=summary.with_condition(free_a)
    )
    right = module.input_logit_corrections(
        **common, summary=summary.with_condition(free_b)
    )
    assert not torch.equal(left, right)
    assert torch.equal(left, torch.zeros_like(left))
    torch.testing.assert_close(left[:, 0], -left[:, 1])
    (left.square().mean() + right.square().mean()).backward()
    assert free_a.grad is not None and bool(torch.isfinite(free_a.grad).all())
    assert free_b.grad is not None and bool(torch.isfinite(free_b.grad).all())


def test_output_reads_all_and_own_type_but_keeps_one_joint_candidate_axis() -> None:
    module, context, state, weights, frame = _fixture()
    values = torch.randn(3, 2, 50, 4, 4)
    native_query = torch.randn(4, 2, 4)
    mean = values.reshape(-1, 4).mean(0)
    summary = module.summarize_output(
        target=0,
        program_event_state=state,
        native_event_query=native_query,
        values=values,
        native_mean=mean,
        frame_measure=frame,
        context=context,
    )
    assert len(summary.by_type) == 4
    with torch.no_grad():
        module.output_condition[TargetFamily.Q.value][-1].weight.normal_(std=0.05)
    correction = module.output_logit_corrections(
        target=0,
        native_event_query=native_query,
        event_weights=weights,
        base_query=torch.randn(4, 4),
        values=values,
        native_mean=mean,
        context=context,
        summary=summary,
    )
    assert correction.shape == (4, 2, 3, 2, 50, 4)
    torch.testing.assert_close(correction[:, 0], -correction[:, 1])


def test_free_summary_tree_does_not_broadcast_across_native_scopes() -> None:
    def summary() -> EventBankSetSummary:
        return EventBankSetSummary(
            mean=torch.zeros(2, 32),
            log_variance=torch.zeros(2, 32),
            induced_positive=torch.zeros(4, 2, 4),
            induced_negative=torch.zeros(4, 2, 4),
            log_partition=torch.zeros(4, 2, 2),
            event_mass=torch.ones(2),
        )

    def output() -> OutputEventBankSetSummary:
        return OutputEventBankSetSummary(
            all_types=summary(), by_type=tuple(summary() for _ in range(4))
        )

    base = ProgramBankSetSummaries(
        inputs=(summary(), summary()), outputs=((output(),), (output(),))
    )
    tokens = [
        torch.full((4, 2, base.inputs[0].condition.shape[-1]), float(index))
        for index in range(12)
    ]
    conditions = ProgramBankSetConditions(
        inputs=(tokens[0], tokens[1]),
        outputs=(
            (
                OutputProgramBankSetConditions(
                    all_types=tokens[2], by_type=tuple(tokens[3:7])
                ),
            ),
            (
                OutputProgramBankSetConditions(
                    all_types=tokens[7], by_type=tuple(tokens[8:12])
                ),
            ),
        ),
    )
    observed = base.with_condition(conditions)
    actual = [*observed.inputs]
    for groups in observed.outputs:
        for group in groups:
            actual.extend((group.all_types, *group.by_type))
    for scope, token in zip(actual, tokens, strict=True):
        assert torch.equal(scope.condition, token)


def test_s1_requires_passed_s0_gate_without_consuming_checkpoint_state(
    tmp_path,
) -> None:
    authority_commit = "b" * 40
    aggregate = {
        "schema_version": BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
        "status": "complete",
        "stage": BANK_SET_S0_STAGE,
        "gate": "pass",
        "authority_commit": authority_commit,
        "tasks": {
            task: {"gate": "pass", "checks": {"margin": True}}
            for task in ("1", "93")
        },
    }
    path = tmp_path / "aggregate.json"
    write_json_atomic(path, aggregate)
    config = {
        "stage": BANK_SET_S1_STAGE,
        "authorities": {
            "required_s0_gate": {
                "path": path.name,
                "bytes": path.stat().st_size,
                "aggregate_schema": BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
                "stage": BANK_SET_S0_STAGE,
                "required_gate": "pass",
                "authority_commit": authority_commit,
            }
        },
    }
    observed = required_s0_gate_authority(config, asset_root=tmp_path)
    assert observed is not None and observed["gate"] == "pass"

    aggregate["tasks"]["93"]["checks"]["margin"] = False
    write_json_atomic(path, aggregate)
    config["authorities"]["required_s0_gate"]["bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="did not pass"):
        required_s0_gate_authority(config, asset_root=tmp_path)
