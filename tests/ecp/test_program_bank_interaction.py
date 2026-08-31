from __future__ import annotations

import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
    EventConditionedBankSetInteraction,
    ProgramBankContext,
)
from ember.ecp.contracts import TargetFamily, TargetOwner


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
        program_event_state=state,
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


def test_free_summary_changes_the_same_film_path_and_has_gradient() -> None:
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
        module.input_correction[TargetFamily.Q.value].weight.normal_(std=0.05)
    free_a = torch.nn.Parameter(torch.zeros_like(summary.condition))
    free_b = torch.nn.Parameter(torch.ones_like(summary.condition))
    common = dict(
        target=0,
        program_event_state=state,
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
        module.output_correction[TargetFamily.Q.value].weight.normal_(std=0.05)
    correction = module.output_logit_corrections(
        target=0,
        program_event_state=state,
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
