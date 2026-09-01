from __future__ import annotations

import inspect

import pytest
import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
    EventConditionedBankSetInteraction,
    ProgramBankContext,
    ProgramBankSetSummaries,
)
from ember.ecp.bank_conditioning.set_summary import OutputEventBankSetSummary
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.joint_program_primal.bank_set_tasklocal_contract import (
    BANK_CONDITIONED_PRIMAL_STAGE,
    BANK_SET_S0_STAGE,
    BANK_SET_S1_STAGE,
    BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
    required_s0_gate_authority,
    required_s1_non_pass_authority,
)
from ember.ecp.native_factors import native_output_group_count
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
    frame = torch.rand(3)
    frame = frame / frame.sum()
    return module, context, state, frame


def _summaries(module, context, state, frame):
    input_values = torch.randn(3, 2, 50, 4)
    input_query = torch.randn(4, 2, 4)
    input_summary = module.summarize_input(
        target=0,
        program_event_state=state,
        native_event_query=input_query,
        values=input_values,
        native_mean=input_values.reshape(-1, 4).mean(0),
        frame_measure=frame,
        context=context,
    )
    group_width = 8 // native_output_group_count(module.owners[0])
    outputs = []
    for _ in range(native_output_group_count(module.owners[0])):
        values = torch.randn(3, 2, 50, 4, group_width)
        outputs.append(
            module.summarize_output(
                target=0,
                program_event_state=state,
                native_event_query=torch.randn(4, 2, group_width),
                values=values,
                native_mean=values.reshape(-1, group_width).mean(0),
                frame_measure=frame,
                context=context,
            )
        )
    return ProgramBankSetSummaries(inputs=(input_summary,), outputs=(tuple(outputs),))


def test_bank_conditioned_primal_is_zero_initialized_and_trainable() -> None:
    module, context, state, frame = _fixture()
    summaries = _summaries(module, context, state, frame)
    base_input = (torch.randn(4, 4),)
    groups = native_output_group_count(module.owners[0])
    base_output = (torch.randn(groups, 4, 8 // groups),)
    inputs, outputs = module.bank_conditioned_primals(
        input_primals=base_input,
        output_primals=base_output,
        summaries=summaries,
    )
    assert torch.equal(inputs[0], base_input[0])
    assert torch.equal(outputs[0], base_output[0])
    (inputs[0].square().mean() + outputs[0].square().mean()).backward()
    for heads in (module.input_primal_gate, module.output_primal_gate):
        head = heads[TargetFamily.Q.value]
        assert head[-1].weight.grad is not None
        assert bool(head[-1].weight.grad.abs().sum() > 0)


def test_program_content_enters_only_the_real_b0_set_query() -> None:
    module, _, state, _ = _fixture()
    changed = state.clone()
    changed[0, 0] = changed[0, 0] + torch.randn_like(changed[0, 0])
    query = module._b0_query_context(target=0, program_event_state=state)
    other = module._b0_query_context(target=0, program_event_state=changed)
    assert not torch.equal(query, other)
    signature = inspect.signature(module.bank_conditioned_primals).parameters
    assert "program_event_state" not in signature
    query.square().mean().backward()
    assert module.owner_slot_context.grad is not None
    assert module.rank_slot_context.grad is not None
    assert module.event_slot_context.grad is not None


def test_real_b0_summary_has_rank_event_native_anchor() -> None:
    module, context, state, frame = _fixture()
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
    assert summary.condition.shape == (4, 2, module.summary_width)
    assert summary.native_anchor.shape == (4, 2, 4)
    changed_state = state.clone()
    changed_state[0, 0] = changed_state[0, 0] + 1.0
    changed = module.summarize_input(
        target=0,
        program_event_state=changed_state,
        native_event_query=native_query,
        values=values,
        native_mean=values.reshape(-1, 4).mean(0),
        frame_measure=frame,
        context=context,
    )
    assert not torch.equal(summary.native_anchor, changed.native_anchor)


def test_output_uses_all_type_anchor_and_type_specific_context_only() -> None:
    module, context, state, frame = _fixture()
    summaries = _summaries(module, context, state, frame)
    assert all(
        scope.native_positive is None
        for group in summaries.outputs[0]
        for scope in group.by_type
    )
    for head in module.output_primal_gate.values():
        torch.nn.init.normal_(head[-1].weight, std=0.05)
    groups = native_output_group_count(module.owners[0])
    base = (torch.randn(4, 4),)
    output = (torch.randn(groups, 4, 8 // groups),)
    _, observed = module.bank_conditioned_primals(
        input_primals=base, output_primals=output, summaries=summaries
    )
    first = summaries.outputs[0][0]
    changed_first = OutputEventBankSetSummary(
        all_types=first.all_types,
        by_type=(
            first.by_type[0].with_condition(
                first.by_type[0].condition + torch.randn_like(first.by_type[0].condition)
            ),
            *first.by_type[1:],
        ),
    )
    changed = ProgramBankSetSummaries(
        inputs=summaries.inputs,
        outputs=((changed_first, *summaries.outputs[0][1:]),),
    )
    _, actual = module.bank_conditioned_primals(
        input_primals=base, output_primals=output, summaries=changed
    )
    assert observed[0].shape == output[0].shape
    assert not torch.equal(observed[0][0], actual[0][0])
    torch.testing.assert_close(observed[0][1:], actual[0][1:])


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


def test_bank_conditioned_primal_requires_the_formal_s1_non_pass(tmp_path) -> None:
    authority_commit = "c" * 40
    expected_passes = {
        "wrong_each": True,
        "margin": True,
        "all_pairs": True,
        "correction_not_broadly_saturated": True,
        "correct_fit_each": False,
        "correct_held": False,
    }
    aggregate = {
        "schema_version": BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
        "status": "complete",
        "stage": BANK_SET_S1_STAGE,
        "gate": "non_pass",
        "authority_commit": authority_commit,
        "tasks": {
            task: {"gate": "non_pass", "checks": dict(expected_passes)}
            for task in ("1", "93")
        },
    }
    path = tmp_path / "aggregate.json"
    write_json_atomic(path, aggregate)
    config = {
        "stage": BANK_CONDITIONED_PRIMAL_STAGE,
        "authorities": {
            "required_s1_non_pass": {
                "path": path.name,
                "bytes": path.stat().st_size,
                "aggregate_schema": BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
                "stage": BANK_SET_S1_STAGE,
                "required_gate": "non_pass",
                "authority_commit": authority_commit,
            }
        },
    }
    observed = required_s1_non_pass_authority(config, asset_root=tmp_path)
    assert observed is not None and observed["gate"] == "non_pass"

    aggregate["tasks"]["93"]["checks"]["correct_held"] = True
    write_json_atomic(path, aggregate)
    config["authorities"]["required_s1_non_pass"]["bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="S1 result changed"):
        required_s1_non_pass_authority(config, asset_root=tmp_path)
