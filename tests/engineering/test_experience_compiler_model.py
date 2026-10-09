"""CPU contracts for ordered real context, fixed coordinates and energy credit."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from ember.experience_compiler.model import ExperienceCompiler, EXPERIENCE_SHAPES
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


TARGETS = ("first", "second", "third")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def mt_state(names=TARGETS):
    generator = torch.Generator().manual_seed(10)
    return {name + suffix: torch.randn(shape, generator=generator) * .02
            for name in names for suffix, shape in
            ((LORA_A_SUFFIX, (128, 64)), (LORA_B_SUFFIX, (80, 128)))}


def teacher(count=2, *, requires_grad=False):
    generator = torch.Generator().manual_seed(20)
    return {"phi": torch.randn(count, 512, 2048, generator=generator, requires_grad=requires_grad),
            "hidden": torch.randn(count, 50, 1024, generator=generator, requires_grad=requires_grad),
            "language": torch.randn(2048, generator=generator, requires_grad=requires_grad),
            "indices": torch.arange(count) * 5}


def experience(count=2, *, requires_grad=False):
    generator = torch.Generator().manual_seed(30)
    result = {name: torch.randn((count, *tail), generator=generator, requires_grad=requires_grad)
              for name, tail in EXPERIENCE_SHAPES.items()
              if name not in {"executed", "episode", "step"}}
    result["executed"] = torch.tensor([True, True, False, False, False]).expand(count, -1).clone()
    result["episode"] = torch.zeros(count, dtype=torch.long)
    result["step"] = torch.arange(count) * 5
    return result


@pytest.fixture
def model():
    return ExperienceCompiler(mt_state(), TARGETS, frame_chunk=1, experience_chunk=1)


def test_fresh_construction_preserves_rng_and_retires_parameter_routing():
    incoming = mt_state()
    before = torch.get_rng_state().clone()
    compiler = ExperienceCompiler(incoming, TARGETS)
    assert torch.equal(before, torch.get_rng_state())
    assert not any(hasattr(compiler, name) for name in
                   ("parameter_encoder", "editor", "decoder", "delta", "update_gate", "update_out"))
    assert not hasattr(compiler.reader, "state_read")
    layers = [layer for layer in compiler.energy if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(384, 256), (256, 128), (128, 1)]
    assert layers[-1].bias is None and torch.count_nonzero(layers[-1].weight) == 0
    assert compiler.action_projection.in_features == 35 and compiler.action_projection.out_features == 128
    assert all(parameter.requires_grad for parameter in compiler.parameters())


def test_fixed_complete_metric_uses_actual_factor_units_and_fp32(model):
    incoming = mt_state()
    assert model.keys == tuple(incoming) and model.shapes == {key: tuple(value.shape) for key, value in incoming.items()}
    direction = {key: torch.ones_like(value, dtype=torch.bfloat16) for key, value in incoming.items()}
    expected = torch.stack([value.square().mean().sqrt() for value in incoming.values()])
    torch.testing.assert_close(model.rms, expected)
    assert not model.rms.requires_grad
    scaled = model.precondition(direction)
    assert set(scaled) == set(incoming)
    for index, key in enumerate(model.keys):
        assert scaled[key].dtype == torch.float32 and scaled[key].shape == incoming[key].shape
        torch.testing.assert_close(scaled[key], torch.full_like(scaled[key], expected[index].square()))
    for value in incoming.values():
        value.zero_()
    torch.testing.assert_close(model.rms, expected)
    zeros = {key: torch.zeros_like(value) for key, value in incoming.items()}
    floored = ExperienceCompiler(zeros, TARGETS)
    torch.testing.assert_close(floored.rms, torch.full_like(floored.rms, 1e-6))
    assert all(torch.count_nonzero(value) > 0 for value in floored.precondition(direction).values())


def test_all38_actual_a_and_b_coordinates_are_registered():
    names = tuple(f"target{index}" for index in range(38))
    incoming = mt_state(names)
    compiler = ExperienceCompiler(incoming, names)
    assert len(compiler.keys) == compiler.rms.numel() == 76
    scaled = compiler.precondition({key: torch.ones_like(value) for key, value in incoming.items()})
    assert len(scaled) == 76 and all(value.shape == incoming[key].shape for key, value in scaled.items())


def test_feedback_changes_teaching_queries_before_spatial_compression(model):
    teaching, evidence = teacher(), experience(2)
    calls = []
    hook = model.reader.spatial.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].detach().clone()))
    with torch.no_grad():
        original = model.context(teaching, evidence, torch.tensor([0]))
        first_queries = torch.cat(calls)
        calls.clear()
        changed = model.context(teaching, {**evidence, "feedback": evidence["feedback"] + 2}, torch.tensor([0]))
        second_queries = torch.cat(calls)
    hook.remove()
    assert first_queries.shape == (2, 8, 256) and not torch.allclose(first_queries, second_queries)
    assert original.shape == (1, 256) and not torch.allclose(original, changed)


def test_context_reads_ordered_teacher_and_all_decisions_not_only_support(model):
    teaching, evidence = teacher(), experience(3)
    reordered = {**teaching, "phi": teaching["phi"].flip(0), "hidden": teaching["hidden"].flip(0)}
    unselected = {**evidence, "feedback": evidence["feedback"].clone()}
    unselected["feedback"][2] += 3
    with torch.no_grad():
        original = model.context(teaching, evidence, torch.tensor([0]))
        assert not torch.allclose(original, model.context(reordered, evidence, torch.tensor([0])))
        assert not torch.allclose(original, model.context(teaching, unselected, torch.tensor([0])))


def test_encoder_keeps_full_actual_hidden_and_ordered_time(model):
    evidence = experience(2)
    late_horizon = {**evidence, "hidden": evidence["hidden"].clone()}
    late_horizon["hidden"][:, :, 49] += 3
    reordered = {name: value.flip(0) if name not in {"episode", "step"} else value
                 for name, value in evidence.items()}
    with torch.no_grad():
        facts, original = model.encoder.encode(evidence)
        assert facts.shape == (2, 256) and original.shape == (16, 256)
        assert not torch.allclose(original, model.encoder(late_horizon))
        assert not torch.allclose(original, model.encoder(reordered))
        empty, slots = model.encoder.encode({})
        assert empty.shape == (0, 256) and slots.shape == (16, 256)
        assert model.reader(teacher(), slots).shape == (17, 256)


def test_unexecuted_action_values_cannot_enter_facts(model):
    evidence = experience(2)
    not_executed = {**evidence, "actions": evidence["actions"].clone()}
    not_executed["actions"][:, 2:] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model.encoder(evidence), model.encoder(not_executed))


def test_zero_action_cotangent_has_first_update_mixed_derivative(model):
    contexts = model.context(teacher(), experience(2), torch.tensor([0, 1]))
    actions = torch.randn(2, 5, 7, requires_grad=True)
    pressure = model.action_cotangent(contexts, actions, create_graph=True)
    assert pressure.shape == (2, 5, 7) and torch.count_nonzero(pressure) == 0
    (pressure * torch.linspace(-1, 1, pressure.numel()).reshape_as(pressure)).sum().backward()
    assert model.energy[-1].weight.grad is not None
    assert torch.isfinite(model.energy[-1].weight.grad).all() and model.energy[-1].weight.grad.norm() > 0
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for name, parameter in model.named_parameters() if name != "energy.4.weight")
    assert actions.grad is None


def test_learned_energy_credits_readers_but_stops_raw_history_and_actions(model):
    teaching, evidence = teacher(requires_grad=True), experience(3, requires_grad=True)
    actions = torch.randn(2, 5, 7, requires_grad=True)
    generator = torch.Generator().manual_seed(40)
    with torch.no_grad():
        model.energy[-1].weight.copy_(torch.randn(model.energy[-1].weight.shape, generator=generator) * .02)
    contexts = model.context(teaching, evidence, torch.tensor([0, 2]))
    pressure = model.action_cotangent(contexts, actions, create_graph=True)
    (pressure * torch.randn(pressure.shape, generator=generator)).sum().backward()
    checked = (model.reader.image.weight, model.reader.hidden.weight, model.reader.language.weight,
               model.encoder.image.weight, model.encoder.hidden.weight, model.encoder.numeric[0].weight,
               model.support_read.query.weight, model.action_projection.weight, model.energy[0].weight)
    for parameter in checked:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert actions.grad is None
    assert all(teaching[field].grad is None for field in ("phi", "hidden", "language"))
    assert all(evidence[field].grad is None for field in ("images", "hidden", "proprio", "actions", "feedback"))


def test_frozen_deployment_can_compute_action_pressure_under_no_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        contexts = model.context(teacher(), experience(1), torch.tensor([0]))
        pressure = model.action_cotangent(contexts, torch.randn(1, 5, 7), create_graph=False)
    assert not pressure.requires_grad and torch.count_nonzero(pressure) == 0


def test_teacher_and_metric_information_walls(model):
    teaching, evidence, incoming = teacher(), experience(1), mt_state()
    with pytest.raises(ValueError, match="only Phi"):
        model.context({**teaching, "teacher_action": torch.zeros(7)}, evidence, torch.tensor([0]))
    with pytest.raises(ValueError, match="complete ordered"):
        model.context({**teaching, "hidden": teaching["hidden"][:, :49]}, evidence, torch.tensor([0]))
    with pytest.raises(ValueError, match="real stride"):
        model.context({**teaching, "indices": torch.tensor([5, 0])}, evidence, torch.tensor([0]))
    with pytest.raises(ValueError, match="complete registered"):
        model.precondition({**incoming, "task_id": torch.zeros(1)})
    with pytest.raises(ValueError, match="actual registered A/B shape"):
        model.precondition({**incoming, "first" + LORA_B_SUFFIX: torch.zeros(128, 80)})
    with pytest.raises(ValueError, match="rank128"):
        ExperienceCompiler({**incoming, "first" + LORA_A_SUFFIX: torch.zeros(64, 64)}, TARGETS)


@pytest.mark.parametrize("indices", (torch.tensor([], dtype=torch.long), torch.arange(17),
                                      torch.tensor([0., 1.]), torch.tensor([0, 0]),
                                      torch.tensor([-1]), torch.tensor([2])))
def test_revision_support_is_bounded_and_only_actual_distinct_decisions(model, indices):
    with pytest.raises(ValueError, match="support indices"):
        model.context(teacher(), experience(2), indices)


def test_empty_e_does_not_manufacture_functional_support(model):
    with pytest.raises(ValueError, match="existing actual"):
        model.context(teacher(), {}, torch.tensor([0]))


def test_experience_requires_actual_mask_and_time_identity(model):
    evidence = experience(2)
    with pytest.raises(ValueError, match="boolean actual-action"):
        model.encoder({**evidence, "executed": evidence["executed"].float()})
    with pytest.raises(ValueError, match="increasing actual"):
        model.encoder({**evidence, "step": torch.tensor([5, 0])})
    with pytest.raises(ValueError, match="only the registered"):
        model.encoder({**evidence, "task_id": torch.zeros(2)})
