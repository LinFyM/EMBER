"""CPU checks of actual parameter routing, ordered facts and editing credit."""
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


class JacobianDecoder(nn.Module):
    """A small assembly mock; separate decoder tests own the nonlinear rule."""

    def __init__(self, mt_state, target_names, **kwargs):
        super().__init__()
        self.target_names = tuple(target_names)

    def forward(self, incoming_state, delta_q):
        result = {}
        for target, name in enumerate(self.target_names):
            change = delta_q[target, :, :1]
            for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
                key = name + suffix
                result[key] = incoming_state[key].detach() + (change if suffix == LORA_A_SUFFIX else change.T)
        return result


@pytest.fixture(autouse=True)
def small_decoder(monkeypatch):
    from ember.experience_compiler import decoder
    monkeypatch.setattr(decoder, "CoordinateDecoder", JacobianDecoder)


def mt_state():
    generator = torch.Generator().manual_seed(10)
    return {name + suffix: torch.randn(shape, generator=generator) * .02
            for name, a_width, b_width in zip(TARGETS, (64, 64, 96), (80, 80, 64), strict=True)
            for suffix, shape in ((LORA_A_SUFFIX, (128, a_width)), (LORA_B_SUFFIX, (b_width, 128)))}


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


def test_fresh_construction_preserves_rng_and_retires_persistent_q():
    incoming = mt_state()
    before = torch.get_rng_state().clone()
    compiler = ExperienceCompiler(incoming, TARGETS)
    assert torch.equal(before, torch.get_rng_state())
    assert not any(hasattr(compiler, name) for name in ("initial", "revise", "decode", "initializer"))
    with torch.no_grad():
        outgoing = compiler(incoming, teacher(), experience(1))
    assert set(outgoing) == set(incoming)
    assert all(torch.equal(outgoing[key], incoming[key]) for key in incoming)
    assert all(parameter.requires_grad for parameter in compiler.parameters())


def test_actual_factor_roles_width_sharing_and_frozen_rms(model):
    encoder, incoming = model.parameter_encoder, mt_state()
    assert set(encoder.a) == {"64", "96"} and set(encoder.b) == {"64", "80"}
    assert encoder.a["64"] is not encoder.b["64"]
    expected = []
    for target, name in enumerate(TARGETS):
        a, b = incoming[name + LORA_A_SUFFIX], incoming[name + LORA_B_SUFFIX]
        torch.testing.assert_close(encoder.rms[target], torch.stack((a.square().mean().sqrt(), b.square().mean().sqrt())))
        expected.append(torch.cat((encoder.a[str(a.shape[1])](a / encoder.rms[target, 0]),
                                   encoder.b[str(b.shape[0])](b.T / encoder.rms[target, 1])), -1))
    expected = encoder.norm(torch.stack(expected) + encoder.target_identity[:, None] + encoder.rank_identity[None])
    torch.testing.assert_close(encoder(incoming), expected)
    before = encoder.rms.clone()
    changed = {**incoming, "first" + LORA_B_SUFFIX: incoming["first" + LORA_B_SUFFIX] * 3}
    assert not torch.allclose(encoder(incoming), encoder(changed))
    assert torch.equal(before, encoder.rms) and not encoder.rms.requires_grad
    zeros = {key: torch.zeros_like(value) for key, value in incoming.items()}
    floor = ExperienceCompiler(zeros, TARGETS).parameter_encoder.rms
    torch.testing.assert_close(floor, torch.full_like(floor, 1e-6))


def test_feedback_and_actual_a_b_change_queries_before_spatial_compression(model):
    teaching, evidence, incoming = teacher(), experience(1), mt_state()
    calls = []
    hook = model.reader.spatial.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].detach().clone()))
    def queries(state, facts):
        calls.clear()
        model.delta(state, teaching, facts)
        return torch.cat(calls)
    with torch.no_grad():
        original = queries(incoming, evidence)
        feedback = queries(incoming, {**evidence, "feedback": evidence["feedback"] + 2})
        changes = []
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            key = "first" + suffix
            changes.append(queries({**incoming, key: incoming[key] * 2}, evidence))
    hook.remove()
    assert original.shape == (2, 8, 256) and not torch.allclose(original, feedback)
    assert all(not torch.allclose(original, changed) for changed in changes)


def test_encoder_observes_late_horizon_and_ordered_facts(model):
    evidence = experience(2)
    late_horizon = {**evidence, "hidden": evidence["hidden"].clone()}
    late_horizon["hidden"][:, :, 49] += 3
    reordered = {name: value.flip(0) if name not in {"episode", "step"} else value
                 for name, value in evidence.items()}
    with torch.no_grad():
        original, changed_hidden = model.encoder(evidence), model.encoder(late_horizon)
        changed_order, empty = model.encoder(reordered), model.encoder(experience(0))
    assert original.shape == empty.shape == (16, 256)
    assert not torch.allclose(original, changed_hidden) and not torch.allclose(original, changed_order)


def test_unexecuted_action_values_cannot_enter_facts(model):
    evidence = experience(2)
    not_executed = {**evidence, "actions": evidence["actions"].clone()}
    not_executed["actions"][:, 2:] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model.encoder(evidence), model.encoder(not_executed))


def test_delta_is_zero_for_variable_evidence_and_has_only_last_projection_credit(model):
    incoming, teaching = mt_state(), teacher()
    with torch.no_grad():
        for count in (0, 1, 4, 2, 0):
            delta = model.delta(incoming, teaching, experience(count))
            assert delta.shape == (3, 128, 256) and torch.count_nonzero(delta) == 0
    torch.testing.assert_close(model.update_gate.bias, torch.full((256,), -2.0))
    assert torch.count_nonzero(model.update_gate.weight) > 0
    outgoing = model(incoming, teaching, experience(2))
    generator = torch.Generator().manual_seed(40)
    loss = sum((value * torch.randn(value.shape, generator=generator)).mean() for value in outgoing.values())
    loss.backward()
    for parameter in (model.update_out.weight, model.update_out.bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for name, p in model.named_parameters() if not name.startswith("update_out."))


def test_learned_delta_backpropagates_to_shared_modules_but_stops_raw_history(model):
    teaching, evidence = teacher(requires_grad=True), experience(3, requires_grad=True)
    incoming = {key: value.detach().requires_grad_() for key, value in mt_state().items()}
    generator = torch.Generator().manual_seed(40)
    with torch.no_grad():
        model.update_out.weight.copy_(torch.randn(model.update_out.weight.shape, generator=generator) * .001)
    outgoing = model(incoming, teaching, evidence)
    sum((value * torch.randn(value.shape, generator=generator)).mean() for value in outgoing.values()).backward()
    checked = (model.reader.image.weight, model.reader.hidden.weight, model.reader.language.weight,
               model.encoder.image.weight, model.encoder.hidden.weight, model.encoder.numeric[0].weight,
               model.parameter_encoder.a["64"].weight, model.parameter_encoder.b["80"].weight,
               model.parameter_encoder.target_identity, model.parameter_encoder.rank_identity,
               model.editor[0].read.query.weight, model.editor[1].read.query.weight)
    for parameter in checked:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(value.grad is None for value in incoming.values())
    assert all(teaching[field].grad is None for field in ("phi", "hidden", "language"))
    assert all(evidence[field].grad is None for field in ("images", "hidden", "proprio", "actions", "feedback"))


def test_teacher_and_parameter_information_walls(model):
    teaching, incoming = teacher(), mt_state()
    with pytest.raises(ValueError, match="only Phi"):
        model.delta(incoming, {**teaching, "teacher_action": torch.zeros(7)}, {})
    with pytest.raises(ValueError, match="complete ordered"):
        model.delta(incoming, {**teaching, "hidden": teaching["hidden"][:, :49]}, {})
    with pytest.raises(ValueError, match="real stride"):
        model.delta(incoming, {**teaching, "indices": torch.tensor([5, 0])}, {})
    with pytest.raises(ValueError, match="complete registered A/B"):
        model.parameter_encoder({**incoming, "task_id": torch.zeros(1)})
    with pytest.raises(ValueError, match="actual registered A/B shape"):
        model.parameter_encoder({**incoming, "first" + LORA_B_SUFFIX: torch.zeros(128, 80)})


def test_experience_requires_actual_mask_and_time_identity(model):
    evidence = experience(2)
    with pytest.raises(ValueError, match="boolean actual-action"):
        model.encoder({**evidence, "executed": evidence["executed"].float()})
    with pytest.raises(ValueError, match="increasing actual"):
        model.encoder({**evidence, "step": torch.tensor([5, 0])})
    with pytest.raises(ValueError, match="only the registered"):
        model.encoder({**evidence, "task_id": torch.zeros(2)})
