"""Small CPU checks of evidence routing, shared revisions and uncut gradients."""
from __future__ import annotations

import pytest
import torch

from ember.experience_compiler.model import ExperienceCompiler, EXPERIENCE_SHAPES
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def mt_state():
    generator = torch.Generator().manual_seed(10)
    return {name + suffix: torch.randn(shape, generator=generator) * 0.02
            for name in ("first", "second")
            for suffix, shape in ((LORA_A_SUFFIX, (128, 64)), (LORA_B_SUFFIX, (64, 128)))}


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
    return ExperienceCompiler(mt_state(), ("first", "second"), frame_chunk=1, experience_chunk=1)


def test_fresh_construction_preserves_rng_and_direct_decode():
    template = mt_state()
    before = torch.get_rng_state().clone()
    compiler = ExperienceCompiler(template, ("first", "second"))
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        q = compiler.initial(teacher())
        assert q.shape == (2, 128, 256)
        decoded = compiler.decode(q)
    assert set(decoded) == set(template)
    for key in template:
        torch.testing.assert_close(decoded[key], template[key], rtol=1e-6, atol=1e-7)
    assert all(parameter.requires_grad for parameter in compiler.parameters())


def test_feedback_and_current_q_change_queries_before_spatial_compression(model):
    teaching, evidence = teacher(), experience(1)
    changed_feedback = {**evidence, "feedback": evidence["feedback"] + 2.0}
    calls = []
    hook = model.reader.spatial.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].detach().clone()))
    with torch.no_grad():
        q = model.initial(teaching)
        calls.clear()
        model.reader(teaching, model.encoder(evidence), q)
        original = torch.cat(calls)
        calls.clear()
        model.reader(teaching, model.encoder(changed_feedback), q)
        feedback_changed = torch.cat(calls)
        calls.clear()
        changed_q = q + torch.randn(q.shape, generator=torch.Generator().manual_seed(42))
        model.reader(teaching, model.encoder(evidence), changed_q)
        q_changed = torch.cat(calls)
    hook.remove()
    assert original.shape == (2, 8, 256)
    assert not torch.allclose(original, feedback_changed)
    assert not torch.allclose(original, q_changed)


def test_encoder_observes_late_horizon_and_ordered_facts(model):
    evidence = experience(2)
    late_horizon = {**evidence, "hidden": evidence["hidden"].clone()}
    late_horizon["hidden"][:, :, 49] += 3.0
    # Swap real observations/facts while keeping their chronological identities.
    reordered = {name: value.flip(0) if name not in {"episode", "step"} else value
                 for name, value in evidence.items()}
    with torch.no_grad():
        original = model.encoder(evidence)
        changed_hidden = model.encoder(late_horizon)
        changed_order = model.encoder(reordered)
        empty = model.encoder(experience(0))
    assert original.shape == empty.shape == (16, 256)
    assert not torch.allclose(original, changed_hidden)
    assert not torch.allclose(original, changed_order)


def test_unexecuted_action_values_cannot_enter_facts(model):
    evidence = experience(2)
    not_executed = {**evidence, "actions": evidence["actions"].clone()}
    not_executed["actions"][:, 2:] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model.encoder(evidence), model.encoder(not_executed))


def test_shared_revision_is_zero_initialized_and_accepts_variable_evidence(model):
    with torch.no_grad():
        teaching = teacher()
        q = model.initial(teaching)
        for count in (0, 1, 4, 2, 0):
            revised = model.revise(q, teaching, experience(count))
            assert torch.equal(q, revised)
            q = revised
    torch.testing.assert_close(model.update_gate.bias, torch.full((256,), -2.0))


def test_revisions_backpropagate_to_initial_state_and_all_evidence_modules(model):
    teaching, evidence = teacher(requires_grad=True), experience(3, requires_grad=True)
    # Represents an update projection that has started learning after identity.
    generator = torch.Generator().manual_seed(40)
    with torch.no_grad():
        model.update_out.weight.copy_(torch.randn(model.update_out.weight.shape, generator=generator) * 0.001)
    q = model.initial(teaching)
    q.retain_grad()
    q1 = model.revise(q, teaching, {name: value[:1] for name, value in evidence.items()})
    q2 = model.revise(q1, teaching, evidence)
    (q2 * torch.randn(q2.shape, generator=generator)).mean().backward()
    assert q.grad is not None and bool((q.grad != 0).any())
    checked = (model.reader.image.weight, model.reader.hidden.weight,
               model.reader.language.weight, model.encoder.image.weight,
               model.encoder.hidden.weight, model.encoder.numeric[0].weight,
               model.target_identity, model.rank_identity,
               model.initializer[0].read.query.weight, model.updater[0].read.query.weight)
    for parameter in checked:
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0).any())
    for field in ("phi", "hidden", "language"):
        assert teaching[field].grad is None
    for field in ("images", "hidden", "proprio", "actions", "feedback"):
        assert evidence[field].grad is None


def test_teacher_information_wall_and_complete_horizon(model):
    teaching = teacher()
    with pytest.raises(ValueError, match="only Phi"):
        model.initial({**teaching, "teacher_action": torch.zeros(7)})
    with pytest.raises(ValueError, match="complete ordered"):
        model.initial({**teaching, "hidden": teaching["hidden"][:, :49]})
    with pytest.raises(ValueError, match="real stride"):
        model.initial({**teaching, "indices": torch.tensor([5, 0])})


def test_experience_requires_actual_mask_and_time_identity(model):
    evidence = experience(2)
    with pytest.raises(ValueError, match="boolean actual-action"):
        model.encoder({**evidence, "executed": evidence["executed"].float()})
    with pytest.raises(ValueError, match="increasing actual"):
        model.encoder({**evidence, "step": torch.tensor([5, 0])})
    with pytest.raises(ValueError, match="only the registered"):
        model.encoder({**evidence, "task_id": torch.zeros(2)})
