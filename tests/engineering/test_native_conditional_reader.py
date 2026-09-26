"""Direct graph and cotangent replay for the bounded native reader."""

import json

import pytest
import torch

from ember.writer.native_conditional_reader import NativeConditionalReader, TeachingMemory
from ember.writer.native_reader_engineering import _resume_prefix


def _policy():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.paligemma_with_expert = torch.nn.Module()
    policy.model.paligemma_with_expert.gemma_expert = torch.nn.Module()
    policy.model.paligemma_with_expert.gemma_expert.model = torch.nn.Module()
    policy.model.paligemma_with_expert.gemma_expert.model.layers = torch.nn.ModuleList(
        [torch.nn.Module() for _ in range(10)]
    )
    target = policy.model.paligemma_with_expert.gemma_expert.model.layers[9]
    target.self_attn = torch.nn.Module()
    target.self_attn.q_proj = torch.nn.Linear(1024, 2048, bias=False)
    target.self_attn.v_proj = torch.nn.Linear(1024, 256, bias=False)
    policy.requires_grad_(False)
    return policy


def _memory(mode, *, core=None, procedure=None):
    core = torch.randn(1, 4, 256) if core is None else core
    mask = torch.tensor([[True, True, True, False]])
    if mode == "R_L":
        return TeachingMemory(mode, core, mask)
    procedure = torch.randn(1, 3, 256) if procedure is None else procedure
    return TeachingMemory(mode, core, mask, procedure,
                          torch.tensor([[0, 5, 10]]), torch.ones(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("mode", ["R_V", "R_L"])
def test_identity_and_real_projection_hook(mode):
    torch.manual_seed(9)
    reader, policy = NativeConditionalReader(), _policy()
    hidden = torch.randn(1, 50, 1024)
    memory = _memory(mode)
    target = policy.model.paligemma_with_expert.gemma_expert.model.layers[9].self_attn
    original = (target.q_proj(hidden), target.v_proj(hidden))
    with reader.installed(policy, memory) as counts:
        actual = (target.q_proj(hidden), target.v_proj(hidden))
    assert counts == {"q": 1, "v": 1}
    assert all(torch.equal(left, right) for left, right in zip(original, actual))
    with torch.no_grad():
        reader.out_q.weight.normal_(std=0.01)
        reader.out_v.weight.normal_(std=0.01)
    with reader.installed(policy, memory):
        changed = (target.q_proj(hidden), target.v_proj(hidden))
    assert all(not torch.equal(left, right) for left, right in zip(original, changed))
    moved = hidden.roll(1, dims=1)
    with reader.installed(policy, memory):
        moved_q = target.q_proj(moved)
    assert not torch.allclose(changed[0] - original[0], moved_q - target.q_proj(moved))
    assert not any(parameter.requires_grad for parameter in policy.parameters())


def test_direct_and_memory_replay_teacher_gradients():
    torch.manual_seed(13)
    reader = NativeConditionalReader().double()
    with torch.no_grad():
        reader.out_q.weight.normal_(std=0.01)
    policy = _policy().double()
    teacher = torch.nn.Linear(7, 256, bias=False).double()
    input_value = torch.randn(1, 3, 7, dtype=torch.double)
    hidden = torch.randn(2, 50, 1024, dtype=torch.double)
    target = policy.model.paligemma_with_expert.gemma_expert.model.layers[9].self_attn.q_proj

    def memory():
        procedure = teacher(input_value)
        return _memory("R_V", core=procedure[:, :2].repeat(1, 2, 1), procedure=procedure)

    with reader.installed(policy, memory()):
        target(hidden).square().mean().backward()
    direct = teacher.weight.grad.detach().clone()
    teacher.zero_grad(set_to_none=True)
    reader.zero_grad(set_to_none=True)
    leaf = memory().leaves()
    with reader.installed(policy, leaf):
        target(hidden).square().mean().backward()
    cotangents = tuple(value.grad.detach().clone() for value in leaf.values())
    teacher.zero_grad(set_to_none=True)
    replay = memory()
    torch.autograd.backward(replay.values(), cotangents)
    assert torch.allclose(teacher.weight.grad, direct, rtol=2e-5, atol=1e-7)


def test_language_memory_rejects_video_and_mask_errors():
    with pytest.raises(ValueError, match="cannot receive video"):
        TeachingMemory("R_L", torch.zeros(1, 1, 256), torch.ones(1, 1, dtype=torch.bool),
                       torch.zeros(1, 1, 256))
    with pytest.raises(ValueError, match="invalid Core"):
        TeachingMemory("R_L", torch.zeros(1, 1, 256), torch.zeros(1, 1, dtype=torch.bool))


def test_resume_uses_checkpoint_history_prefix_without_changing_parent(tmp_path):
    path = tmp_path / "metrics.jsonl"
    complete = "".join(json.dumps({"update": step}) + "\n" for step in range(1, 5))
    path.write_text(complete)
    assert [json.loads(row)["update"] for row in _resume_prefix(tmp_path, 2)] == [1, 2]
    assert path.read_text() == complete
    path.write_text(json.dumps({"update": 1}) + "\n")
    with pytest.raises(ValueError, match="history prefix"):
        _resume_prefix(tmp_path, 2)
