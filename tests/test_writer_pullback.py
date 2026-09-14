"""Small native-autograd oracles for all 38 projected source targets and q credit."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import eager_attention_forward

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget
from ember.writer.correction import NativeCorrectionReader, SourceCoordinates
from ember.writer.factor import native_input_basis
from ember.writer.native import FrozenInputChunk, NativeCondition


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(32, 32, bias=False)

    def forward(self, value):
        return self.base_layer(value)


class _NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj = _Projection(), _Projection()
        self.k_proj, self.o_proj = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
        self.num_key_value_groups = 1

    def forward(self, value, prefix):
        normalized = F.layer_norm(value, (32,))
        query, key, content = self.q_proj(normalized), self.k_proj(normalized), self.v_proj(normalized)
        query, key, content, prefix = (x.reshape(len(x), -1, 2, 16).transpose(1, 2)
                                       for x in (query, key, content, prefix))
        output, _ = eager_attention_forward(self, query, torch.cat((prefix, key), dim=2),
                                             torch.cat((prefix, content), dim=2), None, scaling=.25)
        return value + .1 * self.o_proj(output.reshape_as(value))


class _NativeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_in_proj, self.action_out_proj = _Projection(), _Projection()
        self.layers = nn.ModuleList([_NativeAttention() for _ in range(18)])
        self.calls = 0
        self.prefix_calls = 0

    def forward(self, padding, cache, noise, clock):
        self.calls += 1
        assert not torch.is_autocast_enabled('cpu')
        assert bool((clock == 1).all()) and bool(padding.all())
        value = self.action_in_proj(noise)
        if not isinstance(cache, torch.Tensor):
            cache = cache[0][0]
        for layer in self.layers:
            value = layer(value, cache)
        return self.action_out_proj(F.layer_norm(value, (32,)))

    def denoise_step(self, *args):
        return self(*args)


@pytest.fixture
def source(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    policy = nn.Module()
    policy.model = _NativeCore()
    policy.requires_grad_(False).eval()
    names = ['model.action_in_proj']
    names += [f'model.layers.{index}.{name}' for index in range(18) for name in ('q_proj', 'v_proj')]
    names += ['model.action_out_proj']
    contract = SimpleNamespace(targets=tuple(LoRATarget(name, 32, 32) for name in names), rank=16, alpha=16)
    probe = torch.randn(50, 32, generator=torch.Generator().manual_seed(1729))
    reader = NativeCorrectionReader(policy, contract, probe)
    embeddings = torch.randn(3, 4, 32)
    mask = torch.ones(3, 4, dtype=torch.bool)

    def prepare(owner, prefix, *, native_precision):
        assert native_precision and not torch.is_autocast_enabled('cpu')
        owner.model.prefix_calls += 1
        return prefix.embeddings, ((prefix.embeddings, prefix.embeddings, None),)

    monkeypatch.setattr('ember.writer.correction.prepare_prefix_features_and_cache', prepare)
    yield reader, embeddings, mask
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in policy.parameters())
    assert all(not module._forward_hooks for module in reader.modules)
    torch.set_num_threads(previous)


def _condition(embeddings, mask, chunk):
    chunks = tuple(FrozenInputChunk(embeddings[start:start + chunk], mask[start:start + chunk],
                                   mask[start:start + chunk], mask[start:start + chunk])
                   for start in range(0, len(embeddings), chunk))
    return NativeCondition((chunks,), (torch.tensor([0, 5, 9]),), embeddings[0], mask[0])


def _dense_reference(reader, embeddings, mask, q):
    # Independent small dense weight-leaf oracle only; physical source
    # parameters stay frozen and production never constructs these matrices.
    replacements = {target.name.removeprefix('model.') + '.base_layer.weight':
                    reader.policy.get_submodule(target.name).base_layer.weight.detach().clone().requires_grad_()
                    for target in reader.contract.targets}
    values = torch.func.functional_call(reader.policy.model, replacements,
        (mask, embeddings, reader.probe.expand(len(embeddings), -1, -1), torch.ones(len(embeddings))))
    gradients = torch.autograd.grad(values, tuple(replacements.values()), F.pad(q, (0, 25)), create_graph=True)
    return values, gradients


@pytest.mark.parametrize('chunk', [1, 2, 3])
def test_complete_projection_and_q_adjoint_match_dense_native_autograd(source, chunk):
    reader, embeddings, mask = source
    condition = _condition(embeddings, mask, chunk)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        coordinates = reader.read(condition)
    assert isinstance(coordinates, SourceCoordinates) and not isinstance(coordinates, nn.Module)
    assert coordinates.condition is condition and coordinates.predictions.shape == (3, 50, 7)
    assert coordinates.bases['model.layers.0.q_proj'] is coordinates.bases['model.layers.0.v_proj']
    q = torch.randn_like(coordinates.predictions, requires_grad=True)
    compiled = coordinates.compile(q)
    assert len(compiled) == 76
    reference_q = q.detach().clone().requires_grad_()
    native_x = []
    name = 'model.layers.1.q_proj'
    handle = reader.policy.get_submodule(name).base_layer.register_forward_pre_hook(
        lambda module, args: native_x.append(args[0].detach()))
    try:
        velocity, dense = _dense_reference(reader, embeddings, mask, reference_q)
    finally:
        handle.remove()
    _, _, right = torch.linalg.svd(native_x[0].flatten(0, 1), full_matrices=False)
    a = coordinates.bases[name]
    torch.testing.assert_close(a.T @ a, right[:16].T @ right[:16], rtol=4e-4, atol=5e-6)
    torch.testing.assert_close(coordinates.predictions, reader.probe[None, :, :7] - velocity[:, :, :7],
                               rtol=2e-4, atol=2e-6)
    direct_loss, compiled_loss = 0., 0.
    for target, gradient in zip(reader.contract.targets, dense, strict=True):
        a, b = compiled[target.name + LORA_A_SUFFIX], compiled[target.name + LORA_B_SUFFIX]
        assert not a.requires_grad and b.requires_grad
        torch.testing.assert_close(a @ a.T, torch.eye(16), atol=2e-6, rtol=2e-5)
        projected = gradient @ a.T / 3
        torch.testing.assert_close(b, projected, rtol=3e-4, atol=3e-6)
        torch.testing.assert_close(b @ a, gradient @ (a.T @ a) / 3, rtol=3e-4, atol=3e-6)
        weight = torch.randn_like(b)
        direct_loss = direct_loss + (projected * weight).sum()
        compiled_loss = compiled_loss + (b * weight).sum()
    expected, = torch.autograd.grad(direct_loss, reference_q)
    actual, = torch.autograd.grad(compiled_loss, q)
    torch.testing.assert_close(actual, expected, rtol=4e-4, atol=4e-6)
    torch.testing.assert_close(compiled_loss.detach(), (q.detach() * actual).sum(), rtol=4e-4, atol=5e-6)
    assert actual.norm() > 0 and bool(actual.abs().sum((0, 2)).gt(0).all())
    assert reader.policy.model.prefix_calls == len(condition.videos[0])


def test_no_grad_materialization_identity_and_last_frame_credit(source):
    reader, embeddings, mask = source
    coordinates = reader.read(_condition(embeddings, mask, 2))
    q = torch.zeros_like(coordinates.predictions, requires_grad=True)
    state = coordinates.compile(q)
    assert all(state[target.name + LORA_B_SUFFIX].count_nonzero() == 0 for target in reader.contract.targets)
    # A constant zero q still has the correct nonzero first-update gradient.
    gradient, = torch.autograd.grad(state['model.action_out_proj' + LORA_B_SUFFIX].square().sum() +
                                    state['model.action_out_proj' + LORA_B_SUFFIX].sum(), q)
    assert gradient.norm() > 0
    q = torch.zeros_like(q)
    q[-1] = torch.randn_like(q[-1])
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        materialized = coordinates.compile(q)
    assert all(not value.requires_grad for value in materialized.values())
    assert materialized['model.action_in_proj' + LORA_B_SUFFIX].norm() > 0
    with pytest.raises(ValueError, match='50x7'):
        coordinates.compile(torch.zeros(3, 50, 32))


def test_native_gram_basis_is_the_uncentered_full_input_top_rank_projection():
    torch.manual_seed(19)
    x = torch.randn(150, 32) * torch.linspace(.3, 3, 32)
    basis = native_input_basis(sum(part.T @ part for part in x.split(50)), 16)
    _, _, right = torch.linalg.svd(x, full_matrices=False)
    expected = right[:16]
    torch.testing.assert_close(basis.T @ basis, expected.T @ expected, rtol=3e-4, atol=5e-6)
