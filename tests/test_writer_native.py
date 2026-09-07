"""The new observation boundary: real final Z and post-normalization full H."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

from ember.ecp.policy_effects import ExecutionPolicyPrefix
from ember.writer.native import NativeVideoObserver


def test_native_response_is_actual_action_projection_input_and_meta_vjp():
    class Meta(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))

        @contextmanager
        def installed(self, expert):
            yield

    meta = Meta()

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_out_proj = nn.Linear(1024, 32, bias=False)
            self.action_out_proj.requires_grad_(False)

        def denoise_step(self, padding, cache, noise, timestep):
            pre_norm = noise.repeat(1, 1, 32) * meta.scale
            self.expected = nn.functional.layer_norm(pre_norm, (1024,)).float()
            return self.action_out_proj(self.expected)

    observer = object.__new__(NativeVideoObserver)
    core = Core()
    observer.policy, observer.meta, observer.expert = SimpleNamespace(model=core), meta, None
    observer.device, observer.probe = torch.device("cpu"), torch.randn(50, 32)
    chunk = SimpleNamespace(on_device=lambda device: (torch.ones(2, 4, dtype=torch.bool), None))
    response = observer.capture(chunk)
    assert response.shape == (2, 50, 1024)
    torch.testing.assert_close(response, core.expected)
    gradient = torch.randn_like(response)
    expected = torch.autograd.grad(core.expected, meta.scale, gradient)[0]
    response = observer.capture(chunk)
    response.backward(gradient)
    torch.testing.assert_close(meta.scale.grad, expected)
    assert not core.action_out_proj._forward_pre_hooks


def test_prefix_retains_final_visual_and_exact_span_from_one_forward(monkeypatch):
    observer = object.__new__(NativeVideoObserver)
    observer.policy, observer.device = object(), torch.device("cpu")
    padding = torch.tensor([[True] * 3 + [False] * 3 + [True] * 4]).expand(2, -1)
    embeddings = torch.arange(20).reshape(2, 10, 1).float()
    monkeypatch.setattr("ember.writer.native.prepare_execution_policy_prefix",
                        lambda *args: ExecutionPolicyPrefix(embeddings, padding))
    calls = []

    def forward(policy, prefix):
        calls.append(prefix)
        return prefix.embeddings + 100, ((torch.zeros(2, 1, 7, 1), torch.ones(2, 1, 7, 1), None),)

    monkeypatch.setattr("ember.writer.native.prepare_prefix_features_and_cache", forward)
    chunk = observer.prefix(torch.zeros(2, 3, 8, 8, dtype=torch.uint8), torch.ones(1, 4, dtype=torch.long),
                            torch.ones(1, 4, dtype=torch.bool), torch.tensor([[False, True, True, False]]))
    assert len(calls) == 1
    torch.testing.assert_close(chunk.visual_tokens, embeddings[:, [0, 1, 2, 7, 8]] + 100)
    assert chunk.padding.shape == (2, 7)
    assert chunk.visual_mask.all()
    assert chunk.tensor_bytes == sum(value.numel() * value.element_size() for value in
        (chunk.padding, chunk.visual_tokens, chunk.visual_mask, chunk.layers[0][0], chunk.layers[0][1]))
