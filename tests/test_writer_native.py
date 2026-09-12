"""Frozen input caching and complete joint direct-Z / R-through-KV credit."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

from ember.ecp.policy_effects import ExecutionPolicyPrefix
from ember.writer.native import FrozenInputChunk, NativeCondition, NativeVideoObserver


def test_native_response_and_direct_visual_share_complete_prefix_vjp(monkeypatch):
    class Meta(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.8))

        @contextmanager
        def installed(self, expert):
            yield

    meta, vl_meta = Meta(), Meta()

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_out_proj = nn.Linear(1024, 32, bias=False)
            self.action_out_proj.requires_grad_(False)

        def denoise_step(self, padding, cache, noise, timestep):
            pre_norm = noise.repeat(1, 1, 32) * meta.scale + cache
            self.expected = nn.functional.layer_norm(pre_norm, (1024,)).float()
            return self.action_out_proj(self.expected)

    observer = object.__new__(NativeVideoObserver)
    core = Core()
    observer.policy, observer.meta, observer.expert = SimpleNamespace(model=core), meta, None
    observer.vl_meta, observer.gemma = vl_meta, None
    observer.device, observer.probe = torch.device("cpu"), torch.randn(50, 32)
    chunk = FrozenInputChunk(torch.randn(2, 4, 1024), torch.ones(2, 4, dtype=torch.bool),
                             torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.bool))
    calls = []

    def forward(policy, prefix, *, track_grad):
        calls.append(track_grad)
        visual = prefix.embeddings * vl_meta.scale
        return visual, visual.mean(1, keepdim=True)

    monkeypatch.setattr("ember.writer.native.prepare_prefix_features_and_cache", forward)
    response, visual = observer.capture(chunk)
    assert response.shape == (2, 50, 1024)
    torch.testing.assert_close(response, core.expected)
    response_grad, visual_grad = torch.randn_like(response), torch.randn_like(visual)
    r_only = torch.autograd.grad(response, vl_meta.scale, response_grad, retain_graph=True)[0]
    z_only = torch.autograd.grad(visual, vl_meta.scale, visual_grad, retain_graph=True)[0]
    assert r_only.abs() > 0 and z_only.abs() > 0
    expected = torch.autograd.grad((response, visual), (meta.scale, vl_meta.scale), (response_grad, visual_grad))
    condition = NativeCondition(((chunk,),), (torch.tensor([0, 5]),), torch.zeros(4, 2048), torch.ones(4, dtype=torch.bool),
                                (torch.zeros(2, 5, 7),))
    observer.backward(condition, (response_grad,), (visual_grad,))
    torch.testing.assert_close(meta.scale.grad, expected[0])
    torch.testing.assert_close(vl_meta.scale.grad, expected[1])
    torch.testing.assert_close(vl_meta.scale.grad, r_only + z_only)
    old_visual = visual.detach().clone()
    with torch.no_grad():
        vl_meta.scale.add_(.1)
    _, read_inputs = observer.read(condition)
    assert not torch.allclose(read_inputs[3][0], old_visual)
    assert calls == [True, True, False]
    assert not chunk.embeddings.requires_grad and all(p.grad is None for p in core.parameters())
    assert not core.action_out_proj._forward_pre_hooks


def test_prefix_caches_only_pre_gemma_embeddings_and_exact_masks(monkeypatch):
    observer = object.__new__(NativeVideoObserver)
    observer.policy, observer.device = object(), torch.device("cpu")
    observer.camera_names = ("agentview",)
    padding = torch.tensor([[True] * 3 + [False] * 3 + [True] * 4]).expand(2, -1)
    embeddings = torch.arange(20).reshape(2, 10, 1).float()
    monkeypatch.setattr("ember.writer.native.prepare_execution_policy_prefix",
                        lambda *args: ExecutionPolicyPrefix(embeddings, padding))
    def forbidden(*args, **kwargs):
        raise AssertionError("learned prefix values cannot enter the persistent cache")
    monkeypatch.setattr("ember.writer.native.prepare_prefix_features_and_cache", forbidden)
    chunk = observer.prefix(torch.zeros(2, 3, 8, 8, dtype=torch.uint8), torch.ones(1, 4, dtype=torch.long),
                            torch.ones(1, 4, dtype=torch.bool), torch.tensor([[False, True, True, False]]))
    torch.testing.assert_close(chunk.embeddings, embeddings[:, [0, 1, 2, 6, 7, 8, 9]])
    assert chunk.padding.shape == (2, 7)
    torch.testing.assert_close(chunk.evidence_mask,
                               torch.tensor([[True, True, True, False, True, True, False]]).expand(2, -1))
    torch.testing.assert_close(chunk.task_mask,
                               torch.tensor([[False, False, False, False, True, True, False]]).expand(2, -1))
    assert chunk.tensor_bytes == sum(value.numel() * value.element_size() for value in
        (chunk.embeddings, chunk.padding, chunk.evidence_mask, chunk.task_mask))


def test_dual_prefix_fuses_cameras_once_and_keeps_one_episode(monkeypatch):
    from ember.writer.runtime import FrozenVideoPrefixCache

    observer = object.__new__(NativeVideoObserver)
    observer.device, observer.frame_chunk = torch.device("cpu"), 2
    observer.camera_view, observer.camera_names = "dual", ("agentview", "eye_in_hand")
    observer.policy = SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=SimpleNamespace(
        embed_language_tokens=lambda tokens: torch.zeros(1, 4, 8))))
    tokens, mask = torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool)
    span = torch.tensor([[False, True, True, False]])
    observer.tokenizer = lambda languages: (tokens, mask, span)
    prior_calls = []
    def prior(video):
        prior_calls.append(len(video))
        return torch.zeros(len(video), 5, 7, dtype=torch.bfloat16)
    observer.prior = prior
    calls = []

    def embed(policy, batch):
        front = batch["observation.images.base_0_rgb"]
        wrist = batch["observation.images.left_wrist_0_rgb"]
        calls.append((front, wrist))
        assert front.shape == wrist.shape
        assert torch.all(front == 0) and torch.all(wrist == 1)
        # Two valid tokens per camera, two masked spare-camera tokens, four text tokens.
        padding = torch.tensor([[True] * 4 + [False] * 2 + [True] * 4]).expand(len(front), -1)
        embeddings = torch.arange(10).float().reshape(1, 10, 1).expand(len(front), -1, -1)
        return ExecutionPolicyPrefix(embeddings, padding)

    monkeypatch.setattr("ember.writer.native.prepare_execution_policy_prefix", embed)
    frames = torch.zeros(3, 2, 3, 8, 8, dtype=torch.uint8)
    frames[:, 1] = 255
    indices = torch.tensor([0, 5, 7])
    data = SimpleNamespace(videos=SimpleNamespace(camera_view="dual"),
        tasks={0: SimpleNamespace(authority=SimpleNamespace(language="task"))},
        load_videos=lambda task, demos: ((frames,), (indices,)))
    cache = FrozenVideoPrefixCache(observer, data, byte_limit=10**6)
    condition = cache.condition(0, (0,))
    repeated = cache.condition(0, (0,))
    assert len(calls) == 2 and cache.hits == 1 and cache.misses == 1
    assert len(condition.videos) == 1 and len(condition.videos[0]) == 2
    torch.testing.assert_close(condition.frame_indices[0], indices)
    assert repeated.videos[0] is condition.videos[0]
    assert repeated.prior_tokens[0] is condition.prior_tokens[0] and prior_calls == [3]
    assert cache.bytes == (sum(chunk.tensor_bytes for chunk in condition.videos[0])
                           + condition.prior_tokens[0].numel() * 2)
    chunk = condition.videos[0][0]
    torch.testing.assert_close(chunk.embeddings[0, :, 0], torch.tensor([0., 1., 2., 3., 6., 7., 8., 9.]))
    torch.testing.assert_close(chunk.task_mask[0], torch.tensor([False] * 5 + [True] * 2 + [False]))
    assert chunk.padding.shape == (2, 8)
    assert not chunk.embeddings.requires_grad


def test_camera_contract_rejects_single_rgb_in_dual_observer_and_cache_mismatch():
    import pytest
    from ember.writer.runtime import FrozenVideoPrefixCache

    observer = object.__new__(NativeVideoObserver)
    observer.camera_view, observer.camera_names = "dual", ("agentview", "eye_in_hand")
    with pytest.raises(ValueError, match="declared teacher camera views"):
        observer.prefix(torch.zeros(1, 3, 8, 8), None, None, None)
    with pytest.raises(ValueError, match="camera views differ"):
        FrozenVideoPrefixCache(observer, SimpleNamespace(videos=SimpleNamespace(camera_view="agentview")), 100)
