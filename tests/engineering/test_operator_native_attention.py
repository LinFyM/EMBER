"""Installed PI05/Gemma kernel equivalence and teacher-only replay scoping."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from transformers.models.gemma import modeling_gemma
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks

from ember.operator_writer import native


def visibility_mask():
    # Actual PI05 prefix block, language padding and all-visible action suffix.
    padding = torch.ones(2, 19, dtype=torch.bool)
    padding[:, 8:12] = False
    attention = torch.zeros_like(padding)
    attention[:, 12] = True
    visible = make_att_2d_masks(padding, attention)
    mask = PI05Pytorch._prepare_attention_masks_4d(None, visible)
    assert torch.isfinite(mask).all() and torch.count_nonzero(mask) > 0
    assert visible[0, 0, 3] and not visible[0, 0, 12]
    assert visible[0, 12, 0] and visible[0, 12, 18]
    assert not visible[0, 9].any()  # Finite-mask all-padding query remains covered.
    return mask


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('groups', [1, 4])
def test_installed_eager_sdpa_forward_and_complete_qkv_vjp(dtype, groups):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(812)
    query = torch.randn(2, 8, 19, 16, generator=generator, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 8 // groups, 19, 16, generator=generator, dtype=dtype, requires_grad=True)
    value = torch.randn(2, 8 // groups, 19, 16, generator=generator, dtype=dtype, requires_grad=True)
    cotangent = torch.randn(2, 19, 8, 16, generator=generator, dtype=dtype)
    module = SimpleNamespace(num_key_value_groups=groups, training=False)
    mask, scale = visibility_mask(), 16**-.5
    eager, weights = modeling_gemma.eager_attention_forward(module, query, key, value, mask, scale)
    actual, discarded = native._teacher_sdpa(module, query, key, value, mask, scale)
    assert discarded is None and weights is not None
    expected_grads = torch.autograd.grad(eager, (query, key, value), grad_outputs=cotangent)
    actual_grads = torch.autograd.grad(actual, (query, key, value), grad_outputs=cotangent)
    tolerances = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=.035, atol=.012)
    torch.testing.assert_close(actual, eager, **tolerances)
    for gradient, expected in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        torch.testing.assert_close(gradient, expected, **tolerances)


def test_scope_restores_on_exception_and_other_thread_uses_original(monkeypatch):
    original = modeling_gemma.eager_attention_forward
    q = torch.randn(1, 2, 4, 3)
    args = (SimpleNamespace(num_key_value_groups=1, training=False), q, q, q, None, 3**-.5)
    expected, _ = original(*args)
    calls = []
    kernel = native._teacher_sdpa
    def observed(*a, **k):
        calls.append(1)
        return kernel(*a, **k)
    monkeypatch.setattr(native, '_teacher_sdpa', observed)
    with pytest.raises(RuntimeError, match='consumer failure'):
        with native._teacher_attention_kernel():
            assert modeling_gemma.eager_attention_forward is not original
            result, weights = modeling_gemma.eager_attention_forward(*args)
            assert weights is None and len(calls) == 1
            with ThreadPoolExecutor(max_workers=1) as pool:
                foreign, foreign_weights = pool.submit(modeling_gemma.eager_attention_forward, *args).result()
            assert foreign_weights is not None and len(calls) == 1
            torch.testing.assert_close(foreign, expected)
            raise RuntimeError('consumer failure')
    assert modeling_gemma.eager_attention_forward is original
    assert native._TEACHER_ATTENTION.get() is False
    _, weights = modeling_gemma.eager_attention_forward(*args)
    assert weights is not None


@pytest.mark.parametrize('early_stop', [False, True])
def test_outer_checkpoint_reenters_kernel_with_same_functional_beta_and_restores(early_stop):
    torch.manual_seed(38)
    original = modeling_gemma.eager_attention_forward
    mask = visibility_mask()
    count = []
    class FrameCall(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 16))
            self.b = nn.Parameter(torch.randn(16, 4))
        def forward(self, x, *, teacher=True):
            # Physical values differ from β installed for the complete frame call.
            hidden = x + torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
            q = hidden[:, None].expand(-1, 8, -1, -1)
            k, v = hidden[:, None].expand(-1, 2, -1, -1), hidden[:, None].expand(-1, 2, -1, -1)
            module = SimpleNamespace(num_key_value_groups=4, training=False)
            if teacher:
                with native._teacher_attention_kernel(mask):
                    count.append((self.a, self.b))
                    return modeling_gemma.eager_attention_forward(module, q, k, v, mask, .25)[0]
            return modeling_gemma.eager_attention_forward(module, q, k, v, mask, .25)[0]
    frame = FrameCall()
    x = torch.randn(2, 19, 16)
    beta = {'a': torch.randn_like(frame.a, requires_grad=True), 'b': torch.randn_like(frame.b, requires_grad=True)}
    def call(a, b):
        return torch.func.functional_call(frame, {'a': a, 'b': b}, (x,))
    with set_checkpoint_early_stop(early_stop):
        actual = checkpoint(call, beta['a'], beta['b'], use_reentrant=False, preserve_rng_state=False)
    assert modeling_gemma.eager_attention_forward is original
    cotangent = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, tuple(beta.values()), grad_outputs=cotangent)
    assert len(count) == 2 and all(a is beta['a'] and b is beta['b'] for a, b in count)
    assert modeling_gemma.eager_attention_forward is original
    eager = torch.func.functional_call(frame, beta, (x,), {'teacher': False})
    expected = torch.autograd.grad(eager, tuple(beta.values()), grad_outputs=cotangent)
    torch.testing.assert_close(actual, eager, rtol=3e-5, atol=1e-5)
    for gradient, wanted in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, wanted, rtol=3e-4, atol=3e-5)
        assert gradient.norm() > 0


def test_padding_queries_never_enter_sdpa_and_context_prepares_indices_once(monkeypatch):
    mask = visibility_mask()
    q, k, v = torch.randn(2, 8, 19, 16), torch.randn(2, 2, 19, 16), torch.randn(2, 2, 19, 16)
    module = SimpleNamespace(num_key_value_groups=4, training=False)
    prepare, sdpa = native._padding_query_rows, torch.nn.functional.scaled_dot_product_attention
    counts = []
    def counted(value):
        counts.append(1)
        return prepare(value)
    def visible_only(query, key, value, **kwargs):
        assert query.shape[-2] == 15 and key.shape[-2] == value.shape[-2] == 19
        assert not (kwargs['attn_mask'] < 0).all(-1).any()
        return sdpa(query, key, value, **kwargs)
    monkeypatch.setattr(native, '_padding_query_rows', counted)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', visible_only)
    original = modeling_gemma.eager_attention_forward
    with native._teacher_attention_kernel(mask):
        for _ in range(2):
            modeling_gemma.eager_attention_forward(module, q, k, v, mask, .25)
    assert len(counts) == 1 and modeling_gemma.eager_attention_forward is original
