"""Unified native Z/H reads, exact input roles, and differentiable Meta replay."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GemmaConfig

from lerobot.policies.pi05.modeling_pi05 import compute_layer_complete, make_att_2d_masks
from lerobot.policies.pi_gemma import PiGemmaModel, layernorm_forward
from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.video_program import Pi05UnifiedVideoEncoder, VideoProgramError


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def _backbone(width=16, adaptive=False):
    config = GemmaConfig(
        hidden_size=width, intermediate_size=2 * width, num_hidden_layers=18,
        num_attention_heads=8, num_key_value_heads=1, head_dim=2, vocab_size=40,
        max_position_embeddings=1024, attention_dropout=0.0,
    )
    config.use_adarms = adaptive
    if adaptive:
        config.adarms_cond_dim = width
    config._attn_implementation = 'eager'
    return PiGemmaModel(config)


class _TinyBridge(nn.Module):
    """Actual PI Gemma layers with small widths and real two-camera patch content."""

    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.language_model = _backbone()
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = _backbone(8, adaptive=True)
        self.text_calls = 0
        self.text_inputs = []

    def embed_image(self, images):
        pixels = F.adaptive_avg_pool2d(images, (16, 16)).mean(1).flatten(1)
        channels = torch.linspace(.2, 1.0, 16)
        position = torch.arange(256).float()[None, :, None] / 256
        return pixels[..., None] * channels + position * channels.flip(0)

    def embed_language_tokens(self, tokens):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(self, *, inputs_embeds, attention_mask, position_ids, adarms_cond,
                past_key_values=None, use_cache=False):
        assert past_key_values is None and not use_cache
        prefix, suffix = inputs_embeds
        language = self.paligemma.model.language_model
        expert = self.gemma_expert.model
        if suffix is None:
            self.text_calls += 1
            self.text_inputs.append(prefix.detach().clone())
            output = language(inputs_embeds=prefix, attention_mask=attention_mask,
                              position_ids=position_ids, use_cache=False)
            return (output.last_hidden_state, None), None
        hidden = [prefix, suffix]
        for pair in zip(language.layers, expert.layers, strict=True):
            hidden = compute_layer_complete(hidden, attention_mask, position_ids, adarms_cond,
                                            layers=pair, rotary_emb=language.rotary_emb)
        hidden = [layernorm_forward(model.norm, value, condition)[0]
                  for model, value, condition in zip((language, expert), hidden, adarms_cond, strict=True)]
        return hidden, None


class _TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)
        self.paligemma_with_expert = _TinyBridge()
        self.action_in_proj = nn.Linear(32, 8, bias=False)
        self.time_projection = nn.Linear(1, 8)
        self.suffix_calls = 0

    def embed_suffix(self, noise, timestep):
        self.suffix_calls += 1
        value = self.action_in_proj(noise)
        padding = torch.ones(noise.shape[:2], dtype=torch.bool)
        attention = torch.zeros_like(padding)
        attention[:, 0] = True
        return value, padding, attention, self.time_projection(timestep[:, None])

    def _prepare_attention_masks_4d(self, value):
        return torch.where(value[:, None], 0.0, torch.finfo(torch.float32).min)


def _encoder(checkpoint=False):
    policy = nn.Module()
    policy.model = _TinyCore()
    policy.requires_grad_(False)
    bridge = policy.model.paligemma_with_expert
    encoder = Pi05UnifiedVideoEncoder(
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        image_width=16, expert_width=8, program_width=8,
        text_meta_lora_rank=4, vl_meta_lora_rank=4, action_meta_lora_rank=4,
        patch_grounding_heads=2, max_frames_per_encoder_call=2,
        action_horizon=50, padded_action_dim=32, initialization_seed=7,
        activation_checkpointing=checkpoint, native_split_layer=9,
        joint_heads=2, joint_blocks=2,
    )
    frames = torch.empty(3, 2, 3, 8, 8, dtype=torch.uint8)
    for index in range(3):
        frames[index, 0] = 20 + index * 35
        frames[index, 1] = 240 - index * 20
    tokens = torch.tensor([[1, 3, 5, 7, 0], [1, 4, 6, 8, 9]])
    spans = torch.tensor([[False, True, True, False, False],
                          [False, True, True, True, False]])
    args = (frames, torch.tensor([0, 5, 0]), torch.tensor([0, 2, 3]), tokens, tokens.ne(0), spans)
    return policy, encoder, args


def _single_video(args, row=0):
    frames, indices, offsets, tokens, mask, spans = args
    first, last = int(offsets[row]), int(offsets[row + 1])
    return (frames[first:last], indices[first:last], torch.tensor([0, last - first]),
            tokens[row:row + 1], mask[row:row + 1], spans[row:row + 1])


def _open_meta_and_writeback(encoder):
    with torch.no_grad():
        for stack in (encoder.text_meta_lora, encoder.vl_meta_lora, encoder.action_meta_lora):
            for adapter in stack.adapters.values():
                adapter.b.normal_(std=.03)
        encoder.prefix_writeback.weight.normal_(std=.02)
        encoder.horizon_writeback.weight.normal_(std=.02)


def _assert_no_native_hooks(policy):
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in policy.modules())


def test_reading_meta_all_layers_receive_gradients_and_remove_scoped_hooks():
    expert = _backbone().requires_grad_(False)
    meta = MetaLoRAStack(expert.layers, rank=4)
    with meta.installed(expert):
        response = expert(inputs_embeds=torch.randn(2, 50, 16), use_cache=False).last_hidden_state
    (response * torch.randn_like(response)).sum().backward()
    assert all(parameter.grad is None for parameter in expert.parameters())
    assert all(adapter.b.grad is not None and bool(adapter.b.grad.abs().sum() > 0)
               for adapter in meta.adapters.values())
    _assert_no_native_hooks(expert)


def test_zero_writeback_preserves_native_read_with_actual_masks_positions_and_final_norms():
    policy, encoder, args = _encoder()
    core = policy.model
    bridge = core.paligemma_with_expert
    language = bridge.paligemma.model.language_model
    expert = bridge.gemma_expert.model
    frames, _, _, tokens, mask, spans = _single_video(args, row=1)
    suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
        encoder.fixed_suffix_noise[None], torch.ones(1)
    )
    prefix = torch.cat((bridge.embed_image(encoder._prepare_images(frames)).reshape(1, 512, 16),
                        bridge.embed_language_tokens(tokens)), dim=1)
    padding = torch.cat((torch.ones(1, 512, dtype=torch.bool), mask, suffix_padding), dim=1)
    attention = torch.cat((torch.zeros(1, 512 + tokens.shape[1], dtype=torch.bool), suffix_attention), dim=1)
    attention_mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
    positions = padding.long().cumsum(1) - 1
    with encoder.vl_meta_lora.installed(language), encoder.action_meta_lora.installed(expert):
        expected, _ = bridge.forward(inputs_embeds=[prefix, suffix], attention_mask=attention_mask,
                                     position_ids=positions, adarms_cond=[None, adarms])
    norm_calls = []
    handles = [module.register_forward_hook(lambda _m, _i, _o: norm_calls.append(True))
               for module in (language.norm, expert.norm)]
    try:
        middle_z, middle_h = encoder._lower_native(core, encoder._embed_prefix(core, frames, tokens), mask, suffix,
                                                  suffix_padding, suffix_attention, adarms)
        assert not norm_calls
        written = encoder._writeback(middle_z, middle_h, torch.randn(1, 53, 8), spans, 3)
        torch.testing.assert_close(written, (middle_z, middle_h))
        observed = encoder._native_layers(core, *written, mask, suffix_padding,
                                          suffix_attention, adarms, 9, 18, final_norm=True)
        assert len(norm_calls) == 2
        torch.testing.assert_close(observed, tuple(expected), rtol=2e-5, atol=2e-6)
    finally:
        for handle in handles:
            handle.remove()
    _assert_no_native_hooks(policy)


def test_writeback_updates_only_task_span_and_every_horizon_slot():
    _, encoder, args = _encoder()
    spans = args[-1]
    prefix = torch.randn(2, 517, 16)
    suffix = torch.randn(2, 50, 8)
    middle = torch.randn(2, 53, 8, requires_grad=True)
    _open_meta_and_writeback(encoder)
    changed_z, changed_h = encoder._writeback(prefix, suffix, middle, spans, 3)
    untouched = torch.cat((torch.ones(2, 512, dtype=torch.bool), ~spans), dim=1)
    torch.testing.assert_close(changed_z[untouched], prefix[untouched])
    assert (changed_z[:, 512:][spans] - prefix[:, 512:][spans]).abs().sum(-1).gt(0).all()
    assert (changed_h - suffix).abs().sum(-1).gt(0).all()
    (changed_z.square().sum() + changed_h.square().sum()).backward()
    assert middle.grad[:, 3:].abs().sum(-1).gt(0).all()
    assert middle.grad[0, 2].count_nonzero() == 0


def test_one_exact_text_read_exposes_matching_middle_and_final_queries():
    policy, encoder, args = _encoder()
    core = policy.model
    bridge = core.paligemma_with_expert
    tokens, _, spans = args[3:]
    middle, final = encoder._encode_text(core, tokens, spans, 3)
    assert bridge.text_calls == 1
    expected_tokens = torch.tensor([[1, 3, 5, 0], [1, 4, 6, 8]])
    torch.testing.assert_close(bridge.text_inputs[0], bridge.embed_language_tokens(expected_tokens))
    assert middle.shape == final.shape == (2, 3, 16)
    assert middle[0, 2].count_nonzero() == final[0, 2].count_nonzero() == 0
    assert not torch.allclose(middle[:, :2], final[:, :2])
    _assert_no_native_hooks(policy)


def test_real_dual_cameras_all_horizon_positions_and_raw_frame_indices_reach_joint_stacks():
    policy, encoder, args = _encoder()
    recorded_positions = []
    handles = [stack.register_forward_pre_hook(lambda _m, inputs: recorded_positions.append(inputs[1].clone()))
               for stack in (encoder.middle_stack, encoder.final_stack)]
    try:
        memory, valid_frames, valid_roles, addresses, semantic = encoder(policy, *args)
    finally:
        for handle in handles:
            handle.remove()
    assert memory.shape == (2, 2, 53, 8) and semantic == 3
    assert valid_frames.tolist() == [[True, True], [True, False]]
    assert valid_roles[0, :3].tolist() == [True, True, False]
    assert bool(valid_roles[:, 3:].all()) and addresses.shape == (53, 8)
    assert memory[1, 1].count_nonzero() == memory[0, :, 2].count_nonzero() == 0
    assert [value.tolist() for value in recorded_positions] == [[[0, 5], [0, 0]]] * 2
    assert policy.model.suffix_calls == 1 and policy.model.paligemma_with_expert.text_calls == 1
    for camera in range(2):
        changed = args[0].clone()
        changed[:, camera] = 255 - changed[:, camera]
        observed = encoder(policy, changed, *args[1:])[0]
        assert not torch.allclose(observed, memory)
    assert not ({id(value) for value in encoder.parameters()} & {id(value) for value in policy.parameters()})
    assert all(not value.requires_grad for value in policy.parameters())
    with pytest.raises(VideoProgramError, match='video-language'):
        encoder(policy, args[0][:, 0], *args[1:])


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_replay_preserves_gradients_through_all_three_meta_stacks(checkpoint, dtype):
    policy, encoder, args = _encoder(checkpoint)
    policy.to(dtype)
    _open_meta_and_writeback(encoder)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        memory = encoder(policy, *args)[0]
    assert bool(memory.isfinite().all())
    (memory * torch.randn_like(memory)).sum().backward()
    for stack in (encoder.text_meta_lora, encoder.vl_meta_lora, encoder.action_meta_lora):
        assert all(value.grad is not None and value.grad.norm() > 0 for value in stack.parameters())
    for module in (encoder.middle_read, encoder.final_read, encoder.middle_stack, encoder.final_stack):
        assert all(value.grad is not None and value.grad.norm() > 0 for value in module.parameters())
    assert encoder.prefix_writeback.weight.grad.norm() > 0
    assert encoder.horizon_writeback.weight.grad.norm() > 0
    assert all(value.grad is None for value in policy.parameters())
    _assert_no_native_hooks(policy)


def test_middle_stack_gets_functional_credit_after_zero_writeback_opens():
    policy, encoder, args = _encoder()
    memory = encoder(policy, *args)[0]
    memory.square().sum().backward()
    assert encoder.prefix_writeback.weight.grad.norm() > 0
    assert encoder.horizon_writeback.weight.grad.norm() > 0
    assert all(value.grad is None or value.grad.count_nonzero() == 0
               for value in encoder.middle_stack.parameters())
    encoder.zero_grad(set_to_none=True)
    with torch.no_grad():
        encoder.prefix_writeback.weight.add_(.01 * torch.randn_like(encoder.prefix_writeback.weight))
        encoder.horizon_writeback.weight.add_(.01 * torch.randn_like(encoder.horizon_writeback.weight))
    encoder(policy, *args)[0].square().sum().backward()
    assert all(value.grad is not None and value.grad.norm() > 0 for value in encoder.middle_stack.parameters())


def test_physical_chunks_and_padded_condition_batch_preserve_the_same_video_memory():
    policy, encoder, args = _encoder()
    _open_meta_and_writeback(encoder)
    with torch.no_grad():
        expected = encoder(policy, *args)[0]
        encoder.max_frames_per_encoder_call = 1
        observed = encoder(policy, *args)[0]
        torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
        for row in range(2):
            single, _, _, _, semantic = encoder(policy, *_single_video(args, row))
            frame_count = int(args[2][row + 1] - args[2][row])
            packed = torch.cat((expected[row:row + 1, :frame_count, :semantic],
                                expected[row:row + 1, :frame_count, 3:]), dim=2)
            torch.testing.assert_close(single, packed, rtol=3e-5, atol=3e-6)


def test_identical_real_frames_remain_static_through_native_writeback_and_both_stacks():
    policy, encoder, args = _encoder()
    _open_meta_and_writeback(encoder)
    frames, _, _, tokens, mask, spans = _single_video(args, 1)
    with torch.no_grad():
        one = encoder(policy, frames, torch.tensor([0]), torch.tensor([0, 1]), tokens, mask, spans)[0]
        repeated = encoder(policy, frames.expand(3, -1, -1, -1, -1), torch.tensor([0, 5, 13]),
                           torch.tensor([0, 3]), tokens, mask, spans)[0]
    torch.testing.assert_close(repeated, one.expand_as(repeated), rtol=3e-5, atol=3e-6)


def test_meta_hooks_are_removed_when_native_checkpoint_replay_raises(monkeypatch):
    policy, encoder, args = _encoder(checkpoint=True)
    memory = encoder(policy, *_single_video(args, 1))[0]
    from lerobot.policies.pi05 import modeling_pi05

    def failure(*_args, **_kwargs):
        raise RuntimeError('native replay stopped')

    monkeypatch.setattr(modeling_pi05, 'compute_layer_complete', failure)
    with pytest.raises(RuntimeError, match='native replay stopped'):
        memory.square().sum().backward()
    _assert_no_native_hooks(policy)


def test_order_and_condition_boundaries_are_validated():
    policy, encoder, args = _encoder()
    invalid = list(args)
    invalid[1] = torch.tensor([5, 0, 0])
    with pytest.raises(VideoProgramError, match='natural order'):
        encoder(policy, *invalid)
    invalid = list(args)
    invalid[2] = torch.tensor([0, 0, 3])
    with pytest.raises(VideoProgramError, match='video-language'):
        encoder(policy, *invalid)
