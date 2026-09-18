"""Registered camera/horizon reads, Meta replay, and complete LoRA output."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.video_program import LearnedHorizonRead, Pi05LanguageAxialEncoder, VideoProgramError


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


class _TinyBackbone(nn.Module):
    """Small frozen projections exercise the actual temporary Meta hooks."""

    def __init__(self, width=8):
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in range(18))
        for layer in self.layers:
            layer.self_attn = nn.Module()
            for name in MetaLoRAStack.PROJECTIONS:
                setattr(layer.self_attn, name, nn.Linear(width, width, bias=False))

    def forward(self, value):
        for layer in self.layers:
            attn = layer.self_attn
            content = attn.q_proj(value) + attn.k_proj(value).mean(1, keepdim=True) + attn.v_proj(value)
            value = value + .1 * attn.o_proj(content.tanh())
        return value


class _TinyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.language_model = _TinyBackbone()
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = _TinyBackbone()
        self.embedding = nn.Embedding(40, 8)
        self.calls = []
        self.image_prefixes = []

    def embed_image(self, images):
        pixels = F.adaptive_avg_pool2d(images, (16, 16)).mean(1).flatten(1)
        channels = torch.linspace(.2, 1.0, 8)
        position = torch.arange(256).float()[None, :, None] / 256
        return pixels[..., None] * channels + position * channels.flip(0)

    def embed_language_tokens(self, tokens):
        return self.embedding(tokens)

    def forward(self, *, inputs_embeds, **kwargs):
        prefix, suffix = inputs_embeds
        language = self.paligemma.model.language_model
        expert = self.gemma_expert.model
        self.calls.append((suffix is not None,
                           bool(language.layers[0].self_attn.q_proj._forward_hooks),
                           bool(expert.layers[0].self_attn.q_proj._forward_hooks)))
        if suffix is not None:
            self.image_prefixes.append(prefix.detach().clone())
            assert kwargs['position_ids'].shape[1] == prefix.shape[1] + 50
        prefix = language(prefix)
        if suffix is not None:
            suffix = expert(suffix + prefix.mean(1, keepdim=True))
        return (prefix, suffix), None


class _TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)
        self.paligemma_with_expert = _TinyBridge()
        self.action_in_proj = nn.Linear(32, 8, bias=False)

    def embed_suffix(self, noise, timestep):
        value = self.action_in_proj(noise) + timestep[:, None, None]
        padding = torch.ones(noise.shape[:2], dtype=torch.bool)
        return value, padding, torch.zeros_like(padding), timestep

    def _prepare_attention_masks_4d(self, value):
        return value[:, None]


def _encoder(checkpoint=False, *, camera_view="dual", horizon_read="learned"):
    policy = nn.Module()
    policy.model = _TinyCore()
    policy.requires_grad_(False)
    bridge = policy.model.paligemma_with_expert
    encoder = Pi05LanguageAxialEncoder(
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        image_width=8, expert_width=8, program_width=8,
        text_meta_lora_rank=4, vl_meta_lora_rank=4, action_meta_lora_rank=4,
        patch_grounding_heads=2, max_frames_per_encoder_call=2,
        action_horizon=50, padded_action_dim=32, initialization_seed=7,
        activation_checkpointing=checkpoint,
        camera_view=camera_view, horizon_read=horizon_read,
    )
    frames = torch.empty(3, 2, 3, 8, 8, dtype=torch.uint8)
    for index in range(3):
        frames[index, 0] = 20 + index * 35
        frames[index, 1] = 240 - index * 20
    if camera_view == "agentview":
        frames = frames[:, 0]
    tokens = torch.tensor([[1, 3, 5, 7, 0], [1, 4, 6, 8, 9]])
    spans = torch.tensor([[False, True, True, False, False], [False, True, True, True, False]])
    args = (frames, torch.tensor([0, 0, 1]), tokens, tokens.ne(0), spans)
    return policy, encoder, args


@pytest.mark.parametrize('mode', ['fixed_mean', 'learned'])
def test_complete_horizon_read_starts_as_mean_and_preserves_its_registered_learning_mode(mode):
    read = LearnedHorizonRead(width=1024, horizon=50, mode=mode)
    hidden = torch.randn(2, 50, 1024, requires_grad=True)
    pooled = read(hidden)
    torch.testing.assert_close(pooled, hidden.mean(1))
    pooled.square().sum().backward()
    assert hidden.grad.abs().sum(-1).gt(0).all()
    if mode == 'learned':
        assert read.query.grad.norm() > 0 and read.bias.grad.abs().gt(0).all()
        with torch.no_grad():
            read.bias[-1] = 30
        torch.testing.assert_close(read(hidden), hidden[:, -1])
    else:
        assert all(not value.requires_grad and value.grad is None for value in read.parameters())
        torch.optim.AdamW(read.parameters(), lr=.1, weight_decay=.1).step()
        assert all(value.count_nonzero() == 0 for value in read.parameters())
        torch.testing.assert_close(read(hidden), hidden.mean(1))
    with pytest.raises(VideoProgramError, match='complete native horizon'):
        read(hidden[:, :49])


@pytest.mark.parametrize('camera_view,horizon_read,patches', [('agentview', 'fixed_mean', 256), ('dual', 'learned', 512)])
def test_declared_cameras_supply_only_their_real_patch_values_to_core(camera_view, horizon_read, patches):
    policy, encoder, args = _encoder(camera_view=camera_view, horizon_read=horizon_read)
    patch_inputs = []
    horizon_inputs = []
    encoder.patch_grounding.register_forward_pre_hook(lambda module, values: patch_inputs.append(values[1]))
    encoder.horizon_read.register_forward_pre_hook(lambda module, values: horizon_inputs.append(values[0]))
    text, evidence, interactions, valid = encoder(policy, *args)
    assert text.shape == (2, 3, 8) and evidence.shape == (3, 3, 8)
    assert interactions.shape == (3, 8) and valid.tolist() == [[True, True, False], [True] * 3]
    assert [value.shape[1] for value in patch_inputs] == [patches, patches]
    assert all(value.shape[1:] == (50, 8) for value in horizon_inputs)
    bridge = policy.model.paligemma_with_expert
    expected = bridge.embed_image(encoder._prepare_images(args[0][:2])).reshape(2, patches, 8)
    assert bridge.image_prefixes[0].shape[1] == patches + args[2].shape[1]
    torch.testing.assert_close(bridge.image_prefixes[0][:, :patches], expected)
    if camera_view == 'dual':
        assert not torch.allclose(expected[:, :256], expected[:, 256:])
    for camera in range(encoder.camera_count):
        changed = args[0].clone()
        if camera_view == 'dual':
            changed[:, camera] = 255 - changed[:, camera]
        else:
            changed = 255 - changed
        observed = encoder(policy, changed, *args[1:])
        assert not torch.allclose(observed[1], evidence)
    assert not ({id(value) for value in encoder.parameters()} & {id(value) for value in policy.parameters()})
    assert all(not value.requires_grad for value in policy.parameters())
    wrong_frames = args[0][:, 0] if camera_view == 'dual' else args[0][:, None].expand(-1, 2, -1, -1, -1)
    with pytest.raises(VideoProgramError, match='invalid frame-language'):
        encoder(policy, wrong_frames, *args[1:])


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('camera_view,horizon_read', [('agentview', 'fixed_mean'), ('dual', 'learned')])
def test_frame_replay_keeps_all_three_meta_stacks_in_gradient_path(checkpoint, camera_view, horizon_read):
    policy, encoder, args = _encoder(checkpoint, camera_view=camera_view, horizon_read=horizon_read)
    stacks = (encoder.text_meta_lora, encoder.vl_meta_lora, encoder.action_meta_lora)
    assert all(isinstance(stack, MetaLoRAStack) for stack in stacks)
    assert all(adapter.b.count_nonzero() == 0 for stack in stacks for adapter in stack.adapters.values())
    with torch.no_grad():
        for stack in stacks:
            for adapter in stack.adapters.values():
                adapter.b.normal_(std=.03)
    output = encoder(policy, *args)
    sum(value.square().mean() for value in output[:3]).backward()
    for stack in stacks:
        assert all(value.grad is not None and value.grad.norm() > 0 for value in stack.parameters())
    if horizon_read == 'learned':
        assert encoder.horizon_read.query.grad.norm() > 0
        assert encoder.horizon_read.bias.grad.norm() > 0
    else:
        assert encoder.horizon_read.query.grad is None and encoder.horizon_read.bias.grad is None
    assert encoder.language_projection.weight.grad.norm() > 0
    assert encoder.interaction_projection.weight.grad.norm() > 0
    bridge = policy.model.paligemma_with_expert
    assert len(bridge.calls) > 3 if checkpoint else len(bridge.calls) == 3
    assert all(text_hooks and (expert_hooks if is_video else not expert_hooks)
               for is_video, text_hooks, expert_hooks in bridge.calls)
    assert all(not module._forward_hooks for module in bridge.modules())
    assert all(value.grad is None for value in policy.parameters())


def test_camera_modes_share_fresh_parameters_and_consume_the_same_rng_stream():
    # Start after unrelated upstream construction, as runtime does after its LoRA template.
    torch.rand(19)
    initial_rng = torch.get_rng_state()
    policy_a, encoder_a, _ = _encoder(camera_view='agentview', horizon_read='fixed_mean')
    after_a = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    policy_b, encoder_b, _ = _encoder()
    assert torch.equal(torch.get_rng_state(), after_a)
    torch.testing.assert_close(policy_a.state_dict(), policy_b.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(encoder_a.state_dict(), encoder_b.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize('camera_view,horizon_read', [('dual', 'fixed_mean'), ('agentview', 'learned'),
                                                     ('eye_in_hand', 'fixed_mean'), ('dual', 'truncated')])
def test_unregistered_camera_read_pairs_are_rejected(camera_view, horizon_read):
    with pytest.raises(VideoProgramError, match='dimensions'):
        _encoder(camera_view=camera_view, horizon_read=horizon_read)


def test_fixed_mean_checkpoint_refuses_learned_nonuniform_weights():
    fixed = LearnedHorizonRead(width=8, horizon=50, mode='fixed_mean')
    fixed.load_state_dict(fixed.state_dict(), strict=True)
    for name in ('query', 'bias'):
        state = {key: value.clone() for key, value in fixed.state_dict().items()}
        state[name][-1] = .1
        with pytest.raises(VideoProgramError, match='fixed-mean horizon checkpoint'):
            fixed.load_state_dict(state, strict=True)


def test_checkpoint_and_physical_chunk_preserve_the_same_condition_read():
    policy, encoder, args = _encoder()
    encoder.eval()
    with torch.no_grad():
        expected = encoder(policy, *args)
    encoder.max_frames_per_encoder_call = 1
    encoder.activation_checkpointing = True
    encoder.train()
    observed = encoder(policy, *args)
    for left, right in zip(expected, observed, strict=True):
        torch.testing.assert_close(left, right)
    sum(value.square().mean() for value in observed[:3]).backward()
    assert encoder.action_meta_lora.adapters['17_o_proj'].b.grad.norm() > 0




def _parameter_gradients(module):
    return torch.cat([(value.grad if value.grad is not None else torch.zeros_like(value)).flatten()
                      for value in module.parameters()])


def _check_frame_parallel_case(rank, size, frame_count):
    torch.manual_seed(32)
    policy, parallel, args = _encoder(checkpoint=True, camera_view='agentview', horizon_read='fixed_mean')
    with torch.no_grad():
        for stack in (parallel.text_meta_lora, parallel.vl_meta_lora, parallel.action_meta_lora):
            for adapter in stack.adapters.values():
                adapter.b.normal_(std=.03)
    serial = deepcopy(parallel)
    frames = args[0][:1].expand(frame_count, -1, -1, -1).clone()
    for index in range(frame_count):
        frames[index] = 15 + index * 31
    ids = torch.tensor([0, 0, 0, 1, 1] if frame_count == 5 else [0, 1])
    args = (frames, ids, *args[2:])
    expected = serial(policy, *args)
    generator = torch.Generator().manual_seed(913)
    cotangents = [torch.randn((size, *value.shape), generator=generator) for value in expected[:3]]
    torch.autograd.backward(expected[:3], [value.sum(0) for value in cotangents])
    expected_gradient = _parameter_gradients(serial)

    image_batches = []
    bridge = policy.model.paligemma_with_expert
    embed_image = bridge.embed_image

    def count_images(images):
        image_batches.append(images.shape[0])
        return embed_image(images)

    bridge.embed_image = count_images
    observed = parallel(policy, *args, frame_parallel_group=dist.group.WORLD)
    torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-6)
    local_frames = frame_count // size + int(rank < frame_count % size)
    assert sum(image_batches) == local_frames
    torch.autograd.backward(observed[:3], [value[rank] for value in cotangents])
    observed_gradient = _parameter_gradients(parallel)
    dist.all_reduce(observed_gradient, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(observed_gradient, expected_gradient, rtol=3e-4, atol=3e-5)
    # Checkpoint replay reads only this owner's real frames; an empty owner has
    # neither initial nor replay native calls, while contributing text gradients.
    assert sum(image_batches) == 2 * local_frames
    assert all(value.grad is None for value in policy.parameters())
    assert all(not module._forward_hooks for module in bridge.modules())


def _frame_parallel_worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=size, timeout=timedelta(seconds=90))
    try:
        for count in (5, 2):
            _check_frame_parallel_case(rank, size, count)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is required')
def test_three_rank_frame_shards_match_serial_outputs_and_aggregate_gradients(tmp_path):
    mp.spawn(_frame_parallel_worker, args=(3, str(tmp_path / 'native-gloo')), nprocs=3, join=True)
