from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ember.writer.backbone_memory import (
    BackboneMemoryError,
    Pi05BackboneMemoryEncoder,
    make_backbone_memory_mask,
)


class _Norm(torch.nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del cond
        return value, None


class _Mlp(torch.nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.up_proj.weight.copy_(torch.eye(width))
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up_proj(value) * self.scale


class _Attention(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.head_dim = 2
        self.num_key_value_groups = 8
        self.scaling = self.head_dim**-0.5
        self.q_proj = torch.nn.Linear(width, 16, bias=False)
        self.k_proj = torch.nn.Linear(width, 2, bias=False)
        self.v_proj = torch.nn.Linear(width, 2, bias=False)
        self.o_proj = torch.nn.Linear(16, width, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.input_layernorm = _Norm()
        self.self_attn = _Attention(width)
        self.post_attention_layernorm = _Norm()
        self.mlp = _Mlp(width, 0.01 * (index + 1))


class _Rotary(torch.nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        _position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(value), torch.zeros_like(value)


class _Backbone(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _Layer(width, index)
            for index in range(Pi05BackboneMemoryEncoder.LAYER_COUNT)
        )
        self.norm = _Norm()
        self.rotary_emb = _Rotary()


class _Bridge(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=_Backbone(width))
        )
        self.gemma_expert = SimpleNamespace(model=_Backbone(width))
        self.language_embedding = torch.nn.Embedding(128, width)
        self.image_calls = 0
        self.language_calls = 0

    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        self.image_calls += 1
        frame_value = images.mean(dim=(1, 2, 3), keepdim=False)
        return frame_value[:, None, None].expand(
            -1,
            Pi05BackboneMemoryEncoder.NATIVE_IMAGE_TOKENS,
            self.language_embedding.embedding_dim,
        )

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        self.language_calls += 1
        return self.language_embedding(tokens)

    def forward(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("backbone memory must own the single joint loop")


class _Core(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)
        self.paligemma_with_expert = _Bridge(width)
        self.action_in_proj = torch.nn.Linear(32, width, bias=False)

    @staticmethod
    def _prepare_attention_masks_4d(mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[:, None], 0.0, -2.0e9)

    def embed_suffix(
        self,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        suffix = self.action_in_proj(noise)
        padding = torch.ones(
            suffix.shape[:2],
            dtype=torch.bool,
            device=suffix.device,
        )
        markers = torch.zeros_like(padding)
        markers[:, 0] = True
        condition = timestep[:, None].expand(-1, suffix.shape[-1])
        return suffix, padding, markers, condition


class _Policy(torch.nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.model = _Core(width)
        self.requires_grad_(False)


def _encoder(
    policy: _Policy,
    *,
    microbatch: int = 8,
    activation_checkpointing: bool = False,
) -> Pi05BackboneMemoryEncoder:
    bridge = policy.model.paligemma_with_expert
    return Pi05BackboneMemoryEncoder(
        bridge=bridge,
        image_width=4,
        expert_width=4,
        max_frames_per_encoder_call=microbatch,
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=17,
        activation_checkpointing=activation_checkpointing,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(4 * 3 * 3 * 3, dtype=torch.uint8).reshape(4, 3, 3, 3)
    condition_ids = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    tokens = torch.tensor(
        [[1, 10, 11, 12, 0], [1, 20, 21, 22, 23]],
        dtype=torch.long,
    )
    language_mask = tokens.ne(0)
    task_span = torch.tensor(
        [
            [False, True, True, False, False],
            [False, True, True, True, False],
        ]
    )
    return frames, condition_ids, tokens, language_mask, task_span


def _pinned_reference_without_memory(
    policy: _Policy,
    encoder: Pi05BackboneMemoryEncoder,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    from lerobot.policies.pi05.modeling_pi05 import (
        compute_layer_complete,
        make_att_2d_masks,
    )
    from lerobot.policies.pi_gemma import layernorm_forward

    frames, condition_ids, tokens, language_mask, task_span = inputs
    core = policy.model
    bridge = core.paligemma_with_expert
    selected_tokens = tokens.index_select(0, condition_ids)
    selected_mask = language_mask.index_select(0, condition_ids)
    selected_span = task_span.index_select(0, condition_ids)
    with torch.no_grad():
        image = bridge.embed_image(encoder._prepare_images(frames))
        language = bridge.embed_language_tokens(selected_tokens)
    prefix = torch.cat((image, language), dim=1)
    prefix_padding = torch.cat(
        (torch.ones(image.shape[:2], dtype=torch.bool), selected_mask), dim=1
    )
    noise = encoder.fixed_suffix_noise[None].expand(frames.shape[0], -1, -1)
    timestep = torch.ones(frames.shape[0])
    suffix, suffix_padding, suffix_markers, condition = core.embed_suffix(
        noise, timestep
    )
    padding = torch.cat((prefix_padding, suffix_padding), dim=1)
    markers = torch.cat((torch.zeros_like(prefix_padding), suffix_markers), dim=1)
    attention_mask = core._prepare_attention_masks_4d(
        make_att_2d_masks(padding, markers)
    )
    position_ids = torch.cumsum(padding, dim=1) - 1
    language_model = bridge.paligemma.model.language_model
    expert_model = bridge.gemma_expert.model
    target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
    streams = [prefix.to(target_dtype), suffix.to(target_dtype)]
    for layers in zip(language_model.layers, expert_model.layers, strict=True):
        streams = compute_layer_complete(
            streams,
            attention_mask,
            position_ids,
            [None, condition],
            layers,
            language_model.rotary_emb,
        )
    prefix_hidden, _ = layernorm_forward(language_model.norm, streams[0], None)
    suffix_hidden, _ = layernorm_forward(expert_model.norm, streams[1], condition)
    task_hidden = encoder._pack_task_hidden(
        prefix_hidden[:, encoder.NATIVE_IMAGE_TOKENS :],
        selected_span,
        int(task_span.sum(dim=1).max()),
    )
    return suffix_hidden, task_hidden


def test_three_block_mask_visibility_and_padding() -> None:
    prefix_padding = torch.tensor([[True, True, False]])
    mask = make_backbone_memory_mask(
        prefix_padding,
        action_horizon=50,
        memory_tokens=8,
    )[0]
    action = 3
    memory = 53
    assert mask.shape == (61, 61)
    assert mask[0, :3].tolist() == [True, True, False]
    assert not mask[0, action:].any()
    assert mask[action, :memory].sum().item() == 52
    assert not mask[action, memory:].any()
    assert mask[memory, :].sum().item() == 60
    assert not mask[2].any()
    assert not mask[:, 2].any()


def test_joint_forward_shapes_layer_capture_and_shared_condition_language() -> None:
    torch.manual_seed(5)
    policy = _Policy()
    encoder = _encoder(policy)
    output = encoder(policy, *_inputs())
    assert output.layer_memory.shape == (4, 18, 8, 4)
    assert output.probe_hidden.shape == (4, 50, 4)
    assert output.task_hidden.shape == (4, 3, 4)
    assert output.valid_task_tokens.tolist() == [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, True],
    ]
    assert not torch.equal(output.layer_memory[:, 0], output.layer_memory[:, -1])
    bridge = policy.model.paligemma_with_expert
    assert bridge.image_calls == bridge.language_calls == 1
    assert not any(
        projection._forward_hooks
        for layer in bridge.gemma_expert.model.layers
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        )
    )


def test_action_probe_and_prefix_are_invariant_to_later_memory_values() -> None:
    torch.manual_seed(7)
    policy = _Policy()
    encoder = _encoder(policy)
    encoder.eval()
    with torch.no_grad():
        inputs = _inputs()
        first = encoder(policy, *inputs)
        reference_probe, reference_task = _pinned_reference_without_memory(
            policy, encoder, inputs
        )
        encoder.memory_tokens.add_(100.0)
        second = encoder(policy, *inputs)
    torch.testing.assert_close(first.probe_hidden, reference_probe, rtol=0, atol=1e-6)
    torch.testing.assert_close(first.task_hidden, reference_task, rtol=0, atol=1e-6)
    torch.testing.assert_close(first.probe_hidden, second.probe_hidden, rtol=0, atol=0)
    torch.testing.assert_close(first.task_hidden, second.task_hidden, rtol=0, atol=0)
    assert not torch.equal(first.layer_memory, second.layer_memory)


def test_gradients_reach_memory_and_action_meta_lora_only() -> None:
    torch.manual_seed(11)
    policy = _Policy()
    encoder = _encoder(policy, activation_checkpointing=True)
    output = encoder(policy, *_inputs())
    output.layer_memory.float().square().mean().backward()
    assert encoder.memory_tokens.grad is not None
    assert encoder.memory_tokens.grad.abs().sum() > 0
    assert any(
        adapter.b.grad is not None and adapter.b.grad.abs().sum() > 0
        for adapter in encoder.action_meta_lora.adapters.values()
    )
    assert all(parameter.grad is None for parameter in policy.parameters())
    names = tuple(name for name, _ in encoder.named_parameters())
    assert names[0] == "memory_tokens"
    assert all("text_meta" not in name and "vl_meta" not in name for name in names)


def test_rejects_video_ids_used_as_condition_ids() -> None:
    policy = _Policy()
    encoder = _encoder(policy)
    frames, _, tokens, language_mask, task_span = _inputs()
    video_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with pytest.raises(BackboneMemoryError, match="invalid frame-language"):
        encoder(policy, frames, video_ids, tokens, language_mask, task_span)
