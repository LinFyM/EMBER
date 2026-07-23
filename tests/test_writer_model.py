from __future__ import annotations

import torch

from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


def _template() -> dict[str, torch.Tensor]:
    rank = 16
    state: dict[str, torch.Tensor] = {}
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, input_width, output_width in (
            ("q_proj", 1024, 2048),
            ("v_proj", 1024, 256),
        ):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                rank, input_width
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width, rank
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(rank, input_width)
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, rank)
    return state


class _FakeActionMemory(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        *,
        frame_microbatch: int,
    ) -> torch.Tensor:
        assert frame_microbatch == 4
        image = frames.to(torch.float32).mean(dim=(1, 2, 3))
        language = language_tokens.to(torch.float32).mean(dim=1)
        value = image + language.index_select(0, condition_ids)
        layer = torch.arange(18, dtype=torch.float32) / 18
        slot = torch.arange(16, dtype=torch.float32) / 16
        return (
            value[:, None, None, None]
            + layer[None, :, None, None]
            + slot[None, None, :, None]
        ).expand(-1, 18, 16, 1024)


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    torch.manual_seed(3)
    template = _template()
    action_in = torch.nn.Linear(32, 1024)
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        action_in_projection=action_in,
        expert_layers=18,
        memory_slots=16,
        expert_width=1024,
        action_code_width=32,
        meta_lora_rank=2,
        hidden_dim=32,
        attention_heads=4,
        temporal_blocks=1,
        decoder_hidden_dim=16,
        frame_microbatch=4,
        conditional_linear_bias=True,
    )
    model.action_memory = _FakeActionMemory()
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(5, 3, 4, 4)
    frame_indices = torch.tensor([0, 4, 0, 4, 8], dtype=torch.long)
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    tokens = torch.tensor([[1, 2, 0], [4, 5, 6]], dtype=torch.long)
    masks = tokens.ne(0)
    return frames, frame_indices, offsets, tokens, masks


def test_action_memory_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_action_memory_writer_restores_only_internal_conditional_biases() -> None:
    model, _ = _model()
    encoder = model.task_encoder
    assert encoder.input_projection.bias is not None
    assert encoder.time_modulation.bias is not None
    for block in (*encoder.temporal, encoder.layer_mixer, encoder.slot_mixer):
        assert block.attention.in_proj_bias is not None
        assert block.attention.out_proj.bias is not None
        assert block.ffn[0].bias is not None
        assert block.ffn[2].bias is not None
    for head in model.factor_heads.values():
        assert head.network[1].bias is not None
        assert head.network[-1].bias is not None
        assert torch.count_nonzero(head.network[-1].bias) == 0
    assert not hasattr(model, "shared_lora")


def test_action_memory_writer_accepts_variable_video_batch() -> None:
    model, _ = _model()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert all(value.shape[0] == 2 for value in output.values())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())


def test_action_memory_initialization_is_deterministic_in_action_input_manifold() -> None:
    torch.manual_seed(11)
    action_in = torch.nn.Linear(32, 1024)
    template = _template()
    kwargs = dict(
        tensor_specs=build_lora_tensor_specs(template),
        template_state=template,
        action_in_projection=action_in,
        expert_layers=18,
        memory_slots=16,
        expert_width=1024,
        action_code_width=32,
        meta_lora_rank=2,
        hidden_dim=32,
        attention_heads=4,
        temporal_blocks=1,
        decoder_hidden_dim=16,
        frame_microbatch=4,
        conditional_linear_bias=True,
    )
    left = CompleteLoRAWriter(**kwargs)
    right = CompleteLoRAWriter(**kwargs)
    assert torch.equal(
        left.action_memory.memory_tokens,
        right.action_memory.memory_tokens,
    )
    assert left.action_memory.memory_tokens.shape == (16, 1024)
    assert (
        float(torch.sigmoid(left.action_memory.memory_timestep_logit).detach())
        == 0.5
    )
