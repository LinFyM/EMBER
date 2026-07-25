from __future__ import annotations

import torch

from ember.writer.action_forecast import VisualStateTokenDecoder
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.temporal import (
    LoRAQueryDecoder,
    PlanRevisionEncoder,
    VariableTimeTemporalEncoder,
)


class _Projection(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.in_features = input_width
        self.out_features = output_width


class _Layer(torch.nn.Module):
    def __init__(self, dimensions: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        for name, (input_width, output_width) in dimensions.items():
            setattr(
                self.self_attn,
                name,
                _Projection(input_width, output_width),
            )


class _Backbone(torch.nn.Module):
    def __init__(self, dimensions: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _Layer(dimensions) for _ in range(18)
        )


def _backbones() -> tuple[_Backbone, _Backbone]:
    return (
        _Backbone(
            {
                "q_proj": (2048, 2048),
                "k_proj": (2048, 256),
                "v_proj": (2048, 256),
                "o_proj": (2048, 2048),
            }
        ),
        _Backbone(
            {
                "q_proj": (1024, 2048),
                "k_proj": (1024, 256),
                "v_proj": (1024, 256),
                "o_proj": (2048, 1024),
            }
        ),
    )


def _template() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, output_width in (("q_proj", 2048), ("v_proj", 256)):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                16,
                1024,
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width,
                16,
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(16, input_width)
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, 16)
    return state


class _FakeForecast(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _state_positions: torch.Tensor,
        flow_noise: torch.Tensor,
    ) -> torch.Tensor:
        image = frames.to(torch.float32).mean(dim=(1, 2, 3))
        language = language_tokens.to(torch.float32).mean(dim=1)
        value = image + language.index_select(0, condition_ids)
        value = value + flow_noise[:, 0, 0].index_select(0, condition_ids)
        return value[:, None, None].expand(-1, 50, 7)


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    torch.manual_seed(3)
    template = _template()
    pali, expert = _backbones()
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=pali,
        expert_model=expert,
        image_width=2048,
        state_width=128,
        state_slots=28,
        state_heads=4,
        state_blocks=2,
        vl_meta_lora_rank=4,
        action_meta_lora_rank=8,
        frame_microbatch_size=1,
        num_flow_steps=10,
        action_horizon=50,
        padded_action_dim=32,
        output_action_dim=7,
        maximum_revision_count=10,
        temporal_width=256,
        temporal_heads=8,
        temporal_blocks=2,
        query_decoder_blocks=2,
        factor_hidden_width=256,
        initialization_seed=7,
        activation_checkpointing=True,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_125_376
    model.action_forecast = _FakeForecast()
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(5, 3, 4, 4)
    frame_indices = torch.tensor([0, 5, 0, 5, 10], dtype=torch.long)
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    tokens = torch.tensor([[1, 2, 0], [4, 5, 6]], dtype=torch.long)
    masks = tokens.ne(0)
    state_positions = torch.zeros(2, 28, dtype=torch.long)
    noise = torch.stack((torch.zeros(50, 32), torch.ones(50, 32)))
    return frames, frame_indices, offsets, tokens, masks, state_positions, noise


def test_action_forecast_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_action_forecast_writer_is_conditioned_after_factor_heads_move() -> None:
    model, _ = _model()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_virtual_state_slots_have_no_routing_only_output_bypass() -> None:
    decoder = VisualStateTokenDecoder(
        image_width=16,
        state_width=8,
        state_slots=4,
        heads=2,
        blocks=2,
        initialization_seed=7,
    )
    empty = decoder(torch.zeros(2, 6, 16))
    assert empty.shape == (2, 4, 16)
    assert torch.count_nonzero(empty) == 0

    image = torch.randn(2, 6, 16, requires_grad=True)
    conditioned = decoder(image)
    assert float(conditioned.detach().abs().sum()) > 0
    conditioned.square().mean().backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0


def test_lora_query_routing_cannot_reach_output_without_memory_content() -> None:
    decoder = LoRAQueryDecoder(
        width=32,
        heads=4,
        blocks=2,
        initialization_seed=7,
    )
    valid = torch.ones(2, 5, dtype=torch.bool)
    empty = decoder(torch.zeros(2, 5, 32), valid)
    assert all(torch.count_nonzero(value) == 0 for value in empty)

    conditioned = decoder(torch.randn(2, 5, 32), valid)
    assert all(float(value.detach().abs().sum()) > 0 for value in conditioned)


def test_plan_revision_temporal_and_queries_are_variable_time_and_differentiable() -> None:
    torch.manual_seed(17)
    plans = torch.randn(2, 3, 50, 7, requires_grad=True)
    indices = torch.tensor([[0, 5, 10], [0, 10, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    plan_revision = PlanRevisionEncoder(
        action_width=7,
        horizon=50,
        width=256,
        heads=8,
        maximum_revision_count=10,
    )
    tokens, positions, valid = plan_revision(plans, indices, mask)
    temporal = VariableTimeTemporalEncoder(width=256, heads=8, blocks=2)
    memory = temporal(tokens, positions, valid)
    decoder = LoRAQueryDecoder(
        width=256,
        heads=8,
        blocks=2,
        initialization_seed=7,
    )
    expert, action_in, action_out = decoder(memory, valid)
    assert tokens.shape == (2, 120, 256)
    assert expert.shape == (2, 18, 16, 256)
    assert action_in.shape == action_out.shape == (2, 16, 256)
    sum(value.square().mean() for value in (expert, action_in, action_out)).backward()
    assert plans.grad is not None and bool(torch.isfinite(plans.grad).all())
    assert float(plans.grad.abs().sum()) > 0
    assert torch.count_nonzero(plan_revision.stability_gate[-1].weight) == 0
    assert torch.count_nonzero(plan_revision.stability_gate[-1].bias) == 0


def test_reversing_frame_content_changes_absolute_time_memory() -> None:
    torch.manual_seed(23)
    encoder = PlanRevisionEncoder(
        action_width=7,
        horizon=50,
        width=256,
        heads=8,
        maximum_revision_count=10,
    )
    plans = torch.randn(1, 4, 50, 7)
    indices = torch.tensor([[0, 5, 10, 15]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    forward = encoder(plans, indices, mask)[0]
    reversed_content = encoder(plans.flip(1), indices, mask)[0]
    assert not torch.allclose(forward, reversed_content)
