from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from ember.pi05_processing import PI05_DIGIT_TOKEN_IDS
from ember.writer.action_forecast import Pi05ActionForecastEncoder
from ember.writer.visual_state import (
    AnchoredVisualState,
)
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.temporal import (
    ForecastBeliefEncoder,
    LoRAQueryDecoder,
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
        self._input_embeddings = _DigitEmbedding(2048)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self._input_embeddings


class _DigitEmbedding(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(41)
        self.register_buffer(
            "table",
            torch.randn(10, width, generator=generator),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        rows = torch.full_like(token_ids, -1)
        for row, token_id in enumerate(PI05_DIGIT_TOKEN_IDS):
            rows = torch.where(token_ids == token_id, row, rows)
        if bool((rows < 0).any()) or bool((rows >= 10).any()):
            raise RuntimeError("test digit token changed")
        return self.table.index_select(0, rows)


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
        state_slots=32,
        state_coordinates=8,
        state_heads=4,
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
        factor_hidden_width=411,
        initialization_seed=7,
        activation_checkpointing=True,
    )
    assert all(
        parameter.requires_grad
        for parameter in model.action_forecast.visual_state.parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_299_072
    assert any(
        parameter.requires_grad
        for parameter in model.action_forecast.vl_meta_lora.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in model.action_forecast.action_meta_lora.parameters()
    )
    model.action_forecast = _FakeForecast()
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(5, 3, 4, 4)
    frame_indices = torch.tensor([0, 5, 0, 5, 10], dtype=torch.long)
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    tokens = torch.tensor([[1, 2, 0], [4, 5, 6]], dtype=torch.long)
    masks = tokens.ne(0)
    state_positions = torch.zeros(2, 32, dtype=torch.long)
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


def test_anchored_visual_state_is_native_at_init_and_change_is_content_only() -> None:
    state = AnchoredVisualState(
        image_width=16,
        state_width=8,
        heads=2,
        digit_basis=torch.randn(10, 16),
        initialization_seed=7,
    )
    anchor = torch.randn(2, 6, 16, requires_grad=True)
    current = torch.randn(2, 6, 16, requires_grad=True)
    initialized = state(current, anchor, anchor)
    assert initialized.shape == (2, 32, 16)
    assert torch.count_nonzero(initialized) == 0

    with torch.no_grad():
        state.change_reader.output_gate.fill_(1.0)
    key = torch.randn(2, 6, 8)
    signed = torch.randn(2, 6, 8)
    forward_change = state.change_reader(key, signed)
    backward_change = state.change_reader(key, -signed)
    assert torch.allclose(
        forward_change,
        -backward_change,
        atol=1e-6,
        rtol=1e-5,
    )
    unchanged = state(anchor, anchor, anchor)
    assert torch.count_nonzero(unchanged) == 0
    changed = state(current, anchor, anchor)
    assert float(changed.detach().abs().sum()) > 0
    grouped = changed.reshape(2, 8, 4, 16)
    assert torch.count_nonzero(grouped[:, :, 0]) == 0
    assert torch.count_nonzero(grouped[:, :, 1:]) > 0
    changed.square().mean().backward()
    assert current.grad is not None and float(current.grad.abs().sum()) > 0
    assert anchor.grad is not None and float(anchor.grad.abs().sum()) > 0


def test_action_forecast_uses_fixed_size_padded_microbatches() -> None:
    pali, expert = _backbones()
    encoder = Pi05ActionForecastEncoder(
        paligemma_model=pali,
        expert_model=expert,
        image_width=2048,
        state_width=128,
        state_slots=32,
        state_coordinates=8,
        state_heads=4,
        vl_meta_lora_rank=4,
        action_meta_lora_rank=8,
        frame_microbatch_size=4,
        num_flow_steps=10,
        action_horizon=50,
        padded_action_dim=32,
        output_action_dim=7,
        initialization_seed=7,
        activation_checkpointing=False,
    )
    observed: list[tuple[list[int], list[int], list[int]]] = []

    def fake_forecast(
        _self: Pi05ActionForecastEncoder,
        _core: torch.nn.Module,
        frames: torch.Tensor,
        current_map: torch.Tensor,
        anchor_map: torch.Tensor,
        previous_map: torch.Tensor,
        _language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _state_positions: torch.Tensor,
        _noise: torch.Tensor,
    ) -> torch.Tensor:
        current = frames.index_select(0, current_map)
        anchor = frames.index_select(0, anchor_map)
        previous = frames.index_select(0, previous_map)
        observed.append(
            (
                current[:, 0, 0, 0].tolist(),
                anchor[:, 0, 0, 0].tolist(),
                previous[:, 0, 0, 0].tolist(),
            )
        )
        return current[:, :1, :1, :1].to(torch.float32).reshape(-1, 1, 1).expand(
            -1,
            50,
            7,
        )

    encoder._forecast_microbatch = MethodType(fake_forecast, encoder)
    frames = torch.arange(6, dtype=torch.uint8).reshape(6, 1, 1, 1).expand(
        -1,
        3,
        1,
        1,
    )
    plans = encoder(
        SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(chunk_size=50, max_action_dim=32)
            )
        ),
        frames,
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 32, dtype=torch.long),
        torch.zeros(2, 50, 32),
    )
    assert observed == [
        ([0, 1, 2, 3], [0, 0, 0, 3], [0, 0, 1, 3]),
        ([4, 5, 5, 5], [3, 3, 3, 3], [3, 4, 4, 4]),
    ]
    assert plans.shape == (6, 50, 7)
    assert plans[:, 0, 0].tolist() == [0, 1, 2, 3, 4, 5]


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


def test_belief_temporal_and_queries_are_variable_time_and_differentiable() -> None:
    torch.manual_seed(17)
    plans = torch.randn(2, 3, 50, 7, requires_grad=True)
    indices = torch.tensor([[0, 5, 10], [0, 10, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    belief_encoder = ForecastBeliefEncoder(
        action_width=7,
        horizon=50,
        width=256,
        maximum_revision_count=10,
    )
    tokens, positions, valid, routing = belief_encoder(plans, indices, mask)
    temporal = VariableTimeTemporalEncoder(width=256, heads=8, blocks=2)
    memory = temporal(tokens, positions, valid, routing)
    decoder = LoRAQueryDecoder(
        width=256,
        heads=8,
        blocks=2,
        initialization_seed=7,
    )
    expert, action_in, action_out = decoder(memory, valid)
    assert tokens.shape == (2, 60, 256)
    assert expert.shape == (2, 18, 16, 256)
    assert action_in.shape == action_out.shape == (2, 16, 256)
    sum(value.square().mean() for value in (expert, action_in, action_out)).backward()
    assert plans.grad is not None and bool(torch.isfinite(plans.grad).all())
    assert float(plans.grad.abs().sum()) > 0


def test_reversing_frame_content_changes_absolute_time_memory() -> None:
    torch.manual_seed(23)
    encoder = ForecastBeliefEncoder(
        action_width=7,
        horizon=50,
        width=256,
        maximum_revision_count=10,
    )
    plans = torch.randn(1, 4, 50, 7)
    indices = torch.tensor([[0, 5, 10, 15]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    forward = encoder(plans, indices, mask)[0]
    reversed_content = encoder(plans.flip(1), indices, mask)[0]
    assert not torch.allclose(forward, reversed_content)


def test_belief_revision_is_plan_relative_zero_and_strength_preserving() -> None:
    torch.manual_seed(29)
    encoder = ForecastBeliefEncoder(
        action_width=7,
        horizon=50,
        width=256,
        maximum_revision_count=10,
    )
    indices = torch.tensor([[0, 5]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    agreement = torch.zeros(1, 2, 50, 7)
    agreed_belief, _, _, agreed_routing = encoder(agreement, indices, mask)
    assert torch.count_nonzero(agreed_belief[..., 128:]) == 0
    assert torch.count_nonzero(agreed_routing[..., 2]) == 0

    one_frame_indices = torch.tensor([[0]])
    one_frame_mask = torch.ones(1, 1, dtype=torch.bool)
    small_plan = torch.full((1, 1, 50, 7), 0.1)
    large_plan = torch.full((1, 1, 50, 7), 1.0)
    small_plan_tokens = encoder(
        small_plan,
        one_frame_indices,
        one_frame_mask,
    )[0][..., :128]
    large_plan_tokens = encoder(
        large_plan,
        one_frame_indices,
        one_frame_mask,
    )[0][..., :128]
    assert torch.allclose(
        large_plan_tokens,
        10.0 * small_plan_tokens,
        atol=1e-5,
        rtol=1e-5,
    )

    small = agreement.clone()
    small[:, 0] = -0.1
    small.requires_grad_(True)
    large = agreement.clone()
    large[:, 0] = -1.0
    small_belief, _, _, small_routing = encoder(small, indices, mask)
    large_belief, _, _, large_routing = encoder(large, indices, mask)
    overlap = slice(5, 50)
    assert torch.allclose(
        small_belief[:, overlap, :128],
        large_belief[:, overlap, :128],
    )
    assert bool(
        (
            large_routing[:, overlap, 2]
            > small_routing[:, overlap, 2]
        ).all()
    )
    assert torch.allclose(
        small_routing[:, overlap, 2],
        torch.full_like(small_routing[:, overlap, 2], 0.1),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        large_routing[:, overlap, 2],
        torch.ones_like(large_routing[:, overlap, 2]),
        atol=1e-6,
        rtol=0.0,
    )
    assert not small_routing.requires_grad
    small_revision_rms = small_belief[:, overlap, 128:].square().mean().sqrt()
    large_revision_rms = large_belief[:, overlap, 128:].square().mean().sqrt()
    assert float(large_revision_rms.detach()) > float(small_revision_rms.detach())

    trainable = torch.randn(1, 2, 50, 7, requires_grad=True)
    trainable_revision = encoder(trainable, indices, mask)[0][..., 128:]
    direction_probe = torch.linspace(
        -1.0,
        1.0,
        trainable_revision.shape[-1],
    )
    (trainable_revision * direction_probe).sum().backward()
    assert trainable.grad is not None
    assert bool(torch.isfinite(trainable.grad).all())
    assert float(trainable.grad.abs().sum()) > 0


def test_temporal_metadata_cannot_create_content_from_zero_beliefs() -> None:
    temporal = VariableTimeTemporalEncoder(width=32, heads=4, blocks=2)
    beliefs = torch.zeros(2, 7, 32)
    positions = torch.arange(7)[None].expand(2, -1)
    valid = torch.ones(2, 7, dtype=torch.bool)
    routing = torch.randn(2, 7, 3)
    output = temporal(beliefs, positions, valid, routing)
    assert torch.count_nonzero(output) == 0


def test_temporal_starts_as_masked_belief_identity() -> None:
    temporal = VariableTimeTemporalEncoder(width=32, heads=4, blocks=2)
    beliefs = torch.randn(2, 7, 32)
    positions = torch.arange(7)[None].expand(2, -1)
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, False, False],
        ]
    )
    routing = torch.randn(2, 7, 3)
    output = temporal(beliefs, positions, valid, routing)
    assert torch.equal(
        output,
        beliefs.masked_fill(~valid[..., None], 0.0),
    )
