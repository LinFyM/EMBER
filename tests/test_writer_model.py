from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.fewshot_m2p import (
    PolicyLayerGroup,
    PolicyLayerTraceM2P,
    PolicyTargetSpec,
)
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.task_gradient import parameter_layout
from ember.writer.video_program import temporal_trace_tokens


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
            setattr(self.self_attn, name, _Projection(input_width, output_width))


class _Backbone(torch.nn.Module):
    def __init__(self, dimensions: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer(dimensions) for _ in range(18))


class _FakeConditionDescriptor(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_video_ids: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _task_span_mask: torch.Tensor,
    ) -> torch.Tensor:
        videos = language_tokens.shape[0]
        frame_seed = frames.to(torch.float32).mean(dim=(1, 2, 3))
        pooled = torch.zeros(videos, dtype=torch.float32)
        pooled.index_add_(0, frame_video_ids, frame_seed)
        counts = video_offsets.diff().to(torch.float32)
        pooled = pooled / counts
        group = torch.arange(20, dtype=torch.float32)[None, :, None, None]
        temporal = torch.arange(16, dtype=torch.float32)[None, None, :, None]
        width = torch.arange(1024, dtype=torch.float32)[None, None, None]
        return pooled[:, None, None, None] + group + temporal + width


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
    generator = torch.Generator(device="cpu").manual_seed(13)
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, output_width in (("q_proj", 2048), ("v_proj", 256)):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                16, 1024, generator=generator
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width, 16
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(
            16, input_width, generator=generator
        )
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, 16)
    return state


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    template = _template()
    pali, expert = _backbones()
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=pali,
        expert_model=expert,
        image_width=2048,
        expert_width=1024,
        policy_groups=20,
        trace_temporal_terms=16,
        memory_slots=68,
        m2p_width=1024,
        m2p_heads=8,
        m2p_blocks=4,
        m2p_ffn_expansion=2,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        videos_per_condition=4,
        initialization_seed=7,
    )
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(16 * 3 * 4 * 4, dtype=torch.int32).remainder(256).to(
        torch.uint8
    ).reshape(16, 3, 4, 4)
    return (
        frames,
        torch.tensor([0, 5] * 8, dtype=torch.long),
        torch.arange(0, 17, 2, dtype=torch.long),
        torch.tensor([0, 4, 8], dtype=torch.long),
        torch.tensor(
            [[1, 10, 11, 12, 13, 0], [1, 20, 21, 22, 23, 24]],
            dtype=torch.long,
        ),
        torch.tensor(
            [[True, True, True, True, True, False], [True] * 6]
        ),
        torch.tensor(
            [
                [False, False, True, True, False, False],
                [False, True, True, True, True, False],
            ]
        ),
    )


def test_policy_trace_dct_is_zero_preserving_and_order_sensitive() -> None:
    frames = torch.arange(2 * 16 * 3 * 5, dtype=torch.float32).reshape(32, 3, 5)
    offsets = torch.tensor([0, 16, 32], dtype=torch.long)
    expected = temporal_trace_tokens(frames, offsets, terms=16)
    assert expected.shape == (2, 3, 16, 5)
    assert torch.equal(
        temporal_trace_tokens(torch.zeros_like(frames), offsets, terms=16),
        torch.zeros_like(expected),
    )
    reversed_frames = torch.cat((frames[:16].flip(0), frames[16:].flip(0)))
    observed = temporal_trace_tokens(reversed_frames, offsets, terms=16)
    assert not torch.allclose(observed, expected)


def test_k4_writer_parameter_ownership_is_end_to_end_and_exact() -> None:
    model, _ = _model()
    layout = parameter_layout(model)
    assert {row.block for row in layout} == {"policy_layer_reader", "axis_m2p"}
    assert layout[-1].stop == sum(value.numel() for value in model.parameters())
    assert model.layer_m2p.group_output_weight.shape == (20, 1024, 1024)
    assert model.condition_descriptor.fixed_suffix_noise.shape == (50, 32)
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == contract["trainable_parameter_count"]
    assert contract["source_policy_trainable_parameter_count"] == 0


def test_fresh_k4_writer_is_exact_functional_identity() -> None:
    model, template = _model()
    model.condition_descriptor = _FakeConditionDescriptor()
    output = model(*_inputs(), policy=torch.nn.Identity())
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def _tiny_layer_m2p() -> PolicyLayerTraceM2P:
    targets = tuple(
        PolicyTargetSpec(
            module_index=index,
            module=f"module_{index}",
            a_name=f"target_{index}.A",
            b_name=f"target_{index}.B",
            rank=2,
            input_width=2,
            output_width=2,
        )
        for index in range(3)
    )
    groups = tuple(
        PolicyLayerGroup(f"group_{index}", (target,))
        for index, target in enumerate(targets)
    )
    template = {
        name: torch.randn(2, 2) if name.endswith(".A") else torch.zeros(2, 2)
        for target in targets
        for name in (target.a_name, target.b_name)
    }
    return PolicyLayerTraceM2P(
        groups,
        template_state=template,
        width=16,
        memory_slots=2,
        temporal_terms=4,
        heads=4,
        blocks=4,
        ffn_expansion=2,
        initialization_seed=3,
    )


def test_layer_trace_reader_is_video_owned_and_shot_permutation_invariant() -> None:
    decoder = _tiny_layer_m2p()
    video = torch.randn(8, 3, 4, 16)
    offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    expected = decoder.encode(video, offsets)
    permutation = torch.tensor([2, 0, 3, 1, 7, 5, 4, 6])
    observed = decoder.encode(video[permutation], offsets)
    assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(
        decoder.encode(torch.zeros_like(video), offsets),
        torch.zeros_like(expected),
    )


def test_group_output_bootstrap_then_opens_reader_and_axis_m2p() -> None:
    decoder = _tiny_layer_m2p()
    video = torch.randn(4, 3, 4, 16)
    offsets = torch.tensor([0, 4], dtype=torch.long)
    output = decoder(video, offsets)
    loss = output["target_0.B"].sum()
    loss.backward()
    assert float(decoder.group_output_weight.grad.norm()) > 0
    assert decoder.query.weight.grad is not None
    assert float(decoder.query.weight.grad.norm()) == 0
    assert decoder.axis_blocks[0].query.weight.grad is not None
    assert float(decoder.axis_blocks[0].query.weight.grad.norm()) == 0

    decoder.zero_grad(set_to_none=True)
    decoder.group_output_weight.data.normal_(std=0.01)
    output = decoder(video, offsets)
    output["target_0.B"].sum().backward()
    assert float(decoder.query.weight.grad.norm()) > 0
    assert float(decoder.axis_blocks[0].query.weight.grad.norm()) > 0


def test_axis_m2p_preserves_small_dynamic_amplitude() -> None:
    decoder = _tiny_layer_m2p()
    generator = torch.Generator(device="cpu").manual_seed(17)
    memory = torch.randn(1, 3, 2, 16, generator=generator) * 1e-4
    first = decoder._axis_m2p(memory)
    second = decoder._axis_m2p(2 * memory)
    ratio = float((second.norm() / first.norm()).detach())
    assert 1.8 < ratio < 2.2
