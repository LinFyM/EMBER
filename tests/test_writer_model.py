from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


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
    task_descriptor_width = 2048
    video_descriptor_width = 512

    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        _video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = language_tokens.shape[0]
        task_seed = language_tokens.to(torch.float32).sum(dim=1)
        task = task_seed[:, None] + torch.arange(2048, dtype=torch.float32)[None]
        frame_seed = frames.to(torch.float32).mean(dim=(1, 2, 3))
        video = torch.zeros(count, 512, dtype=torch.float32)
        video.index_add_(0, frame_condition_ids, frame_seed[:, None].expand(-1, 512))
        return task, video + torch.arange(512, dtype=torch.float32)[None]


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


def _authority() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(19)
    return {
        "task_center": torch.zeros(2048),
        "task_frequencies": torch.randn(16, 2048, generator=generator),
        "video_frequencies": torch.randn(16, 512, generator=generator),
    }


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    template = _template()
    pali, expert = _backbones()
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=pali,
        expert_model=expert,
        condition_authority=_authority(),
        image_width=2048,
        expert_width=1024,
        program_width=256,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        factor_hidden_width=256,
        condition_task_rff_frequencies=16,
        condition_video_rff_frequencies=16,
        initialization_seed=7,
    )
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(5, 3, 4, 4)
    return (
        frames,
        torch.tensor([0, 5, 0, 5, 10], dtype=torch.long),
        torch.tensor([0, 2, 5], dtype=torch.long),
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


def test_condition_kernel_writer_parameter_ownership_is_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 86_065_152
    assert model.program_memory.value.numel() == 83_886_080
    assert not model.program_memory.value.requires_grad
    assert sum(parameter.numel() for parameter in model.factor_heads.parameters()) == 2_179_072
    assert all(parameter.requires_grad for parameter in model.factor_heads.parameters())
    assert model.condition_descriptor.fixed_suffix_noise.shape == (50, 32)
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 86_065_152
    assert contract["trainable_parameter_count"] == 2_179_072
    assert contract["source_policy_trainable_parameter_count"] == 0


def test_fresh_writer_is_exact_identity_despite_nonzero_memory() -> None:
    model, template = _model()
    assert torch.count_nonzero(model.program_memory.value)
    model.condition_descriptor = _FakeConditionDescriptor()
    output = model(*_inputs(), policy=torch.nn.Identity())
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_encode_condition_and_decode_are_exact_forward_composition() -> None:
    model, _ = _model()
    model.condition_descriptor = _FakeConditionDescriptor()
    inputs = _inputs()
    feature, program = model.encode_condition(*inputs, policy=torch.nn.Identity())
    direct = model(*inputs, policy=torch.nn.Identity())
    decoded = model.decode_program(program)
    assert feature.shape == (2, 1024)
    assert program.shape == (2, 320, 256)
    assert torch.allclose(feature.norm(dim=1), torch.ones(2))
    assert all(torch.equal(direct[name], decoded[name]) for name in direct)
