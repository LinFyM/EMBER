from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.fewshot_m2p import InvariantProgramEncoder
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.task_gradient import parameter_layout


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
    FRAME_DESCRIPTOR_WIDTH = 128

    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_video_ids: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        videos = language_tokens.shape[0]
        task_seed = language_tokens.to(torch.float32).sum(dim=1)
        task = task_seed[:, None] + torch.arange(2048, dtype=torch.float32)[None]
        frame_seed = frames.to(torch.float32).mean(dim=(1, 2, 3))
        pooled = torch.zeros(videos, dtype=torch.float32)
        pooled.index_add_(0, frame_video_ids, frame_seed)
        counts = video_offsets.diff().to(torch.float32)
        pooled = pooled / counts
        temporal = torch.arange(4, dtype=torch.float32)[None, :, None]
        width = torch.arange(128, dtype=torch.float32)[None, None]
        return task, pooled[:, None, None] + temporal + width


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
        program_width=256,
        program_slots=32,
        program_heads=8,
        program_blocks=2,
        m2p_heads=8,
        m2p_blocks=3,
        factor_hidden_width=256,
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


def test_k4_writer_parameter_ownership_is_end_to_end_and_exact() -> None:
    model, _ = _model()
    layout = parameter_layout(model)
    assert {row.block for row in layout} == {
        "invariant_program",
        "m2p_shared",
        "m2p_a_heads",
        "m2p_b_heads",
    }
    assert layout[-1].stop == sum(value.numel() for value in model.parameters())
    assert model.m2p._routing().shape == (608, 256)
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


def test_invariant_program_is_video_owned_and_shot_permutation_invariant() -> None:
    encoder = InvariantProgramEncoder(
        task_width=8,
        video_width=6,
        program_width=16,
        program_slots=3,
        heads=4,
        blocks=1,
        initialization_seed=3,
    )
    task = torch.randn(8, 8)
    video = torch.randn(8, 4, 6)
    offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    expected = encoder(task, video, offsets)
    permutation = torch.tensor([2, 0, 3, 1, 7, 5, 4, 6])
    observed = encoder(task[permutation], video[permutation], offsets)
    assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(encoder(task, torch.zeros_like(video), offsets), torch.zeros_like(expected))


def test_program_reaches_lora_after_zero_final_bootstrap_is_opened() -> None:
    model, _ = _model()
    head = model.m2p.b_heads["target_000"].output
    head.weight.data.normal_(std=0.01)
    program = torch.randn(1, 32, 256, requires_grad=True)
    output = model.decode_program(program)
    loss = output[model.m2p.targets[0].b_name].square().sum()
    loss.backward()
    assert program.grad is not None and float(program.grad.norm()) > 0
    assert model.invariant_program.latent_route.requires_grad
