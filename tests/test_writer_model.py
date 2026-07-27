from __future__ import annotations

import torch

from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.temporal import (
    CausalProcedureEncoder,
    CoreProcedureLoRACompiler,
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
            setattr(self.self_attn, name, _Projection(input_width, output_width))


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
    generator = torch.Generator(device="cpu").manual_seed(13)
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, output_width in (("q_proj", 2048), ("v_proj", 256)):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                16,
                1024,
                generator=generator,
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width,
                16,
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(
            16,
            input_width,
            generator=generator,
        )
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, 16)
    return state


class _FakeFrameSemantics(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = frames.to(torch.float32).mean(dim=(1, 2, 3))
        language = language_tokens.to(torch.float32).mean(dim=1)
        value = image + language.index_select(0, frame_condition_ids)
        core = value[:, None, None].expand(-1, 64, 256)
        interaction = value[:, None].expand(-1, 256)
        return core, interaction


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
        expert_width=1024,
        program_width=256,
        spatial_pool_grid=8,
        vl_meta_lora_rank=4,
        action_meta_lora_rank=8,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        procedure_heads=8,
        procedure_blocks=2,
        core_compiler_blocks=1,
        procedure_refiner_blocks=1,
        factor_hidden_width=420,
        initialization_seed=7,
        activation_checkpointing=True,
    )
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(
        5,
        3,
        4,
        4,
    )
    frame_indices = torch.tensor([0, 5, 0, 5, 10], dtype=torch.long)
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    tokens = torch.tensor([[1, 2, 0], [4, 5, 6]], dtype=torch.long)
    masks = tokens.ne(0)
    return frames, frame_indices, offsets, tokens, masks


def test_v5_writer_parameter_budget_and_fixed_probe_noise_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_301_440
    assert model.frame_semantics.fixed_suffix_noise.shape == (50, 32)
    assert "frame_semantics.fixed_suffix_noise" in model.state_dict()
    assert not model.frame_semantics.fixed_suffix_noise.requires_grad


def test_v5_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    model.frame_semantics = _FakeFrameSemantics()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_v5_writer_becomes_video_conditioned_after_heads_open() -> None:
    model, _ = _model()
    model.frame_semantics = _FakeFrameSemantics()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_semantic_core_compiler_is_permutation_invariant() -> None:
    torch.manual_seed(17)
    compiler = CoreProcedureLoRACompiler(
        width=32,
        heads=4,
        core_blocks=1,
        procedure_blocks=1,
        initialization_seed=7,
    )
    core = torch.randn(2, 11, 32)
    valid_core = torch.ones(2, 11, dtype=torch.bool)
    procedure = torch.randn(2, 5, 32)
    positions = torch.arange(5)[None].expand(2, -1)
    valid_procedure = torch.ones(2, 5, dtype=torch.bool)
    baseline = compiler(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    permutation = torch.tensor([7, 2, 10, 0, 5, 1, 9, 4, 8, 3, 6])
    shuffled = compiler(
        core[:, permutation],
        valid_core[:, permutation],
        procedure,
        positions,
        valid_procedure,
    )
    for left, right in zip(baseline, shuffled, strict=True):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_causal_procedure_preserves_prefix_and_uses_order() -> None:
    torch.manual_seed(23)
    encoder = CausalProcedureEncoder(width=32, heads=4, blocks=2)
    content = torch.randn(1, 6, 32)
    changed_future = content.clone()
    changed_future[:, 4:] = torch.randn_like(changed_future[:, 4:])
    positions = torch.arange(6)[None]
    valid = torch.ones(1, 6, dtype=torch.bool)
    baseline = encoder(content, positions, valid)
    future = encoder(changed_future, positions, valid)
    reverse = encoder(content.flip(1), positions, valid)
    assert torch.allclose(baseline[:, :4], future[:, :4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(baseline, reverse)


def test_routing_and_positions_cannot_create_lora_content_from_zero_values() -> None:
    compiler = CoreProcedureLoRACompiler(
        width=32,
        heads=4,
        core_blocks=1,
        procedure_blocks=1,
        initialization_seed=7,
    )
    core = torch.zeros(2, 7, 32)
    procedure = torch.zeros(2, 5, 32)
    valid_core = torch.ones(2, 7, dtype=torch.bool)
    valid_procedure = torch.ones(2, 5, dtype=torch.bool)
    positions = torch.tensor([[0, 5, 10, 15, 20], [0, 3, 8, 13, 21]])
    output = compiler(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    assert all(torch.count_nonzero(value) == 0 for value in output)


def test_procedure_refinement_is_zero_at_init_then_order_sensitive_when_opened() -> None:
    torch.manual_seed(29)
    compiler = CoreProcedureLoRACompiler(
        width=32,
        heads=4,
        core_blocks=1,
        procedure_blocks=1,
        initialization_seed=7,
    )
    core = torch.randn(1, 9, 32)
    valid_core = torch.ones(1, 9, dtype=torch.bool)
    procedure = torch.randn(1, 5, 32)
    positions = torch.arange(5)[None]
    valid_procedure = torch.ones(1, 5, dtype=torch.bool)
    baseline = compiler(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    reversed_at_init = compiler(
        core,
        valid_core,
        procedure.flip(1),
        positions,
        valid_procedure,
    )
    for left, right in zip(baseline, reversed_at_init, strict=True):
        assert torch.equal(left, right)

    output = compiler.procedure_blocks[0].cross_attention.output
    torch.nn.init.eye_(output.weight)
    normal = compiler(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    reverse = compiler(
        core,
        valid_core,
        procedure.flip(1),
        positions,
        valid_procedure,
    )
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(normal, reverse, strict=True)
    )
