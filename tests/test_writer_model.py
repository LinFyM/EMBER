from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.compiler import (
    CoreProgramCompiler,
    ZeroPreservingSlotBlock,
)
from ember.writer.model import (
    CompleteLoRAWriter,
    FactorHead,
    build_lora_tensor_specs,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    TaskGroundedVisualTransitionFusion,
)
from ember.writer.video_program import TaskQueriedPatchGrounding


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


class _FakeSemanticEncoder(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        counts = task_span_mask.sum(dim=1)
        maximum = int(counts.max())
        valid = torch.arange(maximum)[None] < counts[:, None]
        language = language_tokens.to(torch.float32).mean(dim=1)
        text = language[:, None, None].expand(-1, maximum, 256).clone()
        image = frames.to(torch.float32).mean(dim=(1, 2, 3))
        frame_value = image + language.index_select(0, frame_condition_ids)
        channels = torch.linspace(0.01, 1.0, 256)
        evidence = torch.sin(
            frame_value[:, None, None] * channels[None, None]
        ).expand(-1, maximum, -1).clone()
        grounded = evidence + torch.cos(
            frame_value[:, None, None] * channels[None, None]
        ).expand(-1, maximum, -1)
        action = torch.cos(frame_value[:, None] * channels[None])
        return text, evidence, grounded, action, valid


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
        text_meta_lora_rank=4,
        vl_meta_lora_rank=4,
        action_meta_lora_rank=4,
        patch_grounding_heads=8,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        semantic_core_heads=8,
        semantic_core_blocks=2,
        visual_transition_heads=8,
        procedure_heads=8,
        procedure_blocks=2,
        fusion_heads=8,
        bilinear_hidden_width=512,
        factor_hidden_width=256,
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
    tokens = torch.tensor(
        [[1, 10, 11, 12, 13, 0], [1, 20, 21, 22, 23, 24]],
        dtype=torch.long,
    )
    masks = tokens.ne(0)
    task_spans = torch.tensor(
        [
            [False, False, True, True, False, False],
            [False, True, True, True, True, False],
        ]
    )
    return frames, frame_indices, offsets, tokens, masks, task_spans


def test_core_program_writer_parameter_budget_and_native_probe_contract_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_905_856
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 10_905_856
    assert contract["source_policy_trainable_parameter_count"] == 0
    expected = {
        "text_meta_lora": (model.semantic_encoder.text_meta_lora, 921_600),
        "vl_meta_lora": (model.semantic_encoder.vl_meta_lora, 921_600),
        "action_meta_lora": (model.semantic_encoder.action_meta_lora, 626_688),
        "language_projection": (
            model.semantic_encoder.language_projection,
            524_288,
        ),
        "patch_grounding": (
            model.semantic_encoder.patch_grounding,
            197_120,
        ),
        "semantic_core_set_fusion": (
            model.semantic_core.semantic_set_fusion,
            262_656,
        ),
        "semantic_core_blocks": (model.semantic_core.blocks, 1_573_888),
        "interaction_projection": (
            model.semantic_encoder.interaction_projection,
            262_144,
        ),
        "visual_transition": (model.visual_transition, 197_120),
        "procedure": (model.procedure, 1_573_888),
        "compiler": (model.compiler, 1_665_792),
        "factor_heads": (model.factor_heads, 2_179_072),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert "semantic_encoder.fixed_suffix_noise" in model.state_dict()
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad
    assert not hasattr(model.semantic_encoder, "action_probe_positions")
    assert model.compiler.bilinear_fusion.core_projection.out_features == 512
    assert model.compiler.bilinear_fusion.program_projection.out_features == 512


def test_core_program_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_core_program_writer_becomes_video_conditioned_after_heads_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_core_program_gradient_staging_reaches_both_branches_and_transition() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()

    first = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in first.values()).backward()
    assert all(
        head.network[-1].weight.grad is not None
        and bool(torch.count_nonzero(head.network[-1].weight.grad))
        for head in model.factor_heads.values()
    )
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
        for parameter in model.semantic_core.parameters()
    )

    model.zero_grad(set_to_none=True)
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    second = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in second.values()).backward()
    for module in (
        model.semantic_core,
        model.visual_transition,
        model.procedure,
        model.compiler,
    ):
        assert any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )


def test_task_queried_patch_grounding_uses_patch_content_without_order_geometry() -> None:
    torch.manual_seed(23)
    grounding = TaskQueriedPatchGrounding(width=32, heads=4)
    queries = torch.randn(2, 5, 32)
    patches = torch.randn(2, 256, 32)
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    baseline = grounding(queries, patches, valid)
    permuted = grounding(
        queries,
        patches[:, torch.randperm(256)],
        valid,
    )
    changed = grounding(queries, patches + 0.25, valid)
    assert torch.allclose(baseline, permuted, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(baseline, changed)
    assert not bool(baseline[0, 3:].count_nonzero())
    assert not hasattr(grounding, "value")


def test_language_core_is_frame_permutation_invariant() -> None:
    torch.manual_seed(17)
    core = LanguageSemanticCore(
        width=32,
        heads=4,
        blocks=2,
    )
    evidence = torch.randn(2, 5, 7, 32)
    text_queries = torch.randn(2, 7, 32)
    valid_frames = torch.ones(2, 5, dtype=torch.bool)
    valid_tokens = torch.ones(2, 7, dtype=torch.bool)
    baseline, _ = core(text_queries, evidence, valid_frames, valid_tokens)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    shuffled, _ = core(
        text_queries,
        evidence[:, permutation],
        valid_frames[:, permutation],
        valid_tokens,
    )
    assert torch.allclose(baseline, shuffled, atol=1e-5, rtol=1e-5)


def test_language_core_preserves_mean_when_learned_residuals_are_zero() -> None:
    torch.manual_seed(18)
    core = LanguageSemanticCore(width=32, heads=4, blocks=1)
    for module in core.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
    torch.nn.init.eye_(core.semantic_set_fusion.mean.weight)
    evidence = torch.randn(1, 4, 3, 32)
    text_queries = torch.randn(1, 3, 32)
    valid_frames = torch.tensor([[True, True, True, False]])
    valid_tokens = torch.tensor([[True, True, False]])
    output, weights = core(
        text_queries,
        evidence,
        valid_frames,
        valid_tokens,
    )
    expected = evidence[:, :3].mean(dim=1)
    expected[:, 2] = 0
    assert torch.allclose(output, expected)
    assert output.shape == (1, 3, 32)
    assert weights.shape == (1, 4, 4, 3)
    assert not bool(output[:, 2].count_nonzero())
    assert not bool(weights[..., 2].count_nonzero())


def test_visual_transition_is_uncapped_and_recomputed_in_input_order() -> None:
    torch.manual_seed(19)
    fusion = TaskGroundedVisualTransitionFusion(width=32, heads=4)
    action = torch.full((2, 4, 32), 1e-4)
    grounded = 10.0 * torch.randn(2, 4, 5, 32)
    valid_frames = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    valid_tokens = torch.ones(2, 5, dtype=torch.bool)
    fused, transition = fusion(
        action,
        grounded,
        valid_frames,
        valid_tokens,
    )
    assert torch.equal(
        transition[:, 0],
        torch.zeros_like(transition[:, 0]),
    )
    residual_rms = (fused - action).square().mean(dim=-1).sqrt()
    action_rms = action.square().mean(dim=-1).sqrt()
    assert bool((residual_rms[valid_frames] > 0.25 * action_rms[valid_frames]).any())
    assert not bool(fused[1, 3].count_nonzero())

    zero_action = torch.zeros_like(action)
    zero_fused, _ = fusion(
        zero_action,
        grounded,
        valid_frames,
        valid_tokens,
    )
    assert bool(zero_fused[:, 1:].count_nonzero())


def test_visual_transition_zero_first_delta_has_finite_backward() -> None:
    torch.manual_seed(21)
    fusion = TaskGroundedVisualTransitionFusion(width=32, heads=4)
    action = torch.randn(2, 4, 32, requires_grad=True)
    grounded = torch.randn(2, 4, 5, 32, requires_grad=True)
    valid_frames = torch.ones(2, 4, dtype=torch.bool)
    valid_tokens = torch.ones(2, 5, dtype=torch.bool)
    fused, transition = fusion(
        action,
        grounded,
        valid_frames,
        valid_tokens,
    )
    assert not bool(transition[:, 0].count_nonzero())
    fused[:, 0].sum().backward()
    assert action.grad is not None and bool(torch.isfinite(action.grad).all())
    assert grounded.grad is not None and bool(torch.isfinite(grounded.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in fusion.parameters()
    )


def test_causal_procedure_does_not_leak_future_content() -> None:
    torch.manual_seed(23)
    encoder = CausalProcedureEncoder(width=32, heads=4, blocks=2)
    content = torch.randn(1, 4, 32)
    changed = content.clone()
    changed[:, 3] = torch.randn_like(changed[:, 3])
    positions = torch.tensor([[0, 5, 10, 15]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    baseline = encoder(content, positions, valid)
    modified = encoder(changed, positions, valid)
    assert torch.allclose(
        baseline[:, :3],
        modified[:, :3],
        atol=1e-6,
        rtol=1e-5,
    )


def _compiler(width: int = 32) -> CoreProgramCompiler:
    return CoreProgramCompiler(
        width=width,
        heads=4,
        bilinear_hidden_width=64,
        initialization_seed=7,
    )


def test_compiler_requires_both_core_and_procedure() -> None:
    torch.manual_seed(29)
    compiler = _compiler()
    core = torch.randn(2, 5, 32)
    procedure = torch.randn(2, 4, 32)
    valid_core = torch.ones(2, 5, dtype=torch.bool)
    positions = torch.tensor([[0, 5, 10, 15], [0, 5, 10, 15]])
    valid = torch.ones(2, 4, dtype=torch.bool)
    complete, diagnostics = compiler.fused_slots(
        core,
        valid_core,
        procedure,
        positions,
        valid,
    )
    core_only, _ = compiler.fused_slots(
        core,
        valid_core,
        torch.zeros_like(procedure),
        positions,
        valid,
    )
    procedure_only, _ = compiler.fused_slots(
        torch.zeros_like(core),
        valid_core,
        procedure,
        positions,
        valid,
    )
    assert bool(complete.count_nonzero())
    assert not bool(core_only.count_nonzero())
    assert not bool(procedure_only.count_nonzero())
    assert not hasattr(compiler.core_reader.attention, "value")
    assert not hasattr(compiler.procedure_reader.attention, "value")
    assert diagnostics["core_basis"].shape[-1] == 64
    assert diagnostics["procedure_program"].shape[-1] == 64


def test_compiler_preserves_constant_procedure_dc_and_open_head_uses_it() -> None:
    torch.manual_seed(30)
    compiler = _compiler()
    core = torch.randn(1, 5, 32)
    constant = torch.randn(1, 1, 32)
    procedure = constant.expand(-1, 4, -1).clone()
    slots, diagnostics = compiler.fused_slots(
        core,
        torch.ones(1, 5, dtype=torch.bool),
        procedure,
        torch.tensor([[0, 5, 10, 15]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert bool(diagnostics["procedure_slots"].count_nonzero())
    assert bool(slots.count_nonzero())
    head = FactorHead(32, 32, 8)
    torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    assert bool(head(slots).count_nonzero())


def test_compiler_zero_branch_backward_is_finite() -> None:
    torch.manual_seed(31)
    compiler = _compiler()
    core = torch.randn(1, 5, 32, requires_grad=True)
    procedure = torch.zeros(1, 4, 32, requires_grad=True)
    slots, _ = compiler.fused_slots(
        core,
        torch.ones(1, 5, dtype=torch.bool),
        procedure,
        torch.tensor([[0, 5, 10, 15]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert not bool(slots.count_nonzero())
    slots.sum().backward()
    assert core.grad is not None and bool(torch.isfinite(core.grad).all())
    assert procedure.grad is not None and bool(torch.isfinite(procedure.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.parameters()
    )


def test_zero_preserving_slot_block_keeps_routing_out_of_content() -> None:
    torch.manual_seed(33)
    block = ZeroPreservingSlotBlock(width=32, heads=4)
    routing = torch.randn(1, 6, 32)
    zero = torch.zeros(1, 6, 32, requires_grad=True)
    zero_output, _ = block(zero, routing)
    assert not bool(zero_output.count_nonzero())
    zero_output.sum().backward()
    assert zero.grad is not None
    assert bool(torch.isfinite(zero.grad).all())

    content = torch.randn(1, 6, 32)
    first, _ = block(content, routing)
    second, _ = block(content, routing + 0.25)
    assert not torch.allclose(first, second)
    assert not hasattr(block, "output_norm")
    assert not hasattr(block, "input_rms")
