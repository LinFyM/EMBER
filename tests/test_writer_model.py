from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.temporal import CausalProcedureEncoder, LanguageSemanticCore


class _Projection(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.in_features = input_width
        self.out_features = output_width


class _Layer(torch.nn.Module):
    def __init__(self, dimensions: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        for name, dimensions_pair in dimensions.items():
            setattr(self.self_attn, name, _Projection(*dimensions_pair))


class _Backbone(torch.nn.Module):
    def __init__(self, dimensions: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer(dimensions) for _ in range(18))


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
    state = {}
    generator = torch.Generator().manual_seed(13)
    for layer in range(18):
        prefix = f"model.paligemma_with_expert.gemma_expert.model.layers.{layer}.self_attn."
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


class _FakeSemanticEncoder(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        counts = task_span_mask.sum(dim=1)
        maximum = int(counts.max())
        valid = torch.arange(maximum)[None] < counts[:, None]
        language = language_tokens.to(torch.float32).mean(dim=1)
        text = language[:, None, None].expand(-1, maximum, 256).clone()
        image = frames.to(torch.float32).mean(dim=(1, 2, 3))
        value = image + language.index_select(0, frame_condition_ids)
        evidence = value[:, None, None].expand(-1, maximum, 256).clone()
        grounded = evidence.clone()
        interaction = value[:, None].expand(-1, 256).clone()
        return text, evidence, grounded, interaction, valid


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
        procedure_heads=8,
        procedure_blocks=2,
        visual_transition_heads=8,
        fusion_heads=8,
        factor_hidden_width=256,
        videos_per_condition=4,
        phase_slots=16,
        initialization_seed=7,
        activation_checkpointing=True,
    )
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(16 * 3 * 4 * 4, dtype=torch.int64).remainder(251).to(torch.uint8)
    frames = frames.reshape(16, 3, 4, 4)
    frame_indices = torch.tensor([0, 5] * 8, dtype=torch.long)
    video_offsets = torch.arange(0, 17, 2, dtype=torch.long)
    condition_offsets = torch.tensor([0, 4, 8], dtype=torch.long)
    tokens = torch.tensor(
        [[1, 10, 11, 12, 13, 0], [1, 20, 21, 22, 23, 24]], dtype=torch.long
    )
    masks = tokens.ne(0)
    spans = torch.tensor(
        [
            [False, False, True, True, False, False],
            [False, True, True, True, True, False],
        ]
    )
    return frames, frame_indices, video_offsets, condition_offsets, tokens, masks, spans


def test_phase_aligned_writer_parameter_budget_and_identity_are_exact() -> None:
    model, template = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_775_296
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 10_775_296
    assert contract["source_policy_trainable_parameter_count"] == 0
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_phase_alignment_is_video_set_permutation_invariant() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    inputs = _inputs()
    baseline = model.encode_program(*inputs, policy=torch.nn.Identity())
    order = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    frame_rows = torch.cat((2 * order[:, None], 2 * order[:, None] + 1), dim=1).reshape(-1)
    changed = model.encode_program(
        inputs[0].index_select(0, frame_rows),
        inputs[1],
        inputs[2],
        inputs[3],
        *inputs[4:],
        policy=torch.nn.Identity(),
    )
    assert torch.allclose(baseline, changed, atol=2e-5, rtol=2e-5)


def test_four_videos_jointly_change_the_generated_lora_after_heads_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    baseline = model(*_inputs(), policy=torch.nn.Identity())
    changed_inputs = list(_inputs())
    changed_inputs[0] = changed_inputs[0].clone()
    changed_inputs[0][4:6] = 0
    changed = model(*changed_inputs, policy=torch.nn.Identity())
    assert any(not torch.allclose(baseline[name], changed[name]) for name in baseline)


def test_semantic_core_is_frame_set_invariant_and_procedure_is_causal() -> None:
    torch.manual_seed(17)
    core = LanguageSemanticCore(width=32, heads=4, blocks=2)
    text = torch.randn(2, 5, 32)
    evidence = torch.randn(2, 12, 5, 32)
    valid_frames = torch.ones(2, 12, dtype=torch.bool)
    valid_tokens = torch.ones(2, 5, dtype=torch.bool)
    baseline, _ = core(text, evidence, valid_frames, valid_tokens)
    changed, _ = core(text, evidence[:, torch.randperm(12)], valid_frames, valid_tokens)
    assert torch.allclose(baseline, changed, atol=1e-5, rtol=1e-5)
    procedure = CausalProcedureEncoder(width=32, heads=4, blocks=2)
    values = torch.randn(1, 8, 32)
    positions = torch.arange(8)[None]
    valid = torch.ones(1, 8, dtype=torch.bool)
    first = procedure(values, positions, valid)
    altered = values.clone()
    altered[:, 5:] += 10
    second = procedure(altered, positions, valid)
    assert torch.allclose(first[:, :5], second[:, :5], atol=1e-5, rtol=1e-5)


def test_gradient_staging_reaches_video_semantics_after_compiler_opens() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    torch.nn.init.normal_(model.compiler.modulation.weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in output.values()).backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.semantic_core.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.procedure.parameters()
    )
