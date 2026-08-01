from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.pi05_lora import load_pi05_lora_contract, pi05_target_names
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.program_compiler import (
    SemanticProgramError,
    TargetRankProgramReader,
)
from ember.writer.semantic_program import UnifiedCausalProgram
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
        evidence = frame_value[:, None, None].expand(-1, maximum, 256).clone()
        interaction = frame_value[:, None].expand(-1, 256).clone()
        grounded = evidence * 0.1
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
        program_heads=8,
        program_blocks=2,
        compiler_heads=8,
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


def test_ucp_writer_parameter_budget_and_fixed_probe_noise_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_683_328
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 7_683_328
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
        "interaction_projection": (
            model.semantic_encoder.interaction_projection,
            262_144,
        ),
        "semantic_program": (model.semantic_program, 1_838_592),
        "compiler": (model.compiler, 212_224),
        "factor_heads": (model.factor_heads, 2_179_072),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert "semantic_encoder.fixed_suffix_noise" in model.state_dict()
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad


def test_ucp_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_ucp_target_ordinals_follow_sealed_policy_not_state_key_sort() -> None:
    model, _ = _model()
    observed = {}
    for item in model.tensor_specs:
        observed.setdefault(item.module, model._decoding[item.name][1])
    assert tuple(
        module for module, _ in sorted(observed.items(), key=lambda row: row[1])
    ) == pi05_target_names()


def test_ucp_writer_becomes_video_conditioned_after_heads_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_ucp_gradient_staging_opens_all_major_paths_after_heads_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()

    first = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in first.values()).backward()
    assert all(
        head.network[-1].weight.grad is not None
        and bool(torch.count_nonzero(head.network[-1].weight.grad))
        for head in model.factor_heads.values()
    )
    for module in (model.semantic_program, model.compiler):
        assert all(
            parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )

    model.zero_grad(set_to_none=True)
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    second = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in second.values()).backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.semantic_program.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.compiler.parameters()
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


def test_unified_program_preserves_interval_prefix_and_uses_order() -> None:
    torch.manual_seed(23)
    encoder = UnifiedCausalProgram(width=32, heads=4, blocks=2, initialization_seed=7)
    absolute = torch.randn(1, 6, 4, 32)
    grounded = torch.randn(1, 6, 4, 32)
    action = torch.randn(1, 6, 32)
    future_absolute = absolute.clone()
    future_grounded = grounded.clone()
    future_action = action.clone()
    future_absolute[:, 4:] = torch.randn_like(future_absolute[:, 4:])
    future_grounded[:, 4:] = torch.randn_like(future_grounded[:, 4:])
    future_action[:, 4:] = torch.randn_like(future_action[:, 4:])
    positions = torch.arange(6)[None]
    valid_frames = torch.ones(1, 6, dtype=torch.bool)
    valid_tokens = torch.ones(1, 4, dtype=torch.bool)
    baseline = encoder(
        absolute, grounded, action, positions, valid_frames, valid_tokens
    )[0]
    future = encoder(
        future_absolute,
        future_grounded,
        future_action,
        positions,
        valid_frames,
        valid_tokens,
    )[0]
    reverse = encoder(
        absolute.flip(1),
        grounded.flip(1),
        action.flip(1),
        positions,
        valid_frames,
        valid_tokens,
    )[0]
    assert torch.allclose(baseline[:, :3], future[:, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(baseline, reverse)


def test_unified_program_aligns_absolute_action_and_outgoing_patch_change() -> None:
    encoder = UnifiedCausalProgram(width=32, heads=4, blocks=1, initialization_seed=7)
    encoder.blocks = torch.nn.ModuleList()
    absolute = torch.randn(1, 5, 3, 32)
    grounded = torch.randn(1, 5, 3, 32)
    action = torch.randn(1, 5, 32)
    positions = torch.tensor([[0, 5, 10, 15, 17]])
    valid_frames = torch.ones(1, 5, dtype=torch.bool)
    valid_tokens = torch.tensor([[True, True, False]])
    program, endpoints, valid_intervals, valid_semantics = encoder(
        absolute,
        grounded,
        action,
        positions,
        valid_frames,
        valid_tokens,
    )
    assert torch.equal(program[:, :, :2], absolute[:, :-1, :2])
    assert not bool(program[:, :, 2].count_nonzero())
    assert torch.equal(program[:, :, 3], action[:, :-1])
    assert torch.equal(
        program[:, :, 4:6],
        grounded[:, 1:, :2] - grounded[:, :-1, :2],
    )
    assert not bool(program[:, :, 6].count_nonzero())
    assert torch.equal(endpoints, positions[:, 1:])
    assert torch.equal(valid_intervals, torch.ones_like(valid_intervals))
    assert torch.equal(
        valid_semantics,
        torch.tensor([[True, True, False, True, True, True, False]]),
    )


def test_program_identities_cannot_create_value_from_zero_content() -> None:
    encoder = UnifiedCausalProgram(width=32, heads=4, blocks=2, initialization_seed=7)
    absolute = torch.zeros(2, 6, 4, 32)
    grounded = torch.zeros(2, 6, 4, 32)
    action = torch.zeros(2, 6, 32)
    positions = torch.tensor(
        [[0, 5, 10, 15, 20, 25], [0, 3, 8, 13, 21, 28]]
    )
    valid_frames = torch.ones(2, 6, dtype=torch.bool)
    valid_tokens = torch.ones(2, 4, dtype=torch.bool)
    output = encoder(
        absolute, grounded, action, positions, valid_frames, valid_tokens
    )[0]
    assert torch.count_nonzero(output) == 0


def test_routing_and_positions_cannot_create_lora_content_from_zero_values() -> None:
    reader = TargetRankProgramReader(
        width=32,
        heads=4,
        target_count=38,
        rank=16,
        initialization_seed=7,
    )
    program = torch.zeros(2, 5, 9, 32)
    valid_intervals = torch.ones(2, 5, dtype=torch.bool)
    valid_semantics = torch.ones(2, 9, dtype=torch.bool)
    positions = torch.tensor([[0, 5, 10, 15, 20], [0, 3, 8, 13, 21]])
    output = reader(program, positions, valid_intervals, valid_semantics)
    assert output.shape == (2, 38, 16, 32)
    assert torch.count_nonzero(output) == 0


def test_target_rank_reader_reads_program_order_without_terminal_gate() -> None:
    torch.manual_seed(29)
    reader = TargetRankProgramReader(
        width=32,
        heads=4,
        target_count=38,
        rank=16,
        initialization_seed=7,
    )
    program = torch.randn(1, 5, 9, 32)
    positions = torch.arange(5)[None]
    valid_intervals = torch.ones(1, 5, dtype=torch.bool)
    valid_semantics = torch.ones(1, 9, dtype=torch.bool)
    baseline = reader(program, positions, valid_intervals, valid_semantics)
    reverse = reader(
        program.flip(1), positions, valid_intervals, valid_semantics
    )
    assert not torch.allclose(baseline, reverse)


def test_target_and_rank_identities_route_without_entering_values() -> None:
    torch.manual_seed(37)
    reader = TargetRankProgramReader(
        width=32,
        heads=4,
        target_count=5,
        rank=4,
        initialization_seed=7,
    )
    program = torch.randn(1, 4, 7, 32)
    positions = torch.arange(4)[None]
    valid_intervals = torch.ones(1, 4, dtype=torch.bool)
    valid_semantics = torch.ones(1, 7, dtype=torch.bool)
    baseline = reader(program, positions, valid_intervals, valid_semantics)
    target_swap = torch.tensor([1, 0, 2, 3, 4])
    with torch.no_grad():
        reader.target_identity.copy_(reader.target_identity[target_swap])
    target_permuted = reader(program, positions, valid_intervals, valid_semantics)
    assert torch.allclose(target_permuted, baseline[:, target_swap], atol=1e-6, rtol=1e-5)

    rank_swap = torch.tensor([2, 1, 0, 3])
    with torch.no_grad():
        reader.rank_identity.copy_(reader.rank_identity[rank_swap])
    both_permuted = reader(program, positions, valid_intervals, valid_semantics)
    assert torch.allclose(
        both_permuted,
        target_permuted[:, :, rank_swap],
        atol=1e-6,
        rtol=1e-5,
    )


def test_program_and_reader_ignore_ragged_padding_content() -> None:
    torch.manual_seed(31)
    encoder = UnifiedCausalProgram(width=32, heads=4, blocks=2, initialization_seed=7)
    reader = TargetRankProgramReader(
        width=32,
        heads=4,
        target_count=38,
        rank=16,
        initialization_seed=7,
    )
    absolute = torch.randn(1, 6, 4, 32)
    grounded = torch.randn(1, 6, 4, 32)
    action = torch.randn(1, 6, 32)
    changed_absolute = absolute.clone()
    changed_grounded = grounded.clone()
    changed_action = action.clone()
    changed_absolute[:, 4:] = 100.0 * torch.randn_like(changed_absolute[:, 4:])
    changed_grounded[:, 4:] = 100.0 * torch.randn_like(changed_grounded[:, 4:])
    changed_action[:, 4:] = 100.0 * torch.randn_like(changed_action[:, 4:])
    positions = torch.tensor([[0, 5, 10, 15, 0, 0]])
    valid_frames = torch.tensor([[True, True, True, True, False, False]])
    valid_tokens = torch.ones(1, 4, dtype=torch.bool)
    baseline = encoder(
        absolute, grounded, action, positions, valid_frames, valid_tokens
    )
    padded = encoder(
        changed_absolute,
        changed_grounded,
        changed_action,
        positions,
        valid_frames,
        valid_tokens,
    )
    assert torch.equal(baseline[2], padded[2])
    assert torch.allclose(baseline[0], padded[0], atol=2e-6, rtol=1e-5)

    compiled_baseline = reader(*baseline)
    compiled_padded = reader(*padded)
    assert torch.allclose(
        compiled_baseline, compiled_padded, atol=2e-6, rtol=1e-5
    )


def test_target_rank_reader_rejects_missing_content_and_float_endpoints() -> None:
    reader = TargetRankProgramReader(
        width=32,
        heads=4,
        target_count=38,
        rank=16,
        initialization_seed=7,
    )
    program = torch.randn(1, 2, 3, 32)
    endpoints = torch.tensor([[5, 10]])
    valid_intervals = torch.ones(1, 2, dtype=torch.bool)
    valid_semantics = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(SemanticProgramError, match="target/rank Program memory"):
        reader(
            program,
            endpoints.to(torch.float32),
            valid_intervals,
            valid_semantics,
        )
    with pytest.raises(SemanticProgramError, match="target/rank Program memory"):
        reader(
            program,
            endpoints,
            torch.zeros_like(valid_intervals),
            valid_semantics,
        )
