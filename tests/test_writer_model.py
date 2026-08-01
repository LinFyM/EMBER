from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.pi05_lora import load_pi05_lora_contract, pi05_target_names
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.internal_analysis import capture_writer, counterfactual_states
from ember.writer.internal_metrics import effective_metrics, rank_gauge_permute
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.program_compiler import AsymmetricDualReader, SemanticProgramError
from ember.writer.semantic_core import MeanBackedSemanticCore
from ember.writer.semantic_program import OutgoingSemanticProgram
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
            setattr(
                self.self_attn,
                name,
                _Projection(input_width, output_width),
            )


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
        action = frame_value[:, None].expand(-1, 256).clone()
        grounded = evidence * 0.1
        return text, evidence, grounded, action, valid


class _AnalysisSemanticEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interaction_projection = torch.nn.Linear(1024, 256, bias=False)

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
        language = language_tokens.float().mean(dim=1)
        query = language[:, None, None].expand(-1, maximum, 256).clone()
        image = frames.float().mean(dim=(1, 2, 3))
        content = image + language.index_select(0, frame_condition_ids)
        evidence = content[:, None, None].expand(-1, maximum, 256).clone()
        grounded = 0.1 * evidence
        raw_action = content[:, None].expand(-1, 1024).clone()
        return query, evidence, grounded, self.interaction_projection(raw_action), valid


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
        program_heads=8,
        program_blocks=2,
        compiler_heads=8,
        factor_hidden_width=256,
        initialization_seed=7,
        activation_checkpointing=True,
    )
    return model, template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(5 * 3 * 4 * 4, dtype=torch.uint8).reshape(5, 3, 4, 4)
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


def _analysis_inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(10 * 3 * 4 * 4, dtype=torch.int64).remainder(255).to(torch.uint8)
    frames = frames.reshape(10, 3, 4, 4)
    indices = torch.tensor([0, 5] * 5, dtype=torch.long)
    offsets = torch.arange(0, 11, 2, dtype=torch.long)
    tokens = torch.tensor([[1, 10, 11, 12, 0]] * 5, dtype=torch.long)
    masks = tokens.ne(0)
    spans = torch.tensor([[False, True, True, True, False]] * 5)
    return frames, indices, offsets, tokens, masks, spans


def test_ap_adr_parameter_budget_and_module_enumeration_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_241_024
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 10_241_024
    assert contract["source_policy_trainable_parameter_count"] == 0
    expected = {
        "text_meta_lora": (model.semantic_encoder.text_meta_lora, 921_600),
        "vl_meta_lora": (model.semantic_encoder.vl_meta_lora, 921_600),
        "action_meta_lora": (model.semantic_encoder.action_meta_lora, 626_688),
        "language_projection": (model.semantic_encoder.language_projection, 524_288),
        "patch_grounding": (model.semantic_encoder.patch_grounding, 197_120),
        "interaction_projection": (
            model.semantic_encoder.interaction_projection,
            262_144,
        ),
        "semantic_core": (model.semantic_core, 1_836_544),
        "semantic_program": (model.semantic_program, 1_838_592),
        "compiler": (model.compiler, 409_088),
        "factor_heads": (model.factor_heads, 2_703_360),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad


def test_ap_adr_starts_at_exact_public_lora_identity() -> None:
    model, template = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_target_ordinals_follow_sealed_policy_and_writer_becomes_conditioned() -> None:
    model, _ = _model()
    observed = {}
    for item in model.tensor_specs:
        observed.setdefault(item.module, model._decoding[item.name][1])
    assert tuple(
        module for module, _ in sorted(observed.items(), key=lambda row: row[1])
    ) == pi05_target_names()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_gradient_staging_reaches_core_program_readers_and_frontend() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    first = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in first.values()).backward()
    assert all(
        head.network[-1].weight.grad is not None
        and bool(torch.count_nonzero(head.network[-1].weight.grad))
        for head in model.factor_heads.values()
    )
    for module in (model.semantic_core, model.semantic_program, model.compiler):
        assert all(
            parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )

    model.zero_grad(set_to_none=True)
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    second = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in second.values()).backward()
    for module in (model.semantic_core, model.semantic_program, model.compiler):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )


def test_task_queried_patch_grounding_uses_unordered_raw_patch_values() -> None:
    torch.manual_seed(23)
    grounding = TaskQueriedPatchGrounding(width=32, heads=4)
    queries = torch.randn(2, 5, 32)
    patches = torch.randn(2, 256, 32)
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    baseline = grounding(queries, patches, valid)
    permuted = grounding(queries, patches[:, torch.randperm(256)], valid)
    changed = grounding(queries, patches + 0.25, valid)
    assert torch.allclose(baseline, permuted, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(baseline, changed)
    assert not bool(baseline[0, 3:].count_nonzero())
    assert not hasattr(grounding, "value")


def test_semantic_core_is_strictly_frame_set_permutation_invariant() -> None:
    torch.manual_seed(31)
    core = MeanBackedSemanticCore(width=32, heads=4, blocks=2)
    text = torch.randn(2, 4, 32)
    evidence = torch.randn(2, 6, 4, 32)
    valid_frames = torch.tensor(
        [[True, True, True, True, False, False], [True] * 6]
    )
    valid_tokens = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    baseline = core(text, evidence, valid_frames, valid_tokens)[0]
    permuted = core(
        text,
        evidence[:, permutation],
        valid_frames[:, permutation],
        valid_tokens,
    )[0]
    assert torch.allclose(baseline, permuted, atol=2e-6, rtol=1e-5)


def test_outgoing_program_preserves_causal_prefix_and_uses_order() -> None:
    torch.manual_seed(23)
    program = OutgoingSemanticProgram(
        width=32, heads=4, blocks=2, initialization_seed=7
    )
    grounded = torch.randn(1, 6, 4, 32)
    action = torch.randn(1, 6, 32)
    future_grounded = grounded.clone()
    future_action = action.clone()
    future_grounded[:, 4:] = torch.randn_like(future_grounded[:, 4:])
    future_action[:, 4:] = torch.randn_like(future_action[:, 4:])
    positions = torch.arange(6)[None]
    valid_frames = torch.ones(1, 6, dtype=torch.bool)
    valid_tokens = torch.ones(1, 4, dtype=torch.bool)
    baseline = program(grounded, action, positions, valid_frames, valid_tokens)[0]
    future = program(
        future_grounded, future_action, positions, valid_frames, valid_tokens
    )[0]
    reverse = program(
        grounded.flip(1), action.flip(1), positions, valid_frames, valid_tokens
    )[0]
    assert torch.allclose(baseline[:, :3], future[:, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(baseline, reverse)


def test_outgoing_program_aligns_action_endpoint_effect_and_change() -> None:
    program = OutgoingSemanticProgram(
        width=32, heads=4, blocks=1, initialization_seed=7
    )
    program.blocks = torch.nn.ModuleList()
    grounded = torch.randn(1, 5, 3, 32)
    action = torch.randn(1, 5, 32)
    positions = torch.tensor([[0, 5, 10, 15, 17]])
    valid_frames = torch.ones(1, 5, dtype=torch.bool)
    valid_tokens = torch.tensor([[True, True, False]])
    key, value, endpoints, valid_intervals, valid_semantics = program(
        grounded, action, positions, valid_frames, valid_tokens
    )
    assert torch.equal(key, value)
    assert torch.equal(value[:, :, 0], action[:, :-1])
    assert torch.equal(value[:, :, 1:3], grounded[:, 1:, :2])
    assert not bool(value[:, :, 3].count_nonzero())
    assert torch.equal(
        value[:, :, 4:6], grounded[:, 1:, :2] - grounded[:, :-1, :2]
    )
    assert not bool(value[:, :, 6].count_nonzero())
    assert torch.equal(endpoints, positions[:, 1:])
    assert torch.equal(valid_intervals, torch.ones_like(valid_intervals))
    assert torch.equal(
        valid_semantics,
        torch.tensor([[True, True, True, False, True, True, False]]),
    )


def _dual_reader(width: int = 32, targets: int = 38, rank: int = 16) -> AsymmetricDualReader:
    return AsymmetricDualReader(
        width=width,
        heads=4,
        target_count=targets,
        rank=rank,
        initialization_seed=7,
    )


def test_identities_cannot_create_core_program_or_coordinate_values() -> None:
    core = MeanBackedSemanticCore(width=32, heads=4, blocks=2)
    valid_frames = torch.ones(2, 6, dtype=torch.bool)
    valid_tokens = torch.ones(2, 4, dtype=torch.bool)
    core_value = core(
        torch.randn(2, 4, 32),
        torch.zeros(2, 6, 4, 32),
        valid_frames,
        valid_tokens,
    )[0]
    assert not bool(core_value.count_nonzero())
    program = OutgoingSemanticProgram(
        width=32, heads=4, blocks=2, initialization_seed=7
    )
    grounded = torch.zeros(2, 6, 4, 32)
    action = torch.zeros(2, 6, 32)
    positions = torch.tensor(
        [[0, 5, 10, 15, 20, 25], [0, 3, 8, 13, 21, 28]]
    )
    memory = program(grounded, action, positions, valid_frames, valid_tokens)
    assert not bool(memory[0].count_nonzero())
    assert not bool(memory[1].count_nonzero())
    coordinates = _dual_reader()(
        core_value, valid_tokens, *memory
    )
    assert coordinates.shape == (2, 38, 16, 64)
    assert not bool(coordinates.count_nonzero())


def test_dual_reader_has_target_only_core_and_target_rank_program_reads() -> None:
    torch.manual_seed(37)
    reader = _dual_reader(targets=5, rank=4)
    core = torch.randn(1, 3, 32)
    valid_core = torch.ones(1, 3, dtype=torch.bool)
    key = torch.randn(1, 4, 7, 32)
    value = torch.randn(1, 4, 7, 32)
    endpoints = torch.arange(4)[None]
    intervals = torch.ones(1, 4, dtype=torch.bool)
    semantics = torch.ones(1, 7, dtype=torch.bool)
    baseline, diagnostics = reader.compile_with_diagnostics(
        core, valid_core, key, value, endpoints, intervals, semantics
    )
    assert diagnostics["core_read"].shape == (1, 5, 32)
    assert diagnostics["program_read"].shape == (1, 5, 4, 32)
    assert torch.equal(
        baseline[..., :32],
        diagnostics["core_read"][:, :, None].expand(-1, -1, 4, -1),
    )
    assert not hasattr(reader, "gate")
    assert not hasattr(reader, "mixer")

    target_swap = torch.tensor([1, 0, 2, 3, 4])
    with torch.no_grad():
        reader.target_identity.copy_(reader.target_identity[target_swap])
    target_permuted = reader(
        core, valid_core, key, value, endpoints, intervals, semantics
    )
    assert torch.allclose(
        target_permuted, baseline[:, target_swap], atol=1e-6, rtol=1e-5
    )
    rank_swap = torch.tensor([2, 1, 0, 3])
    with torch.no_grad():
        reader.rank_identity.copy_(reader.rank_identity[rank_swap])
    both_permuted = reader(
        core, valid_core, key, value, endpoints, intervals, semantics
    )
    assert torch.allclose(
        both_permuted,
        target_permuted[:, :, rank_swap],
        atol=1e-6,
        rtol=1e-5,
    )


def test_program_reader_preserves_raw_value_amplitude_with_fixed_keys() -> None:
    torch.manual_seed(41)
    reader = _dual_reader(targets=3, rank=2)
    core = torch.randn(1, 4, 32)
    valid_core = torch.ones(1, 4, dtype=torch.bool)
    key = torch.randn(1, 5, 7, 32)
    value = torch.randn(1, 5, 7, 32)
    endpoints = torch.arange(5)[None]
    intervals = torch.ones(1, 5, dtype=torch.bool)
    semantics = torch.ones(1, 7, dtype=torch.bool)
    first = reader.compile_with_diagnostics(
        core, valid_core, key, value, endpoints, intervals, semantics
    )[1]["program_read"]
    doubled = reader.compile_with_diagnostics(
        core, valid_core, key, 2.0 * value, endpoints, intervals, semantics
    )[1]["program_read"]
    assert torch.allclose(doubled, 2.0 * first, atol=2e-6, rtol=1e-5)


def test_core_program_and_reader_ignore_ragged_padding_content() -> None:
    torch.manual_seed(43)
    core = MeanBackedSemanticCore(width=32, heads=4, blocks=2)
    program = OutgoingSemanticProgram(
        width=32, heads=4, blocks=2, initialization_seed=7
    )
    reader = _dual_reader()
    text = torch.randn(1, 4, 32)
    evidence = torch.randn(1, 6, 4, 32)
    grounded = torch.randn(1, 6, 4, 32)
    action = torch.randn(1, 6, 32)
    changed_evidence = evidence.clone()
    changed_grounded = grounded.clone()
    changed_action = action.clone()
    changed_evidence[:, 4:] = 100.0 * torch.randn_like(changed_evidence[:, 4:])
    changed_grounded[:, 4:] = 100.0 * torch.randn_like(changed_grounded[:, 4:])
    changed_action[:, 4:] = 100.0 * torch.randn_like(changed_action[:, 4:])
    positions = torch.tensor([[0, 5, 10, 15, 0, 0]])
    valid_frames = torch.tensor([[True, True, True, True, False, False]])
    valid_tokens = torch.ones(1, 4, dtype=torch.bool)
    base_core = core(text, evidence, valid_frames, valid_tokens)[0]
    padded_core = core(text, changed_evidence, valid_frames, valid_tokens)[0]
    base_program = program(
        grounded, action, positions, valid_frames, valid_tokens
    )
    padded_program = program(
        changed_grounded, changed_action, positions, valid_frames, valid_tokens
    )
    assert torch.allclose(base_core, padded_core, atol=2e-6, rtol=1e-5)
    assert torch.allclose(base_program[0], padded_program[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(
        reader(base_core, valid_tokens, *base_program),
        reader(padded_core, valid_tokens, *padded_program),
        atol=2e-6,
        rtol=1e-5,
    )


def test_dual_reader_rejects_missing_content_and_float_endpoints() -> None:
    reader = _dual_reader()
    core = torch.randn(1, 2, 32)
    valid_core = torch.ones(1, 2, dtype=torch.bool)
    key = torch.randn(1, 2, 3, 32)
    value = torch.randn_like(key)
    endpoints = torch.tensor([[5, 10]])
    intervals = torch.ones(1, 2, dtype=torch.bool)
    semantics = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(SemanticProgramError, match="dual-reader memory"):
        reader(
            core,
            valid_core,
            key,
            value,
            endpoints.to(torch.float32),
            intervals,
            semantics,
        )
    with pytest.raises(SemanticProgramError, match="dual-reader memory"):
        reader(
            core,
            valid_core,
            key,
            value,
            endpoints,
            torch.zeros_like(intervals),
            semantics,
        )


def test_internal_analyzer_recomputes_canonical_ap_path_and_counterfactuals() -> None:
    model, _ = _model()
    model.semantic_encoder = _AnalysisSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.002)
    model.eval()
    captured = capture_writer(
        model,
        torch.nn.Identity(),
        *_analysis_inputs(),
    )
    assert captured["a_raw"].shape == (5, 2, 1024)
    assert captured["a"].shape == (5, 2, 256)
    assert captured["program"]["value"].shape[2] == 7
    assert captured["compiled"]["coordinates"].shape == (5, 38, 16, 512)
    assert captured["compiled"]["recomputed_coordinates"].shape == (
        5,
        38,
        16,
        512,
    )
    assert captured["parity"]["public"]["relative_l2"] <= 2e-5
    assert captured["parity"]["compiler_coordinates"]["relative_l2"] <= 2e-5
    assert captured["compiled"]["parity"]["core_read"]["relative_l2"] <= 2e-5
    assert captured["compiled"]["parity"]["program_read"]["relative_l2"] <= 2e-5
    for block in captured["program"]["attention"]:
        assert block["interval_local"]["probability_sum_error_max"] < 1e-5
        assert block["semantic_column_causal"]["probability_sum_error_max"] < 1e-5

    variants = counterfactual_states(model, captured)
    required = {
        "full",
        "core_only",
        "program_only",
        "aed/A",
        "aed/E",
        "aed/D",
        "aed/A+E+D",
        "aed_fixed_key/A",
        "scale/A/0.5",
        "scale/A/1",
        "scale/A/2",
        "core_carrier/no_mean",
        "core_carrier/no_centered",
        "identity/target",
        "identity/rank",
        "temporal_keys/order_permuted",
    }
    assert required <= set(variants)
    for name in ("aed/A+E+D", "scale/A/1", "scale/E/1", "scale/D/1"):
        assert effective_metrics(
            model, variants["full"]["public"], variants[name]["public"]
        )["relative_l2"] <= 2e-5
    assert any(
        effective_metrics(
            model,
            variants["full"]["public"],
            variants[name]["public"],
        )["relative_l2"]
        > 1e-6
        for name in ("core_only", "program_only", "aed/A", "identity/target")
    )
    authority = variants["full"]["temporal_key_authority"]
    assert authority["initialization_keys"]["status"] == "unsupported"
    assert authority["initialization_keys"]["fail_closed"] is True
    assert len(authority["trained_program_state_sha256"]) == 64
    for name in ("full", "temporal_keys/order_permuted"):
        routing = variants[name]["attention"]["program_target_rank_routing"]
        assert routing["target_centered_energy"] >= 0
        assert routing["rank_centered_energy"] >= 0


def test_internal_analyzer_public_rank_gauge_preserves_effective_ba() -> None:
    model, template = _model()
    generator = torch.Generator().manual_seed(91)
    state = {
        name: value + 0.01 * torch.randn(value.shape, generator=generator)
        for name, value in template.items()
    }
    permutation = torch.roll(torch.arange(16), -1)
    permuted, changes = rank_gauge_permute(model, state, permutation)
    assert effective_metrics(model, state, permuted)["relative_l2"] < 2e-5
    assert all(
        value["public_a"]["relative_l2"] > 0
        and value["public_b"]["relative_l2"] > 0
        for value in changes.values()
    )
