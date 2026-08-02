from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.pi05_lora import load_pi05_lora_contract, pi05_target_names
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.internal_analysis import (
    _paired_diagnostics,
    capture_writer,
    counterfactual_states,
)
from ember.writer.internal_metrics import effective_metrics, rank_gauge_permute
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.program_compiler import SemanticProgramError, TargetBoundRoleCompiler
from ember.writer.semantic_core import MeanBackedSemanticCore
from ember.writer.semantic_program import TargetBoundRoleProgram
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


def test_target_bound_role_parameter_budget_and_modules_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_092_224
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 11_092_224
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
        "semantic_program": (model.semantic_program, 1_641_216),
        "compiler": (model.compiler, 409_088),
        "factor_heads": (model.factor_heads, 3_751_936),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad


def test_target_bound_role_starts_at_exact_public_lora_identity() -> None:
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


def _target_context(
    batch: int,
    targets: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, targets, width), torch.randn(batch, targets, width)


def test_target_role_program_preserves_core_conditional_prefix_and_uses_order() -> None:
    torch.manual_seed(23)
    program = TargetBoundRoleProgram(
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
    target_query, target_core = _target_context(1, 5, 32)
    arguments = (valid_frames, valid_tokens, target_query, target_core)
    baseline = program(grounded, action, positions, *arguments)[0]
    future = program(
        future_grounded, future_action, positions, *arguments
    )[0]
    reverse = program(
        grounded.flip(1), action.flip(1), positions, *arguments
    )[0]
    assert baseline.shape == (1, 5, 5, 3, 32)
    assert torch.allclose(baseline[:, :, :3], future[:, :, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(baseline, reverse)


class _CaptureEvidenceReader(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memories: list[torch.Tensor] = []

    def forward(
        self,
        address: torch.Tensor,
        memory: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        _role_identity: torch.Tensor,
    ) -> torch.Tensor:
        self.memories.append(memory.detach().clone())
        mask = valid_task_tokens[:, None, :, None]
        mean = memory.masked_fill(~mask, 0.0).sum(dim=2)
        mean = mean / mask.sum(dim=2).clamp_min(1)
        return mean[:, None].expand_as(address)


def test_target_role_program_aligns_action_endpoint_effect_and_change() -> None:
    program = TargetBoundRoleProgram(
        width=32, heads=4, blocks=1, initialization_seed=7
    )
    program.blocks = torch.nn.ModuleList()
    capture = _CaptureEvidenceReader()
    program.evidence_reader = capture
    grounded = torch.randn(1, 5, 3, 32)
    action = torch.randn(1, 5, 32)
    positions = torch.tensor([[0, 5, 10, 15, 17]])
    valid_frames = torch.ones(1, 5, dtype=torch.bool)
    valid_tokens = torch.tensor([[True, True, False]])
    target_query, target_core = _target_context(1, 4, 32)
    memory, endpoints, valid_intervals = program(
        grounded,
        action,
        positions,
        valid_frames,
        valid_tokens,
        target_query,
        target_core,
    )
    assert torch.equal(memory[:, :, :, 0], action[:, None, :-1].expand(-1, 4, -1, -1))
    assert torch.equal(capture.memories[0], grounded[:, 1:])
    assert torch.equal(capture.memories[1], grounded[:, 1:] - grounded[:, :-1])
    assert torch.equal(endpoints, positions[:, 1:])
    assert torch.equal(valid_intervals, torch.ones_like(valid_intervals))


def _compiler(
    width: int = 32,
    targets: int = 38,
    rank: int = 16,
) -> TargetBoundRoleCompiler:
    return TargetBoundRoleCompiler(
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
    compiler = _compiler()
    target_query, target_core = compiler.read_target_core(core_value, valid_tokens)
    program = TargetBoundRoleProgram(
        width=32, heads=4, blocks=2, initialization_seed=7
    )
    grounded = torch.zeros(2, 6, 4, 32)
    action = torch.zeros(2, 6, 32)
    positions = torch.tensor(
        [[0, 5, 10, 15, 20, 25], [0, 3, 8, 13, 21, 28]]
    )
    memory = program(
        grounded,
        action,
        positions,
        valid_frames,
        valid_tokens,
        target_query,
        target_core,
    )
    assert not bool(memory[0].count_nonzero())
    coordinates = compiler(target_query, target_core, *memory)
    assert coordinates.shape == (2, 38, 16, 128)
    assert not bool(coordinates.count_nonzero())


def test_compiler_has_target_core_and_private_target_rank_role_reads() -> None:
    torch.manual_seed(37)
    reader = _compiler(targets=5, rank=4)
    core = torch.randn(1, 3, 32)
    valid_core = torch.ones(1, 3, dtype=torch.bool)
    target_query, target_core = reader.read_target_core(core, valid_core)
    program = torch.randn(1, 5, 4, 3, 32)
    endpoints = torch.arange(4)[None]
    intervals = torch.ones(1, 4, dtype=torch.bool)
    baseline, diagnostics = reader.compile_with_diagnostics(
        target_query, target_core, program, endpoints, intervals
    )
    assert diagnostics["core_read"].shape == (1, 5, 32)
    assert diagnostics["role_read"].shape == (1, 5, 4, 3, 32)
    assert torch.equal(
        baseline[..., :32],
        diagnostics["core_read"][:, :, None].expand(-1, -1, 4, -1),
    )
    assert not hasattr(reader, "gate")
    assert not hasattr(reader, "mixer")

    target_swap = torch.tensor([1, 0, 2, 3, 4])
    target_permuted = reader(
        target_query[:, target_swap],
        target_core[:, target_swap],
        program[:, target_swap],
        endpoints,
        intervals,
    )
    assert torch.allclose(
        target_permuted, baseline[:, target_swap], atol=1e-6, rtol=1e-5
    )
    rank_swap = torch.tensor([2, 1, 0, 3])
    with torch.no_grad():
        reader.rank_identity.copy_(reader.rank_identity[rank_swap])
    both_permuted = reader(
        target_query[:, target_swap],
        target_core[:, target_swap],
        program[:, target_swap],
        endpoints,
        intervals,
    )
    assert torch.allclose(
        both_permuted,
        target_permuted[:, :, rank_swap],
        atol=1e-6,
        rtol=1e-5,
    )


def test_private_rank_reads_reach_each_role_without_value_cross_talk() -> None:
    torch.manual_seed(39)
    reader = _compiler(width=32, targets=3, rank=2)
    core = torch.randn(1, 4, 32)
    valid_core = torch.ones(1, 4, dtype=torch.bool)
    target_query, target_core = reader.read_target_core(core, valid_core)
    program = torch.randn(
        1, 3, 5, 3, 32, requires_grad=True
    )
    coordinates = reader(
        target_query,
        target_core,
        program,
        torch.arange(5)[None],
        torch.ones(1, 5, dtype=torch.bool),
    )
    for role in range(3):
        left = 32 * (role + 1)
        selected = coordinates[..., left : left + 32].square().sum()
        gradient = torch.autograd.grad(
            selected,
            program,
            retain_graph=role < 2,
        )[0]
        assert torch.isfinite(gradient).all()
        assert bool(gradient[..., role, :].count_nonzero())
        assert not bool(
            gradient[
                ...,
                [other for other in range(3) if other != role],
                :,
            ].count_nonzero()
        )


def test_role_readers_accept_one_contextual_memory_for_keys_and_values() -> None:
    torch.manual_seed(41)
    reader = _compiler(targets=3, rank=2)
    core = torch.randn(1, 4, 32)
    valid_core = torch.ones(1, 4, dtype=torch.bool)
    target_query, target_core = reader.read_target_core(core, valid_core)
    program = torch.randn(1, 3, 5, 3, 32)
    endpoints = torch.arange(5)[None]
    intervals = torch.ones(1, 5, dtype=torch.bool)
    first = reader.compile_with_diagnostics(
        target_query, target_core, program, endpoints, intervals
    )[1]["role_read"]
    changed = reader.compile_with_diagnostics(
        target_query, target_core, 2.0 * program, endpoints, intervals
    )[1]["role_read"]
    assert not torch.allclose(changed, first)
    with pytest.raises(TypeError):
        reader.compile_with_diagnostics(
            target_query,
            target_core,
            program,
            torch.randn_like(program),
            endpoints,
            intervals,
        )


def test_core_program_and_reader_ignore_ragged_padding_content() -> None:
    torch.manual_seed(43)
    core = MeanBackedSemanticCore(width=32, heads=4, blocks=2)
    program = TargetBoundRoleProgram(
        width=32, heads=4, blocks=2, initialization_seed=7
    )
    reader = _compiler()
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
    base_query, base_target_core = reader.read_target_core(base_core, valid_tokens)
    padded_query, padded_target_core = reader.read_target_core(
        padded_core, valid_tokens
    )
    base_program = program(
        grounded,
        action,
        positions,
        valid_frames,
        valid_tokens,
        base_query,
        base_target_core,
    )
    padded_program = program(
        changed_grounded,
        changed_action,
        positions,
        valid_frames,
        valid_tokens,
        padded_query,
        padded_target_core,
    )
    assert torch.allclose(base_core, padded_core, atol=2e-6, rtol=1e-5)
    assert torch.allclose(base_program[0], padded_program[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(
        reader(base_query, base_target_core, *base_program),
        reader(padded_query, padded_target_core, *padded_program),
        atol=2e-6,
        rtol=1e-5,
    )


def test_role_compiler_rejects_missing_content_and_float_endpoints() -> None:
    reader = _compiler()
    core = torch.randn(1, 2, 32)
    valid_core = torch.ones(1, 2, dtype=torch.bool)
    target_query, target_core = reader.read_target_core(core, valid_core)
    program = torch.randn(1, 38, 2, 3, 32)
    endpoints = torch.tensor([[5, 10]])
    intervals = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(SemanticProgramError, match="role compiler memory"):
        reader(
            target_query,
            target_core,
            program,
            endpoints.to(torch.float32),
            intervals,
        )
    with pytest.raises(SemanticProgramError, match="role compiler memory"):
        reader(
            target_query,
            target_core,
            program,
            endpoints,
            torch.zeros_like(intervals),
        )


def test_internal_analyzer_recomputes_target_bound_role_path() -> None:
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
    assert captured["program"]["memory"].shape == (5, 38, 1, 3, 256)
    assert captured["compiled"]["coordinates"].shape == (5, 38, 16, 1024)
    assert captured["compiled"]["recomputed_coordinates"].shape == (
        5,
        38,
        16,
        1024,
    )
    for key, targets in captured["decoded"]["head_target_indices"].items():
        expected = tuple(
            sorted(
                target
                for spec in model.tensor_specs
                for owner, target in (model._decoding[spec.name],)
                if owner == key
            )
        )
        assert targets == expected
        assert captured["decoded"]["heads"][key].shape[1] == len(expected)
    assert captured["parity"]["public"]["relative_l2"] <= 2e-5
    assert captured["parity"]["compiler_coordinates"]["relative_l2"] <= 2e-5
    assert captured["compiled"]["parity"]["core_read"]["relative_l2"] <= 2e-5
    assert captured["compiled"]["parity"]["role_read"]["relative_l2"] <= 2e-5
    for block in captured["program"]["attention"]:
        assert block["probability_sum_error_max"] < 1e-5

    variants = counterfactual_states(model, captured)
    required = {
        "full",
        "coordinate/core_only",
        "coordinate/program_only",
        "program_role/remove_A",
        "program_role/remove_E",
        "program_role/remove_D",
        "program_role/A_only",
        "program_role/E_only",
        "program_role/D_only",
        "program_input/remove_A",
        "program_input/remove_E",
        "program_input/remove_D",
        "action_router/zero",
        "core_carrier/no_mean",
        "core_carrier/no_centered",
        "identity/target",
        "identity/rank",
        "program_memory/order_reversed",
    }
    assert required <= set(variants)
    assert any(
        effective_metrics(
            model,
            variants["full"]["public"],
            variants[name]["public"],
        )["relative_l2"]
        > 1e-6
        for name in (
            "coordinate/core_only",
            "coordinate/program_only",
            "program_role/remove_A",
            "program_input/remove_D",
            "identity/target",
        )
    )
    authority = variants["full"]["program_memory_authority"]
    assert "private softmax" in authority["key_value_coupling"]
    assert len(authority["trained_program_state_sha256"]) == 64
    for name in ("full", "program_memory/order_reversed"):
        routing = variants[name]["attention"]["program_target_rank_routing"]
        assert set(routing) == {"A", "E", "D"}
        assert all(
            value["target_centered_energy"] >= 0
            and value["rank_centered_energy"] >= 0
            for value in routing.values()
        )
    paired = _paired_diagnostics(
        model,
        captured,
        [torch.randn(1, 5, 7) for _ in range(5)],
    )
    assert set(paired["comparisons"]) == {
        "correct",
        "same_task_other",
        "cross_suite_wrong",
        "shuffled",
        "reversed",
    }
    assert "memory_to_role_read" in paired["comparisons"]["reversed"][
        "change_retention"
    ]


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
