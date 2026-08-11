from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from safetensors.torch import save_file

from ember.expert_manifold.contract import ExpertTask
from ember.expert_manifold.v6_prior import (
    COUNTERFACTUAL_KINDS,
    counterfactual_frame_order,
    counterfactual_kind,
    cross_suite_wrong_task,
    freeze_v6_prior_writer,
    load_v6_prior_warm_start_,
)
from ember.writer.architecture import V6_WRITER_PARAMETER_COUNT
from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    SlotNormalizedCoreProcedureCompiler,
    TaskGroundedVisualTransitionFusion,
    TaskSelectedSemanticSetFusion,
)
from ember.writer.video_program import TaskQueriedPatchGrounding, VideoProgramError
from fixtures.writer_model import _FakeSemanticEncoder, _inputs, _model


def test_v6_writer_parameter_budget_and_fixed_probe_noise_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        V6_WRITER_PARAMETER_COUNT
    )
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
        "semantic_set_fusion": (
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
        "compiler": (model.compiler, 1_535_232),
        "factor_heads": (model.factor_heads, 2_179_072),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert "semantic_encoder.fixed_suffix_noise" in model.state_dict()
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad


def test_v6_prior_strict_warm_start_and_all_frozen_contract(tmp_path) -> None:
    source, _ = _model()
    checkpoint = tmp_path / "writer.safetensors"
    state = {
        name: value.detach().contiguous() for name, value in source.state_dict().items()
    }
    save_file(state, str(checkpoint))
    target, _ = _model()
    for parameter in target.parameters():
        torch.nn.init.zeros_(parameter)
    warm_start = load_v6_prior_warm_start_(target, checkpoint)
    assert warm_start.state_tensor_count == 600
    assert all(
        torch.equal(value, target.state_dict()[name]) for name, value in state.items()
    )
    ownership = freeze_v6_prior_writer(target)
    assert ownership.frozen_parameter_count == V6_WRITER_PARAMETER_COUNT
    assert ownership.trainable_parameter_count == 0
    assert ownership.state_tensor_count == 600
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert not target.training


def test_v6_prior_counterfactual_schedule_is_balanced_and_temporal() -> None:
    for visit in range(5):
        kinds = [counterfactual_kind(task, visit) for task in range(24)]
        assert {kind: kinds.count(kind) for kind in COUNTERFACTUAL_KINDS} == {
            kind: 8 for kind in COUNTERFACTUAL_KINDS
        }
    offsets = (0, 5, 9)
    reverse = counterfactual_frame_order(
        "reversed",
        offsets,
        seed=7,
        task_ordinal=2,
        task_visit=3,
        teacher_demo=11,
        device="cpu",
    )
    assert torch.equal(reverse, torch.tensor([4, 3, 2, 1, 0, 8, 7, 6, 5]))
    shuffled = counterfactual_frame_order(
        "shuffled",
        offsets,
        seed=7,
        task_ordinal=2,
        task_visit=3,
        teacher_demo=11,
        device="cpu",
    )
    assert shuffled is not None
    assert torch.equal(shuffled[:5].sort().values, torch.arange(5))
    assert torch.equal(shuffled[5:].sort().values, torch.arange(5, 9))
    assert not torch.equal(shuffled, torch.arange(9))
    assert (
        counterfactual_frame_order(
            "wrong",
            offsets,
            seed=7,
            task_ordinal=2,
            task_visit=3,
            teacher_demo=11,
            device="cpu",
        )
        is None
    )


def test_v6_prior_wrong_video_cycles_across_suites() -> None:
    suites = ("spatial", "object", "goal", "libero10")
    tasks = tuple(
        ExpertTask(
            ordinal=ordinal,
            global_task_id=ordinal,
            suite=suites[ordinal // 6],
            task_id=ordinal % 6,
            split_role="train",
            language=f"task {ordinal}",
            authority=object(),
        )
        for ordinal in range(24)
    )
    source = tasks[7]
    selected = [
        cross_suite_wrong_task(tasks, task_ordinal=source.ordinal, task_visit=visit)
        for visit in range(9)
    ]
    assert all(task.suite != source.suite for task in selected)
    assert {task.suite for task in selected} == set(suites) - {source.suite}


def test_v6_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_v6_writer_becomes_video_conditioned_after_heads_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def test_staged_evidence_path_matches_forward_and_reorders_only_temporal_memory() -> (
    None
):
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    torch.nn.init.normal_(model.compiler.modulation.weight, std=0.01)
    for head in model.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    frames, indices, offsets, tokens, masks, spans = _inputs()
    direct = model(
        frames,
        indices,
        offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    evidence = model.encode_video_evidence(
        torch.nn.Identity(),
        frames,
        offsets,
        tokens,
        masks,
        spans,
    )
    normal_memory = model.build_memories(evidence, indices)
    staged = model.decode_memories(normal_memory)
    slots = model.compile_slots(normal_memory)
    split = model.decode_slots(slots)
    assert all(torch.equal(direct[name], staged[name]) for name in direct)
    assert all(torch.equal(staged[name], split[name]) for name in staged)

    reversed_order = torch.tensor([1, 0, 4, 3, 2], dtype=torch.long)
    reversed_memory = model.build_memories(
        evidence,
        indices,
        frame_order=reversed_order,
    )
    reversed_state = model.decode_memories(reversed_memory)
    assert torch.allclose(
        normal_memory.core,
        reversed_memory.core,
        atol=1e-5,
        rtol=1e-5,
    )
    assert not torch.allclose(normal_memory.procedure, reversed_memory.procedure)
    assert any(
        not torch.allclose(staged[name], reversed_state[name]) for name in staged
    )


def test_staged_frame_order_cannot_cross_video_conditions() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    frames, indices, offsets, tokens, masks, spans = _inputs()
    evidence = model.encode_video_evidence(
        torch.nn.Identity(),
        frames,
        offsets,
        tokens,
        masks,
        spans,
    )
    with pytest.raises(WriterModelError, match="crossed video conditions"):
        model.build_memories(
            evidence,
            indices,
            frame_order=torch.tensor([0, 2, 1, 3, 4], dtype=torch.long),
        )


def test_writer_offsets_and_frame_ordinals_fail_closed_at_batch_boundary() -> None:
    assert CompleteLoRAWriter._validated_offsets(
        torch.tensor([0, 2, 5], dtype=torch.long),
        5,
    ) == (0, 2, 5)
    for invalid in (
        torch.tensor([0, 2, 5], dtype=torch.float32),
        torch.tensor([0, 0, 5], dtype=torch.long),
        torch.tensor([0, 2, 4], dtype=torch.long),
    ):
        with pytest.raises(WriterModelError, match="offsets are invalid"):
            CompleteLoRAWriter._validated_offsets(invalid, 5)

    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    frames, indices, offsets, tokens, masks, spans = _inputs()
    evidence = model.encode_video_evidence(
        torch.nn.Identity(), frames, offsets, tokens, masks, spans
    )
    for invalid in (
        torch.tensor([0, 5, 1, 5, 10], dtype=torch.long),
        torch.tensor([0, 5, 0, 5, 5], dtype=torch.long),
    ):
        with pytest.raises(WriterModelError, match="start at zero and increase"):
            model.build_memories(evidence, invalid)


def test_semantic_batch_validates_all_value_ownership_in_one_outer_gate() -> None:
    model, _ = _model()
    encoder = model.semantic_encoder
    frames, _, _, tokens, masks, spans = _inputs()
    condition_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    policy = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32))
    )
    _, valid, counts = encoder._validate_forward_batch(
        policy,
        frames,
        condition_ids,
        tokens,
        masks,
        spans,
    )
    assert valid.shape == (2, 4)
    assert counts.tolist() == [2, 4]

    invalid_spans = spans.clone()
    invalid_spans[0, 5] = True
    empty_spans = spans.clone()
    empty_spans[1] = False
    bos_spans = spans.clone()
    bos_spans[0, 0] = True
    cases = (
        (condition_ids, invalid_spans),
        (condition_ids, empty_spans),
        (condition_ids, bos_spans),
        (torch.tensor([0, 1, 0, 1, 1]), spans),
        (torch.zeros(5, dtype=torch.long), spans),
    )
    for invalid_conditions, invalid_task_spans in cases:
        with pytest.raises(
            VideoProgramError,
            match="invalid frame-language semantic batch",
        ):
            encoder._validate_forward_batch(
                policy,
                frames,
                invalid_conditions,
                tokens,
                masks,
                invalid_task_spans,
            )


def test_v6_gradient_staging_opens_only_intended_paths() -> None:
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
    assert model.compiler.modulation.weight.grad is not None
    assert bool(torch.count_nonzero(model.compiler.modulation.weight.grad))
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.semantic_core.parameters()
    )
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
        for parameter in model.procedure.parameters()
    )
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
        for parameter in model.visual_transition.parameters()
    )

    model.zero_grad(set_to_none=True)
    torch.nn.init.normal_(model.compiler.modulation.weight, std=0.01)
    third = model(*_inputs(), policy=torch.nn.Identity())
    sum(value.to(torch.float32).sum() for value in third.values()).backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.procedure.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.visual_transition.parameters()
    )


def test_task_queried_patch_grounding_uses_patch_content_without_order_geometry() -> (
    None
):
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
    text = torch.randn(2, 7, 32)
    evidence = torch.randn(2, 5, 7, 32)
    valid_frames = torch.ones(2, 5, dtype=torch.bool)
    valid_tokens = torch.ones(2, 7, dtype=torch.bool)
    baseline, _ = core(text, evidence, valid_frames, valid_tokens)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    shuffled, _ = core(
        text,
        evidence[:, permutation],
        valid_frames[:, permutation],
        valid_tokens,
    )
    assert torch.allclose(baseline, shuffled, atol=1e-5, rtol=1e-5)


def test_semantic_set_fusion_has_mean_backbone_and_raw_centered_values() -> None:
    torch.manual_seed(18)
    fusion = TaskSelectedSemanticSetFusion(width=32, heads=4)
    text = torch.randn(1, 3, 32)
    evidence = torch.randn(1, 4, 3, 32)
    valid_frames = torch.tensor([[True, True, True, False]])
    valid_tokens = torch.tensor([[True, True, False]])
    output, weights = fusion(
        text,
        evidence,
        valid_frames,
        valid_tokens,
    )
    assert output.shape == (1, 3, 32)
    assert weights.shape == (1, 4, 4, 3)
    assert not bool(output[:, 2].count_nonzero())
    assert not bool(weights[:, :, :, 2].count_nonzero())
    assert not hasattr(fusion, "value")
    assert not hasattr(fusion, "gate_logits")
    full_width = TaskSelectedSemanticSetFusion(width=256, heads=8)
    assert sum(parameter.numel() for parameter in full_width.parameters()) == 262_656


def test_visual_transition_recomputes_after_order_change_without_static_value_path() -> (
    None
):
    torch.manual_seed(19)
    fusion = TaskGroundedVisualTransitionFusion(width=32, heads=4)
    probe = torch.randn(1, 4, 32)
    grounded = torch.randn(1, 4, 3, 32, requires_grad=True)
    valid_frames = torch.tensor([[True, True, True, False]])
    valid_tokens = torch.tensor([[True, True, False]])
    normal, transition = fusion(
        probe,
        grounded,
        valid_frames,
        valid_tokens,
    )
    reversed_output, reversed_transition = fusion(
        probe,
        grounded.flip(1),
        valid_frames,
        valid_tokens,
    )
    assert not bool(transition[:, 0].count_nonzero())
    assert torch.allclose(normal[:, 0], probe[:, 0])
    assert torch.allclose(
        transition[:, 1, :2],
        grounded[:, 1, :2] - grounded[:, 0, :2],
    )
    assert not bool(transition[:, :, 2:].count_nonzero())
    assert not bool(transition[:, 3].count_nonzero())
    assert torch.allclose(
        reversed_transition[:, 1, :2],
        grounded[:, 2, :2] - grounded[:, 3, :2],
    )
    assert not torch.allclose(normal[:, :3], reversed_output[:, :3])
    assert not hasattr(fusion, "value")
    constant, constant_transition = fusion(
        probe,
        grounded[:, :1].expand_as(grounded),
        valid_frames,
        valid_tokens,
    )
    assert not bool(constant_transition.count_nonzero())
    assert torch.allclose(
        constant,
        probe.masked_fill(~valid_frames[..., None], 0.0),
    )
    normal.sum().backward()
    assert grounded.grad is not None
    assert bool(grounded.grad[:, :3, :2].count_nonzero())
    full_width = TaskGroundedVisualTransitionFusion(width=256, heads=8)
    assert sum(parameter.numel() for parameter in full_width.parameters()) == 197_120


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
    compiler = SlotNormalizedCoreProcedureCompiler(
        width=32,
        heads=4,
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


def test_procedure_modulation_is_zero_at_init_then_order_sensitive_when_opened() -> (
    None
):
    torch.manual_seed(29)
    compiler = SlotNormalizedCoreProcedureCompiler(
        width=32,
        heads=4,
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

    torch.nn.init.normal_(compiler.modulation.weight, std=0.01)
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
