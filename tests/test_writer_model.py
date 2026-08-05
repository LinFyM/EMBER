from __future__ import annotations

from pathlib import Path

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import writer_trainable_contract
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.policy_dictionary import (
    PolicyCoordinateComposer,
    PolicyWideAtomDictionary,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    TaskGroundedVisualTransitionFusion,
    TaskSelectedSemanticSetFusion,
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
        evidence = frame_value[:, None, None].expand(-1, maximum, 256).clone()
        grounded = evidence.clone()
        interaction = frame_value[:, None].expand(-1, 256).clone()
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
        policy_coordinate_heads=8,
        policy_atom_count=64,
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


def test_policy_wide_atom_writer_parameter_budget_and_fixed_probe_noise_are_exact() -> None:
    model, _ = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_033_728
    contract = writer_trainable_contract(
        model,
        torch.nn.Identity(),
        load_pi05_lora_contract(
            Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
        ),
    )
    assert contract["parameter_count"] == 13_033_728
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
        "composer": (model.composer, 824_064),
        "policy_atoms": (model.policy_atoms, 5_148_672),
    }
    assert {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, (module, _) in expected.items()
    } == {name: count for name, (_, count) in expected.items()}
    assert model.semantic_encoder.fixed_suffix_noise.shape == (50, 32)
    assert "semantic_encoder.fixed_suffix_noise" in model.state_dict()
    assert not model.semantic_encoder.fixed_suffix_noise.requires_grad


def test_policy_wide_atom_writer_starts_at_exact_identity_template() -> None:
    model, template = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert set(output) == set(template)
    for name, value in output.items():
        assert value.shape == (2, *template[name].shape)
        assert torch.equal(value[0], template[name])
        assert torch.equal(value[1], template[name])


def test_policy_wide_atom_writer_becomes_video_conditioned_after_atoms_open() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()
    for atom in model.policy_atoms.b_atoms:
        torch.nn.init.normal_(atom, std=0.01)
    output = model(*_inputs(), policy=torch.nn.Identity())
    assert any(not torch.equal(value[0], value[1]) for value in output.values())
    assert not hasattr(model, "shared_lora")


def _functional_ba_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float32)
    for name, value_a in output.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        name_b = name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        result = result + torch.matmul(
            output[name_b].to(torch.float32),
            value_a.to(torch.float32),
        ).sum()
    return result


def _has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_policy_wide_atom_functional_gradient_staging_opens_intended_paths() -> None:
    model, _ = _model()
    model.semantic_encoder = _FakeSemanticEncoder()

    first = model(*_inputs(), policy=torch.nn.Identity())
    _functional_ba_loss(first).backward()
    assert all(atom.grad is not None and bool(torch.count_nonzero(atom.grad))
               for atom in model.policy_atoms.b_atoms)
    assert all(atom.grad is None or not bool(torch.count_nonzero(atom.grad))
               for atom in model.policy_atoms.a_atoms)
    assert not _has_nonzero_gradient(model.composer)
    assert not _has_nonzero_gradient(model.semantic_core)

    model.zero_grad(set_to_none=True)
    for atom in model.policy_atoms.b_atoms:
        torch.nn.init.normal_(atom, std=0.01)
    second = model(*_inputs(), policy=torch.nn.Identity())
    _functional_ba_loss(second).backward()
    assert _has_nonzero_gradient(model.composer)
    assert _has_nonzero_gradient(model.semantic_core)
    assert _has_nonzero_gradient(model.procedure)
    assert _has_nonzero_gradient(model.visual_transition)
    assert all(atom.grad is not None and bool(torch.count_nonzero(atom.grad))
               for atom in model.policy_atoms.a_atoms)
    assert model.composer.a_query.weight.grad is None or not bool(
        torch.count_nonzero(model.composer.a_query.weight.grad)
    )

    model.zero_grad(set_to_none=True)
    for atom in model.policy_atoms.a_atoms:
        torch.nn.init.normal_(atom, std=0.01)
    third = model(*_inputs(), policy=torch.nn.Identity())
    _functional_ba_loss(third).backward()
    assert model.composer.a_query.weight.grad is not None
    assert bool(torch.count_nonzero(model.composer.a_query.weight.grad))
    assert model.composer.atom_a_keys.grad is not None
    assert bool(torch.count_nonzero(model.composer.atom_a_keys.grad))


def test_policy_dictionary_uses_one_mix_for_all_target_shapes() -> None:
    dictionary = PolicyWideAtomDictionary(
        target_widths=((5, 7), (3, 11)), rank=2, atom_count=4
    )
    with torch.no_grad():
        for atom in (*dictionary.a_atoms, *dictionary.b_atoms):
            torch.nn.init.normal_(atom)
    mix_a = torch.randn(3, 2, 4)
    mix_b = torch.randn(3, 2, 4)
    states = dictionary(mix_a, mix_b)
    assert [tuple(value.shape) for pair in states for value in pair] == [
        (3, 2, 5),
        (3, 7, 2),
        (3, 2, 3),
        (3, 11, 2),
    ]


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


def test_visual_transition_recomputes_after_order_change_without_static_value_path() -> None:
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


def test_policy_coordinate_composer_is_conditioned_and_finite() -> None:
    torch.manual_seed(29)
    composer = PolicyCoordinateComposer(
        width=32,
        heads=4,
        rank=4,
        atom_count=8,
        initialization_seed=7,
    )
    core = torch.randn(2, 9, 32)
    procedure = torch.randn(2, 5, 32)
    valid_core = torch.ones(2, 9, dtype=torch.bool)
    valid_procedure = torch.ones(2, 5, dtype=torch.bool)
    mix_a, mix_b = composer(core, valid_core, procedure, valid_procedure)
    assert mix_a.shape == (2, 4, 8)
    assert mix_b.shape == mix_a.shape
    assert torch.isfinite(mix_a).all() and torch.isfinite(mix_b).all()
    assert not torch.equal(mix_a[0], mix_a[1])
    assert not torch.equal(mix_b[0], mix_b[1])
