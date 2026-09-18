"""Ordered recurrent reads, boundary masks, and the complete A/B generator."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ember.writer.procedure import RecurrentProcedureEncoder
from ember.writer.temporal import (
    LanguageSemanticCore, SlotNormalizedCoreProcedureCompiler, VariableEpisodeInputError,
)


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(previous)


def procedure_inputs():
    return (
        torch.randn(2, 5, 16, requires_grad=True),
        torch.randn(2, 5, 50, 8, requires_grad=True),
        torch.randn(2, 5, 3, 16, requires_grad=True),
        torch.tensor([[0, 5, 9, 0, 0], [0, 5, 10, 15, 19]]),
        torch.tensor([[True, True, True, False, False], [True] * 5]),
        torch.tensor([[True, True, False], [True] * 3]),
    )


def procedure(*, opened=True, blocks=2):
    model = RecurrentProcedureEncoder(width=16, expert_width=8, heads=4, blocks=blocks)
    if opened:
        with torch.no_grad():
            for block in model.blocks:
                block.horizon_read.output.weight.normal_(std=.1)
                block.visual_read.output.weight.normal_(std=.1)
    return model


@pytest.mark.parametrize('blocks', [2, 3])
def test_zero_new_reads_recover_the_ordered_a_function(blocks):
    model = procedure(opened=False, blocks=blocks)
    x, h, e, positions, valid, tokens = procedure_inputs()
    expected = x.masked_fill(~valid[..., None], 0)
    for block in model.blocks:
        assert block.horizon_read.output.weight.count_nonzero() == 0
        assert block.visual_read.output.weight.count_nonzero() == 0
        assert block.temporal.causal
        expected = deepcopy(block.temporal)(expected, positions, valid)
    observed = model(x, h, e, positions, valid, tokens)
    torch.testing.assert_close(observed, expected)
    observed.square().sum().backward()
    for block in model.blocks:
        assert block.horizon_read.output.weight.grad.norm() > 0
        assert block.visual_read.output.weight.grad.norm() > 0


def test_full_horizon_content_and_its_position_affect_real_reads():
    model, args = procedure(), procedure_inputs()
    observed = model(*args)
    permuted = model(args[0], args[1].flip(2), *args[2:])
    assert not torch.allclose(permuted, observed)
    with pytest.raises(VariableEpisodeInputError, match='complete Procedure'):
        model(args[0], args[1][:, :, :49], *args[2:])
    (observed * torch.randn_like(observed)).sum().backward()
    _, h, e, _, valid, tokens = args
    assert h.grad[valid].norm(dim=-1).gt(0).all()  # Includes native horizon 49.
    evidence_valid = valid[..., None] & tokens[:, None]
    assert e.grad[evidence_valid].norm(dim=-1).gt(0).all()
    for block in model.blocks:
        for read in (block.horizon_read, block.visual_read):
            assert all(value.grad is not None and value.grad.norm() > 0 for value in read.parameters())


def test_temporal_reads_are_causal_after_the_explicit_next_frame_read():
    model, args = procedure(), procedure_inputs()
    args = tuple(value[1:] for value in args)
    observed = model(*args)
    h = args[1].detach().clone()
    h[:, 3:] += torch.randn_like(h[:, 3:]) * 4
    changed_h = model(args[0], h, *args[2:])
    torch.testing.assert_close(changed_h[:, :3], observed[:, :3])
    assert not torch.allclose(changed_h[:, 3:], observed[:, 3:])
    e = args[2].detach().clone()
    e[:, 3:] += torch.randn_like(e[:, 3:]) * 4
    changed_e = model(*args[:2], e, *args[3:])
    torch.testing.assert_close(changed_e[:, :2], observed[:, :2])
    assert not torch.allclose(changed_e[:, 2], observed[:, 2])


def test_visual_values_keep_both_real_endpoints_and_mask_the_boundary():
    model, args = procedure(), procedure_inputs()
    captured = []
    model.blocks[0].visual_read.register_forward_pre_hook(lambda _, values: captured.append(values))
    model(*args)
    _, keys, values, valid, positions = captured[0]
    e = args[2]
    torch.testing.assert_close(values[0, :3], e[0, 0])
    torch.testing.assert_close(values[0, 3:], e[0, 1])
    torch.testing.assert_close(values[2, :3], e[0, 2])
    assert valid[2].tolist() == [True, True, False, False, False, False]
    assert valid[9].tolist() == [True, True, True, False, False, False]
    assert positions[0].tolist() == [0, 1, 2, 0, 1, 2]
    same = e[:, :1].expand_as(e)
    captured.clear()
    model(*args[:2], same, *args[3:])
    assert not torch.allclose(captured[0][1][0, :3], captured[0][1][0, 3:])


def test_padding_has_no_content_or_gradient_and_does_not_change_valid_reads():
    model, args = procedure(), procedure_inputs()
    x, h, e, positions, valid, tokens = args
    observed = model(*args)
    single = model(x[:1, :3], h[:1, :3], e[:1, :3, :2], positions[:1, :3],
                   valid[:1, :3], tokens[:1, :2])
    torch.testing.assert_close(observed[:1, :3], single)
    assert observed[~valid].count_nonzero() == 0
    observed.square().sum().backward()
    assert x.grad[~valid].count_nonzero() == h.grad[~valid].count_nonzero() == 0
    assert e.grad[~(valid[..., None] & tokens[:, None])].count_nonzero() == 0


def test_video_positions_address_both_procedure_and_core_conditioned_slot_read():
    model, args = procedure(), procedure_inputs()
    compiler = SlotNormalizedCoreProcedureCompiler(width=16, heads=4, initialization_seed=8)
    with torch.no_grad():
        compiler.modulation.weight.normal_(std=.1)
    assert compiler.procedure_reader.attention.rotary_keys
    positions = args[3].square()
    p = model(*args)
    assert not torch.allclose(model(*args[:3], positions, *args[4:]), p)
    c = torch.randn(2, 3, 16)
    original = compiler(c, args[5], p, args[3], args[4])
    changed = compiler(c, args[5], p, positions, args[4])
    assert not torch.allclose(original[0], changed[0])


def test_language_positions_are_retained():
    core = LanguageSemanticCore(width=16, heads=4, blocks=2, frame_attention_initial_lambda=.05)
    q, evidence = torch.randn(1, 4, 16), torch.randn(1, 3, 4, 16)
    valid, tokens = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)
    permutation = [2, 0, 3, 1]
    expected, _ = core(q, evidence, valid, tokens)
    observed, _ = core(q[:, permutation], evidence[:, :, permutation], valid, tokens)
    assert not torch.allclose(observed, expected[:, permutation], atol=1e-5, rtol=1e-5)


class EvidenceEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(256) * .1)
        self.horizon_scale = nn.Parameter(torch.randn(1024) * .1)
        self.interaction_projection = nn.Linear(1024, 256, bias=False)

    def forward(self, policy, frames, ids, tokens, masks, spans, *, frame_parallel_group=None):
        self.group = frame_parallel_group
        signal = frames.float().mean(dim=tuple(range(1, frames.ndim))) / 255
        axis = torch.linspace(.1, 2., 256)
        evidence = (signal[:, None, None] + axis[None, None] + torch.arange(3)[None, :, None] * .2).sin() * self.scale
        horizon = (signal[:, None, None] * axis.repeat(4) + torch.arange(50)[None, :, None] * .03).cos() * self.horizon_scale
        interactions = self.interaction_projection(horizon.mean(1))
        query = (tokens[:, :3, None].float() + axis).sin()
        return query, evidence, interactions, horizon, torch.ones(tokens.shape[0], 3, dtype=torch.bool)


@pytest.mark.parametrize('batch', [1, 2])
def test_complete_writer_identity_shape_order_and_gradient_contract(monkeypatch, batch):
    from ember.lora import identity_lora_state, validate_lora_state
    from ember.pi05_lora import load_pi05_lora_contract
    from ember.writer import model as module

    monkeypatch.setattr(module, 'Pi05LanguageAxialEncoder', EvidenceEncoder)
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / 'configs/pi05_lora_v1.json')
    template = identity_lora_state(contract)
    backbone = SimpleNamespace(layers=range(18))
    model = module.CompleteLoRAWriter(module.build_lora_tensor_specs(template), template_state=template,
        paligemma_model=backbone, expert_model=backbone, image_width=2048, expert_width=1024,
        program_width=256, text_meta_lora_rank=4, vl_meta_lora_rank=4, action_meta_lora_rank=4,
        patch_grounding_heads=8, max_frames_per_encoder_call=8, action_horizon=50, padded_action_dim=32,
        semantic_core_heads=8, semantic_core_blocks=2, frame_attention_initial_lambda=.05,
        procedure_heads=8, procedure_blocks=2, fusion_heads=8, factor_hidden_width=216,
        initialization_seed=7, activation_checkpointing=True)
    assert sum(p.numel() for p in model.factor_heads.parameters()) == 1_838_592
    frames = torch.arange(batch * 3, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 4, 4) * 37
    indices = torch.tensor([0, 5, 8] * batch)
    tokens = torch.arange(1, 4).expand(batch, -1)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    args = (frames, indices, torch.arange(batch + 1) * 3, tokens, masks, masks)
    state = model(*args, policy=None)
    assert len(state) == 76
    validate_lora_state(state if batch == 1 else {k: v[0] for k, v in state.items()}, contract)
    expected = template if batch == 1 else {k: v[None].expand(batch, -1, -1) for k, v in template.items()}
    torch.testing.assert_close(state, expected)
    with torch.no_grad():
        for head in model.factor_heads.values():
            head.network[-1].weight.normal_(std=.01)
        model.compiler.modulation.weight.normal_(std=.01)
        for block in model.procedure.blocks:
            block.horizon_read.output.weight.normal_(std=.01)
            block.visual_read.output.weight.normal_(std=.01)
    learned = model(*args, policy=None, frame_parallel_group='forwarded-group')
    assert model.semantic_encoder.group == 'forwarded-group'
    permutation = torch.tensor([2, 0, 1])[None] + torch.arange(batch)[:, None] * 3
    permuted = model(frames[permutation.flatten()], *args[1:], policy=None)
    assert any(not torch.allclose(permuted[k], learned[k]) for k in learned)
    assert any(not torch.allclose(learned[k], expected[k]) for k in learned)
    sum(value.square().mean() for value in learned.values()).backward()
    assert model.semantic_encoder.scale.grad.norm() > 0
    assert model.semantic_encoder.horizon_scale.grad.norm() > 0
    assert model.compiler.modulation.weight.grad.norm() > 0
