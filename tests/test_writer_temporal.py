"""Behavioral contracts for joint video tokens and complete-LoRA parameter decoding."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ember.lora import identity_lora_state, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.temporal import (
    ContinuousParameterDecoder, JointVideoStack, VariableEpisodeInputError,
    token_role_addresses,
)


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(previous)


def _components(*, width=16, semantic=3, horizon=5):
    types = nn.Parameter(torch.randn(2, width) * .02)
    return (JointVideoStack(width, 4, 2), ContinuousParameterDecoder(width, 4, 2, 7),
            types, token_role_addresses(types, semantic, horizon))


def _read(stack, decoder, memory, positions, frames, roles, addresses, semantic):
    joint = stack(memory, positions, frames, roles, addresses)
    return joint, decoder(joint, frames, roles, addresses, semantic)


def test_role_addresses_share_types_and_restart_the_within_type_index():
    types = nn.Parameter(torch.randn(2, 16))
    short = token_role_addresses(types, 2, 3)
    long = token_role_addresses(types, 5, 3)
    torch.testing.assert_close(short[:2], long[:2])
    torch.testing.assert_close(short[2:], long[5:])
    torch.testing.assert_close(short[0] - types[0], short[2] - types[1])
    short.square().sum().backward()
    assert types.grad is not None and bool((types.grad.norm(dim=1) > 0).all())


@pytest.mark.parametrize('repeat', [2, 7])
def test_static_repetition_preserves_joint_content_and_complete_parameter_slots(repeat):
    stack, decoder, _, addresses = _components()
    memory = torch.randn(1, 1, 8, 16)
    roles = torch.ones(1, 8, dtype=torch.bool)
    once, expected = _read(stack, decoder, memory, torch.zeros(1, 1, dtype=torch.long),
                           torch.ones(1, 1, dtype=torch.bool), roles, addresses, 3)
    positions = (torch.arange(repeat) * 5)[None]
    joint, observed = _read(stack, decoder, memory.expand(-1, repeat, -1, -1), positions,
                           torch.ones(1, repeat, dtype=torch.bool), roles, addresses, 3)
    torch.testing.assert_close(joint, once.expand_as(joint), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(observed, expected, atol=3e-6, rtol=2e-5)
    assert bool(observed[0].norm() > 0)  # Repetition invariance does not erase static content.


def test_padding_matches_individual_unpadded_conditions_and_receives_no_gradient():
    stack, decoder, types, addresses = _components(semantic=4, horizon=3)
    frames = torch.tensor([[True, True, True, False, False], [True] * 5])
    roles = torch.tensor([[True, True, False, False, True, True, True], [True] * 7])
    mask = frames[:, :, None] & roles[:, None, :]
    memory = torch.randn(2, 5, 7, 16).masked_fill(~mask[..., None], 10_000.).requires_grad_()
    positions = torch.tensor([[0, 5, 8, 0, 0], [0, 5, 10, 15, 19]])
    joint, observed = _read(stack, decoder, memory, positions, frames, roles, addresses, 4)
    assert joint[~mask].count_nonzero() == 0
    for row, semantic in enumerate((2, 4)):
        selected = memory[row:row + 1, frames[row]][:, :, roles[row]]
        small_frames = torch.ones(selected.shape[:2], dtype=torch.bool)
        small_roles = torch.ones(1, selected.shape[2], dtype=torch.bool)
        _, expected = _read(stack, decoder, selected, positions[row:row + 1, frames[row]],
                            small_frames, small_roles, token_role_addresses(types, semantic, 3), semantic)
        torch.testing.assert_close(tuple(value[row:row + 1] for value in observed), expected,
                                   atol=3e-6, rtol=2e-5)
    sum((value * torch.randn_like(value)).sum() for value in observed).backward()
    assert memory.grad[~mask].count_nonzero() == 0
    assert bool(torch.isfinite(memory.grad).all())


def test_real_frame_reordering_changes_content_but_reindexing_the_same_sequence_does_not():
    stack, decoder, _, addresses = _components()
    memory = torch.randn(1, 4, 8, 16)
    positions = torch.tensor([[0, 5, 10, 14]])
    frames, roles = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)
    _, expected = _read(stack, decoder, memory, positions, frames, roles, addresses, 3)
    permutation = torch.tensor([3, 1, 0, 2])
    _, reindexed = _read(stack, decoder, memory[:, permutation], positions[:, permutation],
                        frames, roles, addresses, 3)
    torch.testing.assert_close(reindexed, expected, atol=4e-6, rtol=3e-5)
    _, reordered = _read(stack, decoder, memory[:, permutation], positions, frames, roles, addresses, 3)
    assert max(float((left - right).detach().abs().max()) for left, right in zip(expected, reordered)) > 1e-4


def test_nonzero_addresses_cannot_create_parameter_content_from_zero_memory():
    stack, decoder, _, addresses = _components()
    memory = torch.zeros(2, 4, 8, 16)
    positions = torch.tensor([[0, 5, 10, 14], [0, 5, 10, 14]])
    frames, roles = torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool)
    joint, output = _read(stack, decoder, memory, positions, frames, roles, addresses * 100., 3)
    assert joint.count_nonzero() == 0
    assert all(value.count_nonzero() == 0 for value in output)


def test_semantic_initialization_precedes_a_read_of_all_action_positions():
    _, decoder, _, addresses = _components()
    first_read = ContinuousParameterDecoder(16, 4, 1, 7)
    memory = torch.randn(1, 3, 8, 16)
    memory[:, :, :3] = 0
    frames, roles = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)
    assert all(value.count_nonzero() == 0 for value in first_read(memory, frames, roles, addresses, 3))
    assert all(bool(value.norm() > 0) for value in decoder(memory, frames, roles, addresses, 3))


@pytest.mark.parametrize('bf16', [False, True])
def test_full_horizon_content_and_temporal_readers_have_finite_functional_credit(bf16):
    stack, decoder, types, addresses = _components(horizon=50)
    memory = torch.randn(1, 3, 53, 16, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        _, output = _read(stack, decoder, memory, torch.tensor([[0, 5, 9]]),
                          torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 53, dtype=torch.bool), addresses, 3)
    sum((value * torch.randn_like(value)).mean() for value in output).backward()
    for module in (stack, decoder):
        assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                   for parameter in module.parameters())
    assert bool((torch.linalg.vector_norm(memory.grad[:, :, 3:], dim=(0, 1, 3)) > 0).all())
    assert bool(types.grad.norm() > 0)
    assert bool(stack.blocks[0].time_attention.query.weight.grad.norm() > 0)
    assert bool(decoder.blocks[1].cross_attention.value.weight.grad.norm() > 0)


def test_parameter_counts_match_the_registered_blocks_and_copy_depth():
    stack = JointVideoStack(256, 8, 2)
    decoder = ContinuousParameterDecoder(256, 8, 2, 7)
    assert sum(p.numel() for p in stack.parameters()) == 2 * 1_049_344
    assert sum(p.numel() for p in decoder.parameters()) == 2 * 1_049_600 + 91_904
    assert sum(p.numel() for p in stack.blocks[0].parameters()) == 1_049_344
    assert sum(p.numel() for p in decoder.blocks[0].parameters()) == 1_049_600


@pytest.mark.parametrize('invalid', ['frames', 'roles', 'positions', 'semantic'])
def test_invalid_memory_contracts_fail_before_all_masked_attention(invalid):
    stack, decoder, _, addresses = _components()
    memory = torch.randn(1, 2, 8, 16)
    frames, roles = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)
    positions, semantic = torch.tensor([[0, 5]]), 3
    if invalid == 'frames':
        frames[:] = False
    elif invalid == 'roles':
        roles[:] = False
    elif invalid == 'positions':
        positions = positions.float()
    else:
        roles[:, :semantic] = False
    with pytest.raises(VariableEpisodeInputError):
        _read(stack, decoder, memory, positions, frames, roles, addresses, semantic)


class _MemoryEncoder(nn.Module):
    """A unit-test boundary supplying joint memory without loading source weights."""

    def __init__(self, **kwargs):
        super().__init__()
        self.constructor = kwargs
        self.content = nn.Parameter(torch.randn(1, 2, 53, 256) * .1)
        self.types = nn.Parameter(torch.randn(2, 256) * .02)

    def forward(self, policy, frames, indices, offsets, tokens, masks, spans):
        batch = tokens.shape[0]
        self.received_indices = indices
        return (self.content.expand(batch, -1, -1, -1), torch.ones(batch, 2, dtype=torch.bool),
                torch.ones(batch, 53, dtype=torch.bool), token_role_addresses(self.types, 3, 50), 3)


@pytest.mark.parametrize('batch', [1, 2])
def test_complete_writer_keeps_identity_all_targets_and_decoder_credit(monkeypatch, batch):
    from ember.writer import video_program

    monkeypatch.setattr(video_program, 'Pi05UnifiedVideoEncoder', _MemoryEncoder, raising=False)
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / 'configs/pi05_lora_v1.json')
    template = identity_lora_state(contract)
    backbone = SimpleNamespace(layers=range(18))
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template), template_state=template,
        paligemma_model=backbone, expert_model=backbone, image_width=2048, expert_width=1024,
        program_width=256, text_meta_lora_rank=4, vl_meta_lora_rank=4, action_meta_lora_rank=4,
        patch_grounding_heads=8, max_frames_per_encoder_call=4, action_horizon=50, padded_action_dim=32,
        factor_hidden_width=216, initialization_seed=7, activation_checkpointing=True,
    )
    assert 'decoder_blocks' not in model.semantic_encoder.constructor
    assert model.semantic_encoder.constructor['native_split_layer'] == 9
    assert sum(p.numel() for p in model.factor_heads.parameters()) == 1_838_592
    frames = torch.randint(0, 256, (2 * batch, 2, 3, 4, 4), dtype=torch.uint8)
    indices = torch.tensor([0, 5] * batch)
    tokens = torch.ones(batch, 3, dtype=torch.long)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    args = (frames, indices, torch.arange(batch + 1) * 2, tokens, masks, masks)
    state = model(*args, policy=None)
    assert model.semantic_encoder.received_indices is indices
    assert len(state) == 76
    one = state if batch == 1 else {name: value[0] for name, value in state.items()}
    validate_lora_state(one, contract)
    expected = template if batch == 1 else {name: value[None].expand(batch, -1, -1) for name, value in template.items()}
    torch.testing.assert_close(state, expected)
    sum((value - .1).square().mean() for value in state.values()).backward()
    assert all(bool(head.network[-1].weight.grad.norm() > 0) for head in model.factor_heads.values())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for head in model.factor_heads.values():
            head.network[-1].weight.normal_(std=.01)
    learned = model(*args, policy=None)
    sum(value.square().mean() for value in learned.values()).backward()
    assert bool(model.semantic_encoder.content.grad.norm() > 0)
    assert bool(model.compiler.blocks[0].cross_attention.value.weight.grad.norm() > 0)
    assert bool(model.compiler.blocks[1].cross_attention.value.weight.grad.norm() > 0)
