"""Behavioral contracts for local process fields and their complete LoRA effects."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget, LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.video import VideoConditionedWriter, VideoWriterConfig


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def writer(**kwargs):
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json")
    contract = replace(contract, targets=tuple(LoRATarget(target.name, 3, 4) for target in contract.targets),
                       rank=2, alpha=2)
    config = replace(VideoWriterConfig(width=12, heads=3, horizon=4, native_width=6, language_width=8,
                                       factor_width=5, field_unit=.1, query_chunk=2), **kwargs)
    return VideoConditionedWriter(contract, config)


def inputs(lengths=(5,)):
    response = tuple(torch.randn(length, 4, 6) for length in lengths)
    indices = tuple(torch.arange(length) * 5 for length in lengths)
    language = torch.randn(6, 8)
    language_mask = torch.tensor([False, True, True, False, True, False])
    visual = tuple(torch.randn(length, 8, 8) for length in lengths)
    valid = tuple(torch.tensor([True, True, True, True, True, True, False, False]).expand(length, -1)
                  for length in lengths)
    task = tuple(torch.tensor([False, False, False, True, True, True, False, False]).expand(length, -1)
                 for length in lengths)
    return response, indices, language, language_mask, visual, valid, task


def native_inputs(model, args):
    return tuple({target.name: torch.randn(*response.shape[:2], target.in_features)
                  for target in model.contract.targets} for response in args[0])


def unlock(model):
    with torch.no_grad():
        for group in model.decoder.groups:
            group.coefficients[-1].weight.normal_(std=.03)


def permute(args, order):
    return tuple((value[0][order],) if index in (0, 1, 4, 5, 6) else value
                 for index, value in enumerate(args))


def test_complete_legal_identity_then_video_conditioned_factors():
    model, args = writer(), inputs()
    native = native_inputs(model, args)
    initial = model(*args, native_inputs=native)
    validate_lora_state(initial, model.contract)
    assert len(initial) == 76
    for target in model.contract.targets:
        assert initial[target.name + LORA_A_SUFFIX].count_nonzero() == 0
        assert initial[target.name + LORA_B_SUFFIX].count_nonzero() > 0
    unlock(model)
    changed = list(args)
    changed[0] = (args[0][0] + torch.randn_like(args[0][0]) * 3,)
    original, perturbed = model(*args, native_inputs=native), model(*changed, native_inputs=native)
    assert all(not torch.allclose(original[name], perturbed[name]) for name in original)


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_supervised_field_is_exactly_the_full_parameter_effect(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), inputs()
    unlock(model)
    native = native_inputs(model, args)
    videos = model.encode(*args)
    state, fields = model.decode_with_fields(videos, native, torch.arange(len(args[0][0])))
    torch.testing.assert_close(state, model.decode(videos, native))
    for target in model.contract.targets:
        effect = state[target.name + LORA_B_SUFFIX] @ state[target.name + LORA_A_SUFFIX]
        contracted = torch.einsum('tho,thd->od', fields[target.name], native[0][target.name]) / len(args[0][0])
        torch.testing.assert_close(effect, contracted, rtol=2e-5, atol=2e-7)
        query = torch.randn(target.in_features)
        kernel = torch.einsum('tho,th->o', fields[target.name], native[0][target.name] @ query) / len(args[0][0])
        torch.testing.assert_close(effect @ query, kernel, rtol=2e-5, atol=2e-7)
    selected_state, selected = model.decode_with_fields(videos, native, torch.tensor([1, 4]))
    torch.testing.assert_close(selected_state, state)
    torch.testing.assert_close(selected, {name: value[[1, 4]] for name, value in fields.items()})


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_complete_horizon_visual_and_process_receive_real_compiler_gradient(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), inputs()
    unlock(model)
    args[0][0].requires_grad_()
    args[4][0].requires_grad_()
    state = model(*args, native_inputs=native_inputs(model, args))
    sum(value.square().mean() for value in state.values()).backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()
    assert args[4][0].grad[args[5][0]].abs().sum(-1).gt(0).all()
    assert model.encoder.local_read.query.weight.grad.norm() > 0
    assert model.encoder.temporal_blocks[0].attention.query.weight.grad.norm() > 0
    assert model.encoder.native_projection.weight.grad.norm() > 0


def test_exact_task_span_and_masked_padding_are_respected():
    model, args = writer(), inputs()
    unlock(model)
    native = native_inputs(model, args)
    reference = model(*args, native_inputs=native)
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][~args[3]] += 100
    changed[4] = (args[4][0].clone(),)
    changed[4][0][~args[5][0]] += 100
    torch.testing.assert_close(model(*changed, native_inputs=native), reference)
    invalid = list(args)
    invalid[6] = (args[6][0].clone(),)
    invalid[6][0][:, 3] = False
    with pytest.raises(ValueError, match="task-token"):
        model(*invalid, native_inputs=native)


def test_frame_set_uses_all_frames_but_is_permutation_and_time_invariant():
    model, args = writer(process_mode="frame_set"), inputs()
    unlock(model)
    native = native_inputs(model, args)
    original = model(*args, native_inputs=native)
    order = torch.tensor([3, 0, 4, 1, 2])
    changed = list(permute(args, order))
    changed[1] = (torch.tensor([37, 11, 2, 0, 999]),)
    permuted_native = ({name: value[order] for name, value in native[0].items()},)
    torch.testing.assert_close(model(*changed, native_inputs=permuted_native), original, rtol=2e-5, atol=2e-6)
    changed = list(args)
    changed[4] = (args[4][0].clone(),)
    changed[4][0][2, :3] += torch.randn(3, 8) * 4
    assert not torch.allclose(model(*changed, native_inputs=native)[next(iter(original))], original[next(iter(original))])


def test_ordered_and_set_have_matching_capacity_and_different_temporal_reading():
    model, args = writer(), inputs()
    baseline = writer(process_mode="frame_set")
    baseline.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
    local, roles = model.encode(*args)[0]
    static, unordered = baseline.encode(*args)[0]
    assert local.shape == static.shape == (5, 4, 12)
    assert roles.shape == unordered.shape == (5, 3, 12)
    assert not torch.allclose(local, static) and not torch.allclose(roles, unordered)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.encode(*permute(args, torch.tensor([3, 0, 4, 1, 2])))


def test_all_fifty_native_horizon_slots_remain_live():
    model = writer(horizon=50)
    args = list(inputs((3,)))
    args[0] = (torch.randn(3, 50, 6, requires_grad=True),)
    local, roles = model.encode(*args)[0]
    (local.square().mean() + roles.square().mean()).backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()


def test_k1_only_and_native_input_information_wall():
    model, args = writer(), inputs()
    native = native_inputs(model, args)
    with pytest.raises(ValueError, match="K1"):
        model.encode(*inputs((3, 7)))
    target = model.contract.targets[0]
    native[0][target.name].requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        model(*args, native_inputs=native)


def test_parameter_partition_covers_exactly_encoder_and_compiler():
    model = writer()
    encoder, compiler = map(lambda it: {id(value) for value in it},
                            (model.encoder_parameters(), model.compiler_parameters()))
    assert encoder and compiler and not encoder & compiler
    assert encoder | compiler == {id(value) for value in model.parameters()}
