"""Behavioral contracts for semantic paths and complete video-conditioned LoRAs."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget, LORA_A_SUFFIX, LORA_B_SUFFIX, identity_lora_state, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.video import VideoConditionedWriter, VideoWriterConfig, path_statistics


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
                                       factor_width=5, state_width=3, query_chunk=2), **kwargs)
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


def unlock(model):
    with torch.no_grad():
        for group in model.decoder.groups:
            group.a[-1].weight.normal_(std=.03)
            group.b[-1].weight.normal_(std=.03)


def permute(args, order):
    return tuple((value[0][order],) if index in (0, 1, 4, 5, 6) else value
                 for index, value in enumerate(args))


def test_path_area_is_directed_and_invariant_to_translation_pause_and_refinement():
    # Analytic state-space paths only; these are not teacher-video interventions.
    path = torch.tensor([[0., 0.], [1., 0.], [1., 1.]])[:, None]
    expected = torch.tensor([[1., 1., .5]])
    torch.testing.assert_close(path_statistics(path, "ordered"), expected)
    other = torch.tensor([[0., 0.], [0., 1.], [1., 1.]])[:, None]
    torch.testing.assert_close(path_statistics(other, "ordered"), expected * torch.tensor([1., 1., -1.]))
    refined = torch.tensor([[0., 0.], [.5, 0.], [1., 0.], [1., 0.], [1., .5], [1., 1.]])[:, None]
    torch.testing.assert_close(path_statistics(refined + 7, "ordered"), expected)
    assert path_statistics(path[:1], "ordered").count_nonzero() == 0


def test_path_integral_matches_pairwise_definition_and_has_gradients():
    state = torch.randn(6, 2, 4, requires_grad=True)
    d = state[1:] - state[:-1]
    area = sum((d[i, :, :, None] * d[j, :, None, :] - d[j, :, :, None] * d[i, :, None, :]) / 2
               for i in range(len(d)) for j in range(i + 1, len(d)))
    row, col = torch.triu_indices(4, 4, 1)
    expected = torch.cat((state[-1] - state[0], area[:, row, col]), -1)
    actual = path_statistics(state, "ordered")
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    direct = torch.autograd.grad((expected * weight).sum(), state, retain_graph=True)[0]
    compiled = torch.autograd.grad((actual * weight).sum(), state)[0]
    torch.testing.assert_close(compiled, direct)
    assert compiled.abs().sum(-1).gt(0).all()


def test_complete_legal_identity_then_video_conditioned_factors():
    model, args = writer(), inputs()
    initial = model(*args)
    validate_lora_state(initial, model.contract)
    assert len(initial) == 76
    for target in model.contract.targets:
        assert initial[target.name + LORA_A_SUFFIX].count_nonzero() > 0
        assert initial[target.name + LORA_B_SUFFIX].count_nonzero() == 0
    target = model.contract.targets[0]
    torch.testing.assert_close(initial[target.name + LORA_A_SUFFIX],
                               identity_lora_state(model.contract)[target.name + LORA_A_SUFFIX])
    unlock(model)
    changed = list(args)
    changed[0] = (args[0][0] + torch.randn_like(args[0][0]) * 3,)
    original, perturbed = model(*args), model(*changed)
    assert all(not torch.allclose(original[name], perturbed[name]) for name in original)


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_complete_horizon_visual_and_process_receive_real_compiler_gradient(activation_checkpoint):
    model, args = writer(activation_checkpoint=activation_checkpoint), inputs()
    unlock(model)
    args[0][0].requires_grad_()
    args[4][0].requires_grad_()
    state = model(*args)
    sum(value.square().mean() for value in state.values()).backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()
    assert args[4][0].grad[args[5][0]].abs().sum(-1).gt(0).all()
    assert model.encoder.state_projection.weight.grad.norm() > 0
    assert model.process_projection.weight.grad.norm() > 0
    assert model.encoder.native_projection.weight.grad.norm() > 0


def test_exact_task_span_and_masked_padding_are_respected():
    model, args = writer(), inputs()
    unlock(model)
    reference = model(*args)
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][~args[3]] += 100
    changed[4] = (args[4][0].clone(),)
    changed[4][0][~args[5][0]] += 100
    actual = model(*changed)
    torch.testing.assert_close(actual, reference)
    invalid = list(args)
    invalid[6] = (args[6][0].clone(),)
    invalid[6][0][:, 3] = False
    with pytest.raises(ValueError, match="task-token"):
        model(*invalid)


def test_frame_set_uses_all_frames_but_is_permutation_and_time_invariant():
    model, args = writer(process_mode="frame_set"), inputs()
    unlock(model)
    original = model(*args)
    changed = list(permute(args, torch.tensor([3, 0, 4, 1, 2])))
    changed[1] = (torch.tensor([37, 11, 2, 0, 999]),)
    torch.testing.assert_close(model(*changed), original, rtol=2e-5, atol=2e-6)
    changed = list(args)
    changed[4] = (args[4][0].clone(),)
    changed[4][0][2, :3] += torch.randn(3, 8) * 4
    assert not torch.allclose(model(*changed)[next(iter(original))], original[next(iter(original))])


def test_ordered_and_set_have_matching_capacity_and_shared_semantic_interpretation():
    model, args = writer(), inputs()
    baseline = writer(process_mode="frame_set")
    baseline.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
    semantic, path = model.encode(*args)[0]
    static, moments = baseline.encode(*args)[0]
    torch.testing.assert_close(semantic, static)
    assert path.shape == moments.shape == (3, 6)
    assert not torch.allclose(path, moments)
    with pytest.raises(ValueError, match="strictly increasing"):
        model(*permute(args, torch.tensor([3, 0, 4, 1, 2])))


def test_all_fifty_native_horizon_slots_remain_live():
    model = writer(horizon=50)
    args = list(inputs((3,)))
    args[0] = (torch.randn(3, 50, 6, requires_grad=True),)
    semantic, path = model.encode(*args)[0]
    (semantic.square().mean() + path.square().mean()).backward()
    assert args[0][0].grad.abs().sum(-1).gt(0).all()


@pytest.mark.parametrize("mode", ["ordered", "frame_set"])
def test_video_set_order_invariance_after_independent_encoding(mode):
    model, args = writer(process_mode=mode), inputs((3, 7))
    unlock(model)
    videos = model.encode(*args)
    torch.testing.assert_close(model.decode(videos), model.decode(videos[::-1]), rtol=2e-5, atol=2e-6)


def test_parameter_partition_covers_exactly_encoder_and_compiler():
    model = writer()
    encoder, compiler = map(lambda it: {id(value) for value in it},
                            (model.encoder_parameters(), model.compiler_parameters()))
    assert encoder and compiler and not encoder & compiler
    assert encoder | compiler == {id(value) for value in model.parameters()}
