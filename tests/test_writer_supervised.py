"""Direct-autograd oracles for complete main LoRA and native replay credit."""
import copy

import pytest
import torch

from ember.writer.function_credit import mean_velocity_loss
from ember.writer.supervised import replay_functional_credit
from test_video_program import inputs, native_inputs, small_cpu_work, unlock, writer


@pytest.mark.parametrize('activation_checkpoint', [False, True])
def test_joint_replay_matches_direct_autograd(activation_checkpoint):
    model = writer(activation_checkpoint=activation_checkpoint)
    unlock(model)
    reference = copy.deepcopy(model)
    args = inputs((3,))
    native = native_inputs(model, args)
    direct_responses = tuple(value.detach().requires_grad_() for value in args[0])
    direct_visuals = tuple(value.detach().requires_grad_() for value in args[4])
    video = reference.encode(direct_responses, *args[1:4], direct_visuals, *args[5:])
    state = reference.decode(video, native)
    targets = {name: torch.randn_like(value) for name, value in state.items()}
    main = .125 * sum((value - targets[name]).square().mean() for name, value in state.items())
    # The FM policy differentiates all complete LoRA leaves, including fixed A.
    # Replay must use only their actual dependence on q and native observations.
    lora_leaves = {name: value.detach().requires_grad_() for name, value in state.items()}
    leaf_loss = .125 * sum((value - targets[name]).square().mean() for name, value in lora_leaves.items())
    compiled = dict(zip(state, torch.autograd.grad(leaf_loss, tuple(lora_leaves.values()))))
    expected = torch.autograd.grad(main, (*reference.parameters(), *direct_responses, *direct_visuals))
    response_grads, visual_grads = replay_functional_credit(
        model, args[0], args[1:], compiled, native)
    actual = [p.grad for p in model.parameters()] + list(response_grads) + list(visual_grads)
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, target, rtol=3e-4, atol=2e-6)


def test_velocity_loss_uses_all_horizon_only_real_action_dimensions():
    predicted = torch.zeros(2, 50, 32, requires_grad=True)
    loss = mean_velocity_loss(predicted, torch.ones_like(predicted), 7)
    loss.backward()
    assert loss == 1 and (predicted.grad[..., :7] != 0).all()
    assert predicted.grad[..., 7:].count_nonzero() == 0


class _OfficialFullFM(torch.nn.Module):
    """Independent oracle: unchanged installed PI05 joint prefix/suffix forward."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, sample):
        return self.policy.model(*sample.arguments)[..., :sample.action_width].float().mean()


@pytest.fixture
def native_fm_policy():
    """Real installed PI05/PiGemma math, with small vision embeddings and widths."""
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace
    from torch import nn
    from transformers import GemmaConfig
    from lerobot.policies.pi_gemma import PiGemmaForCausalLM
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, PaliGemmaWithExpertModel
    from lerobot.utils.constants import ACTION
    from ember.lora import LoRATarget
    from ember.pi05_lora import load_pi05_lora_contract

    class Bridge(nn.Module):
        forward = PaliGemmaWithExpertModel.forward
        embed_language_tokens = PaliGemmaWithExpertModel.embed_language_tokens
        to_bfloat16_for_selected_params = PaliGemmaWithExpertModel.to_bfloat16_for_selected_params

        def __init__(self):
            super().__init__()
            spec = dict(vocab_size=64, hidden_size=32, intermediate_size=64, head_dim=4,
                        num_attention_heads=8, num_key_value_heads=1, num_hidden_layers=18,
                        hidden_activation='gelu_pytorch_tanh', attention_dropout=0.0)
            self.paligemma = nn.Module()
            self.paligemma.model = nn.Module()
            self.paligemma.model.language_model = PiGemmaForCausalLM(
                GemmaConfig(**spec, use_adarms=False)).model
            self.paligemma.model.multi_modal_projector = nn.Linear(3, 32)
            self.gemma_expert = PiGemmaForCausalLM(
                GemmaConfig(**spec, use_adarms=True, adarms_cond_dim=32))
            self.gemma_expert.model.embed_tokens = None

        def embed_image(self, value):
            return self.paligemma.model.multi_modal_projector(value.flatten(2).transpose(1, 2))

    class Core(nn.Module):
        forward = PI05Pytorch.forward
        embed_prefix = PI05Pytorch.embed_prefix
        embed_suffix = PI05Pytorch.embed_suffix
        denoise_step = PI05Pytorch.denoise_step
        sample_noise = PI05Pytorch.sample_noise
        sample_time = PI05Pytorch.sample_time
        _apply_checkpoint = PI05Pytorch._apply_checkpoint
        _prepare_attention_masks_4d = PI05Pytorch._prepare_attention_masks_4d

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=50, min_period=0.004, max_period=4.0,
                time_sampling_beta_alpha=1.5, time_sampling_beta_beta=1.0,
                time_sampling_scale=0.999, time_sampling_offset=0.001)
            self.gradient_checkpointing_enabled = False
            self.paligemma_with_expert = Bridge()
            self.action_in_proj, self.action_out_proj = nn.Linear(32, 32), nn.Linear(32, 32)
            self.time_mlp_in, self.time_mlp_out = nn.Linear(32, 32), nn.Linear(32, 32)

    class Policy(nn.Module):
        prepare_action = PI05Policy.prepare_action

        def __init__(self):
            super().__init__()
            self.model = Core()
            self.config = SimpleNamespace(max_action_dim=32,
                                          output_features={ACTION: SimpleNamespace(shape=(7,))})

        def _preprocess_images(self, batch):
            return [batch['front'], batch['wrist']], [batch['front_mask'], batch['wrist_mask']]

    policy = Policy().eval()
    contract = load_pi05_lora_contract(Path(__file__).resolve().parents[1] / 'configs/pi05_lora_v1.json')
    contract = replace(contract, targets=tuple(LoRATarget(target.name,
        policy.get_submodule(target.name).in_features, policy.get_submodule(target.name).out_features)
        for target in contract.targets), rank=2, alpha=2)
    return policy, contract


def _fm_batch():
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    # Masked cameras and interior language padding exercise native attention and
    # per-row suffix position offsets, beyond a uniform all-visible prefix.
    return {ACTION: torch.randn(5, 50, 7), 'front': torch.randn(5, 3, 2, 2),
            'wrist': torch.randn(5, 3, 2, 2), 'front_mask': torch.ones(5, dtype=torch.bool),
            'wrist_mask': torch.tensor([True, False, True, False, True]),
            OBS_LANGUAGE_TOKENS: torch.randint(1, 64, (5, 6)),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False, True, False, False],
                [True, True, True, True, True, False], [True, False, True, False, True, False],
                [True, True, True, False, False, False], [True, True, True, True, True, True]])}


@pytest.mark.parametrize('microbatch', [1, 3, 5])
@pytest.mark.parametrize('bf16', [False, True])
def test_native_main_credit_matches_official_joint_forward_and_query_slicing(native_fm_policy, microbatch, bf16):
    from ember.writer.function_credit import flow_sample, paired_functional_credit
    from ember.writer.functional import prepare_frozen_writer_policy

    policy, contract = native_fm_policy
    if bf16:
        policy.model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    template = prepare_frozen_writer_policy(policy, contract)
    state = {name: value.detach().clone().requires_grad_() for name, value in template.items()}
    with torch.no_grad():
        for name, value in state.items():
            if 'lora_B' in name:
                value.normal_(std=.03)
    batch = _fm_batch()
    sample = flow_sample(policy, batch, seed=37, device='cpu', random_batch=64, offset=13)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        expected_loss = torch.func.functional_call(_OfficialFullFM(policy),
            {'policy.' + name: value for name, value in state.items()}, (sample,))
        expected = torch.autograd.grad(expected_loss * .25, tuple(state.values()))
        credit = paired_functional_credit(policy, state, contract, batch, seed=37, device='cpu',
            random_batch=64, offset=13, microbatch=microbatch, condition_weight=.25)
    assert len(credit['lora_cotangent']) == 76
    assert credit['flow_loss'] == pytest.approx(float(expected_loss.detach()), rel=2e-3 if bf16 else 2e-5)
    actual_vector = torch.cat([value.flatten() for value in credit['lora_cotangent'].values()])
    expected_vector = torch.cat([value.flatten() for value in expected])
    if bf16:
        assert torch.nn.functional.cosine_similarity(actual_vector, expected_vector, dim=0) > .999
        assert actual_vector.norm() == pytest.approx(float(expected_vector.norm()), rel=.01)
    else:
        torch.testing.assert_close(actual_vector, expected_vector, rtol=4e-4, atol=2e-6)
    assert all(value.norm() > 0 for value in expected)
    assert credit['source_forward_calls'] == 0 and credit['compiled_forward_calls'] == (5 + microbatch - 1) // microbatch
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in policy.parameters())


@pytest.mark.parametrize('backward', [False, True])
def test_native_fm_prefix_stays_frozen_with_and_without_credit(native_fm_policy, backward):
    from ember.writer.function_credit import paired_functional_credit
    from ember.writer.functional import prepare_frozen_writer_policy

    policy, contract = native_fm_policy
    state = prepare_frozen_writer_policy(policy, contract)
    contexts = []
    prefix = policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    handle = prefix.register_forward_pre_hook(lambda *args: contexts.append(torch.is_grad_enabled()))
    try:
        credit = paired_functional_credit(policy, state, contract, _fm_batch(), seed=37, device='cpu',
            random_batch=64, offset=13, microbatch=3, condition_weight=.25, backward=backward)
    finally:
        handle.remove()
    assert contexts == [False, False]
    assert len(credit['lora_cotangent']) == (76 if backward else 0)
    assert credit['compiled_forward_calls'] == 2
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in policy.parameters())
