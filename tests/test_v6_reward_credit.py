from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    program_cotangent,
)
from ember.expert_manifold.v6_reward_credit import (
    episode_equal_chunk_weights,
    functional_reward_lora_gradient,
    leave_one_out_binary_advantages,
    reward_program_cotangent,
)
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)


@pytest.mark.parametrize("outcomes", tuple(itertools.product((0.0, 1.0), repeat=4)))
def test_k4_binary_loo_has_exact_zero_sum_and_expected_signs(outcomes) -> None:
    successes = torch.tensor(outcomes)
    advantages = leave_one_out_binary_advantages(successes)
    torch.testing.assert_close(
        advantages.sum(), torch.tensor(0.0), rtol=0, atol=1e-7
    )
    if len(set(outcomes)) == 1:
        assert torch.count_nonzero(advantages) == 0
    else:
        assert bool((advantages[successes == 1] > 0).all())
        assert bool((advantages[successes == 0] < 0).all())


def test_episode_chunk_weights_do_not_favor_long_episodes() -> None:
    episode_ids = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    successes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    per_episode = torch.stack(
        [weights[episode_ids == episode].sum() for episode in range(4)]
    )
    torch.testing.assert_close(per_episode, advantages / 4, rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(), torch.tensor(0.0), rtol=0, atol=1e-7)


def test_direct_credit_matches_old_current_aspo_first_derivative() -> None:
    advantages = leave_one_out_binary_advantages(
        torch.tensor([1.0, 0.0, 0.0, 1.0])
    )
    current = torch.tensor([0.3, -0.2, 0.5, 0.1], requires_grad=True)
    old = current.detach().clone()
    historical_aspo = (-torch.exp(old - current) * advantages).mean()
    direct_signed_cfm = (current * advantages).mean()
    historical_gradient = torch.autograd.grad(historical_aspo, current)[0]
    direct_gradient = torch.autograd.grad(direct_signed_cfm, current)[0]
    torch.testing.assert_close(historical_gradient, direct_gradient, rtol=0, atol=0)


def test_fp32_lora_gradient_transports_through_bfloat16_decoder_output() -> None:
    program = torch.randn(1, 320, 256, requires_grad=True)
    decoded = {"projection.lora_A": program[0, :, :4].to(torch.bfloat16)}
    graph = GeneratedConditionGraph(
        correct_lora=decoded,
        program_leaf=program,
        program_input_before=program.detach(),
        correct_feature=torch.ones(256),
        negative_feature=torch.zeros(256),
        correct_raw_frames=10,
        correct_sampled_frames=3,
        negative_raw_frames=10,
        negative_sampled_frames=3,
    )
    cotangent = program_cotangent(
        graph,
        {"projection.lora_A": torch.ones((320, 4), dtype=torch.float32)},
    )
    assert cotangent.dtype == torch.float32
    torch.testing.assert_close(cotangent[:, :4], torch.ones((320, 4)))
    assert torch.count_nonzero(cotangent[:, 4:]) == 0


class _ProjectedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 7, bias=False)
        self.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )

    def forward(self, images, image_masks, tokens, masks, actions, noise, time):
        del images, image_masks, tokens, masks
        signal = actions + noise.mul(0.125) + time[:, None, None].mul(0.25)
        return self.projection(signal).square()


class _ProjectedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ProjectedModel()
        self.config = SimpleNamespace(
            output_features={ACTION: SimpleNamespace(shape=(7,))},
            max_action_dim=7,
            chunk_size=50,
        )

    def _preprocess_images(self, batch):
        size = batch[ACTION].shape[0]
        return [torch.zeros((size, 3, 2, 2))], [torch.ones(size, dtype=torch.bool)]

    @staticmethod
    def prepare_action(batch):
        return batch[ACTION]


def _policy_contract_state():
    policy = _ProjectedPolicy()
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("model.projection", 7, 7),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=29,
    )
    inject_task_lora(policy, contract)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    shapes = expected_lora_state_shapes(contract)
    return policy, contract, shapes


def _replay_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    episode_ids = torch.tensor([0, 1, 1, 2, 3, 3])
    actions = torch.zeros((episode_ids.numel(), 50, 7))
    actions[:, :5] = torch.arange(1, episode_ids.numel() + 1)[:, None, None]
    batch = {
        ACTION: actions,
        "executed_action_steps": torch.tensor([5, 5, 3, 4, 5, 2]),
        "action_is_pad": torch.zeros((episode_ids.numel(), 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((episode_ids.numel(), 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(
            (episode_ids.numel(), 2), dtype=torch.bool
        ),
    }
    return batch, episode_ids, torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_reward_lora_gradient_is_microbatch_semantic_and_skips_homogeneous() -> None:
    policy, contract, shapes = _policy_contract_state()
    generator = torch.Generator().manual_seed(17)
    state = {
        name: torch.randn(shape, generator=generator, requires_grad=True)
        for name, shape in shapes.items()
    }
    batch, episode_ids, successes = _replay_batch()
    first, first_details = functional_reward_lora_gradient(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        mc_samples=4,
        physical_microbatch_size=1,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    second, second_details = functional_reward_lora_gradient(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert first_details["functional_policy_forwards"] == 24
    assert second_details["functional_policy_forwards"] == 12
    for name in state:
        torch.testing.assert_close(first[name], second[name], rtol=2e-6, atol=2e-6)
    full, full_details = functional_reward_lora_gradient(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        mc_samples=4,
        physical_microbatch_size=6,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert full_details["functional_policy_forwards"] == 4
    for name in state:
        torch.testing.assert_close(first[name], full[name], rtol=2e-6, atol=2e-6)
    changed, _ = functional_reward_lora_gradient(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=1,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert any(not torch.equal(second[name], changed[name]) for name in state)
    assert any(bool(torch.count_nonzero(value)) for value in first.values())
    zero, details = functional_reward_lora_gradient(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        torch.ones(4),
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert details["functional_policy_forwards"] == 0
    assert all(torch.count_nonzero(value) == 0 for value in zero.values())
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_mixed_reward_credit_reaches_program_and_homogeneous_is_exact_zero() -> None:
    policy, contract, shapes = _policy_contract_state()
    program = torch.randn(1, 320, 256, generator=torch.Generator().manual_seed(19))
    program.requires_grad_(True)
    offset = 0
    state = {}
    flat = program.flatten()
    for name, shape in shapes.items():
        count = shape[0] * shape[1]
        state[name] = flat[offset : offset + count].reshape(shape)
        offset += count
    graph = GeneratedConditionGraph(
        correct_lora=state,
        program_leaf=program,
        program_input_before=program.detach(),
        correct_feature=torch.ones(256),
        negative_feature=torch.zeros(256),
        correct_raw_frames=10,
        correct_sampled_frames=3,
        negative_raw_frames=10,
        negative_sampled_frames=3,
    )
    batch, episode_ids, successes = _replay_batch()
    cotangent, summary = reward_program_cotangent(
        graph,
        policy=policy,
        contract=contract,
        batch=batch,
        episode_ids=episode_ids,
        successes=successes,
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert cotangent.shape == (320, 256)
    assert summary.mixed and summary.program_cotangent_rms > 0
    zero, zero_summary = reward_program_cotangent(
        graph,
        policy=policy,
        contract=contract,
        batch={"executed_action_steps": batch["executed_action_steps"]},
        episode_ids=episode_ids,
        successes=torch.zeros(4),
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert torch.count_nonzero(zero) == 0
    assert not zero_summary.mixed
    assert zero_summary.functional_policy_forwards == 0
