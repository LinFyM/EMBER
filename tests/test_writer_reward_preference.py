from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)
from ember.writer.reward_preference import (
    episode_equal_chunk_weights,
    functional_reward_lora_gradient,
    leave_one_out_binary_advantages,
)


@pytest.mark.parametrize("outcomes", tuple(itertools.product((0.0, 1.0), repeat=4)))
def test_k4_binary_loo_is_zero_sum_and_signed(outcomes) -> None:
    successes = torch.tensor(outcomes)
    advantages = leave_one_out_binary_advantages(successes)
    torch.testing.assert_close(advantages.sum(), torch.tensor(0.0), rtol=0, atol=1e-7)
    if len(set(outcomes)) == 1:
        assert not torch.count_nonzero(advantages)
    else:
        assert bool((advantages[successes == 1] > 0).all())
        assert bool((advantages[successes == 0] < 0).all())


def test_episode_weights_make_each_rollout_equal() -> None:
    episode_ids = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    successes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights, advantages = episode_equal_chunk_weights(episode_ids, successes)
    per_episode = torch.stack(
        [weights[episode_ids == episode].sum() for episode in range(4)]
    )
    torch.testing.assert_close(per_episode, advantages / 4, rtol=0, atol=0)


class _Model(torch.nn.Module):
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


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Model()
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


def test_reward_lora_gradient_is_microbatch_semantic() -> None:
    policy = _Policy()
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("model.projection", 7, 7),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=29,
    )
    inject_task_lora(policy, contract)
    policy.requires_grad_(False)
    state = {
        name: torch.randn(shape, generator=torch.Generator().manual_seed(17))
        for name, shape in expected_lora_state_shapes(contract).items()
    }
    episode_ids = torch.tensor([0, 1, 1, 2, 3, 3])
    batch = {
        ACTION: torch.ones((6, 50, 7)),
        "executed_action_steps": torch.tensor([5, 5, 3, 4, 5, 2]),
        "action_is_pad": torch.zeros((6, 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((6, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((6, 2), dtype=torch.bool),
    }
    kwargs = dict(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        episode_ids=episode_ids,
        successes=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        mc_samples=4,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    first, first_summary = functional_reward_lora_gradient(
        **kwargs, physical_microbatch_size=2
    )
    second, second_summary = functional_reward_lora_gradient(
        **kwargs, physical_microbatch_size=8
    )
    assert first_summary.functional_policy_forwards == 12
    assert second_summary.functional_policy_forwards == 4
    for name in state:
        torch.testing.assert_close(first[name], second[name], rtol=2e-6, atol=2e-6)
    assert any(bool(torch.count_nonzero(value)) for value in first.values())
    assert all(parameter.grad is None for parameter in policy.parameters())
