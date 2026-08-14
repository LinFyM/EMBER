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
from ember.writer.as_step import parameter_layout
from ember.writer.reward_cycle import _apply_projected_step
from ember.writer.reward_preference import (
    episode_equal_chunk_weights,
    functional_reward_lora_gradients,
    leave_one_out_binary_advantages,
    project_final_parameter_delta,
    successful_episode_chunk_weights,
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


def test_success_support_weights_make_each_success_equal() -> None:
    episode_ids = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    successes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights = successful_episode_chunk_weights(episode_ids, successes)
    per_episode = torch.stack(
        [weights[episode_ids == episode].sum() for episode in range(4)]
    )
    torch.testing.assert_close(
        per_episode,
        torch.tensor([0.5, 0.0, 0.5, 0.0]),
        rtol=0,
        atol=0,
    )


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


def test_reward_lora_preference_and_support_are_microbatch_semantic() -> None:
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
    (
        first_preference,
        first_support,
        first_summary,
    ) = functional_reward_lora_gradients(**kwargs, physical_microbatch_size=2)
    (
        second_preference,
        second_support,
        second_summary,
    ) = functional_reward_lora_gradients(**kwargs, physical_microbatch_size=8)
    assert first_summary.functional_policy_forwards == 12
    assert second_summary.functional_policy_forwards == 4
    assert first_preference is not None and second_preference is not None
    assert first_support is not None and second_support is not None
    for name in state:
        torch.testing.assert_close(
            first_preference[name],
            second_preference[name],
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            first_support[name],
            second_support[name],
            rtol=2e-6,
            atol=2e-6,
        )
    assert any(
        bool(torch.count_nonzero(value)) for value in first_preference.values()
    )
    assert any(bool(torch.count_nonzero(value)) for value in first_support.values())
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_all_success_produces_support_without_preference() -> None:
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
    batch = {
        ACTION: torch.ones((4, 50, 7)),
        "executed_action_steps": torch.full((4,), 5),
        "action_is_pad": torch.zeros((4, 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((4, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((4, 2), dtype=torch.bool),
    }
    preference, support, summary = functional_reward_lora_gradients(
        policy,
        state,
        contract,
        batch,
        torch.arange(4),
        torch.ones(4),
        mc_samples=4,
        physical_microbatch_size=8,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert preference is None
    assert support is not None
    assert summary.preference_policy_backwards == 0
    assert summary.support_policy_backwards == 4
    assert summary.functional_policy_forwards == 4
    assert summary.preference_lora_gradient_rms == 0
    assert summary.support_lora_gradient_rms > 0


def test_final_projection_has_exact_raw_fallbacks() -> None:
    raw = torch.tensor([-1.0, 0.5, 0.25])
    preference = torch.tensor([1.0, -1.0, 0.5])
    empty_rows = torch.zeros(2, 3)
    empty_mask = torch.zeros(2, dtype=torch.bool)
    projected, summary = project_final_parameter_delta(
        raw, empty_rows, empty_mask, preference
    )
    assert torch.equal(projected, raw)
    assert summary.support_constraints == 0
    assert not summary.projection_changed

    feasible_rows = torch.tensor([[1.0, 0.0, 0.0]])
    feasible, feasible_summary = project_final_parameter_delta(
        raw,
        feasible_rows,
        torch.ones(1, dtype=torch.bool),
        preference,
    )
    assert torch.equal(feasible, raw)
    assert feasible_summary.raw_violation_count == 0


def test_final_projection_closes_duplicate_rank_deficient_constraints() -> None:
    raw = torch.tensor([1.0, -1.0, 0.5])
    preference = torch.tensor([0.0, 1.0, 0.0])
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mask = torch.ones(3, dtype=torch.bool)
    projected, summary = project_final_parameter_delta(
        raw, rows, mask, preference
    )
    torch.testing.assert_close(
        projected,
        torch.tensor([0.0, -1.0, 0.0]),
        atol=1e-6,
        rtol=0,
    )
    assert summary.constraint_rank == 2
    assert summary.raw_violation_count == 3
    assert summary.final_violation_count == 0
    assert summary.projection_changed
    assert summary.projected_preference_directional_derivative < 0

    permuted, _ = project_final_parameter_delta(
        raw, rows[[2, 0, 1]], mask, preference
    )
    torch.testing.assert_close(projected, permuted, atol=1e-6, rtol=0)


def test_actual_adamw_parameter_delta_is_projected_but_moments_remain_raw() -> None:
    class _Writer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.procedure_set = torch.nn.Linear(2, 1, bias=False)

    writer = _Writer()
    with torch.no_grad():
        writer.procedure_set.weight.copy_(torch.tensor([[0.5, -0.25]]))
    parameter = writer.procedure_set.weight
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        config={"optimization": {"optimizer": {"gradient_clip_norm": 10.0}}},
        writer=writer,
        optimizer=optimizer,
        trainable_parameters=(parameter,),
        gradient_layout=parameter_layout(writer),
    )
    step = _apply_projected_step(
        runtime,
        torch.tensor([-1.0, 0.0]),
        torch.tensor([[1.0, -1.0]]),
        torch.ones(1, dtype=torch.int32),
        1,
    )
    torch.testing.assert_close(
        parameter,
        torch.tensor([[0.54875, -0.20125]]),
        rtol=0,
        atol=2e-7,
    )
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"],
        torch.tensor([[-0.1, 0.0]]),
        rtol=0,
        atol=1e-7,
    )
    assert step.projection.projection_changed
    assert step.projection.raw_violation_count == 1
    assert step.projection.final_violation_count == 0
