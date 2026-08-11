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

from ember.expert_manifold.v6_prior_step import GeneratedConditionGraph, program_cotangent
from ember.expert_manifold.v6_reward_tangent import (
    landmark_credit_batch,
    landmark_reward_program_cotangent,
    leave_one_out_binary_advantages,
)
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)
from ember.reward.rollout import RewardOccupancyLandmark, RewardRolloutOutcome


@pytest.mark.parametrize("values", tuple(itertools.product((0.0, 1.0), repeat=4)))
def test_k4_landmark_advantage_has_zero_sum_and_expected_signs(values) -> None:
    successes = torch.tensor(values)
    advantages = leave_one_out_binary_advantages(successes)
    torch.testing.assert_close(advantages.sum(), torch.tensor(0.0), rtol=0, atol=1e-7)
    if len(set(values)) == 1:
        assert torch.count_nonzero(advantages) == 0
    else:
        assert bool((advantages[successes == 1] > 0).all())
        assert bool((advantages[successes == 0] < 0).all())


def _outcomes(successes=(True, False, True, False)):
    result = []
    for episode, success in enumerate(successes):
        landmarks = tuple(
            RewardOccupancyLandmark(
                replan_ordinal=row,
                observation={
                    OBS_LANGUAGE_TOKENS: torch.ones((1, 2), dtype=torch.long),
                    OBS_LANGUAGE_ATTENTION_MASK: torch.ones(
                        (1, 2), dtype=torch.bool
                    ),
                },
                normalized_action_chunk=torch.full(
                    (1, 50, 7), float(episode + row + 1)
                ),
                executed_action_steps=5 - row,
                policy_noise_seed=episode * 10 + row,
            )
            for row in range(episode + 1)
        )
        result.append(
            RewardRolloutOutcome(
                suite="libero_goal",
                task_id=2,
                global_task_id=22,
                adaptation_seed=3,
                rollout_cursor=episode,
                env_seed=episode + 10,
                policy_seed_root=19,
                success=success,
                steps=5 * len(landmarks),
                reward_sum=float(success),
                dummy_settling_steps=10,
                policy_noise_seeds=tuple(row.policy_noise_seed for row in landmarks),
                landmarks=landmarks,
            )
        )
    return tuple(result)


def test_landmark_weights_are_episode_equal_despite_different_row_counts() -> None:
    _, successes, advantages, weights = landmark_credit_batch(
        _outcomes(), device=torch.device("cpu")
    )
    counts = (1, 2, 3, 4)
    offsets = (0, 1, 3, 6, 10)
    per_episode = torch.stack(
        [weights[offsets[index] : offsets[index + 1]].sum() for index in range(4)]
    )
    torch.testing.assert_close(per_episode, advantages / 4, rtol=0, atol=1e-7)
    assert torch.equal(successes, torch.tensor([1.0, 0.0, 1.0, 0.0]))


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


class _ProgramDecoder(torch.nn.Module):
    def __init__(self, contract: SmolVLALoRAContract) -> None:
        super().__init__()
        self.contract = contract
        self.decode_calls = 0

    def decode_slots(self, program: torch.Tensor) -> dict[str, torch.Tensor]:
        self.decode_calls += 1
        flat = program.flatten()
        offset = 0
        state = {}
        for name, shape in expected_lora_state_shapes(self.contract).items():
            count = shape[0] * shape[1]
            state[name] = flat[offset : offset + count].reshape(shape)
            offset += count
        return state


def _graph_and_policy():
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
    program = torch.randn(1, 320, 256, generator=torch.Generator().manual_seed(19))
    program.requires_grad_(True)
    flat = program.flatten()
    offset = 0
    state = {}
    for name, shape in expected_lora_state_shapes(contract).items():
        count = shape[0] * shape[1]
        state[name] = flat[offset : offset + count].reshape(shape)
        offset += count
    writer = SimpleNamespace(base_writer=_ProgramDecoder(contract))
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
    return graph, writer, policy, contract


def test_mixed_landmark_credit_reaches_program_and_homogeneous_skips_forward() -> None:
    graph, writer, policy, contract = _graph_and_policy()
    blind = program_cotangent(
        graph,
        {name: torch.ones_like(value) for name, value in graph.correct_lora.items()},
    )
    assert torch.count_nonzero(blind) > 0
    cotangent, summary = landmark_reward_program_cotangent(
        graph,
        writer=writer,  # type: ignore[arg-type]
        policy=policy,
        contract=contract,
        outcomes=_outcomes(),
        flow_seed_root=31,
        schedule_macro=0,
        global_task_id=22,
        device=torch.device("cpu"),
    )
    assert cotangent.shape == (320, 256)
    assert summary.mixed
    assert summary.functional_policy_forwards == 4
    assert summary.selected_landmarks == 10
    assert summary.maximum_landmarks_per_episode == 4
    assert summary.program_cotangent_rms > 0
    assert writer.base_writer.decode_calls == 1
    assert all(parameter.grad is None for parameter in policy.parameters())

    zero, zero_summary = landmark_reward_program_cotangent(
        graph,
        writer=writer,  # type: ignore[arg-type]
        policy=policy,
        contract=contract,
        outcomes=_outcomes((True, True, True, True)),
        flow_seed_root=31,
        schedule_macro=0,
        global_task_id=22,
        device=torch.device("cpu"),
    )
    assert torch.count_nonzero(zero) == 0
    assert zero_summary.mixed is False
    assert zero_summary.functional_policy_forwards == 0
    assert writer.base_writer.decode_calls == 1
