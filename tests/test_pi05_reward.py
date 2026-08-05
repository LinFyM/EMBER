from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.pi05_processing import libero_policy_input
from ember.reward.loss import (
    Pi05ExecutedPrefixFlowLoss,
    equal_episode_loss,
    functional_executed_prefix_flow_loss,
)
from ember.reward.ledger import (
    InteractionCursors,
    ledger_prefix_summary,
    write_rollout_once,
)
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    environment_seed,
    policy_noise_seed,
    task_local_video_demo,
    update_seed,
)
from ember.reward.rollout import (
    complete_trajectory_batch,
    RandomResetEnvironmentPool,
    RewardTrajectory,
    collect_randomized_reward_trajectory,
    successful_trajectory_batch,
)


def test_random_reset_pool_keeps_counterfactual_lanes_independent(
    tmp_path, monkeypatch
) -> None:
    payload = b"(define (problem paired-lanes))\n"
    task_root = tmp_path / "libero_spatial"
    task_root.mkdir()
    path = task_root / "paired_lanes.bddl"
    path.write_bytes(payload)
    task = RewardTask(
        suite="libero_spatial",
        task_id=0,
        global_task_id=0,
        split_role="train",
        language="put the bowl on the tray",
        problem_folder="libero_spatial",
        bddl_file=path.name,
        bddl_bytes=len(payload),
        bddl_sha256=hashlib.sha256(payload).hexdigest(),
        horizon=220,
    )

    class FakeEnvironment:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pool = object.__new__(RandomResetEnvironmentPool)
    pool.bddl_root = tmp_path
    pool.render_resolution = 256
    pool._envs = {}
    pool._validated_tasks = set()
    created: list[FakeEnvironment] = []

    def create(_path):
        value = FakeEnvironment()
        created.append(value)
        return value

    monkeypatch.setattr(pool, "_new_environment", create)
    plus = pool.get(task, lane=0)
    minus = pool.get(task, lane=1)
    assert plus is pool.get(task, lane=0)
    assert minus is pool.get(task, lane=1)
    assert plus is not minus
    assert len(created) == 2
    with pytest.raises(RewardProtocolError, match="lane"):
        pool.get(task, lane=2)
    pool.close()
    assert all(value.closed for value in created)


def _observation(marker: int = 0) -> dict[str, np.ndarray]:
    base = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + marker
    return {
        "agentview_image": base,
        "robot0_eye_in_hand_image": base + 20,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
    }


class _FakeEnvironment:
    def __init__(self, *, success_after_policy_steps: int | None) -> None:
        self.success_after_policy_steps = success_after_policy_steps
        self.events: list[tuple[str, object]] = []
        self.policy_steps = 0

    def seed(self, seed: int) -> None:
        self.events.append(("seed", seed))

    def reset(self) -> dict[str, np.ndarray]:
        self.events.append(("reset", None))
        return _observation()

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        if np.array_equal(action, np.array([0, 0, 0, 0, 0, 0, -1], np.float32)):
            self.events.append(("dummy", None))
            return _observation(), 0.0, False, {}
        self.policy_steps += 1
        self.events.append(("policy", self.policy_steps))
        success = self.policy_steps == self.success_after_policy_steps
        return _observation(self.policy_steps), float(success), success, {}


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)
        self.noises: list[torch.Tensor] = []
        self.num_steps: list[int] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch, *, noise, num_steps):
        assert "observation.images.right_wrist_0_rgb" not in batch
        self.noises.append(noise.detach().cpu())
        self.num_steps.append(num_steps)
        return torch.zeros((1, 50, 7), dtype=torch.float32, device=noise.device)


def _preprocess(value: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        key: tensor.unsqueeze(0)
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor) and key.startswith("observation.images.")
    } | {
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 0]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False]]),
    }


def test_policy_input_rotates_both_images_and_keeps_right_wrist_absent() -> None:
    obs = _observation()
    value = libero_policy_input(obs, "do the task")
    assert "observation.images.right_wrist_0_rgb" not in value
    expected = torch.from_numpy(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    expected = expected.permute(2, 0, 1).float().div(255)
    torch.testing.assert_close(value["observation.images.base_0_rgb"], expected)
    torch.testing.assert_close(
        value["observation.state"],
        torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5]),
    )


def test_reward_schedules_exclude_arm_rank_and_execution_order() -> None:
    env = environment_seed(7, "libero_goal", 4, 11, 9)
    noise = policy_noise_seed(13, "libero_goal", 4, 11, 9, 3)
    update = update_seed(17, "libero_goal", 4, 11, 2)
    demo = task_local_video_demo(19, 24, 11)
    assert 0 <= env < 2**32
    assert (env, noise, update, demo) == (
        27031786,
        5069194589751048431,
        8024636704993181904,
        16,
    )
    assert task_local_video_demo(19, 24, 11) == demo
    assert task_local_video_demo(19, 24, 12) != demo


def test_random_reset_rollout_settles_then_executes_five_step_replans() -> None:
    env = _FakeEnvironment(success_after_policy_steps=7)
    policy = _FakePolicy()
    trajectory = collect_randomized_reward_trajectory(
        env=env,
        policy=policy,
        preprocess=_preprocess,
        postprocess=lambda value: value,
        suite="libero_spatial",
        task_id=6,
        global_task_id=6,
        language="put the bowl on the tray",
        adaptation_seed=23,
        rollout_cursor=4,
        env_seed=29,
        policy_seed_root=31,
        device=torch.device("cpu"),
        max_horizon=220,
        dummy_settling_steps=10,
        dummy_action=[0, 0, 0, 0, 0, 0, -1],
        action_execution_horizon=5,
        num_inference_steps=10,
    )
    assert env.events[:2] == [("seed", 29), ("reset", None)]
    assert env.events[2:12] == [("dummy", None)] * 10
    assert [event for event, _ in env.events[12:]] == ["policy"] * 7
    assert trajectory.success and trajectory.steps == 7
    assert trajectory.valid_action_steps == (5, 2)
    assert trajectory.progress_start_frame is not None
    assert trajectory.progress_terminal_frame is not None
    assert trajectory.progress_start_frame.dtype == torch.uint8
    assert trajectory.progress_terminal_frame.dtype == torch.uint8
    assert not torch.equal(
        trajectory.progress_start_frame, trajectory.progress_terminal_frame
    )
    assert len(trajectory.policy_noise_seeds) == 2
    assert policy.num_steps == [10, 10]
    assert not torch.equal(policy.noises[0], policy.noises[1])
    assert trajectory.ledger_row()["fixed_init_state_id"] is None
    assert trajectory.ledger_row()["dummy_settling_steps"] == 10


def test_randomness_cursor_decouples_antithetic_rng_from_artifact_identity() -> None:
    trajectories = []
    policies = []
    for rollout_cursor in (8, 9):
        policy = _FakePolicy()
        policies.append(policy)
        trajectories.append(
            collect_randomized_reward_trajectory(
                env=_FakeEnvironment(success_after_policy_steps=1),
                policy=policy,
                preprocess=_preprocess,
                postprocess=lambda value: value,
                suite="libero_spatial",
                task_id=6,
                global_task_id=6,
                language="put the bowl on the tray",
                adaptation_seed=23,
                rollout_cursor=rollout_cursor,
                randomness_cursor=4,
                env_seed=29,
                policy_seed_root=31,
                device=torch.device("cpu"),
                max_horizon=220,
                dummy_settling_steps=10,
                dummy_action=[0, 0, 0, 0, 0, 0, -1],
                action_execution_horizon=5,
                num_inference_steps=10,
            )
        )
    assert [value.rollout_cursor for value in trajectories] == [8, 9]
    assert trajectories[0].policy_noise_seeds == trajectories[1].policy_noise_seeds
    assert torch.equal(policies[0].noises[0], policies[1].noises[0])


def _trajectory(valid: tuple[int, ...], *, success: bool = True) -> RewardTrajectory:
    observations = tuple(
        {
            OBS_LANGUAGE_TOKENS: torch.ones((1, 2), dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 2), dtype=torch.bool),
            "observation.images.base_0_rgb": torch.zeros((1, 3, 2, 2)),
        }
        for _ in valid
    )
    return RewardTrajectory(
        suite="libero_goal",
        task_id=4,
        global_task_id=24,
        adaptation_seed=3,
        rollout_cursor=5,
        env_seed=7,
        policy_seed_root=11,
        success=success,
        steps=sum(valid),
        reward_sum=1.0,
        dummy_settling_steps=10,
        initial_observation_sha256="a" * 64,
        policy_noise_seeds=tuple(range(len(valid))),
        observations=observations,
        action_chunks=tuple(torch.zeros((1, 50, 7)) for _ in valid),
        valid_action_steps=valid,
    )


def test_success_batch_preserves_exact_executed_prefixes() -> None:
    batch, episode_ids = successful_trajectory_batch(
        (_trajectory((5, 2)), _trajectory((4,))), torch.device("cpu")
    )
    assert batch[ACTION].shape == (3, 50, 7)
    assert batch["executed_action_steps"].tolist() == [5, 2, 4]
    assert batch["action_is_pad"].sum(dim=1).tolist() == [45, 48, 46]
    assert episode_ids.tolist() == [0, 0, 1]


def test_complete_batch_retains_failure_prefixes_and_binary_outcomes() -> None:
    batch, episode_ids, successes = complete_trajectory_batch(
        (_trajectory((5, 2)), _trajectory((5,), success=False)),
        torch.device("cpu"),
    )
    assert batch[ACTION].shape == (3, 50, 7)
    assert batch["executed_action_steps"].tolist() == [5, 2, 5]
    assert episode_ids.tolist() == [0, 0, 1]
    assert successes.tolist() == [1.0, 0.0]


class _LossModel(torch.nn.Module):
    @staticmethod
    def sample_noise(shape, device):
        return torch.zeros(shape, device=device)

    @staticmethod
    def sample_time(size, device):
        return torch.zeros(size, device=device)

    @staticmethod
    def forward(images, image_masks, tokens, masks, actions, noise, time):
        del images, image_masks, tokens, masks, noise, time
        return actions.square()


class _LossPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _LossModel()
        self.config = SimpleNamespace(
            output_features={ACTION: SimpleNamespace(shape=(7,))},
            max_action_dim=7,
        )

    def _preprocess_images(self, batch):
        size = batch[ACTION].shape[0]
        return [torch.zeros((size, 3, 2, 2))], [torch.ones(size, dtype=torch.bool)]

    @staticmethod
    def prepare_action(batch):
        return batch[ACTION]


def test_executed_prefix_loss_ignores_unexecuted_45_of_50_actions() -> None:
    policy = _LossPolicy()
    loss = Pi05ExecutedPrefixFlowLoss(policy)
    actions = torch.zeros((2, 50, 7))
    actions[:, :5] = 2
    batch = {
        ACTION: actions.clone(),
        "executed_action_steps": torch.tensor([5, 5]),
        OBS_LANGUAGE_TOKENS: torch.ones((2, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((2, 2), dtype=torch.bool),
    }
    before, details = loss(batch)
    batch[ACTION][:, 5:] = 10_000
    after, _ = loss(batch)
    torch.testing.assert_close(before, torch.full((2,), 4.0))
    torch.testing.assert_close(before, after)
    assert details["executed_action_steps"] == 10
    assert details["masked_unexecuted_action_steps"] == 90


def test_equal_episode_loss_does_not_overweight_long_successes() -> None:
    loss, details = equal_episode_loss(
        torch.tensor([1.0, 3.0, 9.0]), torch.tensor([0, 0, 1])
    )
    torch.testing.assert_close(loss, torch.tensor(5.5))
    assert details == {"successful_episodes": 2, "successful_chunks": 3}


def test_immutable_ledger_prefix_binds_three_distinct_cursors(tmp_path) -> None:
    first = _trajectory((5,)).ledger_row()
    first["rollout_cursor"] = 0
    second = _trajectory((5, 2)).ledger_row()
    second["rollout_cursor"] = 1
    path = write_rollout_once(tmp_path, "task_024_identity_seed_003", first)
    assert write_rollout_once(tmp_path, "task_024_identity_seed_003", first) == path
    write_rollout_once(tmp_path, "task_024_identity_seed_003", second)
    summary = ledger_prefix_summary(tmp_path, "task_024_identity_seed_003", 2)
    assert summary["rollout_cursor"] == 2
    assert summary["environment_action_cursor"] == 12
    assert summary["successes"] == 2
    cursors = InteractionCursors(
        rollout=summary["rollout_cursor"],
        environment_actions=summary["environment_action_cursor"],
        optimizer_updates=1,
    )
    assert cursors.to_dict() == {
        "rollout_cursor": 2,
        "environment_action_cursor": 12,
        "optimizer_update_cursor": 1,
    }


class _ProjectedLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 7, bias=False)

    @staticmethod
    def sample_noise(shape, device):
        return torch.zeros(shape, device=device)

    @staticmethod
    def sample_time(size, device):
        return torch.zeros(size, device=device)

    def forward(self, images, image_masks, tokens, masks, actions, noise, time):
        del images, image_masks, tokens, masks, noise, time
        return self.projection(actions).square()


class _FunctionalLossPolicy(_LossPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ProjectedLossModel()


def test_functional_prefix_loss_only_backpropagates_to_generated_lora() -> None:
    policy = _FunctionalLossPolicy()
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
    generated = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in task_lora_state_dict(policy).items()
    }
    batch = {
        ACTION: torch.ones((2, 50, 7)),
        "executed_action_steps": torch.tensor([5, 3]),
        OBS_LANGUAGE_TOKENS: torch.ones((2, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((2, 2), dtype=torch.bool),
    }
    per_chunk, _ = functional_executed_prefix_flow_loss(
        policy,
        generated,
        contract,
        batch,
        noise=torch.zeros((2, 50, 7)),
        time=torch.zeros(2),
    )
    per_chunk.mean().backward()
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert any(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in generated.values()
    )
