from __future__ import annotations

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
    functional_executed_prefix_flow_loss,
)
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    environment_seed,
    policy_noise_seed,
    reward_credit_environment_seed,
    reward_credit_policy_noise_seed,
    task_local_video_demo,
    update_seed,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
    RewardTrajectory,
    collect_randomized_reward_trajectories,
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
        bddl_sha256=None,
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
    lanes = tuple(pool.get(task, lane=lane) for lane in range(4))
    assert all(lanes[lane] is pool.get(task, lane=lane) for lane in range(4))
    assert len({id(value) for value in lanes}) == 4
    assert len(created) == 4
    with pytest.raises(RewardProtocolError, match="lane"):
        pool.get(task, lane=4)
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
    def __init__(
        self, *, success_after_policy_steps: int | None, marker: int = 0
    ) -> None:
        self.success_after_policy_steps = success_after_policy_steps
        self.marker = marker
        self.events: list[tuple[str, object]] = []
        self.policy_steps = 0

    def seed(self, seed: int) -> None:
        self.events.append(("seed", seed))

    def reset(self) -> dict[str, np.ndarray]:
        self.events.append(("reset", None))
        return _observation(self.marker)

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        if np.array_equal(action, np.array([0, 0, 0, 0, 0, 0, -1], np.float32)):
            self.events.append(("dummy", None))
            return _observation(), 0.0, False, {}
        self.policy_steps += 1
        self.events.append(("policy", action.copy()))
        success = self.policy_steps == self.success_after_policy_steps
        return _observation(self.marker + self.policy_steps), float(success), success, {}


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
        image = batch["observation.images.base_0_rgb"]
        marker = image.to(device=noise.device).mean(dim=(1, 2, 3))
        value = marker + noise[:, 0, 0] * 1e-3
        return value[:, None, None].expand(-1, 50, 7).contiguous()


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
    assert reward_credit_environment_seed(7, "libero_goal", 4, 11, 9) == 2993136934
    assert (
        reward_credit_policy_noise_seed(13, "libero_goal", 4, 11, 9, 3)
        == 3231831300698984293
    )


def test_k4_rollout_compacts_active_lanes_without_crossing_replay_identity() -> None:
    envs = tuple(
        _FakeEnvironment(success_after_policy_steps=value, marker=lane * 20)
        for lane, value in enumerate((1, 6, None, 11))
    )
    policy = _FakePolicy()
    trajectories = collect_randomized_reward_trajectories(
        envs=envs,
        policy=policy,
        preprocess=_preprocess,
        postprocess=lambda value: value,
        suite="libero_spatial",
        task_id=6,
        global_task_id=6,
        language="put the bowl on the tray",
        adaptation_seed=23,
        rollout_cursors=(0, 1, 2, 3),
        env_seeds=(29, 31, 37, 41),
        policy_seed_root=43,
        device=torch.device("cpu"),
        max_horizon=12,
        dummy_settling_steps=10,
        dummy_action=[0, 0, 0, 0, 0, 0, -1],
        action_execution_horizon=5,
        num_inference_steps=10,
    )
    assert len(trajectories) == 4
    assert [value.success for value in trajectories] == [True, True, False, True]
    assert [value.valid_action_steps for value in trajectories] == [
        (1,),
        (5, 1),
        (5, 5, 2),
        (5, 5, 1),
    ]
    assert [value.rollout_cursor for value in trajectories] == [0, 1, 2, 3]
    assert [noise.shape[0] for noise in policy.noises] == [4, 3, 2]
    assert policy.reset_count == 1
    for lane, (trajectory, env) in enumerate(zip(trajectories, envs, strict=True)):
        expected_seeds = tuple(
            reward_credit_policy_noise_seed(
                43, "libero_spatial", 6, 23, lane, replan
            )
            for replan in range(len(trajectory.policy_noise_seeds))
        )
        assert trajectory.policy_noise_seeds == expected_seeds
        environment_rows = [value for name, value in env.events if name == "policy"]
        replay_rows = [
            action[0, step].numpy()
            for action, valid in zip(
                trajectory.action_chunks,
                trajectory.valid_action_steps,
                strict=True,
            )
            for step in range(valid)
        ]
        assert len(environment_rows) == len(replay_rows)
        for environment, replay in zip(environment_rows, replay_rows, strict=True):
            np.testing.assert_allclose(environment, replay)


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
        policy_noise_seeds=tuple(range(len(valid))),
        observations=observations if success else (),
        action_chunks=(
            tuple(torch.zeros((1, 50, 7)) for _ in valid) if success else ()
        ),
        valid_action_steps=valid if success else (),
    )


def test_success_batch_retains_only_success_prefixes_and_original_episode_ids() -> None:
    batch, episode_ids, successes, panel_rows, panel_chunks = (
        successful_trajectory_batch(
            (
                _trajectory((5, 2)),
                _trajectory((5,), success=False),
                _trajectory((4,)),
                _trajectory((5, 5), success=False),
            ),
            torch.device("cpu"),
        )
    )
    assert batch[ACTION].shape == (3, 50, 7)
    assert batch["executed_action_steps"].tolist() == [5, 2, 4]
    assert episode_ids.tolist() == [0, 0, 2]
    assert successes.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert panel_rows.tolist() == [0, 1, 3]
    assert panel_chunks == 6


def test_all_failure_batch_contains_no_replay_tensors() -> None:
    batch, episode_ids, successes, panel_rows, panel_chunks = (
        successful_trajectory_batch(
            tuple(_trajectory((5, 2), success=False) for _ in range(4)),
            torch.device("cpu"),
        )
    )
    assert batch == {}
    assert episode_ids.numel() == 0
    assert successes.tolist() == [0.0] * 4
    assert panel_rows.numel() == 0
    assert panel_chunks == 8


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
