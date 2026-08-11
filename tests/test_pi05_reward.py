from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ember.pi05_processing import libero_policy_input
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
    collect_paired_reward_arm_outcomes,
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
    lanes = tuple(pool.get(task, lane=lane) for lane in range(2))
    assert all(lanes[lane] is pool.get(task, lane=lane) for lane in range(2))
    assert len({id(value) for value in lanes}) == 2
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
        self.policy_steps = 0
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


def test_paired_k2_arm_compacts_lanes_and_keeps_initial_action() -> None:
    envs = tuple(
        _FakeEnvironment(success_after_policy_steps=value, marker=lane * 20)
        for lane, value in enumerate((1, 6))
    )
    policy = _FakePolicy()
    outcomes = collect_paired_reward_arm_outcomes(
        envs=envs,
        policy=policy,
        preprocess=_preprocess,
        postprocess=lambda value: value,
        suite="libero_spatial",
        task_id=6,
        global_task_id=6,
        language="put the bowl on the tray",
        adaptation_seed=23,
        rollout_cursors=(0, 1),
        env_seeds=(29, 31),
        policy_seed_root=43,
        device=torch.device("cpu"),
        max_horizon=12,
        dummy_settling_steps=10,
        dummy_action=[0, 0, 0, 0, 0, 0, -1],
        action_execution_horizon=5,
        num_inference_steps=10,
    )
    assert len(outcomes) == 2
    assert [value.success for value in outcomes] == [True, True]
    assert [value.steps for value in outcomes] == [1, 6]
    assert [value.rollout_cursor for value in outcomes] == [0, 1]
    assert [noise.shape[0] for noise in policy.noises] == [2, 1]
    assert policy.reset_count == 1
    for lane, (outcome, env) in enumerate(zip(outcomes, envs, strict=True)):
        expected_seeds = tuple(
            reward_credit_policy_noise_seed(
                43, "libero_spatial", 6, 23, lane, replan
            )
            for replan in range(len(outcome.policy_noise_seeds))
        )
        assert outcome.policy_noise_seeds == expected_seeds
        environment_rows = [value for name, value in env.events if name == "policy"]
        assert len(environment_rows) == outcome.steps
        assert outcome.initial_normalized_action_chunk.shape == (1, 50, 7)
        assert set(vars(outcome)) == {
            "suite",
            "task_id",
            "global_task_id",
            "adaptation_seed",
            "rollout_cursor",
            "env_seed",
            "policy_seed_root",
            "success",
            "steps",
            "reward_sum",
            "dummy_settling_steps",
            "policy_noise_seeds",
            "initial_normalized_action_chunk",
        }


def test_repeated_k2_arms_with_same_keys_reproduce_noise_and_initial_actions() -> None:
    def collect():
        return collect_paired_reward_arm_outcomes(
            envs=tuple(
                _FakeEnvironment(success_after_policy_steps=None, marker=lane * 20)
                for lane in range(2)
            ),
            policy=_FakePolicy(),
            preprocess=_preprocess,
            postprocess=lambda value: value,
            suite="libero_goal",
            task_id=2,
            global_task_id=22,
            language="do the task",
            adaptation_seed=23,
            rollout_cursors=(0, 1),
            env_seeds=(29, 31),
            policy_seed_root=43,
            device=torch.device("cpu"),
            max_horizon=40,
            dummy_settling_steps=10,
            dummy_action=[0, 0, 0, 0, 0, 0, -1],
            action_execution_horizon=5,
            num_inference_steps=10,
        )

    first = collect()
    second = collect()
    assert [value.policy_noise_seeds for value in first] == [
        value.policy_noise_seeds for value in second
    ]
    for left, right in zip(first, second, strict=True):
        assert left.env_seed == right.env_seed
        assert left.rollout_cursor == right.rollout_cursor
        torch.testing.assert_close(
            left.initial_normalized_action_chunk,
            right.initial_normalized_action_chunk,
            rtol=0,
            atol=0,
        )
