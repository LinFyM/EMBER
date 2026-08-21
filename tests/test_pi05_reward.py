from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

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
    RewardTrajectory,
    capture_paired_initial_states,
    collect_paired_reward_arm_trajectories,
)
from ember.reward.occupancy_panel import complete_successful_expert_occupancy_batch


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
        return (
            _observation(self.marker + self.policy_steps),
            float(success),
            success,
            {},
        )


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
    expected = torch.from_numpy(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    )
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
    trajectories = collect_paired_reward_arm_trajectories(
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
    assert len(trajectories) == 2
    assert [value.success for value in trajectories] == [True, True]
    assert [value.steps for value in trajectories] == [1, 6]
    assert [value.rollout_cursor for value in trajectories] == [0, 1]
    assert [noise.shape[0] for noise in policy.noises] == [2, 1]
    assert policy.reset_count == 1
    for lane, (trajectory, env) in enumerate(zip(trajectories, envs, strict=True)):
        expected_seeds = tuple(
            reward_credit_policy_noise_seed(43, "libero_spatial", 6, 23, lane, replan)
            for replan in range(len(trajectory.policy_noise_seeds))
        )
        assert trajectory.policy_noise_seeds == expected_seeds
        environment_rows = [value for name, value in env.events if name == "policy"]
        assert len(environment_rows) == trajectory.steps
        assert trajectory.action_chunks[0].shape == (1, 50, 7)
        assert set(vars(trajectory)) == {
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
            "observations",
            "action_chunks",
            "valid_action_steps",
            "goal_predicate_count",
            "goal_predicate_peak",
        }


def test_paired_k2_rollout_retains_executed_prefixes_for_credit() -> None:
    envs = tuple(
        _FakeEnvironment(success_after_policy_steps=value, marker=lane * 20)
        for lane, value in enumerate((1, None))
    )
    policy = _FakePolicy()
    trajectories = collect_paired_reward_arm_trajectories(
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
    assert [value.success for value in trajectories] == [True, False]
    assert [value.valid_action_steps for value in trajectories] == [
        (1,),
        (5, 5, 2),
    ]
    assert [noise.shape[0] for noise in policy.noises] == [2, 1, 1]


def _trajectory(*, success: bool, marker: float) -> RewardTrajectory:
    observation = {
        OBS_LANGUAGE_TOKENS: torch.ones((1, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 2), dtype=torch.bool),
        "observation.images.base_0_rgb": torch.zeros((1, 3, 2, 2)),
    }
    return RewardTrajectory(
        suite="libero_goal",
        task_id=4,
        global_task_id=24,
        adaptation_seed=3,
        rollout_cursor=5,
        env_seed=7,
        policy_seed_root=11,
        success=success,
        steps=5,
        reward_sum=float(success),
        dummy_settling_steps=10,
        policy_noise_seeds=(13,),
        observations=(observation,),
        action_chunks=(torch.full((1, 50, 7), marker),),
        valid_action_steps=(5,),
    )


def test_successful_expert_occupancy_uses_matched_requery_not_stored_action() -> None:
    trajectory = _trajectory(success=True, marker=2.0)
    batch, trajectory_ids, metrics = complete_successful_expert_occupancy_batch(
        (trajectory,),
        {
            "expert": ((torch.full((1, 50, 7), 5.0),),),
            "student": ((torch.full((1, 50, 7), 4.0),),),
        },
        strata_per_trajectory=8,
        device=torch.device("cpu"),
    )
    assert batch[ACTION].shape == (1, 50, 7)
    assert batch["executed_action_steps"].tolist() == [5]
    assert batch["policy_noise_seed"].tolist() == [13]
    assert torch.equal(batch[ACTION][0], torch.full((50, 7), 5.0))
    assert trajectory_ids.tolist() == [0]
    assert metrics["selected_replan_indices"] == [[0]]


def test_successful_expert_occupancy_selects_maximum_in_each_progress_bin() -> None:
    base = _trajectory(success=True, marker=-1.0)
    count = 17
    trajectory = replace(
        base,
        observations=base.observations * count,
        action_chunks=base.action_chunks * count,
        valid_action_steps=(5,) * count,
        policy_noise_seeds=tuple(range(count)),
    )
    student = tuple(torch.zeros((1, 50, 7)) for _ in range(count))
    expert = tuple(
        torch.full((1, 50, 7), float(index + 1)) for index in range(count)
    )
    batch, trajectory_ids, metrics = complete_successful_expert_occupancy_batch(
        (trajectory,),
        {"expert": (expert,), "student": (student,)},
        strata_per_trajectory=8,
        device=torch.device("cpu"),
    )
    assert metrics["selected_replan_indices"] == [[1, 3, 5, 7, 9, 11, 13, 16]]
    assert metrics["selected_credit_states"] == 8
    assert trajectory_ids.tolist() == [0] * 8
    assert batch["policy_noise_seed"].tolist() == [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        16,
    ]
    assert batch[ACTION][:, 0, 0].tolist() == [2, 4, 6, 8, 10, 12, 14, 17]


def test_repeated_k2_arms_with_same_keys_reproduce_noise_and_initial_actions() -> None:
    def collect():
        return collect_paired_reward_arm_trajectories(
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
            left.action_chunks[0],
            right.action_chunks[0],
            rtol=0,
            atol=0,
        )


def test_paired_arms_restore_one_captured_post_settling_state() -> None:
    class StatefulEnvironment(_FakeEnvironment):
        def __init__(self, *, marker: int) -> None:
            super().__init__(success_after_policy_steps=None, marker=marker)
            self.env = self
            self.deterministic_reset = False
            self.reset_generation = 0
            self.model_marker = 0

        def reset(self) -> dict[str, np.ndarray]:
            self.events.append(("reset", None))
            self.policy_steps = 0
            if not self.deterministic_reset:
                self.reset_generation += 1
                self.model_marker += self.reset_generation
            return _observation(self.marker + self.model_marker)

        def get_sim_state(self) -> np.ndarray:
            return np.array([self.marker, self.policy_steps], dtype=np.float64)

        def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
            self.events.append(("restore", None))
            self.marker = int(state[0])
            self.policy_steps = int(state[1])
            return _observation(self.marker + self.model_marker)

    envs = tuple(StatefulEnvironment(marker=lane * 20) for lane in range(2))
    env_seeds = (29, 31)
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    initial_states = capture_paired_initial_states(
        envs,
        env_seeds,
        dummy_action=dummy_action,
        dummy_settling_steps=10,
    )

    def collect():
        return collect_paired_reward_arm_trajectories(
            envs=envs,
            policy=_FakePolicy(),
            preprocess=_preprocess,
            postprocess=lambda value: value,
            suite="libero_goal",
            task_id=2,
            global_task_id=22,
            language="do the task",
            adaptation_seed=23,
            rollout_cursors=(0, 1),
            env_seeds=env_seeds,
            policy_seed_root=43,
            device=torch.device("cpu"),
            max_horizon=5,
            dummy_settling_steps=10,
            dummy_action=dummy_action,
            action_execution_horizon=5,
            num_inference_steps=10,
            initial_states=initial_states,
        )

    reference = collect()
    candidate = collect()
    for left, right, env in zip(reference, candidate, envs, strict=True):
        assert all(
            torch.equal(left.observations[0][key], right.observations[0][key])
            for key in left.observations[0]
        )
        assert sum(name == "dummy" for name, _ in env.events) == 10
        assert sum(name == "restore" for name, _ in env.events) == 2
        assert env.reset_generation == 1
    winner = replace(reference[0], success=True)
    complete_successful_expert_occupancy_batch(
        (winner,),
        {
            "expert": ((winner.action_chunks[0],),),
            "student": ((candidate[0].action_chunks[0],),),
        },
        strata_per_trajectory=8,
        device=torch.device("cpu"),
    )
