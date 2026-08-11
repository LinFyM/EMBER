"""Exact K2 random-reset rollout arm for PCUG paired candidate tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ember.pi05_assets import configure_libero_runtime_assets
from ember.pi05_processing import libero_policy_input
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    reward_credit_policy_noise_seed,
)


def _single_bool(value: Any) -> bool:
    array = np.asarray(value)
    return bool(array.size and array.reshape(-1)[0])


def _transition(value: Any) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise RewardProtocolError("LIBERO reward env must return (obs,reward,done,info)")
    observation, reward, done, info = value
    if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
        raise RewardProtocolError("LIBERO reward transition changed type")
    return observation, float(np.asarray(reward).reshape(-1)[0]), _single_bool(done), info


def _success(done: bool, reward: float, info: Mapping[str, Any]) -> bool:
    if done or reward > 0:
        return True
    return _single_bool(info["is_success"]) if "is_success" in info else False


def _flow_noise_cpu(*, seed: int, chunk_size: int, max_action_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (1, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


class RandomResetEnvironmentPool:
    """Retain two independent environment lanes per train task."""

    def __init__(
        self, *, bddl_root: Path, assets_root: Path, render_resolution: int
    ) -> None:
        if (
            not bddl_root.is_dir()
            or not assets_root.is_dir()
            or render_resolution != 256
        ):
            raise RewardProtocolError("invalid PI05 random-reset environment authority")
        configure_libero_runtime_assets(assets_root)
        self.bddl_root = bddl_root.resolve()
        self.assets_root = assets_root.resolve()
        self.render_resolution = render_resolution
        self._envs: dict[tuple[int, int], Any] = {}
        self._validated_tasks: set[int] = set()

    def _new_environment(self, path: Path) -> Any:
        from libero.libero.envs import OffScreenRenderEnv

        return OffScreenRenderEnv(
            bddl_file_name=path,
            camera_heights=self.render_resolution,
            camera_widths=self.render_resolution,
        )

    def get(self, task: RewardTask, *, lane: int = 0) -> Any:
        if not 0 <= lane < 2:
            raise RewardProtocolError("PCUG random-reset lane must be in 0..1")
        key = (task.global_task_id, lane)
        if key in self._envs:
            return self._envs[key]
        path = self.bddl_root / task.problem_folder / task.bddl_file
        if task.global_task_id not in self._validated_tasks:
            if not path.is_file() or path.stat().st_size != task.bddl_bytes:
                raise RewardProtocolError(
                    f"installed reward BDDL changed: {task.suite}/{task.task_id}"
                )
            self._validated_tasks.add(task.global_task_id)
        env = self._new_environment(path)
        self._envs[key] = env
        return env

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._validated_tasks.clear()

    def __enter__(self) -> "RandomResetEnvironmentPool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class RewardRolloutOutcome:
    suite: str
    task_id: int
    global_task_id: int
    adaptation_seed: int
    rollout_cursor: int
    env_seed: int
    policy_seed_root: int
    success: bool
    steps: int
    reward_sum: float
    dummy_settling_steps: int
    policy_noise_seeds: tuple[int, ...]
    initial_normalized_action_chunk: torch.Tensor


@dataclass
class _RewardLaneState:
    env: Any
    rollout_cursor: int
    env_seed: int
    observation: Mapping[str, Any]
    noise_seeds: list[int]
    reward_sum: float = 0.0
    steps: int = 0
    success: bool = False
    initial_action_chunk: torch.Tensor | None = None

    def is_active(self, max_horizon: int) -> bool:
        return not self.success and self.steps < max_horizon


def _validate_rollout_contract(
    *,
    policy: torch.nn.Module,
    task_id: int,
    global_task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
    env_seed: int,
    policy_seed_root: int,
    max_horizon: int,
    dummy_settling_steps: int,
    action_execution_horizon: int,
    num_inference_steps: int,
    dummy_action: Sequence[float],
) -> np.ndarray:
    identifiers = (
        task_id,
        global_task_id,
        adaptation_seed,
        rollout_cursor,
        env_seed,
        policy_seed_root,
    )
    if (
        min(identifiers) < 0
        or max_horizon <= 0
        or dummy_settling_steps != 10
        or action_execution_horizon != 5
        or int(policy.config.chunk_size) != 50
        or num_inference_steps != 10
    ):
        raise RewardProtocolError("PI05 reward rollout differs from the sealed recipe")
    dummy = np.asarray(dummy_action, dtype=np.float32)
    if dummy.shape != (7,):
        raise RewardProtocolError("PI05 reward dummy action must have seven values")
    return dummy


def _random_reset_with_settling(
    env: Any, *, env_seed: int, dummy: np.ndarray, steps: int
) -> Mapping[str, Any]:
    env.seed(env_seed)
    reset = env.reset()
    observation = reset[0] if isinstance(reset, tuple) else reset
    if not isinstance(observation, Mapping):
        raise RewardProtocolError("LIBERO random reset returned no observation")
    for _ in range(steps):
        observation, reward, done, info = _transition(env.step(dummy))
        if _success(done, reward, info):
            raise RewardProtocolError("LIBERO task succeeded during dummy settling")
    return observation


def _validate_paired_arm_panel(
    envs: Sequence[Any],
    rollout_cursors: Sequence[int],
    env_seeds: Sequence[int],
) -> None:
    if (
        len(envs) != 2
        or len(rollout_cursors) != 2
        or len(env_seeds) != 2
        or len(set(int(value) for value in rollout_cursors)) != 2
        or len(set(int(value) for value in env_seeds)) != 2
        or len({id(env) for env in envs}) != 2
    ):
        raise RewardProtocolError("PCUG rollout arm must be exact K2")


def _initialize_lanes(
    *,
    envs: Sequence[Any],
    policy: torch.nn.Module,
    task_id: int,
    global_task_id: int,
    adaptation_seed: int,
    rollout_cursors: Sequence[int],
    env_seeds: Sequence[int],
    policy_seed_root: int,
    max_horizon: int,
    dummy_settling_steps: int,
    action_execution_horizon: int,
    num_inference_steps: int,
    dummy_action: Sequence[float],
) -> list[_RewardLaneState]:
    dummies = [
        _validate_rollout_contract(
            policy=policy,
            task_id=task_id,
            global_task_id=global_task_id,
            adaptation_seed=adaptation_seed,
            rollout_cursor=int(cursor),
            env_seed=int(env_seed),
            policy_seed_root=policy_seed_root,
            max_horizon=max_horizon,
            dummy_settling_steps=dummy_settling_steps,
            action_execution_horizon=action_execution_horizon,
            num_inference_steps=num_inference_steps,
            dummy_action=dummy_action,
        )
        for cursor, env_seed in zip(rollout_cursors, env_seeds, strict=True)
    ]
    return [
        _RewardLaneState(
            env=env,
            rollout_cursor=int(cursor),
            env_seed=int(env_seed),
            observation=_random_reset_with_settling(
                env, env_seed=int(env_seed), dummy=dummy, steps=dummy_settling_steps
            ),
            noise_seeds=[],
        )
        for env, cursor, env_seed, dummy in zip(
            envs, rollout_cursors, env_seeds, dummies, strict=True
        )
    ]


def _batched_policy_replan(
    *,
    active: Sequence[_RewardLaneState],
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    suite: str,
    task_id: int,
    language: str,
    adaptation_seed: int,
    policy_seed_root: int,
    device: torch.device,
    num_inference_steps: int,
    policy_seed_fn: Callable[..., int],
) -> tuple[torch.Tensor, np.ndarray, tuple[int, ...]]:
    prepared = [preprocess(libero_policy_input(lane.observation, language)) for lane in active]
    keys = set(prepared[0])
    if any(set(batch) != keys for batch in prepared):
        raise RewardProtocolError("batched reward rollout observation keys changed")
    policy_batch = {
        name: torch.cat([batch[name] for batch in prepared], dim=0)
        for name in sorted(keys)
    }
    seeds = tuple(
        policy_seed_fn(
            policy_seed_root,
            suite,
            task_id,
            adaptation_seed,
            lane.rollout_cursor,
            len(lane.noise_seeds),
        )
        for lane in active
    )
    noise = torch.cat(
        [
            _flow_noise_cpu(
                seed=seed,
                chunk_size=int(policy.config.chunk_size),
                max_action_dim=int(policy.config.max_action_dim),
            )
            for seed in seeds
        ],
        dim=0,
    ).to(device=device)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(
            policy_batch, noise=noise, num_steps=num_inference_steps
        )
    if normalized.shape != (len(active), int(policy.config.chunk_size), 7):
        raise RewardProtocolError(
            "batched PI05 reward policy returned an invalid action chunk"
        )
    normalized = normalized.detach()
    actions = postprocess(normalized).to(device="cpu").numpy()
    return normalized, actions, seeds


def _advance_lane(
    *,
    lane: _RewardLaneState,
    normalized_chunk: torch.Tensor,
    environment_actions: np.ndarray,
    noise_seed: int,
    action_execution_horizon: int,
    max_horizon: int,
) -> None:
    lane.noise_seeds.append(noise_seed)
    if lane.initial_action_chunk is None:
        lane.initial_action_chunk = normalized_chunk.detach().clone()
    executed = 0
    for action in environment_actions[:action_execution_horizon]:
        observation, reward, done, info = _transition(lane.env.step(action))
        lane.observation = observation
        lane.reward_sum += reward
        lane.steps += 1
        executed += 1
        lane.success = _success(done, reward, info)
        if lane.success or lane.steps >= max_horizon:
            break
    if executed <= 0:
        raise RewardProtocolError("PCUG reward lane executed no action")


def _finalize_outcomes(
    *,
    lanes: Sequence[_RewardLaneState],
    suite: str,
    task_id: int,
    global_task_id: int,
    adaptation_seed: int,
    policy_seed_root: int,
    dummy_settling_steps: int,
) -> tuple[RewardRolloutOutcome, ...]:
    result = []
    for lane in lanes:
        if not lane.noise_seeds or lane.initial_action_chunk is None:
            raise RewardProtocolError("PCUG reward lane made no policy observation")
        result.append(
            RewardRolloutOutcome(
                suite=suite,
                task_id=task_id,
                global_task_id=global_task_id,
                adaptation_seed=adaptation_seed,
                rollout_cursor=lane.rollout_cursor,
                env_seed=lane.env_seed,
                policy_seed_root=policy_seed_root,
                success=lane.success,
                steps=lane.steps,
                reward_sum=lane.reward_sum,
                dummy_settling_steps=dummy_settling_steps,
                policy_noise_seeds=tuple(lane.noise_seeds),
                initial_normalized_action_chunk=lane.initial_action_chunk,
            )
        )
    return tuple(result)


def collect_paired_reward_arm_outcomes(
    *,
    envs: Sequence[Any],
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    suite: str,
    task_id: int,
    global_task_id: int,
    language: str,
    adaptation_seed: int,
    rollout_cursors: Sequence[int],
    env_seeds: Sequence[int],
    policy_seed_root: int,
    device: torch.device,
    max_horizon: int,
    dummy_settling_steps: int,
    dummy_action: Sequence[float],
    action_execution_horizon: int,
    num_inference_steps: int,
    policy_seed_fn: Callable[..., int] = reward_credit_policy_noise_seed,
) -> tuple[RewardRolloutOutcome, ...]:
    """Run one exact K2 arm; a second call with the same keys forms its pair."""

    _validate_paired_arm_panel(envs, rollout_cursors, env_seeds)
    lanes = _initialize_lanes(
        envs=envs,
        policy=policy,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=adaptation_seed,
        rollout_cursors=rollout_cursors,
        env_seeds=env_seeds,
        policy_seed_root=policy_seed_root,
        max_horizon=max_horizon,
        dummy_settling_steps=dummy_settling_steps,
        action_execution_horizon=action_execution_horizon,
        num_inference_steps=num_inference_steps,
        dummy_action=dummy_action,
    )
    policy.reset()
    while True:
        active = [lane for lane in lanes if lane.is_active(max_horizon)]
        if not active:
            break
        normalized, actions, seeds = _batched_policy_replan(
            active=active,
            policy=policy,
            preprocess=preprocess,
            postprocess=postprocess,
            suite=suite,
            task_id=task_id,
            language=language,
            adaptation_seed=adaptation_seed,
            policy_seed_root=policy_seed_root,
            device=device,
            num_inference_steps=num_inference_steps,
            policy_seed_fn=policy_seed_fn,
        )
        for row, lane in enumerate(active):
            _advance_lane(
                lane=lane,
                normalized_chunk=normalized[row : row + 1],
                environment_actions=actions[row],
                noise_seed=seeds[row],
                action_execution_horizon=action_execution_horizon,
                max_horizon=max_horizon,
            )
    return _finalize_outcomes(
        lanes=lanes,
        suite=suite,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=adaptation_seed,
        policy_seed_root=policy_seed_root,
        dummy_settling_steps=dummy_settling_steps,
    )
