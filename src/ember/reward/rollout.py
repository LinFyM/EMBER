"""Official random-reset LIBERO trajectories for PI05 reward adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from lerobot.utils.constants import ACTION

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
    if "is_success" in info:
        return _single_bool(info["is_success"])
    return False


def _cpu_tensor_rows(
    batch: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], ...]:
    tensors = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    sizes = {int(value.shape[0]) for value in tensors.values() if value.ndim > 0}
    if not tensors or any(value.ndim == 0 for value in tensors.values()) or len(sizes) != 1:
        raise RewardProtocolError("PI05 reward rollout lost its tensor batch")
    size = sizes.pop()
    if size <= 0:
        raise RewardProtocolError("PI05 reward rollout produced an empty tensor batch")
    return tuple(
        {name: value[row : row + 1] for name, value in tensors.items()}
        for row in range(size)
    )


def _flow_noise_cpu(*, seed: int, chunk_size: int, max_action_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        (1, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


class RandomResetEnvironmentPool:
    """Retain independent persistent LIBERO lanes for each task.

    Each of the four reward-credit episodes owns a separate environment lane,
    so keyed random resets never depend on another episode's termination time.
    No fixed benchmark init state is read.
    """

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
        if not 0 <= lane < 4:
            raise RewardProtocolError("random-reset environment lane must be in 0..3")
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


@dataclass
class RewardTrajectory:
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
    observations: tuple[dict[str, torch.Tensor], ...]
    action_chunks: tuple[torch.Tensor, ...]
    valid_action_steps: tuple[int, ...]


@dataclass
class _RewardLaneState:
    env: Any
    rollout_cursor: int
    env_seed: int
    observation: Mapping[str, Any]
    replay_observations: list[dict[str, torch.Tensor]] = field(default_factory=list)
    replay_actions: list[torch.Tensor] = field(default_factory=list)
    valid_action_steps: list[int] = field(default_factory=list)
    noise_seeds: list[int] = field(default_factory=list)
    reward_sum: float = 0.0
    steps: int = 0
    success: bool = False

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


def _trajectory_result(
    *,
    suite: str,
    task_id: int,
    global_task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
    env_seed: int,
    policy_seed_root: int,
    success: bool,
    steps: int,
    reward_sum: float,
    dummy_settling_steps: int,
    noise_seeds: Sequence[int],
    observations: Sequence[dict[str, torch.Tensor]],
    action_chunks: Sequence[torch.Tensor],
    valid_action_steps: Sequence[int],
) -> RewardTrajectory:
    return RewardTrajectory(
        suite=suite,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=adaptation_seed,
        rollout_cursor=rollout_cursor,
        env_seed=env_seed,
        policy_seed_root=policy_seed_root,
        success=success,
        steps=steps,
        reward_sum=reward_sum,
        dummy_settling_steps=dummy_settling_steps,
        policy_noise_seeds=tuple(noise_seeds),
        observations=tuple(observations),
        action_chunks=tuple(action_chunks),
        valid_action_steps=tuple(valid_action_steps),
    )


def _validate_k4_panel(
    envs: Sequence[Any],
    rollout_cursors: Sequence[int],
    env_seeds: Sequence[int],
) -> None:
    lane_count = len(envs)
    if (
        lane_count != 4
        or len(rollout_cursors) != lane_count
        or len(env_seeds) != lane_count
        or len(set(rollout_cursors)) != lane_count
        or len({id(env) for env in envs}) != lane_count
        or len(set(int(seed) for seed in env_seeds)) != lane_count
    ):
        raise RewardProtocolError("reward-credit rollout panel must be exact K4")


def _initialize_reward_lanes(
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
                env,
                env_seed=int(env_seed),
                dummy=dummy,
                steps=dummy_settling_steps,
            ),
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
) -> tuple[
    tuple[dict[str, torch.Tensor], ...], torch.Tensor, np.ndarray, tuple[int, ...]
]:
    prepared: list[dict[str, torch.Tensor]] = []
    noises: list[torch.Tensor] = []
    current_seeds: list[int] = []
    for lane in active:
        prepared.append(preprocess(libero_policy_input(lane.observation, language)))
        seed = policy_seed_fn(
            policy_seed_root,
            suite,
            task_id,
            adaptation_seed,
            lane.rollout_cursor,
            len(lane.noise_seeds),
        )
        current_seeds.append(seed)
        noises.append(
            _flow_noise_cpu(
                seed=seed,
                chunk_size=int(policy.config.chunk_size),
                max_action_dim=int(policy.config.max_action_dim),
            )
        )
    keys = set(prepared[0])
    if any(set(batch) != keys for batch in prepared):
        raise RewardProtocolError("batched reward rollout observation keys changed")
    policy_batch = {
        name: torch.cat([batch[name] for batch in prepared], dim=0)
        for name in sorted(keys)
    }
    stored = _cpu_tensor_rows(policy_batch)
    noise = torch.cat(noises, dim=0).to(device=device)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(
            policy_batch,
            noise=noise,
            num_steps=num_inference_steps,
        )
    if normalized.shape != (len(active), int(policy.config.chunk_size), 7):
        raise RewardProtocolError(
            "batched PI05 reward policy returned an invalid action chunk"
        )
    normalized = normalized.detach()
    staged = torch.cat((normalized, postprocess(normalized)), dim=-1).to(device="cpu")
    return (
        stored,
        staged[:, :, :7].contiguous(),
        staged[:, :, 7:].numpy(),
        tuple(current_seeds),
    )


def _advance_reward_lane(
    *,
    lane: _RewardLaneState,
    stored: dict[str, torch.Tensor],
    normalized_chunk: torch.Tensor,
    environment_actions: np.ndarray,
    noise_seed: int,
    action_execution_horizon: int,
    max_horizon: int,
) -> None:
    lane.replay_observations.append(stored)
    lane.replay_actions.append(normalized_chunk)
    lane.noise_seeds.append(noise_seed)
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
    lane.valid_action_steps.append(executed)


def _finalize_reward_trajectories(
    *,
    lanes: Sequence[_RewardLaneState],
    suite: str,
    task_id: int,
    global_task_id: int,
    adaptation_seed: int,
    policy_seed_root: int,
    dummy_settling_steps: int,
    retain_failure_replay: bool,
) -> tuple[RewardTrajectory, ...]:
    trajectories = []
    for lane in lanes:
        if not lane.replay_observations:
            raise RewardProtocolError("batched reward lane made no policy observation")
        if not lane.success and not retain_failure_replay:
            lane.replay_observations.clear()
            lane.replay_actions.clear()
            lane.valid_action_steps.clear()
        trajectories.append(
            _trajectory_result(
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
                noise_seeds=lane.noise_seeds,
                observations=lane.replay_observations,
                action_chunks=lane.replay_actions,
                valid_action_steps=lane.valid_action_steps,
            )
        )
    return tuple(trajectories)


def collect_randomized_reward_trajectories(
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
    retain_failure_replay: bool = True,
    policy_seed_fn: Callable[..., int] = reward_credit_policy_noise_seed,
) -> tuple[RewardTrajectory, ...]:
    """Collect the sealed K4 task panel with one batched policy forward per replan.

    All lanes share the already-installed task LoRA, while reset and PI05 flow
    noise remain episode-local.  Active-lane compaction only changes the physical
    batch shape; it never changes a lane's keyed randomness or replay identity.
    """

    _validate_k4_panel(envs, rollout_cursors, env_seeds)
    lanes = _initialize_reward_lanes(
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
        stored, normalized, environment_actions, seeds = _batched_policy_replan(
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
        for batch_row, lane in enumerate(active):
            _advance_reward_lane(
                lane=lane,
                stored=stored[batch_row],
                normalized_chunk=normalized[batch_row : batch_row + 1].contiguous(),
                environment_actions=environment_actions[batch_row],
                noise_seed=seeds[batch_row],
                action_execution_horizon=action_execution_horizon,
                max_horizon=max_horizon,
            )
    return _finalize_reward_trajectories(
        lanes=lanes,
        suite=suite,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=adaptation_seed,
        policy_seed_root=policy_seed_root,
        dummy_settling_steps=dummy_settling_steps,
        retain_failure_replay=retain_failure_replay,
    )


def complete_trajectory_batch(
    trajectories: Sequence[RewardTrajectory], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate successful and failed on-policy prefixes for K4 reward credit."""

    if len(trajectories) < 2 or any(
        not trajectory.observations
        or len(trajectory.observations) != len(trajectory.action_chunks)
        or len(trajectory.observations) != len(trajectory.valid_action_steps)
        for trajectory in trajectories
    ):
        raise RewardProtocolError("reward credit requires complete K4 replay")
    chunks = [
        (episode, observation, action, valid)
        for episode, trajectory in enumerate(trajectories)
        for observation, action, valid in zip(
            trajectory.observations,
            trajectory.action_chunks,
            trajectory.valid_action_steps,
            strict=True,
        )
    ]
    valid = torch.tensor(
        [count for _, _, _, count in chunks], dtype=torch.long, device=device
    )
    episode_ids = torch.tensor(
        [episode for episode, _, _, _ in chunks], dtype=torch.long, device=device
    )
    successes = torch.tensor(
        [trajectory.success for trajectory in trajectories],
        dtype=torch.float32,
        device=device,
    )
    if bool((valid <= 0).any()):
        raise RewardProtocolError("PI05 reward replay executed prefix is invalid")
    if bool((successes == successes[0]).all()):
        return {"executed_action_steps": valid}, episode_ids, successes
    keys = set(chunks[0][1])
    if any(set(observation) != keys for _, observation, _, _ in chunks):
        raise RewardProtocolError("PI05 reward replay observation keys changed")
    batch = {
        key: torch.cat([observation[key] for _, observation, _, _ in chunks]).to(device)
        for key in sorted(keys)
    }
    actions = torch.cat([action for _, _, action, _ in chunks]).to(device)
    if actions.ndim != 3 or bool((valid > actions.shape[1]).any()):
        raise RewardProtocolError("PI05 reward replay executed prefix is invalid")
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    batch["action_is_pad"] = (
        torch.arange(actions.shape[1], device=device)[None] >= valid[:, None]
    )
    return batch, episode_ids, successes
