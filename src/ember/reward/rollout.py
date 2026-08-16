"""Persistent random-reset LIBERO rollout lanes and on-policy replay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from lerobot.utils.constants import ACTION

from ember.lora import copy_task_lora_state_
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
        raise RewardProtocolError(
            "LIBERO reward env must return (obs,reward,done,info)"
        )
    observation, reward, done, info = value
    if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
        raise RewardProtocolError("LIBERO reward transition changed type")
    return (
        observation,
        float(np.asarray(reward).reshape(-1)[0]),
        _single_bool(done),
        info,
    )


def _success(done: bool, reward: float, info: Mapping[str, Any]) -> bool:
    if done or reward > 0:
        return True
    return _single_bool(info["is_success"]) if "is_success" in info else False


def _cpu_tensor_rows(
    batch: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], ...]:
    tensors = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    sizes = {int(value.shape[0]) for value in tensors.values() if value.ndim > 0}
    if (
        not tensors
        or any(value.ndim == 0 for value in tensors.values())
        or len(sizes) != 1
    ):
        raise RewardProtocolError("PI05 reward rollout lost its tensor batch")
    size = sizes.pop()
    return tuple(
        {name: value[row : row + 1] for name, value in tensors.items()}
        for row in range(size)
    )


def policy_flow_noise_cpu(*, seed: int, chunk_size: int, max_action_dim: int) -> torch.Tensor:
    """Recreate one PI05 action-sampling noise tensor from its retained seed."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (1, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


class RandomResetEnvironmentPool:
    """Retain the two paired reset lanes used by both PCSD policy arms."""

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
            raise RewardProtocolError("random-reset lane must be in 0..1")
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
class RewardTrajectory:
    """One rollout plus every executed-prefix policy query used for credit."""

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
    noise_seeds: list[int]
    replay_observations: list[dict[str, torch.Tensor]] = field(default_factory=list)
    replay_actions: list[torch.Tensor] = field(default_factory=list)
    valid_action_steps: list[int] = field(default_factory=list)
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


def capture_paired_initial_states(
    envs: Sequence[Any],
    env_seeds: Sequence[int],
    *,
    dummy_action: Sequence[float],
    dummy_settling_steps: int,
) -> tuple[np.ndarray, ...]:
    """Capture the two post-settling simulator states once for both policy arms."""

    if len(envs) != 2 or len(env_seeds) != 2 or dummy_settling_steps != 10:
        raise RewardProtocolError("paired initial-state panel changed")
    dummy = np.asarray(dummy_action, dtype=np.float32)
    states = []
    for env, env_seed in zip(envs, env_seeds, strict=True):
        _random_reset_with_settling(
            env, env_seed=int(env_seed), dummy=dummy, steps=dummy_settling_steps
        )
        states.append(np.asarray(env.get_sim_state()).copy())
    return tuple(states)


def _restore_initial_state(
    env: Any, *, initial_state: np.ndarray
) -> Mapping[str, Any]:
    # A LIBERO hard reset resamples fixture model poses that are absent from the
    # flattened simulator state. Soft deterministic reset clears controllers and
    # observables while retaining that captured model, then restores qpos/qvel.
    control_env = env.env
    deterministic_reset = bool(control_env.deterministic_reset)
    control_env.deterministic_reset = True
    try:
        env.reset()
    finally:
        control_env.deterministic_reset = deterministic_reset
    observation = env.set_init_state(initial_state)
    if not isinstance(observation, Mapping):
        raise RewardProtocolError("LIBERO state restore returned no observation")
    return observation


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
    initial_states: Sequence[np.ndarray] | None,
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
    if initial_states is not None and len(initial_states) != len(envs):
        raise RewardProtocolError("paired initial-state count changed")
    states: Sequence[np.ndarray | None] = (
        (None,) * len(envs) if initial_states is None else initial_states
    )
    return [
        _RewardLaneState(
            env=env,
            rollout_cursor=int(cursor),
            env_seed=int(env_seed),
            observation=(
                _random_reset_with_settling(
                    env,
                    env_seed=int(env_seed),
                    dummy=dummy,
                    steps=dummy_settling_steps,
                )
                if initial_state is None
                else _restore_initial_state(
                    env,
                    initial_state=initial_state,
                )
            ),
            noise_seeds=[],
        )
        for env, cursor, env_seed, dummy, initial_state in zip(
            envs, rollout_cursors, env_seeds, dummies, states, strict=True
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
    capture_replay: bool,
) -> tuple[
    tuple[dict[str, torch.Tensor], ...] | None,
    torch.Tensor,
    np.ndarray,
    tuple[int, ...],
]:
    prepared = [
        preprocess(libero_policy_input(lane.observation, language)) for lane in active
    ]
    keys = set(prepared[0])
    if any(set(batch) != keys for batch in prepared):
        raise RewardProtocolError("batched reward rollout observation keys changed")
    policy_batch = {
        name: torch.cat([batch[name] for batch in prepared], dim=0)
        for name in sorted(keys)
    }
    stored = _cpu_tensor_rows(policy_batch) if capture_replay else None
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
            policy_flow_noise_cpu(
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
    return stored, normalized, actions, seeds


def _advance_lane(
    *,
    lane: _RewardLaneState,
    normalized_chunk: torch.Tensor,
    environment_actions: np.ndarray,
    noise_seed: int,
    action_execution_horizon: int,
    max_horizon: int,
    stored: dict[str, torch.Tensor] | None = None,
) -> None:
    lane.noise_seeds.append(noise_seed)
    if stored is not None:
        lane.replay_observations.append(stored)
        lane.replay_actions.append(
            normalized_chunk.detach().to(device="cpu").contiguous()
        )
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
        raise RewardProtocolError("PCSD reward lane executed no action")
    if stored is not None:
        lane.valid_action_steps.append(executed)


def _validate_trajectory_panel(
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
        raise RewardProtocolError("PCSD trajectory arm must be exact K2")


def _collect_reward_trajectories(
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
    initial_states: Sequence[np.ndarray] | None = None,
    policy_seed_fn: Callable[..., int] = reward_credit_policy_noise_seed,
) -> tuple[RewardTrajectory, ...]:
    """Collect one arm and retain all successful and failed prefixes."""

    _validate_trajectory_panel(envs, rollout_cursors, env_seeds)
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
        initial_states=initial_states,
    )
    policy.reset()
    while True:
        active = [lane for lane in lanes if lane.is_active(max_horizon)]
        if not active:
            break
        stored, normalized, actions, seeds = _batched_policy_replan(
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
            capture_replay=True,
        )
        assert stored is not None
        for row, lane in enumerate(active):
            _advance_lane(
                lane=lane,
                stored=stored[row],
                normalized_chunk=normalized[row : row + 1],
                environment_actions=actions[row],
                noise_seed=seeds[row],
                action_execution_horizon=action_execution_horizon,
                max_horizon=max_horizon,
            )
    return tuple(
        RewardTrajectory(
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
            observations=tuple(lane.replay_observations),
            action_chunks=tuple(lane.replay_actions),
            valid_action_steps=tuple(lane.valid_action_steps),
        )
        for lane in lanes
    )


def collect_paired_reward_arm_trajectories(
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
    initial_states: Sequence[np.ndarray] | None = None,
    policy_seed_fn: Callable[..., int] = reward_credit_policy_noise_seed,
) -> tuple[RewardTrajectory, ...]:
    """Run one exact K2 arm; a second call with the same keys forms its pair."""

    return _collect_reward_trajectories(
        envs=envs,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        suite=suite,
        task_id=task_id,
        global_task_id=global_task_id,
        language=language,
        adaptation_seed=adaptation_seed,
        rollout_cursors=rollout_cursors,
        env_seeds=env_seeds,
        policy_seed_root=policy_seed_root,
        device=device,
        max_horizon=max_horizon,
        dummy_settling_steps=dummy_settling_steps,
        dummy_action=dummy_action,
        action_execution_horizon=action_execution_horizon,
        num_inference_steps=num_inference_steps,
        initial_states=initial_states,
        policy_seed_fn=policy_seed_fn,
    )


_SuccessfulOccupancyRecord = tuple[int, int, Mapping[str, torch.Tensor], int]


def _successful_occupancy_records(
    trajectories: Sequence[RewardTrajectory],
) -> list[_SuccessfulOccupancyRecord]:
    if len(trajectories) not in {1, 2}:
        raise RewardProtocolError("successful expert occupancy panel changed")
    records: list[_SuccessfulOccupancyRecord] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        if (
            not trajectory.success
            or len(trajectory.observations) != len(trajectory.policy_noise_seeds)
            or len(trajectory.action_chunks) != len(trajectory.observations)
            or len(trajectory.valid_action_steps) != len(trajectory.observations)
        ):
            raise RewardProtocolError("successful expert trajectory lost replay")
        records.extend(
            (trajectory_index, replan, observation, seed)
            for replan, (observation, seed) in enumerate(
                zip(
                    trajectory.observations,
                    trajectory.policy_noise_seeds,
                    strict=True,
                )
            )
        )
    return records


def _matched_occupancy_policy_batch(
    *,
    policy: torch.nn.Module,
    panel: Sequence[_SuccessfulOccupancyRecord],
    device: torch.device,
    num_inference_steps: int,
) -> torch.Tensor:
    keys = set(panel[0][2])
    if any(set(observation) != keys for _, _, observation, _ in panel):
        raise RewardProtocolError("matched occupancy observation keys changed")
    batch = {
        name: torch.cat([observation[name] for _, _, observation, _ in panel]).to(
            device, non_blocking=True
        )
        for name in sorted(keys)
    }
    noise = torch.cat(
        [
            policy_flow_noise_cpu(
                seed=seed,
                chunk_size=int(policy.config.chunk_size),
                max_action_dim=int(policy.config.max_action_dim),
            )
            for _, _, _, seed in panel
        ]
    ).to(device, non_blocking=True)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        actions = policy.predict_action_chunk(
            batch, noise=noise, num_steps=num_inference_steps
        )
    if actions.shape != (len(panel), int(policy.config.chunk_size), 7):
        raise RewardProtocolError("matched occupancy policy returned an invalid action chunk")
    return actions.detach().to(device="cpu").contiguous()


def query_successful_expert_occupancy_actions(
    *,
    policy: torch.nn.Module,
    lora_contract: Any,
    identity_state: Mapping[str, torch.Tensor],
    trajectories: Sequence[RewardTrajectory],
    expert_lora: Mapping[str, torch.Tensor],
    student_lora: Mapping[str, torch.Tensor],
    device: torch.device,
    microbatch_size: int,
    num_inference_steps: int,
) -> tuple[dict[str, tuple[tuple[torch.Tensor, ...], ...]], dict[str, Any]]:
    """Re-query expert and Writer with identical successful-occupancy batches."""

    if microbatch_size <= 0:
        raise RewardProtocolError("successful expert action batch size changed")
    records = _successful_occupancy_records(trajectories)
    outputs: dict[str, dict[tuple[int, int], torch.Tensor]] = {
        "expert": {},
        "student": {},
    }
    batch_sizes: dict[str, list[int]] = {"expert": [], "student": []}
    started = time.monotonic()
    try:
        for arm, lora in (("expert", expert_lora), ("student", student_lora)):
            copy_task_lora_state_(policy, lora, lora_contract)
            policy.reset()
            for start in range(0, len(records), microbatch_size):
                panel = records[start : start + microbatch_size]
                actions = _matched_occupancy_policy_batch(
                    policy=policy,
                    panel=panel,
                    device=device,
                    num_inference_steps=num_inference_steps,
                )
                for row, (trajectory_index, replan, _, _) in enumerate(panel):
                    outputs[arm][(trajectory_index, replan)] = actions[row : row + 1]
                batch_sizes[arm].append(len(panel))
    finally:
        copy_task_lora_state_(policy, identity_state, lora_contract)
    nested = {
        arm: tuple(
            tuple(
                outputs[arm][(trajectory_index, replan)]
                for replan in range(len(trajectory.observations))
            )
            for trajectory_index, trajectory in enumerate(trajectories)
        )
        for arm in ("expert", "student")
    }
    complete_chunks = sum(len(trajectory.observations) for trajectory in trajectories)
    if (
        batch_sizes["expert"] != batch_sizes["student"]
        or any(len(outputs[arm]) != complete_chunks for arm in outputs)
    ):
        raise RewardProtocolError("successful expert action query lost batch identity")
    stored, matched = [], []
    for trajectory_index, trajectory in enumerate(trajectories):
        for replan, valid in enumerate(trajectory.valid_action_steps):
            stored.append(trajectory.action_chunks[replan].float()[:, :valid])
            matched.append(
                nested["expert"][trajectory_index][replan].float()[:, :valid]
            )
    expert_requery_rms = torch.cat(
        [
            (left - right).flatten()
            for left, right in zip(stored, matched, strict=True)
        ]
    ).square().mean().sqrt()
    return nested, {
        "complete_occupancy_chunks": complete_chunks,
        "matched_policy_forwards": 2 * len(batch_sizes["expert"]),
        "matched_query_batch_sizes": batch_sizes["expert"],
        "stored_expert_to_matched_requery_rms": float(expert_requery_rms),
        "matched_action_seconds": time.monotonic() - started,
    }
