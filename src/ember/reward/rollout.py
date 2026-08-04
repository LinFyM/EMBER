"""Official random-reset LIBERO trajectories for PI05 reward adaptation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from lerobot.utils.constants import ACTION

from ember.pi05_assets import configure_libero_runtime_assets
from ember.pi05_processing import libero_policy_input
from ember.pi05_source_checkpoint import sha256_file
from ember.reward.protocol import RewardProtocolError, RewardTask, policy_noise_seed


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


def _cpu_tensor_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors or any(value.ndim == 0 or value.shape[0] != 1 for value in tensors.values()):
        raise RewardProtocolError("PI05 reward rollout lost the single-env batch")
    return tensors


def tensor_batch_sha256(batch: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name].detach().to(device="cpu").contiguous()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _flow_noise(
    *, seed: int, chunk_size: int, max_action_dim: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        (1, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device)


class RandomResetEnvironmentPool:
    """Lazily retain one raw LIBERO environment per task without init-state access."""

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
        self._envs: dict[int, Any] = {}

    def get(self, task: RewardTask) -> Any:
        if task.global_task_id in self._envs:
            return self._envs[task.global_task_id]
        path = self.bddl_root / task.problem_folder / task.bddl_file
        if (
            not path.is_file()
            or path.stat().st_size != task.bddl_bytes
            or sha256_file(path) != task.bddl_sha256
        ):
            raise RewardProtocolError(
                f"installed reward BDDL changed: {task.suite}/{task.task_id}"
            )
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=path,
            camera_heights=self.render_resolution,
            camera_widths=self.render_resolution,
        )
        self._envs[task.global_task_id] = env
        return env

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()

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
    initial_observation_sha256: str
    policy_noise_seeds: tuple[int, ...]
    observations: tuple[dict[str, torch.Tensor], ...]
    action_chunks: tuple[torch.Tensor, ...]
    valid_action_steps: tuple[int, ...]

    def ledger_row(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "global_task_id": self.global_task_id,
            "adaptation_seed": self.adaptation_seed,
            "rollout_cursor": self.rollout_cursor,
            "env_seed": self.env_seed,
            "policy_seed_root": self.policy_seed_root,
            "policy_noise_seeds": list(self.policy_noise_seeds),
            "initial_observation_sha256": self.initial_observation_sha256,
            "official_random_reset": True,
            "fixed_init_state_id": None,
            "set_init_state_called": False,
            "dummy_settling_steps": self.dummy_settling_steps,
            "success": self.success,
            "steps": self.steps,
            "reward_sum": self.reward_sum,
            "action_chunk_count": len(self.valid_action_steps),
            "valid_action_steps": list(self.valid_action_steps),
        }


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
    initial_observation_sha256: str,
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
        initial_observation_sha256=initial_observation_sha256,
        policy_noise_seeds=tuple(noise_seeds),
        observations=tuple(observations),
        action_chunks=tuple(action_chunks),
        valid_action_steps=tuple(valid_action_steps),
    )


def _plan_action_chunk(
    *,
    observation: Mapping[str, Any],
    language: str,
    preprocess: Any,
    postprocess: Any,
    policy: torch.nn.Module,
    policy_seed_root: int,
    suite: str,
    task_id: int,
    adaptation_seed: int,
    rollout_cursor: int,
    replan_index: int,
    device: torch.device,
    num_inference_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, np.ndarray, int]:
    batch = preprocess(libero_policy_input(observation, language))
    stored = _cpu_tensor_batch(batch)
    noise_seed = policy_noise_seed(
        policy_seed_root,
        suite,
        task_id,
        adaptation_seed,
        rollout_cursor,
        replan_index,
    )
    noise = _flow_noise(
        seed=noise_seed,
        chunk_size=int(policy.config.chunk_size),
        max_action_dim=int(policy.config.max_action_dim),
        device=device,
    )
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(
            batch, noise=noise, num_steps=num_inference_steps
        )
    if (
        normalized.ndim != 3
        or normalized.shape[0] != 1
        or normalized.shape[1] != int(policy.config.chunk_size)
        or normalized.shape[2] != 7
    ):
        raise RewardProtocolError("PI05 reward policy returned an invalid action chunk")
    normalized = normalized.detach()
    actions = postprocess(normalized).to(device="cpu").numpy()[0]
    return stored, normalized.to(device="cpu").contiguous(), actions, noise_seed


def collect_randomized_reward_trajectory(
    *,
    env: Any,
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    suite: str,
    task_id: int,
    global_task_id: int,
    language: str,
    adaptation_seed: int,
    rollout_cursor: int,
    env_seed: int,
    policy_seed_root: int,
    device: torch.device,
    max_horizon: int,
    dummy_settling_steps: int,
    dummy_action: Sequence[float],
    action_execution_horizon: int,
    num_inference_steps: int,
    retain_failure_replay: bool = False,
) -> RewardTrajectory:
    """Collect one seeded BDDL-reset trajectory with optional failure replay."""

    dummy = _validate_rollout_contract(
        policy=policy,
        task_id=task_id,
        global_task_id=global_task_id,
        adaptation_seed=adaptation_seed,
        rollout_cursor=rollout_cursor,
        env_seed=env_seed,
        policy_seed_root=policy_seed_root,
        max_horizon=max_horizon,
        dummy_settling_steps=dummy_settling_steps,
        action_execution_horizon=action_execution_horizon,
        num_inference_steps=num_inference_steps,
        dummy_action=dummy_action,
    )
    observation = _random_reset_with_settling(
        env,
        env_seed=env_seed,
        dummy=dummy,
        steps=dummy_settling_steps,
    )

    policy.reset()
    observations: list[dict[str, torch.Tensor]] = []
    action_chunks: list[torch.Tensor] = []
    valid_action_steps: list[int] = []
    noise_seeds: list[int] = []
    reward_sum = 0.0
    steps = 0
    success = False
    initial_observation_sha256 = ""

    while not success and steps < max_horizon:
        stored, normalized, environment_actions, noise_seed = _plan_action_chunk(
            observation=observation,
            language=language,
            preprocess=preprocess,
            postprocess=postprocess,
            policy=policy,
            policy_seed_root=policy_seed_root,
            suite=suite,
            task_id=task_id,
            adaptation_seed=adaptation_seed,
            rollout_cursor=rollout_cursor,
            replan_index=len(noise_seeds),
            device=device,
            num_inference_steps=num_inference_steps,
        )
        if not initial_observation_sha256:
            initial_observation_sha256 = tensor_batch_sha256(stored)
        observations.append(stored)
        action_chunks.append(normalized)
        noise_seeds.append(noise_seed)
        executed = 0
        for action in environment_actions[:action_execution_horizon]:
            observation, reward, done, info = _transition(env.step(action))
            reward_sum += reward
            steps += 1
            executed += 1
            success = _success(done, reward, info)
            if success or steps >= max_horizon:
                break
        valid_action_steps.append(executed)

    if not initial_observation_sha256:
        raise RewardProtocolError("PI05 reward trajectory made no policy observation")
    if not success and not retain_failure_replay:
        observations.clear()
        action_chunks.clear()
        valid_action_steps.clear()
    return _trajectory_result(
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
        initial_observation_sha256=initial_observation_sha256,
        noise_seeds=noise_seeds,
        observations=observations,
        action_chunks=action_chunks,
        valid_action_steps=valid_action_steps,
    )


def successful_trajectory_batch(
    trajectories: Sequence[RewardTrajectory], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    successful = [trajectory for trajectory in trajectories if trajectory.success]
    if not successful:
        raise RewardProtocolError("cannot collate an empty successful rollout batch")
    chunks = [
        (episode, observation, action, valid)
        for episode, trajectory in enumerate(successful)
        for observation, action, valid in zip(
            trajectory.observations,
            trajectory.action_chunks,
            trajectory.valid_action_steps,
            strict=True,
        )
    ]
    keys = set(chunks[0][1])
    if any(set(observation) != keys for _, observation, _, _ in chunks):
        raise RewardProtocolError("PI05 reward observation tensor keys changed")
    batch = {
        key: torch.cat([observation[key] for _, observation, _, _ in chunks]).to(device)
        for key in sorted(keys)
    }
    actions = torch.cat([action for _, _, action, _ in chunks]).to(device)
    valid = torch.tensor(
        [count for _, _, _, count in chunks], dtype=torch.long, device=device
    )
    if actions.ndim != 3 or bool((valid <= 0).any()) or bool((valid > actions.shape[1]).any()):
        raise RewardProtocolError("PI05 reward executed-action prefix is invalid")
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    batch["action_is_pad"] = (
        torch.arange(actions.shape[1], device=device)[None] >= valid[:, None]
    )
    episode_ids = torch.tensor(
        [episode for episode, _, _, _ in chunks], dtype=torch.long, device=device
    )
    return batch, episode_ids


def complete_trajectory_batch(
    trajectories: Sequence[RewardTrajectory], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate successful and failed on-policy prefixes for relative credit."""

    if len(trajectories) < 2 or any(
        not trajectory.observations
        or len(trajectory.observations) != len(trajectory.action_chunks)
        or len(trajectory.observations) != len(trajectory.valid_action_steps)
        for trajectory in trajectories
    ):
        raise RewardProtocolError("relative flow credit requires complete replay")
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
    keys = set(chunks[0][1])
    if any(set(observation) != keys for _, observation, _, _ in chunks):
        raise RewardProtocolError("PI05 relative replay observation keys changed")
    batch = {
        key: torch.cat([observation[key] for _, observation, _, _ in chunks]).to(device)
        for key in sorted(keys)
    }
    actions = torch.cat([action for _, _, action, _ in chunks]).to(device)
    valid = torch.tensor(
        [count for _, _, _, count in chunks], dtype=torch.long, device=device
    )
    if actions.ndim != 3 or bool((valid <= 0).any()) or bool((valid > actions.shape[1]).any()):
        raise RewardProtocolError("PI05 relative replay executed prefix is invalid")
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    batch["action_is_pad"] = (
        torch.arange(actions.shape[1], device=device)[None] >= valid[:, None]
    )
    episode_ids = torch.tensor(
        [episode for episode, _, _, _ in chunks], dtype=torch.long, device=device
    )
    successes = torch.tensor(
        [trajectory.success for trajectory in trajectories],
        dtype=torch.float32,
        device=device,
    )
    return batch, episode_ids, successes
