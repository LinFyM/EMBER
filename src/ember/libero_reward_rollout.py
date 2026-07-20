"""Official randomized LIBERO rollouts for reward-only adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from lerobot.envs import preprocess_observation
from lerobot.utils.constants import ACTION

from ember.writer.model import WriterModelError


@dataclass
class RewardTrajectory:
    task_id: int
    env_seed: int
    policy_seed: int
    success: bool
    steps: int
    reward_sum: float
    observations: tuple[dict[str, torch.Tensor], ...]
    action_chunks: tuple[torch.Tensor, ...]
    valid_action_steps: tuple[int, ...]

    def ledger_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_seed": self.env_seed,
            "policy_seed": self.policy_seed,
            "success": self.success,
            "steps": self.steps,
            "reward_sum": self.reward_sum,
            "action_chunk_count": len(self.valid_action_steps),
            "valid_action_steps": list(self.valid_action_steps),
        }


def _seed_policy(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def _single_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size == 0:
        return False
    return bool(array.reshape(-1)[0])


def _success_from_info(info: Mapping[str, Any]) -> bool:
    if "is_success" in info:
        return _single_bool(info["is_success"])
    final = info.get("final_info")
    if isinstance(final, Mapping):
        return _single_bool(final.get("is_success", False))
    if isinstance(final, Sequence) and final:
        item = final[0]
        return bool(isinstance(item, Mapping) and item.get("is_success", False))
    return False


def _cpu_tensor_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors or any(value.ndim == 0 or value.shape[0] != 1 for value in tensors.values()):
        raise WriterModelError("reward rollout lost the single-env batch dimension")
    return tensors


def collect_randomized_reward_trajectory(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_id: int,
    language: str,
    env_seed: int,
    policy_seed: int,
    device: torch.device,
    max_horizon: int,
    action_execution_horizon: int,
    use_bfloat16: bool,
) -> RewardTrajectory:
    """Collect one random-reset trajectory and retain data only when it succeeds."""

    if max_horizon <= 0 or action_execution_horizon <= 0:
        raise WriterModelError("reward rollout horizons must be positive")
    if getattr(env, "num_envs", None) != 1:
        raise WriterModelError("reward training requires exactly one env per policy rank")
    _seed_policy(policy_seed, device)
    policy.reset()
    observation, _ = env.reset(seed=[env_seed])
    observations: list[dict[str, torch.Tensor]] = []
    action_chunks: list[torch.Tensor] = []
    valid_action_steps: list[int] = []
    reward_sum = 0.0
    steps = 0
    success = False
    done = False

    while not done and steps < max_horizon:
        processed = preprocess_observation(observation)
        processed["task"] = [language]
        processed = env_preprocessor(processed)
        processed = preprocessor(processed)
        stored_observation = _cpu_tensor_batch(processed)
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bfloat16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with torch.inference_mode(), precision:
            normalized_chunk = policy.predict_action_chunk(processed)
        if (
            normalized_chunk.ndim != 3
            or normalized_chunk.shape[0] != 1
            or normalized_chunk.shape[1] < action_execution_horizon
        ):
            raise WriterModelError("SmolVLA reward rollout returned an invalid action chunk")
        normalized_chunk = normalized_chunk[:, :action_execution_horizon].detach()
        observations.append(stored_observation)
        action_chunks.append(normalized_chunk.to(device="cpu").contiguous())
        executed = 0
        for action_index in range(action_execution_horizon):
            normalized_action = normalized_chunk[:, action_index]
            environment_action = postprocessor(normalized_action)
            transition = env_postprocessor({ACTION: environment_action})
            action_numpy = transition[ACTION].detach().to(device="cpu").numpy()
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            reward_sum += float(np.asarray(reward).reshape(-1)[0])
            steps += 1
            executed += 1
            success = success or _success_from_info(info)
            done = (
                success
                or _single_bool(terminated)
                or _single_bool(truncated)
                or steps >= max_horizon
            )
            if done:
                break
        valid_action_steps.append(executed)

    if not success:
        observations.clear()
        action_chunks.clear()
        valid_action_steps.clear()
    return RewardTrajectory(
        task_id=task_id,
        env_seed=env_seed,
        policy_seed=policy_seed,
        success=success,
        steps=steps,
        reward_sum=reward_sum,
        observations=tuple(observations),
        action_chunks=tuple(action_chunks),
        valid_action_steps=tuple(valid_action_steps),
    )


def successful_trajectory_batch(
    trajectories: Sequence[RewardTrajectory], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate successful chunk-start observations with exact executed-action masks."""

    successful = [trajectory for trajectory in trajectories if trajectory.success]
    if not successful:
        raise WriterModelError("cannot collate an empty successful rollout batch")
    chunks = [
        (episode_index, observation, action, valid)
        for episode_index, trajectory in enumerate(successful)
        for observation, action, valid in zip(
            trajectory.observations,
            trajectory.action_chunks,
            trajectory.valid_action_steps,
            strict=True,
        )
    ]
    keys = set(chunks[0][1])
    if any(set(observation) != keys for _, observation, _, _ in chunks):
        raise WriterModelError("reward rollout observation tensor keys changed")
    batch = {
        key: torch.cat([observation[key] for _, observation, _, _ in chunks]).to(
            device
        )
        for key in sorted(keys)
    }
    actions = torch.cat([action for _, _, action, _ in chunks]).to(device)
    horizon = actions.shape[1]
    action_is_pad = torch.ones(
        (len(chunks), horizon), dtype=torch.bool, device=device
    )
    for index, (_, _, _, valid) in enumerate(chunks):
        if not 0 < valid <= horizon:
            raise WriterModelError("reward rollout valid-action mask is invalid")
        action_is_pad[index, :valid] = False
    batch[ACTION] = actions
    batch["action_is_pad"] = action_is_pad
    episode_ids = torch.tensor(
        [episode_index for episode_index, _, _, _ in chunks],
        dtype=torch.long,
        device=device,
    )
    return batch, episode_ids
