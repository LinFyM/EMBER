"""Runtime primitives for the source-only task-local LoRA RL recovery."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from ember.gate_zero_task_local_rl.contract import normalized_episode_advantages


class GateZeroTaskLocalRLRuntimeError(RuntimeError):
    """Raised when online task-local RL mechanics differ from the frozen contract."""


class AnchorRecordingEnvPreprocessor:
    """Record only action-replan observations while preserving the processor path."""

    def __init__(
        self, *, base: Callable[[dict[str, Any]], dict[str, Any]], interval: int
    ) -> None:
        if interval <= 0:
            raise GateZeroTaskLocalRLRuntimeError("replan interval must be positive")
        self.base = base
        self.interval = interval
        self.step = 0
        self.anchors: list[dict[str, Any]] = []

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        processed = self.base(batch)
        if self.step % self.interval == 0:
            record: dict[str, Any] = {"step": self.step}
            for key in ("observation.images.camera1", "observation.images.camera2"):
                value = processed.get(key)
                if not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 3:
                    raise GateZeroTaskLocalRLRuntimeError("replan camera observation changed")
                if value.dtype == torch.uint8:
                    camera = value
                elif value.is_floating_point() and torch.all((value >= 0) & (value <= 1)):
                    camera = value.mul(255).round().to(dtype=torch.uint8)
                else:
                    raise GateZeroTaskLocalRLRuntimeError("replan camera range changed")
                record[key] = camera.detach().cpu().contiguous()
            state = processed.get("observation.state")
            task = processed.get("task")
            if (
                not torch.is_tensor(state)
                or state.ndim != 2
                or state.shape[1] != 8
                or not torch.isfinite(state).all()
                or not isinstance(task, list)
                or len(task) != state.shape[0]
            ):
                raise GateZeroTaskLocalRLRuntimeError("replan state/task observation changed")
            record["observation.state"] = state.detach().to(dtype=torch.float32).cpu().contiguous()
            record["task"] = list(task)
            self.anchors.append(record)
        self.step += 1
        return processed


class ExplorationActionProcessor:
    """Apply deterministic common-random Gaussian exploration in raw action space."""

    def __init__(
        self,
        *,
        base: Callable[[dict[str, Any]], dict[str, Any]],
        standard_deviation: Sequence[float],
        low: Sequence[float],
        high: Sequence[float],
        seed: int,
    ) -> None:
        if (
            len(standard_deviation) != 7
            or len(low) != 7
            or len(high) != 7
            or any(value < 0 or not math.isfinite(value) for value in standard_deviation)
            or not any(value > 0 for value in standard_deviation)
            or any(left >= right for left, right in zip(low, high, strict=True))
        ):
            raise GateZeroTaskLocalRLRuntimeError("invalid exploration action contract")
        self.base = base
        self.standard_deviation = torch.tensor(standard_deviation, dtype=torch.float32)
        self.explored_dimensions = self.standard_deviation > 0
        self.low = torch.tensor(low, dtype=torch.float32)
        self.high = torch.tensor(high, dtype=torch.float32)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.saturated_scalars = 0
        self.saturated_scalars_by_dimension = [0] * 7
        self.total_scalars = 0

    @property
    def saturation_fraction(self) -> float:
        return self.saturated_scalars / self.total_scalars if self.total_scalars else 0.0

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        processed = self.base(transition)
        action = processed.get("action")
        if not torch.is_tensor(action) or action.ndim != 2 or action.shape[-1] != 7:
            raise GateZeroTaskLocalRLRuntimeError("exploration received an invalid action")
        noise = torch.randn(action.shape, generator=self.generator, dtype=torch.float32)
        proposed = action + noise.to(action.device) * self.standard_deviation.to(action.device)
        low = self.low.to(action.device)
        high = self.high.to(action.device)
        saturated = ((proposed < low) | (proposed > high)) & self.explored_dimensions.to(
            action.device
        )
        by_dimension = saturated.sum(dim=0).tolist()
        self.saturated_scalars += int(saturated.sum().item())
        self.saturated_scalars_by_dimension = [
            left + int(right)
            for left, right in zip(
                self.saturated_scalars_by_dimension, by_dimension, strict=True
            )
        ]
        self.total_scalars += action.shape[0] * int(self.explored_dimensions.sum())
        return {**processed, "action": proposed.clamp(min=low, max=high)}


def balanced_anchor_slots(anchor_count: int, *, slots: int) -> list[int]:
    """Choose a fixed number of evenly spaced slots, repeating when episodes end early."""

    if anchor_count <= 0 or slots <= 0:
        raise GateZeroTaskLocalRLRuntimeError("anchor count and slots must be positive")
    if slots == 1:
        return [0]
    return [round(index * (anchor_count - 1) / (slots - 1)) for index in range(slots)]


def validate_training_reset_events(
    events: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
    batch_size: int,
    seed_start: int,
) -> bool:
    if round_index < 0 or batch_size <= 0:
        return False
    expected = []
    for reset_index in range(round_index + 1):
        expected.append(
            {
                "before": list(
                    range(reset_index * batch_size, (reset_index + 1) * batch_size)
                ),
                "after": list(
                    range((reset_index + 1) * batch_size, (reset_index + 2) * batch_size)
                ),
                "seeds": list(
                    range(
                        seed_start + reset_index * batch_size,
                        seed_start + (reset_index + 1) * batch_size,
                    )
                ),
            }
        )
    return list(events) == expected


def signed_flow_ratio_loss(
    current_loss: torch.Tensor,
    old_loss: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ratio_clip: float,
    negative_spo_penalty: float,
    log_ratio_clamp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply a per-sample signed conditional-flow ratio trust objective."""

    if (
        current_loss.ndim != 1
        or old_loss.shape != current_loss.shape
        or advantages.shape != current_loss.shape
        or current_loss.numel() == 0
        or not torch.isfinite(current_loss).all()
        or not torch.isfinite(old_loss).all()
        or not torch.isfinite(advantages).all()
        or not 0 < ratio_clip < 1
        or negative_spo_penalty <= 0
        or log_ratio_clamp <= 0
    ):
        raise GateZeroTaskLocalRLRuntimeError("invalid signed flow-ratio inputs")
    log_ratio_raw = old_loss.detach() - current_loss
    log_ratio = log_ratio_raw + (
        log_ratio_raw.clamp(-log_ratio_clamp, log_ratio_clamp) - log_ratio_raw
    ).detach()
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip)
    positive_loss = torch.maximum(-advantages * ratio, -advantages * clipped)
    spo_objective = advantages * ratio - advantages.abs() * (ratio - 1.0).square() / (
        2.0 * negative_spo_penalty
    )
    per_sample_objective = torch.where(advantages > 0, positive_loss, -spo_objective)
    value = per_sample_objective.mean()
    if value.ndim != 0 or not torch.isfinite(value):
        raise GateZeroTaskLocalRLRuntimeError("signed flow-ratio loss is non-finite")
    metrics = {
        "ratio_mean": float(ratio.detach().mean()),
        "ratio_min": float(ratio.detach().min()),
        "ratio_max": float(ratio.detach().max()),
        "ratio_clip_fraction": float(((ratio.detach() - 1.0).abs() > ratio_clip).float().mean()),
        "advantage_mean": float(advantages.detach().mean()),
        "advantage_std": float(advantages.detach().std(unbiased=False)),
    }
    return value, metrics


def validated_flow_action_shape(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: int,
    expected_chunk_size: int,
    input_action_dim: int,
    model_action_dim: int,
) -> tuple[int, int]:
    """Bind noise to SmolVLA's internal padded width while auditing 7D input."""

    action = batch.get("action")
    expected_input = (expected_batch_size, expected_chunk_size, input_action_dim)
    if (
        not torch.is_tensor(action)
        or tuple(action.shape) != expected_input
        or input_action_dim != 7
        or model_action_dim != 32
    ):
        raise GateZeroTaskLocalRLRuntimeError("processed SmolVLA action shape changed")
    return expected_chunk_size, model_action_dim


def _episode_length(done: torch.Tensor) -> int:
    locations = torch.nonzero(done, as_tuple=False).flatten()
    return int(locations[0]) + 1 if locations.numel() else int(done.numel())


def _validate_rollout(
    rollout: Mapping[str, torch.Tensor], *, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = rollout.get("action")
    done = rollout.get("done")
    success = rollout.get("success")
    if (
        not torch.is_tensor(actions)
        or not torch.is_tensor(done)
        or not torch.is_tensor(success)
        or actions.ndim != 3
        or actions.shape[0] != batch_size
        or actions.shape[2] != 7
        or done.shape != actions.shape[:2]
        or success.shape != done.shape
        or not torch.isfinite(actions).all()
    ):
        raise GateZeroTaskLocalRLRuntimeError("rollout tensors changed shape or finiteness")
    return actions.to(dtype=torch.float32), done.bool(), success.bool()


def _validate_anchors(anchors: Sequence[Mapping[str, Any]], *, batch_size: int) -> None:
    required = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "task",
        "step",
    }
    steps = []
    for anchor in anchors:
        if set(anchor) != required:
            raise GateZeroTaskLocalRLRuntimeError("recorded anchor fields changed")
        steps.append(anchor["step"])
        for key in ("observation.images.camera1", "observation.images.camera2"):
            value = anchor[key]
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.uint8
                or value.ndim != 4
                or value.shape[:2] != (batch_size, 3)
            ):
                raise GateZeroTaskLocalRLRuntimeError("recorded camera changed")
        state = anchor["observation.state"]
        if not torch.is_tensor(state) or state.shape != (batch_size, 8):
            raise GateZeroTaskLocalRLRuntimeError("recorded state changed")
        if not isinstance(anchor["task"], list) or len(anchor["task"]) != batch_size:
            raise GateZeroTaskLocalRLRuntimeError("recorded task language changed")
    if not anchors or steps != sorted(set(steps)) or steps[0] != 0:
        raise GateZeroTaskLocalRLRuntimeError("recorded anchor steps changed")


def _action_chunk(
    actions: torch.Tensor, *, start: int, episode_length: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    stop = min(start + chunk_size, episode_length)
    valid = stop - start
    if valid <= 0:
        raise GateZeroTaskLocalRLRuntimeError("anchor escaped the episode")
    observed = actions[start:stop]
    chunk = observed[-1:].repeat(chunk_size, 1)
    chunk[:valid] = observed
    is_pad = torch.ones(chunk_size, dtype=torch.bool)
    is_pad[:valid] = False
    return chunk, is_pad


def build_balanced_replay_batch(
    *,
    anchors: Sequence[Mapping[str, Any]],
    rollout: Mapping[str, torch.Tensor],
    seeds: Sequence[int],
    task_id: int,
    action_chunk_size: int,
    anchors_per_episode: int,
) -> dict[str, Any]:
    """Build equal-episode replay slots from replan observations and executed actions."""

    batch_size = len(seeds)
    if batch_size <= 0 or len(set(seeds)) != batch_size or task_id not in {3, 4}:
        raise GateZeroTaskLocalRLRuntimeError("invalid replay task or seed identity")
    _validate_anchors(anchors, batch_size=batch_size)
    actions, done, success = _validate_rollout(rollout, batch_size=batch_size)
    samples: dict[str, list[Any]] = {
        "observation.images.camera1": [],
        "observation.images.camera2": [],
        "observation.state": [],
        "action": [],
        "action_is_pad": [],
        "task": [],
        "episode_return": [],
        "row_keys": [],
    }
    for episode_index, seed in enumerate(seeds):
        length = _episode_length(done[episode_index])
        valid_anchors = [anchor for anchor in anchors if int(anchor["step"]) < length]
        if not valid_anchors:
            raise GateZeroTaskLocalRLRuntimeError("episode has no valid replay anchor")
        episode_return = float(success[episode_index, :length].any())
        for slot, anchor_index in enumerate(
            balanced_anchor_slots(len(valid_anchors), slots=anchors_per_episode)
        ):
            anchor = valid_anchors[anchor_index]
            start = int(anchor["step"])
            chunk, is_pad = _action_chunk(
                actions[episode_index],
                start=start,
                episode_length=length,
                chunk_size=action_chunk_size,
            )
            for key in ("observation.images.camera1", "observation.images.camera2"):
                samples[key].append(anchor[key][episode_index].clone())
            samples["observation.state"].append(
                anchor["observation.state"][episode_index].to(dtype=torch.float32).clone()
            )
            samples["action"].append(chunk)
            samples["action_is_pad"].append(is_pad)
            samples["task"].append(str(anchor["task"][episode_index]))
            samples["episode_return"].append(episode_return)
            samples["row_keys"].append(
                f"task{task_id}/seed{seed}/anchor{start}/balanced_slot{slot}"
            )
    stacked: dict[str, Any] = {
        key: torch.stack(value)
        for key, value in samples.items()
        if key not in {"task", "row_keys", "episode_return"}
    }
    stacked["episode_return"] = torch.tensor(samples["episode_return"], dtype=torch.float32)
    stacked["task"] = samples["task"]
    stacked["row_keys"] = samples["row_keys"]
    return stacked


def _clone_replay_training_batch(replay: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "action",
        "action_is_pad",
        "task",
    }
    if not keys <= set(replay):
        raise GateZeroTaskLocalRLRuntimeError("replay training fields are incomplete")
    result: dict[str, Any] = {}
    for key in keys:
        value = replay[key]
        result[key] = value.clone() if torch.is_tensor(value) else list(value)
    return result


def _prepare_flow_training_inputs(
    session: Any,
    replay: Mapping[str, Any],
    row_keys: Sequence[str],
    algorithm: Mapping[str, Any],
    *,
    optimizer_step: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Build one deterministic augmented flow batch from the replay authority."""

    from ember.gate_zero_oracle_session import augment_support_images
    from ember.gate_zero_runtime import deterministic_flow_inputs, preprocess_smolvla_batch

    model = session.model
    owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    raw = augment_support_images(
        _clone_replay_training_batch(replay),
        row_keys=list(row_keys),
        optimizer_step=optimizer_step,
        seed=algorithm["augmentation_seed"],
        scale_min=algorithm["augmentation_scale_min"],
        scale_max=algorithm["augmentation_scale_max"],
    )
    batch = preprocess_smolvla_batch(
        raw, session.preprocessor, list(owner.config.image_features)
    )
    flow_action_shape = validated_flow_action_shape(
        batch,
        expected_batch_size=algorithm["effective_replay_batch_size"],
        expected_chunk_size=algorithm["action_chunk_size"],
        input_action_dim=7,
        model_action_dim=owner.config.max_action_dim,
    )
    noise, flow_time = deterministic_flow_inputs(
        list(row_keys),
        action_shape=flow_action_shape,
        noise_seed=algorithm["fixed_flow_noise_seed"] + optimizer_step,
        time_seed=algorithm["fixed_flow_time_seed"] + optimizer_step,
        device=next(model.parameters()).device,
    )
    return batch, noise, flow_time


def _round_start_flow_losses(
    session: Any,
    replay: Mapping[str, Any],
    row_keys: Sequence[str],
    algorithm: Mapping[str, Any],
    *,
    optimizer_step_start: int,
) -> list[torch.Tensor]:
    """Freeze the per-sample reference loss for every update in one rollout round."""

    model = session.model
    losses = []
    model.eval()
    for offset in range(algorithm["optimizer_steps_per_rollout_round"]):
        optimizer_step = optimizer_step_start + offset + 1
        batch, noise, flow_time = _prepare_flow_training_inputs(
            session, replay, row_keys, algorithm, optimizer_step=optimizer_step
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_sample_loss, _ = model.forward(
                batch, noise=noise, time=flow_time, reduction="none"
            )
        losses.append(per_sample_loss.detach().to(dtype=torch.float32, device="cpu"))
        del batch, noise, flow_time, per_sample_loss
    return losses


def train_signed_flow_ratio_round(
    session: Any,
    replay: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    optimizer_step_start: int,
) -> list[dict[str, Any]]:
    """Run fixed signed-ratio updates against a frozen round-start reference."""

    algorithm = spec["algorithm"]
    row_keys = list(replay.get("row_keys", []))
    returns = replay.get("episode_return")
    expected = algorithm["effective_replay_batch_size"]
    if (
        len(row_keys) != expected
        or len(set(row_keys)) != expected
        or not torch.is_tensor(returns)
        or returns.shape != (expected,)
    ):
        raise GateZeroTaskLocalRLRuntimeError("replay batch differs from effective batch authority")
    advantages = normalized_episode_advantages(returns)
    model = session.model
    trainable = [value for value in model.parameters() if value.requires_grad]
    if sum(value.numel() for value in trainable) != spec["lora"]["trainable_parameters"]:
        raise GateZeroTaskLocalRLRuntimeError("RL trainable parameter count changed")
    old_losses = _round_start_flow_losses(
        session,
        replay,
        row_keys,
        algorithm,
        optimizer_step_start=optimizer_step_start,
    )

    records = []
    for offset in range(algorithm["optimizer_steps_per_rollout_round"]):
        optimizer_step = optimizer_step_start + offset + 1
        started = time.perf_counter()
        batch, noise, flow_time = _prepare_flow_training_inputs(
            session, replay, row_keys, algorithm, optimizer_step=optimizer_step
        )
        model.eval()
        session.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_sample_loss, _ = model.forward(
                batch, noise=noise, time=flow_time, reduction="none"
            )
            loss, ratio_metrics = signed_flow_ratio_loss(
                per_sample_loss,
                old_losses[offset].to(per_sample_loss.device),
                advantages.to(per_sample_loss.device),
                ratio_clip=algorithm["ratio_clip"],
                negative_spo_penalty=algorithm["negative_spo_penalty"],
                log_ratio_clamp=algorithm["log_ratio_clamp"],
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, algorithm["gradient_clip_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise GateZeroTaskLocalRLRuntimeError("RL gradient norm is non-finite")
        learning_rate = float(session.optimizer.param_groups[0]["lr"])
        session.optimizer.step()
        session.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        records.append(
            {
                "optimizer_step": optimizer_step,
                "signed_flow_ratio_loss": float(loss.detach()),
                "unweighted_flow_loss": float(per_sample_loss.detach().mean()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                **ratio_metrics,
                "reward_mean": float(returns.mean()),
                "reward_std": float(returns.std(unbiased=False)),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        del batch, noise, flow_time, per_sample_loss, loss
    model.eval()
    return records


def collect_training_round(
    *,
    runtime: tuple[Any, Any, Any, Any, Any],
    task_id: int,
    language: str,
    round_index: int,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect one matched eight-episode source rollout and its bounded replay."""

    from ember.evaluation_identity import _make_condition_env
    from ember.specification_probe import ResetAuditEnv, apply_prompt_override
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.random_utils import set_seed

    training = spec["training_interaction"]
    exploration = spec["exploration"]
    batch_size = training["batch_size"]
    if not 0 <= round_index < training["rounds_maximum"]:
        raise GateZeroTaskLocalRLRuntimeError("training round escaped the contract")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = runtime
    recorder = AnchorRecordingEnvPreprocessor(
        base=env_preprocessor, interval=spec["algorithm"]["action_chunk_size"]
    )
    explorer = ExplorationActionProcessor(
        base=env_postprocessor,
        standard_deviation=exploration["standard_deviation"],
        low=exploration["clip_low"],
        high=exploration["clip_high"],
        seed=exploration["exploration_seed_start"] + round_index,
    )
    env = ResetAuditEnv(
        _make_condition_env(
            {"task_suite": "libero_90", "task_id": task_id},
            {"name": f"rl_train_t{task_id}_r{round_index}", "batch_size": batch_size, "mode": "async"},
        )
    )
    seed_start = training["seed_start"]
    report_start = seed_start + round_index * batch_size
    seeds = list(range(report_start, report_start + batch_size))
    try:
        override = apply_prompt_override(env, language, batch_size=batch_size)
        for warmup_index in range(round_index):
            warmup_start = seed_start + warmup_index * batch_size
            env.reset(seed=list(range(warmup_start, warmup_start + batch_size)))
        set_seed(training["policy_rng_seed_start"] + round_index)
        started = time.perf_counter()
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=recorder,
            env_postprocessor=explorer,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=seeds,
            return_observations=False,
        )
        elapsed = time.perf_counter() - started
        reset_events = list(env.reset_events)
        final_init_state_ids = list(env.call("init_state_id"))
    finally:
        env.close()
    mechanics_checks = {
        "prompt_override": override["mechanically_valid"],
        "reset_and_init_state_identity": validate_training_reset_events(
            reset_events,
            round_index=round_index,
            batch_size=batch_size,
            seed_start=seed_start,
        ),
        "anchor_steps": [anchor["step"] for anchor in recorder.anchors]
        == list(range(0, 400, spec["algorithm"]["action_chunk_size"])),
    }
    mechanics = all(mechanics_checks.values())
    replay = build_balanced_replay_batch(
        anchors=recorder.anchors,
        rollout=rollout_data,
        seeds=seeds,
        task_id=task_id,
        action_chunk_size=spec["algorithm"]["action_chunk_size"],
        anchors_per_episode=spec["algorithm"]["anchors_per_episode"],
    )
    done = rollout_data["done"].bool()
    episode_steps = [_episode_length(done[index]) for index in range(batch_size)]
    episode_successes = [
        bool(rollout_data["success"][index, : episode_steps[index]].any())
        for index in range(batch_size)
    ]
    return replay, {
        "round_index": round_index,
        "seeds": seeds,
        "init_state_indices": training["train_init_state_indices_by_round"][round_index],
        "reset_events": reset_events,
        "final_init_state_ids": final_init_state_ids,
        "mechanics_valid": mechanics,
        "mechanics_checks": mechanics_checks,
        "episode_steps": episode_steps,
        "environment_steps": sum(episode_steps),
        "episode_successes": episode_successes,
        "success_count": sum(episode_successes),
        "success_rate": sum(episode_successes) / batch_size,
        "replay_rows": len(replay["row_keys"]),
        "unique_replan_anchors": len(set(key.rsplit("/balanced_slot", 1)[0] for key in replay["row_keys"])),
        "saturated_action_scalars": explorer.saturated_scalars,
        "saturated_action_scalars_by_dimension": explorer.saturated_scalars_by_dimension,
        "total_action_scalars": explorer.total_scalars,
        "saturation_fraction": explorer.saturation_fraction,
        "rollout_seconds": elapsed,
    }
