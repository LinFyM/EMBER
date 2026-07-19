"""Temporal-credit primitives for the canonical Gate-0 task-local LoRA RL probe."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TemporalCreditError(RuntimeError):
    """Raised when temporal-credit tensors violate the frozen mechanics contract."""


class TemporalCritic(nn.Module):
    """Small task-local value head over detached frozen-policy observation features."""

    def __init__(self, *, input_dim: int, hidden_dims: Sequence[int] = (512, 256)) -> None:
        super().__init__()
        if input_dim <= 0 or not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise TemporalCreditError("critic dimensions must be positive")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        output = nn.Linear(previous, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise TemporalCreditError("critic features must be a finite matrix")
        values = self.mlp(features).squeeze(-1)
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise TemporalCreditError("critic values are invalid")
        return values


@torch.no_grad()
def calculate_masked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    valid: torch.Tensor,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over fixed action-chunk transitions while excluding padded suffixes."""

    if (
        rewards.ndim != 2
        or values.shape != rewards.shape
        or dones.shape != rewards.shape
        or valid.shape != rewards.shape
        or not torch.isfinite(rewards).all()
        or not torch.isfinite(values).all()
        or not 0 < discount <= 1
        or not 0 <= gae_lambda <= 1
    ):
        raise TemporalCreditError("invalid GAE authority")
    dones = dones.bool()
    valid = valid.bool()
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    last = torch.zeros(rewards.shape[0], dtype=torch.float32, device=rewards.device)
    for index in range(rewards.shape[1] - 1, -1, -1):
        if index + 1 < rewards.shape[1]:
            next_values = values[:, index + 1]
        else:
            next_values = torch.zeros_like(last)
        nonterminal = (~dones[:, index]).to(dtype=torch.float32)
        delta = rewards[:, index] + discount * next_values * nonterminal - values[:, index]
        last = delta + discount * gae_lambda * nonterminal * last
        last = torch.where(valid[:, index], last, torch.zeros_like(last))
        advantages[:, index] = last
    returns = torch.where(valid, advantages + values, torch.zeros_like(advantages))
    advantages = torch.where(valid, advantages, torch.zeros_like(advantages))
    return advantages, returns


def normalize_valid_advantages(
    advantages: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Normalize only real transitions and keep padded rows exactly zero."""

    if advantages.shape != valid.shape or advantages.ndim != 1:
        raise TemporalCreditError("advantage mask shape changed")
    selected = advantages[valid.bool()]
    if selected.numel() < 2 or not torch.isfinite(selected).all():
        raise TemporalCreditError("insufficient finite advantages")
    scale = selected.std(unbiased=False)
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise TemporalCreditError("temporal-credit advantages have no variation")
    result = torch.zeros_like(advantages, dtype=torch.float32)
    result[valid.bool()] = (selected - selected.mean()) / scale
    return result


def _straight_through_clamp(value: torch.Tensor, *, minimum: float, maximum: float) -> torch.Tensor:
    clamped = value.clamp(min=minimum, max=maximum)
    return value + (clamped - value).detach()


def clipped_flow_ppo_loss(
    current_losses: torch.Tensor,
    old_losses: torch.Tensor,
    advantages: torch.Tensor,
    valid: torch.Tensor,
    *,
    ratio_clip: float,
    log_ratio_clamp: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply chunk-level PPO to mean matched conditional-flow losses."""

    if (
        current_losses.ndim != 2
        or old_losses.shape != current_losses.shape
        or advantages.shape != current_losses.shape[:1]
        or valid.shape != advantages.shape
        or current_losses.shape[1] < 2
        or not torch.isfinite(current_losses).all()
        or not torch.isfinite(old_losses).all()
        or not torch.isfinite(advantages).all()
        or not 0 < ratio_clip < 1
        or log_ratio_clamp <= 0
        or not valid.bool().any()
    ):
        raise TemporalCreditError("invalid chunk-level flow PPO inputs")
    old_chunk = old_losses.detach().mean(dim=1)
    current_chunk = current_losses.mean(dim=1)
    log_ratio = _straight_through_clamp(
        old_chunk - current_chunk,
        minimum=-log_ratio_clamp,
        maximum=log_ratio_clamp,
    )
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip)
    first = -advantages * ratio
    second = -advantages * clipped
    selected = torch.maximum(first, second)[valid.bool()]
    loss = selected.mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise TemporalCreditError("flow PPO loss is non-finite")
    detached_ratio = ratio.detach()[valid.bool()]
    approx_kl = ((detached_ratio - 1.0) - detached_ratio.log()).mean()
    return loss, {
        "valid_transitions": int(valid.bool().sum()),
        "ratio_mean": float(detached_ratio.mean()),
        "ratio_min": float(detached_ratio.min()),
        "ratio_max": float(detached_ratio.max()),
        "ratio_clip_fraction": float(
            ((detached_ratio - 1.0).abs() > ratio_clip).to(dtype=torch.float32).mean()
        ),
        "approx_kl": float(approx_kl),
    }


def explained_variance(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Return finite explained variance, using zero when the target is constant."""

    if targets.shape != predictions.shape or targets.ndim != 1:
        raise TemporalCreditError("explained-variance inputs changed")
    variance = targets.var(unbiased=False)
    if not torch.isfinite(variance) or variance <= 1e-12:
        return 0.0
    value = 1.0 - (targets - predictions).var(unbiased=False) / variance
    return float(value) if math.isfinite(float(value)) else 0.0


def build_actor_optimizer(model: nn.Module, algorithm: dict[str, Any]) -> torch.optim.Optimizer:
    """Create fresh matched actor state over task-local LoRA only."""

    trainable = [value for value in model.parameters() if value.requires_grad]
    if not trainable:
        raise TemporalCreditError("task-local actor has no trainable LoRA")
    return torch.optim.AdamW(
        trainable,
        lr=algorithm["actor_learning_rate"],
        betas=tuple(algorithm["actor_betas"]),
        eps=algorithm["actor_epsilon"],
        weight_decay=algorithm["actor_weight_decay"],
    )


def build_task_local_critic(
    algorithm: dict[str, Any], *, device: torch.device, task_id: int
) -> tuple[TemporalCritic, torch.optim.Optimizer]:
    """Create deterministic matched critic state without touching global RNG authority."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(algorithm["critic_initialization_seed"] + task_id)
        critic = TemporalCritic(
            input_dim=algorithm["critic_input_dim"],
            hidden_dims=algorithm["critic_hidden_dims"],
        )
    critic.to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=algorithm["critic_learning_rate"],
        betas=tuple(algorithm["critic_betas"]),
        eps=algorithm["critic_epsilon"],
        weight_decay=algorithm["critic_weight_decay"],
    )
    return critic, optimizer


def _raw_policy_batch(replay: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "action",
        "action_is_pad",
        "task",
    }
    if not keys <= set(replay):
        raise TemporalCreditError("temporal replay policy fields are incomplete")
    return {
        key: value.clone() if torch.is_tensor(value) else list(value)
        for key, value in replay.items()
        if key in keys
    }


def _slice_batch(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    cpu_indices = indices.detach().to(device="cpu").tolist()
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.index_select(0, indices.to(value.device)).clone()
        elif isinstance(value, list):
            result[key] = [value[index] for index in cpu_indices]
        else:
            result[key] = value
    return result


def prepare_temporal_flow_batch(
    session: Any, replay: dict[str, Any], algorithm: dict[str, Any]
) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor]]:
    """Preprocess one on-policy replay and freeze matched flow-noise/time ensembles."""

    from ember.gate_zero_runtime import deterministic_flow_inputs, preprocess_smolvla_batch

    model = session.model
    owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    batch = preprocess_smolvla_batch(
        _raw_policy_batch(replay), session.preprocessor, list(owner.config.image_features)
    )
    expected = algorithm["effective_replay_batch_size"]
    action = batch.get("action")
    if not torch.is_tensor(action) or tuple(action.shape) != (
        expected,
        algorithm["action_chunk_size"],
        7,
    ):
        raise TemporalCreditError("processed temporal replay action shape changed")
    keys = list(replay["row_keys"])
    if len(keys) != expected or len(set(keys)) != expected:
        raise TemporalCreditError("temporal replay row identity changed")
    noises: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    for sample in range(algorithm["flow_samples_per_transition"]):
        sample_keys = [f"{key}/flow_sample{sample}" for key in keys]
        noise, flow_time = deterministic_flow_inputs(
            sample_keys,
            action_shape=(algorithm["action_chunk_size"], owner.config.max_action_dim),
            noise_seed=algorithm["fixed_flow_noise_seed"],
            time_seed=algorithm["fixed_flow_time_seed"],
            device=next(model.parameters()).device,
        )
        noises.append(noise)
        times.append(flow_time)
    return batch, noises, times


@torch.no_grad()
def _select_real_camera_inputs(
    images: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    empty_cameras: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Remove only the policy-declared trailing empty camera slots."""

    if len(images) != len(masks) or not 0 <= empty_cameras < len(images):
        raise TemporalCreditError("policy camera/empty-camera contract changed")
    real_count = len(images) - empty_cameras
    if any(torch.any(mask.to(dtype=torch.bool)) for mask in masks[real_count:]):
        raise TemporalCreditError("declared empty camera became observation-bearing")
    return images[:real_count], masks[:real_count]


@torch.no_grad()
def _encode_frozen_critic_feature_batch(
    session: Any,
    batch: dict[str, Any],
    progress: torch.Tensor,
    *,
    expected_dim: int,
) -> torch.Tensor:
    """Pool frozen SmolVLA visual embeddings and append state plus chunk progress."""

    model = session.model
    owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    images, masks = owner.prepare_images(batch)
    images, masks = _select_real_camera_inputs(
        images,
        masks,
        empty_cameras=int(owner.config.empty_cameras),
    )
    pooled = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for image, mask in zip(images, masks, strict=True):
            embedding = owner.model.vlm_with_expert.embed_image(image).to(dtype=torch.float32)
            value = F.layer_norm(embedding.mean(dim=1), (embedding.shape[-1],))
            pooled.append(value * mask[:, None].to(dtype=value.dtype))
    state = owner.prepare_state(batch).detach().to(dtype=torch.float32)
    progress = progress.to(device=state.device, dtype=torch.float32).reshape(-1, 1)
    features = torch.cat([*pooled, state, progress], dim=1).detach()
    if features.shape != (state.shape[0], expected_dim) or not torch.isfinite(features).all():
        raise TemporalCreditError("frozen critic feature contract changed")
    return features


def encode_frozen_critic_features(
    session: Any,
    batch: dict[str, Any],
    progress: torch.Tensor,
    *,
    expected_dim: int,
    microbatch_size: int | None = None,
) -> torch.Tensor:
    """Encode critic inputs in bounded batches while preserving replay order."""

    first = next((value for value in batch.values() if torch.is_tensor(value)), None)
    if first is None or first.ndim == 0:
        raise TemporalCreditError("critic replay batch is empty")
    size = len(first)
    microbatch_size = microbatch_size or size
    if microbatch_size <= 0 or len(progress) != size:
        raise TemporalCreditError("invalid critic feature microbatch contract")
    chunks = []
    for start in range(0, size, microbatch_size):
        indices = torch.arange(start, min(start + microbatch_size, size), device=first.device)
        chunks.append(
            _encode_frozen_critic_feature_batch(
                session,
                _slice_batch(batch, indices),
                progress.index_select(0, indices.to(progress.device)),
                expected_dim=expected_dim,
            )
        )
    return torch.cat(chunks, dim=0)


def _flow_losses(
    model: nn.Module,
    batch: dict[str, Any],
    noises: list[torch.Tensor],
    times: list[torch.Tensor],
    indices: torch.Tensor,
    *,
    gradient: bool,
) -> torch.Tensor:
    losses = []
    context = torch.enable_grad if gradient else torch.no_grad
    for noise, flow_time in zip(noises, times, strict=True):
        with context(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            value, _ = model.forward(
                _slice_batch(batch, indices),
                noise=noise.index_select(0, indices.to(noise.device)),
                time=flow_time.index_select(0, indices.to(flow_time.device)),
                reduction="none",
            )
        losses.append(value.to(dtype=torch.float32))
    return torch.stack(losses, dim=1)


def _flow_losses_microbatched(
    model: nn.Module,
    batch: dict[str, Any],
    noises: list[torch.Tensor],
    times: list[torch.Tensor],
    indices: torch.Tensor,
    *,
    microbatch_size: int,
) -> torch.Tensor:
    """Capture flow losses without one full-replay model forward."""

    if microbatch_size <= 0 or indices.ndim != 1:
        raise TemporalCreditError("invalid flow-loss microbatch contract")
    chunks = [
        _flow_losses(
            model,
            batch,
            noises,
            times,
            indices[start : start + microbatch_size],
            gradient=False,
        )
        for start in range(0, len(indices), microbatch_size)
    ]
    if not chunks:
        raise TemporalCreditError("flow-loss microbatch is empty")
    return torch.cat(chunks, dim=0)


def _temporal_targets(
    critic: TemporalCritic,
    features: torch.Tensor,
    replay: dict[str, Any],
    algorithm: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    valid = replay["transition_valid"].to(features.device).bool()
    rewards = replay["transition_reward"].to(features.device).reshape(replay["trajectory_shape"])
    dones = replay["transition_done"].to(features.device).reshape(replay["trajectory_shape"])
    with torch.no_grad():
        values = critic(features).reshape(replay["trajectory_shape"])
    advantages, returns = calculate_masked_gae(
        rewards,
        values,
        dones,
        valid.reshape(replay["trajectory_shape"]),
        discount=algorithm["discount"],
        gae_lambda=algorithm["gae_lambda"],
    )
    flat_advantages = normalize_valid_advantages(advantages.flatten(), valid)
    flat_returns = returns.flatten()
    reward_variation = float(advantages.flatten()[valid].std(unbiased=False))
    return flat_advantages, flat_returns, valid, reward_variation


def _actor_update_enabled(algorithm: dict[str, Any], *, round_index: int) -> bool:
    """Return whether the frozen critic-only warmup has completed."""

    critic_only_rounds = algorithm.get("critic_only_rounds")
    if (
        not isinstance(critic_only_rounds, int)
        or isinstance(critic_only_rounds, bool)
        or critic_only_rounds < 0
        or not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 0
    ):
        raise TemporalCreditError("invalid critic-only warmup boundary")
    return round_index >= critic_only_rounds


def _update_minibatch(
    session: Any,
    critic: TemporalCritic,
    critic_optimizer: torch.optim.Optimizer,
    *,
    batch: dict[str, Any],
    noises: list[torch.Tensor],
    times: list[torch.Tensor],
    old_losses: torch.Tensor | None,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    features: torch.Tensor,
    indices: torch.Tensor,
    trainable: list[torch.Tensor],
    algorithm: dict[str, Any],
    epoch: int,
    start: int,
    update_actor: bool,
) -> dict[str, Any]:
    """Apply one memory-bounded actor/critic minibatch update."""

    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    if update_actor:
        if old_losses is None:
            raise TemporalCreditError("actor update lacks frozen old-policy losses")
        model = session.model
        current_detached = _flow_losses(
            model, batch, noises, times, indices, gradient=False
        ).requires_grad_(True)
        proxy_loss, ratio_metrics = clipped_flow_ppo_loss(
            current_detached,
            old_losses.index_select(0, indices),
            advantages.index_select(0, indices),
            valid.index_select(0, indices),
            ratio_clip=algorithm["ratio_clip"],
            log_ratio_clamp=algorithm["log_ratio_clamp"],
        )
        coefficients = torch.autograd.grad(proxy_loss, current_detached)[0].detach()
        session.optimizer.zero_grad(set_to_none=True)
        for sample, (noise, flow_time) in enumerate(zip(noises, times, strict=True)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current, _ = model.forward(
                    _slice_batch(batch, indices),
                    noise=noise.index_select(0, indices),
                    time=flow_time.index_select(0, indices),
                    reduction="none",
                )
                surrogate = (current * coefficients[:, sample]).sum()
            surrogate.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(
            trainable, algorithm["actor_gradient_clip_norm"]
        )
        if not torch.isfinite(actor_grad):
            raise TemporalCreditError("actor gradient is non-finite")
        session.optimizer.step()
        session.optimizer.zero_grad(set_to_none=True)
        actor_loss = float(proxy_loss.detach())
    else:
        actor_grad = torch.zeros((), device=features.device)
        actor_loss = 0.0
        ratio_metrics = {
            "valid_transitions": int(valid.index_select(0, indices).sum()),
            "ratio_mean": 1.0,
            "ratio_min": 1.0,
            "ratio_max": 1.0,
            "ratio_clip_fraction": 0.0,
            "approx_kl": 0.0,
        }

    critic_optimizer.zero_grad(set_to_none=True)
    predicted = critic(features.index_select(0, indices))
    valid_mb = valid.index_select(0, indices)
    critic_loss = 0.5 * F.mse_loss(
        predicted[valid_mb], returns.index_select(0, indices)[valid_mb]
    )
    critic_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), algorithm["critic_gradient_clip_norm"]
    )
    if not torch.isfinite(critic_grad) or not torch.isfinite(critic_loss):
        raise TemporalCreditError("critic update is non-finite")
    critic_optimizer.step()
    finished.record()
    torch.cuda.synchronize()
    return {
        "epoch": epoch,
        "minibatch_start": start,
        "actor_update_enabled": update_actor,
        "actor_loss": actor_loss,
        "critic_loss": float(critic_loss.detach()),
        "actor_gradient_norm": float(actor_grad),
        "critic_gradient_norm": float(critic_grad),
        "wall_milliseconds": float(started.elapsed_time(finished)),
        **ratio_metrics,
    }


def train_temporal_credit_round(
    session: Any,
    replay: dict[str, Any],
    *,
    critic: TemporalCritic,
    critic_optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    """Fit one bounded FPO++-anchored task-local actor/critic round."""

    algorithm = spec["algorithm"]
    model = session.model
    trainable = [value for value in model.parameters() if value.requires_grad]
    if sum(value.numel() for value in trainable) != spec["lora"]["trainable_parameters"]:
        raise TemporalCreditError("RL trainable policy parameter count changed")
    update_actor = _actor_update_enabled(algorithm, round_index=round_index)
    actor_before = [value.detach().clone() for value in trainable]
    batch, noises, times = prepare_temporal_flow_batch(session, replay, algorithm)
    features = encode_frozen_critic_features(
        session,
        batch,
        replay["transition_progress"],
        expected_dim=algorithm["critic_input_dim"],
        microbatch_size=algorithm.get("inference_microbatch_size"),
    )
    advantages, returns, valid, advantage_std = _temporal_targets(
        critic, features, replay, algorithm
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if update_actor:
        all_indices = torch.arange(len(replay["row_keys"]), device=features.device)
        old_losses = _flow_losses_microbatched(
            model,
            batch,
            noises,
            times,
            all_indices,
            microbatch_size=algorithm.get("inference_microbatch_size", len(all_indices)),
        ).detach()
    else:
        old_losses = None
    with torch.no_grad():
        old_values = critic(features)
    records: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(
        algorithm["minibatch_order_seed"] + round_index
    )
    model.eval()
    critic.train()
    stop_for_kl = False
    for epoch in range(algorithm["update_epochs_per_round"]):
        permutation = torch.randperm(len(valid_indices), generator=generator).to(features.device)
        order = valid_indices.index_select(0, permutation)
        for start in range(0, len(order), algorithm["minibatch_size"]):
            indices = order[start : start + algorithm["minibatch_size"]].to(features.device)
            record = _update_minibatch(
                session,
                critic,
                critic_optimizer,
                batch=batch,
                noises=noises,
                times=times,
                old_losses=old_losses,
                advantages=advantages,
                returns=returns,
                valid=valid,
                features=features,
                indices=indices,
                trainable=trainable,
                algorithm=algorithm,
                epoch=epoch,
                start=start,
                update_actor=update_actor,
            )
            records.append(record)
            if update_actor and record["approx_kl"] > algorithm["target_kl"]:
                stop_for_kl = True
                break
        if stop_for_kl:
            break
    with torch.no_grad():
        final_values = critic(features)
    actor_state_unchanged = all(
        torch.equal(before, after.detach())
        for before, after in zip(actor_before, trainable, strict=True)
    )
    valid_returns = returns[valid]
    return {
        "updates": records,
        "optimizer_updates": len(records),
        "actor_optimizer_updates": len(records) if update_actor else 0,
        "critic_optimizer_updates": len(records),
        "actor_update_enabled": update_actor,
        "actor_state_unchanged": actor_state_unchanged,
        "valid_transitions": int(valid.sum()),
        "advantage_std_before_normalization": advantage_std,
        "critic_explained_variance_before": explained_variance(valid_returns, old_values[valid]),
        "critic_explained_variance_after": explained_variance(valid_returns, final_values[valid]),
        "target_kl_early_stop": stop_for_kl,
        "temporal_credit_healthy": bool(
            records
            and advantage_std > 1e-8
            and all(math.isfinite(value["actor_loss"]) for value in records)
            and all(math.isfinite(value["critic_loss"]) for value in records)
            and (
                (not update_actor and actor_state_unchanged)
                or (update_actor and all(value["actor_gradient_norm"] > 0 for value in records))
            )
            and all(value["critic_gradient_norm"] > 0 for value in records)
        ),
    }
