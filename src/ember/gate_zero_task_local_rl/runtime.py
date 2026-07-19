"""Runtime primitives for the source-only task-local LoRA RL recovery."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ember.gate_zero_oracle_artifacts import (
    sha256_file,
    validate_candidate_artifact,
)


class GateZeroTaskLocalRLRuntimeError(RuntimeError):
    """Raised when online task-local RL mechanics differ from the frozen contract."""


@contextmanager
def scoped_policy_execution_horizon(
    policy: Any, *, execution_horizon: int, expected_model_chunk_size: int
) -> Any:
    """Temporarily shorten action execution without changing model output width."""

    config = getattr(policy, "config", None)
    original = getattr(config, "n_action_steps", None)
    model_chunk_size = getattr(config, "chunk_size", None)
    if (
        not isinstance(original, int)
        or isinstance(original, bool)
        or model_chunk_size != expected_model_chunk_size
        or not 0 < execution_horizon <= expected_model_chunk_size
    ):
        raise GateZeroTaskLocalRLRuntimeError("policy execution-horizon contract changed")
    config.n_action_steps = execution_horizon
    policy.reset()
    try:
        yield policy
    finally:
        config.n_action_steps = original
        policy.reset()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroTaskLocalRLRuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise GateZeroTaskLocalRLRuntimeError(f"invalid {label}: {path}")
    return value


def validate_result_authorities(
    spec: Mapping[str, Any],
    *,
    headroom_result: Path,
    diagnostic_result: Path,
    previous_awr_result: Path,
    previous_signed_result: Path,
    previous_temporal_result: Path,
    previous_critic_result: Path | None = None,
    support_replay_result: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Bind the critic-warmup probe to immutable prior negative evidence."""

    authority = spec["authority"]
    paths = {
        "headroom_result_sha256": headroom_result,
        "candidate_diagnostic_result_sha256": diagnostic_result,
        "previous_awr_result_sha256": previous_awr_result,
        "previous_signed_result_sha256": previous_signed_result,
        "previous_temporal_result_sha256": previous_temporal_result,
    }
    horizon_credit = spec.get("schema_version") in (2, 3)
    if horizon_credit:
        if previous_critic_result is None or support_replay_result is None:
            raise GateZeroTaskLocalRLRuntimeError("horizon-credit predecessor is missing")
        paths.update(
            {
                "previous_critic_result_sha256": previous_critic_result,
                "support_replay_result_sha256": support_replay_result,
            }
        )
    for key, path in paths.items():
        if sha256_file(path) != authority[key]:
            raise GateZeroTaskLocalRLRuntimeError(f"upstream result hash changed: {key}")
    headroom = _load_json(headroom_result, "Proposal-A result")
    diagnostic = _load_json(diagnostic_result, "candidate diagnostic result")
    awr = _load_json(previous_awr_result, "previous AWR result")
    signed = _load_json(previous_signed_result, "previous signed-ratio result")
    temporal = _load_json(previous_temporal_result, "previous temporal-credit result")
    critic = (
        _load_json(previous_critic_result, "previous critic-warmup result")
        if previous_critic_result is not None
        else None
    )
    support_replay = (
        _load_json(support_replay_result, "support-replay result")
        if support_replay_result is not None
        else None
    )
    counts = {
        (arm["task_id"], arm["condition"]): sum(arm["successes"])
        for arm in headroom.get("arms", [])
    }
    expected = {
        (3, "frozen_base"): 3,
        (4, "frozen_base"): 3,
        (3, authority["fit_variant"]): 2,
        (4, authority["fit_variant"]): 4,
    }
    valid = (
        headroom.get("status") == "mature_lora_headroom_control_failed_gate_recovery_required"
        and counts == expected
        and diagnostic.get("status") == "candidate_step_magnitude_recovery_not_supported"
        and diagnostic.get("selected_step") is None
        and all(value.get("status") == "task_local_rl_early_check_not_supported" for value in (awr, signed))
        and all(value.get("interaction_episodes_per_task_initialization") == 16 for value in (awr, signed))
        and temporal.get("status") == spec["predecessor_evidence"]["temporal_credit_status"]
        and temporal.get("interaction_episodes_per_task_initialization") == 16
        and temporal.get("aggregate_metrics", {}).get("paired_net_wins_by_arm")
        == {
            "zero_init_rl": spec["predecessor_evidence"]["temporal_credit_zero_init_paired_net_wins"],
            "supervised_init_rl": spec["predecessor_evidence"]["temporal_credit_supervised_init_paired_net_wins"],
        }
        and all(value.get("gate_zero_authorized") is False for value in (headroom, diagnostic, awr, signed, temporal))
        and all(value.get("writer_authorized") is False for value in (awr, signed, temporal))
    )
    if horizon_credit:
        support_nets = {
            (value.get("task_id"), value.get("initialization")): value.get("paired_net_wins")
            for value in (support_replay or {}).get("records", [])
        }
        valid = valid and (
            critic is not None
            and critic.get("status") == "task_local_rl_early_check_not_supported"
            and critic.get("interaction_episodes_per_task_initialization") == 24
            and critic.get("aggregate_metrics", {}).get("paired_net_wins_by_arm")
            == {"zero_init_rl": [0, 0], "supervised_init_rl": [0, -1]}
            and critic.get("gate_zero_authorized") is False
            and critic.get("writer_authorized") is False
            and support_replay is not None
            and support_replay.get("status") == "support_replay_no_improvement"
            and support_replay.get("optimizer_steps") == 0
            and support_nets
            == {
                (3, "supervised_init"): 0,
                (4, "supervised_init"): -1,
                (3, "zero_init"): -1,
                (4, "zero_init"): -1,
            }
            and support_replay.get("gate_zero_authorized") is False
            and support_replay.get("writer_authorized") is False
            and support_replay.get("validation_numeric_access") is False
            and support_replay.get("held_numeric_access") is False
        )
    if not valid:
        raise GateZeroTaskLocalRLRuntimeError("upstream Gate-0 failure boundary changed")
    return headroom, diagnostic, awr, signed, temporal


def initial_successes(
    headroom: Mapping[str, Any], *, task_id: int, initialization: str, variant: str
) -> list[bool]:
    condition = "frozen_base" if initialization == "zero_init" else variant
    matches = [
        arm
        for arm in headroom["arms"]
        if arm["task_id"] == task_id and arm["condition"] == condition
    ]
    if len(matches) != 1 or len(matches[0].get("successes", [])) != 8:
        raise GateZeroTaskLocalRLRuntimeError("initial closed-loop vector changed")
    return [bool(value) for value in matches[0]["successes"]]


def load_supervised_state(
    fit_root: Path, *, spec: Mapping[str, Any], task_id: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = (
        fit_root
        / f"{spec['authority']['fit_variant']}_task{task_id}"
        / "candidates"
        / f"{spec['authority']['supervised_step']:06d}"
    )
    manifest = validate_candidate_artifact(
        path,
        expected={
            "variant": spec["authority"]["fit_variant"],
            "task_id": task_id,
            "step": spec["authority"]["supervised_step"],
        },
    )
    prefix = f"task{task_id}_supervised"
    if (
        sha256_file(path / "candidate_manifest.json")
        != spec["authority"][f"{prefix}_manifest_sha256"]
        or sha256_file(path / "trainable_state.safetensors")
        != spec["authority"][f"{prefix}_state_sha256"]
    ):
        raise GateZeroTaskLocalRLRuntimeError("supervised LoRA state hash changed")
    return load_file(path / "trainable_state.safetensors"), manifest


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
    actions: torch.Tensor,
    *,
    start: int,
    episode_length: int,
    chunk_size: int,
    execution_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < execution_horizon <= chunk_size:
        raise GateZeroTaskLocalRLRuntimeError("execution horizon escaped model action chunk")
    stop = min(start + execution_horizon, episode_length)
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
    execution_horizon: int | None = None,
) -> dict[str, Any]:
    """Build fixed temporal action-chunk trajectories with masked terminal suffixes."""

    batch_size = len(seeds)
    execution_horizon = execution_horizon or action_chunk_size
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
        "transition_reward": [],
        "transition_done": [],
        "transition_valid": [],
        "transition_progress": [],
        "row_keys": [],
    }
    for episode_index, seed in enumerate(seeds):
        length = _episode_length(done[episode_index])
        valid_anchors = [anchor for anchor in anchors if int(anchor["step"]) < length]
        if not valid_anchors:
            raise GateZeroTaskLocalRLRuntimeError("episode has no valid replay anchor")
        for slot in range(anchors_per_episode):
            transition_valid = slot < len(valid_anchors)
            anchor = valid_anchors[min(slot, len(valid_anchors) - 1)]
            start = int(anchor["step"])
            chunk, is_pad = _action_chunk(
                actions[episode_index],
                start=start,
                episode_length=length,
                chunk_size=action_chunk_size,
                execution_horizon=execution_horizon,
            )
            for key in ("observation.images.camera1", "observation.images.camera2"):
                samples[key].append(anchor[key][episode_index].clone())
            samples["observation.state"].append(
                anchor["observation.state"][episode_index].to(dtype=torch.float32).clone()
            )
            samples["action"].append(chunk)
            samples["action_is_pad"].append(is_pad)
            samples["task"].append(str(anchor["task"][episode_index]))
            stop = min(start + execution_horizon, length)
            reward = float(success[episode_index, start:stop].any()) if transition_valid else 0.0
            terminal = (
                bool(done[episode_index, start:stop].any())
                or slot == len(valid_anchors) - 1
                if transition_valid
                else True
            )
            samples["transition_reward"].append(reward)
            samples["transition_done"].append(terminal)
            samples["transition_valid"].append(transition_valid)
            samples["transition_progress"].append(slot / max(anchors_per_episode - 1, 1))
            samples["row_keys"].append(
                f"task{task_id}/seed{seed}/anchor{start}/temporal_slot{slot}/valid{int(transition_valid)}"
            )
    stacked: dict[str, Any] = {
        key: torch.stack(value)
        for key, value in samples.items()
        if key not in {
            "task",
            "row_keys",
            "transition_reward",
            "transition_done",
            "transition_valid",
            "transition_progress",
        }
    }
    stacked["transition_reward"] = torch.tensor(samples["transition_reward"], dtype=torch.float32)
    stacked["transition_done"] = torch.tensor(samples["transition_done"], dtype=torch.bool)
    stacked["transition_valid"] = torch.tensor(samples["transition_valid"], dtype=torch.bool)
    stacked["transition_progress"] = torch.tensor(
        samples["transition_progress"], dtype=torch.float32
    )
    stacked["trajectory_shape"] = [batch_size, anchors_per_episode]
    stacked["task"] = samples["task"]
    stacked["row_keys"] = samples["row_keys"]
    return stacked


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
    batch_size = training["batch_size"]
    if not 0 <= round_index < training["rounds_maximum"]:
        raise GateZeroTaskLocalRLRuntimeError("training round escaped the contract")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = runtime
    execution_horizon = spec["algorithm"].get(
        "execution_horizon", spec["algorithm"]["action_chunk_size"]
    )
    recorder = AnchorRecordingEnvPreprocessor(
        base=env_preprocessor, interval=execution_horizon
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
        with scoped_policy_execution_horizon(
            policy,
            execution_horizon=execution_horizon,
            expected_model_chunk_size=spec["algorithm"]["action_chunk_size"],
        ):
            rollout_data = rollout(
                env=env,
                policy=policy,
                env_preprocessor=recorder,
                env_postprocessor=env_postprocessor,
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
        == list(range(0, 400, execution_horizon)),
    }
    mechanics = all(mechanics_checks.values())
    replay = build_balanced_replay_batch(
        anchors=recorder.anchors,
        rollout=rollout_data,
        seeds=seeds,
        task_id=task_id,
        action_chunk_size=spec["algorithm"]["action_chunk_size"],
        anchors_per_episode=spec["algorithm"]["anchors_per_episode"],
        execution_horizon=execution_horizon,
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
        "execution_horizon": execution_horizon,
        "model_action_chunk_size": spec["algorithm"]["action_chunk_size"],
        "saturated_action_scalars": 0,
        "saturated_action_scalars_by_dimension": [0] * 7,
        "total_action_scalars": 0,
        "saturation_fraction": 0.0,
        "rollout_seconds": elapsed,
    }
