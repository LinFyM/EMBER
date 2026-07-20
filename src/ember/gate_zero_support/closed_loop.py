"""Shared closed-loop rollout mechanics for Gate 0 and Writer validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.evaluation_identity import _make_condition_env
from ember.specification_probe import (
    ResetAuditEnv,
    _run_upstream_eval,
    _set_vector_attr,
    apply_prompt_override,
)


class GateZeroOracleReportRuntimeError(RuntimeError):
    """Raised when locked closed-loop mechanics or authority change."""


class EpisodeDiagnosticEnv:
    """Observe terminal timing without requesting LeRobot's observation archive."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.time_to_success: list[int | None] = [None] * env.num_envs
        self.episode_steps: list[int | None] = [None] * env.num_envs
        self.step_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.time_to_success = [None] * self.env.num_envs
        self.episode_steps = [None] * self.env.num_envs
        self.step_count = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any) -> Any:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict):
                raw = final_info.get("is_success", [False] * self.env.num_envs)
                successes = (
                    raw.tolist()
                    if hasattr(raw, "tolist")
                    else [bool(raw)] * self.env.num_envs
                )
            else:
                successes = [
                    bool(item.get("is_success", False))
                    if isinstance(item, dict)
                    else False
                    for item in final_info
                ]
        elif "is_success" in info:
            raw = info["is_success"]
            successes = (
                raw.tolist()
                if hasattr(raw, "tolist")
                else [bool(raw)] * self.env.num_envs
            )
        else:
            successes = [False] * self.env.num_envs
        for index, success in enumerate(successes):
            if success and self.time_to_success[index] is None:
                self.time_to_success[index] = self.step_count
        for index, done in enumerate(terminated | truncated):
            if bool(done) and self.episode_steps[index] is None:
                self.episode_steps[index] = self.step_count
        return observation, reward, terminated, truncated, info

    def finalized_steps(self) -> list[int]:
        return [int(value or self.step_count) for value in self.episode_steps]


class FirstWorkerRenderEnv:
    """Render one async worker when the gallery retains only one episode."""

    def __init__(self, env: Any, base_env: ResetAuditEnv) -> None:
        self.env = env
        self.base_env = base_env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name != "render":
            return self.env.call(name, *args, **kwargs)
        lazy_env = self.base_env._env
        ensure = getattr(lazy_env, "_ensure", None)
        if callable(ensure):
            ensure()
        vector_env = getattr(lazy_env, "_env", None)
        pipes = getattr(vector_env, "parent_pipes", None)
        state = getattr(vector_env, "_state", None)
        if not pipes or getattr(state, "value", None) != "default":
            return self.env.call(name, *args, **kwargs)
        pipes[0].send(("_call", (name, args, kwargs)))
        result, success = pipes[0].recv()
        if not success:
            statuses = [True] * self.num_envs
            statuses[0] = False
            vector_env._raise_if_errors(statuses)
        return (result,)


def report_warmup_seed_batches(
    *,
    batch_size: int,
    warmup_seed_start: int,
    report_seed_start: int,
    expected_report_init_states: Sequence[int],
) -> list[list[int]]:
    """Plan deterministic warm-up resets needed to reach a frozen init-state batch."""

    expected_states = list(expected_report_init_states)
    if batch_size <= 0 or not expected_states:
        raise GateZeroOracleReportRuntimeError("report init-state batch is invalid")
    if expected_states != list(
        range(expected_states[0], expected_states[0] + batch_size)
    ):
        raise GateZeroOracleReportRuntimeError("report init-state batch is invalid")
    target_start = expected_states[0]
    if target_start % batch_size != 0:
        raise GateZeroOracleReportRuntimeError(
            "report init-state batch is not stride aligned"
        )
    warmup_count = target_start // batch_size - 1
    if warmup_count < 1:
        raise GateZeroOracleReportRuntimeError(
            "report surface requires at least one warm-up reset"
        )
    if warmup_seed_start + batch_size != report_seed_start:
        raise GateZeroOracleReportRuntimeError(
            "last warm-up seeds must precede report seeds"
        )
    first_start = warmup_seed_start - (warmup_count - 1) * batch_size
    return [
        list(range(start, start + batch_size))
        for start in range(first_start, report_seed_start, batch_size)
    ]


def validate_report_reset_identity(
    events: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    warmup_seed_start: int,
    report_seed_start: int,
    expected_report_init_states: Sequence[int],
) -> bool:
    try:
        warmup_batches = report_warmup_seed_batches(
            batch_size=batch_size,
            warmup_seed_start=warmup_seed_start,
            report_seed_start=report_seed_start,
            expected_report_init_states=expected_report_init_states,
        )
    except GateZeroOracleReportRuntimeError:
        return False
    all_seed_batches = [
        *warmup_batches,
        list(range(report_seed_start, report_seed_start + batch_size)),
    ]
    expected = []
    for reset_index, seeds in enumerate(all_seed_batches):
        expected.append(
            {
                "before": list(
                    range(reset_index * batch_size, (reset_index + 1) * batch_size)
                ),
                "after": list(
                    range(
                        (reset_index + 1) * batch_size,
                        (reset_index + 2) * batch_size,
                    )
                ),
                "seeds": seeds,
            }
        )
    expected[-1]["after"] = list(expected_report_init_states)
    return list(events) == expected


def _relative_videos(output_dir: Path, paths: Sequence[str]) -> list[str]:
    values = []
    for raw in paths:
        try:
            values.append(
                Path(raw).resolve().relative_to(output_dir.resolve()).as_posix()
            )
        except ValueError as error:
            raise GateZeroOracleReportRuntimeError(
                "report video escaped output"
            ) from error
    return values


class ClosedLoopMetricsSession:
    """Keep one async simulator pool alive across matched rollout seeds."""

    def __init__(
        self,
        *,
        task_id: int,
        condition: str,
        language: str,
        batch_size: int,
        return_episode_data: bool,
    ) -> None:
        self.task_id = task_id
        self.language = language
        self.batch_size = batch_size
        self.base_env = ResetAuditEnv(
            _make_condition_env(
                {"task_suite": "libero_90", "task_id": task_id},
                {
                    "name": f"{task_id}_{condition}",
                    "batch_size": batch_size,
                    "mode": "async",
                },
            )
        )
        self.diagnostics = (
            EpisodeDiagnosticEnv(self.base_env) if return_episode_data else None
        )
        self.rollout_env = FirstWorkerRenderEnv(
            self.diagnostics or self.base_env, self.base_env
        )
        self.override = apply_prompt_override(
            self.base_env, language, batch_size=batch_size
        )
        self.closed = False

    def _prepare_reset_surface(self, spec: Mapping[str, Any]) -> None:
        report = spec["report"]
        if report["rollout_batch_size"] != self.batch_size:
            raise GateZeroOracleReportRuntimeError(
                "rollout batch changed inside a persistent environment session"
            )
        initial_ids = list(range(self.batch_size))
        _set_vector_attr(self.base_env, "init_state_id", initial_ids)
        if list(self.base_env.call("init_state_id")) != initial_ids:
            raise GateZeroOracleReportRuntimeError(
                "failed to rewind LIBERO init-state cursors"
            )
        self.base_env.reset_events.clear()
        warmup_seed_batches = report_warmup_seed_batches(
            batch_size=self.batch_size,
            warmup_seed_start=report["warmup_seed_start"],
            report_seed_start=report["seed_start"],
            expected_report_init_states=report["official_rollout_init_state_indices"],
        )
        for warmup_seeds in warmup_seed_batches:
            self.base_env.reset(seed=warmup_seeds)

    def evaluate(
        self,
        *,
        runtime: tuple[Any, ...],
        condition: str,
        spec: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        if self.closed:
            raise GateZeroOracleReportRuntimeError("closed rollout session was reused")
        report = spec["report"]
        self._prepare_reset_surface(spec)
        evaluation_spec = {
            "episodes_per_task": self.batch_size,
            "max_videos_per_arm": int(
                spec["resources"]["retain_one_video_per_report_arm"]
            ),
            "seed_start": report["seed_start"],
            "policy_rng_seed": report["policy_rng_seed"],
        }
        metrics, elapsed = _run_upstream_eval(
            spec=evaluation_spec,
            runtime=runtime,
            env=self.rollout_env,
            videos_dir=output_dir / "videos" / f"task_{self.task_id}" / condition,
        )
        final_init_ids = list(self.base_env.call("init_state_id"))
        reset_events = list(self.base_env.reset_events)
        mechanics = self.override["mechanically_valid"] and validate_report_reset_identity(
            reset_events,
            batch_size=self.batch_size,
            warmup_seed_start=report["warmup_seed_start"],
            report_seed_start=report["seed_start"],
            expected_report_init_states=report["official_rollout_init_state_indices"],
        )
        episodes = metrics["per_episode"]
        if len(episodes) != self.batch_size:
            raise GateZeroOracleReportRuntimeError(
                "upstream report episode count changed"
            )
        time_to_success = (
            self.diagnostics.time_to_success
            if self.diagnostics
            else [None] * self.batch_size
        )
        episode_steps = (
            self.diagnostics.finalized_steps()
            if self.diagnostics
            else [0] * self.batch_size
        )
        return {
            "mechanics_valid": mechanics,
            "prompt": self.language,
            "prompt_override": self.override,
            "reset_events": reset_events,
            "official_rollout_init_state_indices": report[
                "official_rollout_init_state_indices"
            ],
            "init_state_ids_after_rollout": final_init_ids,
            "seeds": list(
                range(report["seed_start"], report["seed_start"] + self.batch_size)
            ),
            "successes": [bool(value["success"]) for value in episodes],
            "sum_rewards": [float(value["sum_reward"]) for value in episodes],
            "max_rewards": [float(value["max_reward"]) for value in episodes],
            "episode_steps": episode_steps,
            "time_to_success": time_to_success,
            "video_paths": _relative_videos(
                output_dir, metrics.get("video_paths", [])
            ),
            "eval_seconds": elapsed,
        }

    def close(self) -> None:
        if not self.closed:
            self.base_env.close()
            self.closed = True


def closed_loop_metrics(
    *,
    runtime: tuple[Any, ...],
    task_id: int,
    condition: str,
    language: str,
    spec: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    report = spec["report"]
    session = ClosedLoopMetricsSession(
        task_id=task_id,
        condition=condition,
        language=language,
        batch_size=report["rollout_batch_size"],
        return_episode_data=spec["resources"].get("return_episode_data", False),
    )
    try:
        return session.evaluate(
            runtime=runtime,
            condition=condition,
            spec=spec,
            output_dir=output_dir,
        )
    finally:
        session.close()
