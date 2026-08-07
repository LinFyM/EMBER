"""Per-GPU serialized lifecycle for persistent LIBERO EGL environments."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError, configure_libero_runtime_assets


class PersistentTaskEnvironmentPool:
    """Keep one task's raw environments and serialize EGL transitions per GPU."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        physical_gpu_id: int,
    ) -> None:
        configure_libero_runtime_assets(Path(contract["libero_paths"]["assets"]))
        from libero.libero import benchmark

        if physical_gpu_id < 0:
            raise Pi05EvaluationError("environment pool physical GPU is invalid")
        self.contract = contract
        self.suites = {
            name: benchmark.get_benchmark_dict()[name]()
            for name in contract["environment"]["horizons"]
        }
        self.egl_lock_path = Path(
            f"/tmp/ember_pi05_egl_uid_{os.getuid()}_gpu_{physical_gpu_id}.lock"
        )
        self.current_key: tuple[str, int] | None = None
        self.envs: list[Any] = []
        self.init_states: Any = None

    def _close_unlocked(self) -> None:
        for env in self.envs:
            env.close()
        self.envs = []
        self.init_states = None
        self.current_key = None

    def close(self) -> None:
        with self.egl_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._close_unlocked()

    def _task_assets(self, task: Mapping[str, Any]) -> tuple[Path, Any, Any]:
        key = task["suite"], int(task["task_id"])
        bddl = (
            Path(self.contract["libero_paths"]["bddl_files"])
            / task["problem_folder"]
            / task["bddl_file"]
        )
        init_path = (
            Path(self.contract["libero_paths"]["init_states"])
            / task["suite"]
            / task["init_states_file"]
        )
        if (
            not bddl.is_file()
            or bddl.stat().st_size != int(task["bddl_bytes"])
            or not init_path.is_file()
            or init_path.stat().st_size != int(task["init_states_bytes"])
        ):
            raise Pi05EvaluationError(f"installed task assets changed: {key}")
        suite = self.suites[task["suite"]]
        installed = suite.get_task(int(task["task_id"]))
        if installed.language != task["language"]:
            raise Pi05EvaluationError(f"installed task language changed: {key}")
        return bddl, suite, suite.get_task_init_states(int(task["task_id"]))

    def _create_unlocked(self, task: Mapping[str, Any]) -> tuple[list[Any], Any]:
        from libero.libero.envs import OffScreenRenderEnv

        bddl, _, init_states = self._task_assets(task)
        env_count = min(
            int(self.contract["parallel"]["envs_per_replica"]),
            len(task["init_state_ids"]),
        )
        created = []
        try:
            for _ in range(env_count):
                resolution = int(self.contract["environment"]["render_resolution"])
                created.append(
                    OffScreenRenderEnv(
                        bddl_file_name=bddl,
                        camera_heights=resolution,
                        camera_widths=resolution,
                    )
                )
        except Exception:
            for env in created:
                try:
                    env.close()
                except Exception:
                    pass
            raise
        return created, init_states

    def switch(self, task: Mapping[str, Any]) -> tuple[list[Any], Any]:
        key = task["suite"], int(task["task_id"])
        if self.current_key == key:
            return self.envs, self.init_states
        with self.egl_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._close_unlocked()
            self.envs, self.init_states = self._create_unlocked(task)
            self.current_key = key
        return self.envs, self.init_states
