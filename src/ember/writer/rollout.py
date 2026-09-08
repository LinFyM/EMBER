"""Four independent LIBERO episodes under one frozen Writer parameter version."""

from __future__ import annotations

import random
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_contract import inspect_installed_target_tasks, load_evaluation_authorities
from ember.pi05_processing import libero_policy_input
from ember.pi05_source_checkpoint import read_json
from ember.writer.flow import flow_actions
from ember.writer.rl_math import DecisionReservoir, exploration_covariance


class _EnvironmentRNG:
    """Isolate LIBERO's process-global Python/NumPy RNG for each live lane."""

    def __init__(self, seed: int) -> None:
        self.numpy = np.random.RandomState(seed).get_state()
        self.python = random.Random(seed).getstate()

    def call(self, function, *args):
        old_numpy, old_python = np.random.get_state(), random.getstate()
        np.random.set_state(self.numpy)
        random.setstate(self.python)
        try:
            return function(*args)
        finally:
            self.numpy, self.python = np.random.get_state(), random.getstate()
            np.random.set_state(old_numpy)
            random.setstate(old_python)


@dataclass
class EpisodeTrace:
    seed: dict[str, int]
    reservoir: DecisionReservoir
    success: bool = False
    steps: int = 0


def decision_batch(records, device):
    return {
        key: torch.cat([record["observation"][key] for record in records]).to(device)
        for key in records[0]["observation"]
    }, torch.cat([record["noise"] for record in records]).to(device)


def recorded_flow_batches(records, *, max_batch_size: int = 4):
    """Replay each real decision at its collected numerical batch shape.

    Padding reuses selected observations; returned positions cover real rows
    only, so callers discard padded means or supply zero padded cotangents.
    """
    groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        size = int(record["flow_batch_size"])
        if not 1 <= size <= max_batch_size:
            raise ValueError("collected flow batch exceeds replay capacity")
        groups.setdefault(size, []).append(index)
    for size, indices in groups.items():
        for start in range(0, len(indices), size):
            positions = indices[start:start + size]
            padded = [positions[index % len(positions)] for index in range(size)]
            yield [records[index] for index in padded], positions


class WriterRollouts:
    def __init__(self, runtime, data, asset_root: Path, output: Path, context, config) -> None:
        self.runtime, self.config = runtime, config
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        physical = int(devices[context.local_rank]) if devices[0] else context.local_rank
        reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
        # EGL enumerates physical GPUs independently of CUDA's visible remap.
        # Configure before importing LIBERO/robosuite through benchmark setup.
        os.environ.update(
            MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(physical),
            EMBER_LIBERO_ASSETS_ROOT=str(asset_root / reuse["libero_assets"]),
        )
        authorities = load_evaluation_authorities(asset_root / "configs/pi05_target_evaluation_v1.json", asset_root)
        targets, paths = inspect_installed_target_tasks(
            authorities, role="development_train", state_count=32,
            libero_config_dir=output / f"libero_rank_{context.rank}",
        )
        by_key = {(target.suite, target.task_id): asdict(target) for target in targets}
        self.tasks = {task: by_key[(row.suite, row.suite_task_id)] for task, row in data.tasks.items()}
        pool_contract = {
            "libero_paths": paths, "environment": authorities.config["environment"],
            "parallel": {"envs_per_replica": 4},
        }
        self.environment = pool_contract["environment"]
        self.pool = PersistentTaskEnvironmentPool(pool_contract, physical_gpu_id=physical)
        self.cholesky = torch.linalg.cholesky(exploration_covariance(device="cpu"))

    @torch.no_grad()
    def collect(self, task_id: int, state, episode_seeds) -> list[EpisodeTrace]:
        if len(episode_seeds) != 4:
            raise ValueError("same-condition LOO requires four independent episodes")
        task = self.tasks[task_id]
        envs, init_states = self.pool.switch(task)
        if len(envs) != 4:
            raise ValueError("RL environment pool lost independent lanes")
        traces, observations, rngs, flows, explorations = [], [], [], [], []
        dummy = np.asarray(self.environment["dummy_action"], dtype=np.float32)
        for env, seeds in zip(envs, episode_seeds, strict=True):
            if not 0 <= seeds["initial_state"] < 32:
                raise ValueError("RL initial state crossed into diagnostic range")
            rng = _EnvironmentRNG(seeds["environment"])
            rng.call(env.seed, seeds["environment"])
            rng.call(env.reset)
            observation = rng.call(env.set_init_state, init_states[seeds["initial_state"]])
            for _ in range(int(self.environment["dummy_settling_steps"])):
                observation, _, _, _ = rng.call(env.step, dummy)
            observations.append(observation)
            rngs.append(rng)
            flows.append(torch.Generator(device="cpu").manual_seed(seeds["flow"]))
            explorations.append(torch.Generator(device="cpu").manual_seed(seeds["exploration"]))
            traces.append(EpisodeTrace(dict(seeds), DecisionReservoir(
                capacity=16, rng=random.Random(seeds["reservoir"]),
            )))
        active = list(range(4))
        while active:
            records = []
            for lane in active:
                processed = self.runtime.processor(libero_policy_input(observations[lane], task["language"]))
                records.append({"observation": {key: value.detach().cpu() for key, value in processed.items()},
                                "noise": torch.randn(1, 50, 32, generator=flows[lane])})
            means = []
            microbatch = int(self.config["runtime"]["rollout_microbatch"])
            for start in range(0, len(records), microbatch):
                chunk = records[start:start + microbatch]
                for record in chunk:
                    record["flow_batch_size"] = len(chunk)
                batch, noise = decision_batch(chunk, self.runtime.observer.device)
                actions = flow_actions(self.runtime.policy, state, self.runtime.lora, batch, noise)
                means.append(actions[:, :5, :7].flatten(1).float().cpu())
            means = torch.cat(means)
            plans = {}
            for lane, record, mean in zip(active, records, means, strict=True):
                z = mean + self.cholesky @ torch.randn(35, generator=explorations[lane])
                if not bool(torch.isfinite(z).all()):
                    raise ValueError("nonfinite sampled normalized action")
                record.update(old_mean=mean, z=z)
                traces[lane].reservoir.add(record)
                plans[lane] = self.runtime.processor.unnormalize_action(
                    z.reshape(5, 7).to(self.runtime.observer.device),
                ).cpu().numpy()
            for action_index in range(5):
                for lane in tuple(active):
                    observation, _, done, _ = rngs[lane].call(envs[lane].step, plans[lane][action_index])
                    observations[lane] = observation
                    traces[lane].steps += 1
                    if bool(done) or traces[lane].steps >= task["horizon"]:
                        traces[lane].success = bool(done)
                        active.remove(lane)
        return traces

    def close(self) -> None:
        self.pool.close()
