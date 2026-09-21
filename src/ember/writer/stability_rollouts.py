"""Closed-loop panels for the fixed Writer stability diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.batched_lora import BatchedLoRAInference
from ember.eval_adapters import HORIZON_WRITER_KIND
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_contract import inspect_installed_target_tasks, load_evaluation_authorities
from ember.pi05_evaluation import rollout_shard
from ember.task_protocol import load_task_authorities
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import RawTeacherVideoStore
from ember.writer.runtime import autocast
from ember.writer.stability_diagnostics import FrozenPolicy, LoadedWriter


COMMON_HELD_TASKS = (3, 11, 23, 26, 31)
TRAIN_PANEL_TASKS = (5, 7, 12, 37, 43, 96, 51, 73)
E1_TRAIN_MODELS = {"O1500", "N1000", "N1800", "M300", "S1000"}
EVALUATION_CONFIG = "configs/libero_24_8_8_coverage_v1/evaluation.json"
CURRENT_PROTOCOL = "configs/libero_24_8_8_coverage_v1/protocol.json"


@dataclass(frozen=True)
class PreparedDiagnosticLoRA:
    key: str
    evidence: dict[str, Any]


class DiagnosticWriterAdapter:
    """Execute one precompiled complete LoRA per registered task/state."""

    def __init__(self, loaded: LoadedWriter, states: Mapping[str, Mapping[str, torch.Tensor]],
                 evidence: Mapping[str, Mapping[str, Any]]) -> None:
        self.policy = loaded.runtime.policy
        self.lora = loaded.runtime.lora
        self.states = {key: {name: value.detach().cpu() for name, value in state.items()}
                       for key, state in states.items()}
        self.evidence = {key: dict(value) for key, value in evidence.items()}
        self.batched = BatchedLoRAInference(self.policy, self.lora)

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedDiagnosticLoRA:
        key = f"{suite}:{task_id}:{init_state_id}"
        if key not in self.states:
            raise ValueError(f"diagnostic Writer condition is missing: {key}")
        return PreparedDiagnosticLoRA(key, self.evidence[key])

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if len(prepared) != len(noise):
            raise ValueError("diagnostic Writer policy and LoRA batches differ")
        with self.batched.activate([self.states[item.key] for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()


def _task_global_id(suite: str, task_id: int) -> int:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    return 40 + task_id if suite == "libero_90" else suites.index(suite) * 10 + task_id


def _selected_task_contracts(asset_root: Path, output: Path) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    authorities = load_evaluation_authorities(asset_root / EVALUATION_CONFIG, asset_root)
    target, paths = inspect_installed_target_tasks(
        authorities, role="all_targets", state_count=4, libero_config_dir=output / "libero_config"
    )
    meta, meta_paths = inspect_installed_target_tasks(
        authorities, role="nonheld_meta", state_count=4, libero_config_dir=output / "libero_config"
    )
    if paths != meta_paths:
        raise ValueError("target and LIBERO-90 evaluation installations differ")
    rows = {**{_task_global_id(row.suite, row.task_id): asdict(row) for row in target},
            **{_task_global_id(row.suite, row.task_id): asdict(row) for row in meta}}
    selected = set(COMMON_HELD_TASKS) | set(TRAIN_PANEL_TASKS)
    if not selected <= rows.keys():
        raise ValueError("diagnostic rollout tasks are absent from installed authorities")
    return {task: rows[task] for task in selected}, paths


def _condition_tasks(asset_root: Path, task_ids: Sequence[int]):
    protocol, _manifest = load_task_authorities(asset_root, CURRENT_PROTOCOL)
    del protocol
    held = [task for task in task_ids if task in COMMON_HELD_TASKS]
    train = [task for task in task_ids if task in TRAIN_PANEL_TASKS]
    tasks = {}
    if held:
        tasks.update(load_learning_tasks(asset_root, held, role="validation", protocol_path=CURRENT_PROTOCOL))
    if train:
        tasks.update(load_learning_tasks(asset_root, train, role="train", protocol_path=CURRENT_PROTOCOL))
    if set(tasks) != set(task_ids):
        raise ValueError("diagnostic video authorities do not cover the requested panel")
    return tasks


@torch.no_grad()
def compile_panel(
    loaded: LoadedWriter, *, asset_root: Path, task_ids: Sequence[int]
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    tasks = _condition_tasks(asset_root, task_ids)
    store = RawTeacherVideoStore(
        tuple(task.authority for task in tasks.values()), frame_stride=5,
        camera_view=loaded.run["config"]["observer"]["camera_view"],
    )
    states, evidence = {}, {}
    try:
        for global_task_id in task_ids:
            task = tasks[global_task_id]
            for init_state_id in range(4):
                teacher_demo = init_state_id
                video = store.load(global_task_id, teacher_demo)
                condition = loaded.runtime.prepare(
                    (torch.from_numpy(video.frames),),
                    (torch.from_numpy(video.frame_indices),),
                    task.authority.language,
                )
                with autocast(loaded.runtime.device):
                    state = loaded.runtime.compile(condition)
                key = f"{task.suite}:{task.suite_task_id}:{init_state_id}"
                states[key] = {name: value.detach().cpu() for name, value in state.items()}
                evidence[key] = {
                    "schema_version": "ember_writer_stability_rollout_condition_v1",
                    "asset": loaded.name,
                    "global_task_id": global_task_id,
                    "suite": task.suite,
                    "task_id": task.suite_task_id,
                    "init_state_id": init_state_id,
                    "teacher_demo": teacher_demo,
                    "teacher_sampled_frames": int(len(video.frame_indices)),
                    "teacher_raw_frames": int(video.raw_frame_count),
                    "outcome_dependence": False,
                    "checkpoint_selection_use": False,
                    "test_use": False,
                }
    finally:
        store.close()
    return states, evidence


def _rollout_contract(
    *, base: Mapping[str, Any], paths: Mapping[str, str], tasks: Sequence[Mapping[str, Any]],
    trajectory_root: Path, writer: bool,
) -> dict[str, Any]:
    return {
        "environment": dict(base["environment"]),
        "policy": dict(base["policy"]),
        "rng": dict(base["rng"]),
        "parallel": {"envs_per_replica": 4},
        "libero_paths": dict(paths),
        "tasks": list(tasks),
        "adapter": {"kind": HORIZON_WRITER_KIND} if writer else None,
        "diagnostic_occupancy_capture": {
            "schema_version": "ember_writer_stability_trajectory_capture_v1",
            "trajectory_root": str(trajectory_root),
            "checkpoint_selection_use": False,
            "test_use": False,
        },
        "diagnostic_stage_predicates": {
            "schema_version": "ember_writer_stability_stage_predicates_v1",
            "checkpoint_selection_use": False,
            "test_use": False,
        },
    }


def run_asset_rollouts(
    *, asset: str, asset_root: Path, output: Path, physical_gpu_id: int,
    current_evaluation_contract: Path, loaded_writer: LoadedWriter | None,
    frozen_policy: FrozenPolicy | None,
) -> list[dict[str, Any]]:
    """Run exactly the E1-B panels registered for one frozen asset."""
    if (loaded_writer is None) == (frozen_policy is None):
        raise ValueError("rollout needs exactly one Writer or physical frozen policy")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(physical_gpu_id),
        LIBERO_CONFIG_PATH=str((output / "libero_config").resolve()),
    )
    installed, paths = _selected_task_contracts(asset_root, output)
    panel_ids = list(COMMON_HELD_TASKS)
    if asset in E1_TRAIN_MODELS:
        panel_ids += list(TRAIN_PANEL_TASKS)
    base = json.loads(current_evaluation_contract.read_text())
    writer = loaded_writer is not None
    adapter = None
    if writer:
        states, evidence = compile_panel(loaded_writer, asset_root=asset_root, task_ids=panel_ids)
        adapter = DiagnosticWriterAdapter(loaded_writer, states, evidence)
        policy, processor = loaded_writer.runtime.policy, loaded_writer.runtime.processor
    else:
        policy, processor = frozen_policy.policy, frozen_policy.processor
    contract = _rollout_contract(
        base=base, paths=paths, tasks=[installed[task] for task in panel_ids],
        trajectory_root=output / "trajectories", writer=writer,
    )
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=physical_gpu_id)
    rows = []
    try:
        for global_task_id in panel_ids:
            task = installed[global_task_id]
            envs, init_states = pool.switch(task)
            panel = "common_held" if global_task_id in COMMON_HELD_TASKS else "train"
            for row in rollout_shard(
                envs=envs, init_states=init_states, task=task, state_ids=tuple(range(4)),
                contract=contract, policy=policy, preprocess=processor,
                postprocess=processor.unnormalize_action, task_adapter=adapter,
            ):
                rows.append({
                    **row,
                    "asset": asset,
                    "panel": panel,
                    "global_task_id": global_task_id,
                    "old_writer_unseen_auxiliary": bool(asset.startswith("O") and global_task_id in {43, 96, 51, 73}),
                })
    finally:
        pool.close()
        if adapter is not None:
            adapter.close()
    expected = 20 + (32 if asset in E1_TRAIN_MODELS else 0)
    if len(rows) != expected:
        raise ValueError(f"diagnostic rollout row count changed: {len(rows)} != {expected}")
    return rows
