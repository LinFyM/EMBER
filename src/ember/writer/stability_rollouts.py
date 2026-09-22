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
E3_HELD_TASKS = (3, 11, 26, 31)
E3_FULL_CAPTURE_CONDITIONS = ((3, 0), (31, 0))
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


def _selected_task_contracts(
    asset_root: Path, output: Path, *, additional_ids: Sequence[int] = ()
) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
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
    selected = set(COMMON_HELD_TASKS) | set(TRAIN_PANEL_TASKS) | set(E3_HELD_TASKS) | set(map(int, additional_ids))
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
    trajectory_root: Path, writer: bool, compact_capture: bool = False,
    full_capture_conditions: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    capture = {
        "schema_version": "ember_writer_stability_trajectory_capture_v1",
        "trajectory_root": str(trajectory_root),
        "checkpoint_selection_use": False,
        "test_use": False,
    }
    if compact_capture:
        by_global = {_task_global_id(str(row["suite"]), int(row["task_id"])): row for row in tasks}
        if any(task not in by_global for task, _state in full_capture_conditions):
            raise ValueError("full trajectory capture condition is outside the rollout panel")
        capture.update({
            "schema_version": "ember_writer_stability_terminal_trajectory_capture_v2",
            "mode": "compact",
            "full_conditions": [
                {
                    "suite": by_global[task]["suite"],
                    "task_id": int(by_global[task]["task_id"]),
                    "init_state_id": state,
                }
                for task, state in full_capture_conditions
            ],
        })
    return {
        "environment": dict(base["environment"]),
        "policy": dict(base["policy"]),
        "rng": dict(base["rng"]),
        "parallel": {"envs_per_replica": 4},
        "libero_paths": dict(paths),
        "tasks": list(tasks),
        "adapter": {"kind": HORIZON_WRITER_KIND} if writer else None,
        "diagnostic_occupancy_capture": capture,
        "diagnostic_stage_predicates": {
            "schema_version": "ember_writer_stability_stage_predicates_v1",
            "checkpoint_selection_use": False,
            "test_use": False,
        },
    }


def _writer_panel_adapter(
    loaded: LoadedWriter,
    *,
    asset_root: Path,
    selected_ids: Sequence[int],
    state_ids: Sequence[int],
    installed: Mapping[int, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]] | None,
    evidence: Mapping[str, Mapping[str, Any]] | None,
) -> DiagnosticWriterAdapter:
    if states is None:
        states, evidence = compile_panel(loaded, asset_root=asset_root, task_ids=selected_ids)
    else:
        expected = {
            f"{installed[task]['suite']}:{int(installed[task]['task_id'])}:{state}"
            for task in selected_ids for state in state_ids
        }
        if set(states) != expected or set(evidence or {}) != expected:
            raise ValueError("precompiled Writer panel does not match requested task/state rows")
    return DiagnosticWriterAdapter(loaded, states, evidence or {})


def run_asset_rollouts(
    *, asset: str, asset_root: Path, output: Path, physical_gpu_id: int,
    current_evaluation_contract: Path, loaded_writer: LoadedWriter | None,
    frozen_policy: FrozenPolicy | None, panel_ids: Sequence[int] | None = None,
    state_ids: Sequence[int] = (0, 1, 2, 3),
    compact_capture: bool = False,
    full_capture_conditions: Sequence[tuple[int, int]] = (),
    precompiled_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    precompiled_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run a registered stability-diagnostic closed-loop panel."""
    if (loaded_writer is None) == (frozen_policy is None):
        raise ValueError("rollout needs exactly one Writer or physical frozen policy")
    if (precompiled_states is None) != (precompiled_evidence is None):
        raise ValueError("precompiled Writer states and evidence must be supplied together")
    if precompiled_states is not None and loaded_writer is None:
        raise ValueError("precompiled states require a loaded Writer policy")
    output.mkdir(parents=True, exist_ok=False)
    assets_root = (
        asset_root
        / "data/simulation/ember_assets/datasets/libero-assets"
        / "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
    ).resolve()
    os.environ.update(
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(physical_gpu_id),
        LIBERO_CONFIG_PATH=str((output / "libero_config").resolve()),
        EMBER_LIBERO_ASSETS_ROOT=str(assets_root),
    )
    if panel_ids is None:
        selected_ids = list(COMMON_HELD_TASKS)
        if asset in E1_TRAIN_MODELS:
            selected_ids += list(TRAIN_PANEL_TASKS)
    else:
        selected_ids = list(panel_ids)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("diagnostic rollout panel contains duplicate tasks")
    state_ids = tuple(int(value) for value in state_ids)
    if not state_ids or len(state_ids) != len(set(state_ids)) or any(value < 0 for value in state_ids):
        raise ValueError("diagnostic rollout state ids must be unique non-negative integers")
    installed, paths = _selected_task_contracts(asset_root, output, additional_ids=selected_ids)
    base = json.loads(current_evaluation_contract.read_text())
    writer = loaded_writer is not None
    adapter = None
    if writer:
        adapter = _writer_panel_adapter(
            loaded_writer, asset_root=asset_root, selected_ids=selected_ids, state_ids=state_ids, installed=installed,
            states=precompiled_states, evidence=precompiled_evidence,
        )
        policy, processor = loaded_writer.runtime.policy, loaded_writer.runtime.processor
    else:
        policy, processor = frozen_policy.policy, frozen_policy.processor
    contract = _rollout_contract(
        base=base, paths=paths, tasks=[installed[task] for task in selected_ids],
        trajectory_root=output / "trajectories", writer=writer,
        compact_capture=compact_capture, full_capture_conditions=full_capture_conditions,
    )
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=physical_gpu_id)
    rows = []
    try:
        for global_task_id in selected_ids:
            task = installed[global_task_id]
            envs, init_states = pool.switch(task)
            panel = (
                "terminal_held" if panel_ids is not None
                else "common_held" if global_task_id in COMMON_HELD_TASKS
                else "train"
            )
            for row in rollout_shard(
                envs=envs, init_states=init_states, task=task, state_ids=state_ids,
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
    expected = len(state_ids) * len(selected_ids)
    if len(rows) != expected:
        raise ValueError(f"diagnostic rollout row count changed: {len(rows)} != {expected}")
    return rows
