from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertTask,
    ExpertManifoldError,
    load_expert_manifold_config,
    parse_resume_task,
    parse_task_indices,
    resolve_runtime,
    worker_stage_resume_step,
)
from ember.writer.data import WriterTaskAuthority
from ember.expert_manifold.expert_training import _scheduler
from ember.expert_manifold.sampler import TaskLocalEpochSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"


def test_video_expert_manifold_config_keeps_video_as_dynamic_value() -> None:
    config = load_expert_manifold_config(CONFIG)
    assert config["method"]["language_only_lora_path"] is False
    assert config["video_features"]["shots"] == 1
    assert config["topological_writer"]["chunk_count"] == 168
    assert config["topological_writer"]["valid_values"] == 1_287_168
    assert config["information_wall"]["validation_actions_read"] == 0
    assert config["task_experts"]["profile_defaults"]["scheduler_total_steps"] == 2000


def test_profile_runtime_supports_fresh_then_exact_resume_boundary() -> None:
    config = load_expert_manifold_config(CONFIG)
    fresh = Namespace(mode="profile", batch_size=None, stop_after_step=1, resume=None)
    resumed = Namespace(mode="profile", batch_size=None, stop_after_step=3, resume=Path("x"))
    assert resolve_runtime(fresh, config) == (3, 16, (1, 3), 1)
    assert resolve_runtime(resumed, config) == (3, 16, (1, 3), 3)


def test_formal_experts_remain_blocked_until_live_profile() -> None:
    config = load_expert_manifold_config(CONFIG)
    args = Namespace(mode="formal", batch_size=None, stop_after_step=1000, resume=None)
    with pytest.raises(ExpertManifoldError, match="not sealed"):
        resolve_runtime(args, config)


def test_task_local_sampler_is_step_exact_across_epoch_boundary() -> None:
    sampler = TaskLocalEpochSampler(range(11), task_id=7, batch_size=4, seed=19)
    uninterrupted = tuple(value for step in range(9) for value in sampler.batch_for_step(step))
    resumed = tuple(value for step in range(3, 9) for value in sampler.batch_for_step(step))
    assert resumed == uninterrupted[12:]
    assert len(set(uninterrupted[:11])) == 11
    assert len(set(uninterrupted[11:22])) == 11


def test_task_assignment_and_resume_identity_are_explicit() -> None:
    assert parse_task_indices("0,6,12,18", 24) == (0, 6, 12, 18)
    with pytest.raises(ExpertManifoldError):
        parse_task_indices("6,0", 24)
    assert parse_resume_task(
        Path("worker/task_06_global_12/checkpoints/step_00000003")
    ) == (6, 3)


def test_task_expert_scheduler_warms_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = _scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=25,
        peak_lr=5e-5,
        decay_lr=1e-7,
    )
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] < values[24] <= values[25]
    assert values[25] > values[75] > values[-1]
    assert values[-1] == pytest.approx(1e-7)


def test_worker_stage_resume_requires_complete_same_step_bank(tmp_path: Path) -> None:
    tasks = tuple(
        ExpertTask(
            ordinal=ordinal,
            global_task_id=ordinal + 10,
            suite="suite",
            task_id=ordinal,
            split_role="train",
            language=f"task {ordinal}",
            authority=WriterTaskAuthority(
                task_id=ordinal + 10,
                language=f"task {ordinal}",
                path=tmp_path / f"{ordinal}.hdf5",
                expected_bytes=1,
            ),
        )
        for ordinal in range(2)
    )
    rows = []
    for task in tasks:
        checkpoint = (
            tmp_path
            / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"
            / "checkpoints"
            / "step_00001000"
        )
        checkpoint.mkdir(parents=True)
        rows.append(
            {
                "task_ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "completed_steps": 1000,
            }
        )
    from ember.pi05_source_checkpoint import write_json_atomic

    write_json_atomic(
        tmp_path / "worker_summary.json",
        {
            "schema_version": "ember_pi05_task_expert_worker_summary_v1",
            "tasks": rows,
            "completed_task_count": 2,
            "selected_stop_step": 1000,
        },
    )
    assert worker_stage_resume_step(tmp_path, tmp_path, tasks) == 1000
