from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    ExpertTask,
    load_task_expert_config,
    parse_resume_task,
    parse_task_indices,
    resolve_runtime,
    validate_formal_task_assignment,
    worker_stage_resume_step,
)
from ember.expert_manifold.diagnostic_contract import (
    validation_expert_rows,
    validation_worker_assignments,
)
from ember.expert_manifold.evaluation import (
    FrozenTaskExpertAdapter,
    TASK_EXPERT_ADAPTER_KIND,
    TASK_EXPERT_EPISODE_SCHEMA,
    _evaluation_task_rows,
    validate_task_expert_episode,
)
from ember.expert_manifold.expert_training import _scheduler
from ember.expert_manifold.meta_contract import meta_expert_rows
from ember.expert_manifold.sampler import TaskLocalEpochSampler
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.data import WriterTaskAuthority


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_video_expert_manifold_v1.json"
META_CONFIG = ROOT / "configs/pi05_nonheld_meta_expert_bank_v1.json"
VALIDATION_CONFIG = ROOT / "configs/pi05_validation_expert_diagnostic_v1.json"
PARTICLE_CONFIG = ROOT / "configs/pi05_task_expert_lineages_v1.json"


def test_retained_expert_authorities_are_disjoint_and_complete() -> None:
    train = load_task_expert_config(CONFIG)
    assert train["task_experts"]["task_count"] == 24
    assert train["task_experts"]["task_parameter_sharing"] == "none"
    assert train["information_wall"]["validation_actions_read"] == 0

    meta_rows = meta_expert_rows(load_task_expert_config(META_CONFIG))
    meta_train = _evaluation_task_rows(
        meta_rows, is_meta=True, evaluation_role="nonheld_meta_train"
    )
    meta_validation = _evaluation_task_rows(
        meta_rows, is_meta=True, evaluation_role="nonheld_meta_validation"
    )
    assert (len(meta_train), len(meta_validation)) == (56, 15)
    assert {int(row["task_id"]) for row in meta_train}.isdisjoint(
        int(row["task_id"]) for row in meta_validation
    )

    diagnostic = load_task_expert_config(VALIDATION_CONFIG)
    diagnostic_rows = validation_expert_rows(diagnostic)
    assert len(
        _evaluation_task_rows(
            diagnostic_rows,
            is_meta=False,
            evaluation_role="validation",
            is_validation_diagnostic=True,
        )
    ) == 8
    assert set(
        value
        for assignment in validation_worker_assignments(
            diagnostic["task_experts"]["formal_run"]
        )
        for value in assignment
    ) == set(range(8))


def test_particle_experts_use_fixed_independent_lineages() -> None:
    config = load_task_expert_config(PARTICLE_CONFIG)
    experts = config["task_experts"]
    assert experts["formal_run"]["checkpoint_selection"].startswith("fixed_step2000")
    for index in experts["formal_run"]["task_indices"]:
        validate_formal_task_assignment(config, (index,))
    with pytest.raises(ExpertManifoldError):
        validate_formal_task_assignment(config, (3,))


def test_runtime_and_sampler_resume_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_task_expert_config(CONFIG)
    fresh = Namespace(mode="profile", batch_size=None, stop_after_step=1, resume=None)
    resumed = Namespace(
        mode="profile", batch_size=None, stop_after_step=3, resume=Path("x")
    )
    assert resolve_runtime(fresh, config) == (3, 16, (1, 3), 1)
    assert resolve_runtime(resumed, config) == (3, 16, (1, 3), 3)

    monkeypatch.setattr(
        "ember.expert_manifold.contract.git_state",
        lambda _root: {
            "branch": "main",
            "dirty_paths": [],
            "commit": "sealed",
            "upstream": "origin/main",
            "upstream_commit": "sealed",
            "authority_ref": "origin/main",
            "authority_contains_commit": True,
        },
    )
    formal = Namespace(
        mode="formal", batch_size=None, stop_after_step=1000, resume=None
    )
    assert resolve_runtime(formal, config) == (
        2000,
        16,
        (250, 500, 1000, 1500, 2000),
        1000,
    )

    sampler = TaskLocalEpochSampler(range(11), task_id=7, batch_size=4, seed=19)
    uninterrupted = tuple(
        value for step in range(9) for value in sampler.batch_for_step(step)
    )
    resumed_rows = tuple(
        value for step in range(3, 9) for value in sampler.batch_for_step(step)
    )
    assert resumed_rows == uninterrupted[12:]


def test_task_assignment_scheduler_and_worker_resume(tmp_path: Path) -> None:
    assert parse_task_indices("0,6,12,18", 24) == (0, 6, 12, 18)
    with pytest.raises(ExpertManifoldError):
        parse_task_indices("6,0", 24)
    assert parse_resume_task(
        Path("worker/task_06_global_12/checkpoints/step_00000003")
    ) == (6, 3)

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = _scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=25,
        peak_lr=5e-5,
        decay_lr=1e-7,
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] < rates[24] <= rates[25]
    assert rates[25] > rates[75] > rates[-1]
    assert rates[-1] == pytest.approx(1e-7)

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
    for task in tasks:
        (
            tmp_path
            / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"
            / "checkpoints/step_00001000"
        ).mkdir(parents=True)
    write_json_atomic(
        tmp_path / "worker_summary.json",
        {
            "schema_version": "ember_pi05_task_expert_worker_summary_v1",
            "tasks": [
                {
                    "task_ordinal": task.ordinal,
                    "global_task_id": task.global_task_id,
                    "completed_steps": 1000,
                }
                for task in tasks
            ],
            "completed_task_count": 2,
            "selected_stop_step": 1000,
        },
    )
    assert worker_stage_resume_step(tmp_path, tmp_path, tasks) == 1000


def test_task_expert_episode_evidence_is_exact() -> None:
    record = {
        "suite": "libero_goal",
        "task_id": 2,
        "ordinal": 7,
        "global_task_id": 22,
        "language": "perform task",
        "step": 1000,
        "checkpoint": "/bank/task/checkpoints/step_00001000",
        "manifest_bytes": 300,
        "adapter_bytes": 2_654_208,
    }
    adapter = {"kind": TASK_EXPERT_ADAPTER_KIND, "tasks": [record]}
    evidence = {
        "schema_version": TASK_EXPERT_EPISODE_SCHEMA,
        **record,
        "init_state_id": 4,
    }
    assert validate_task_expert_episode(
        adapter, evidence, suite="libero_goal", task_id=2, init_state_id=4
    )
    assert not validate_task_expert_episode(
        adapter, evidence, suite="libero_goal", task_id=2, init_state_id=5
    )


def test_runtime_loads_only_the_declared_bank_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "task-experts.json"
    observed = {"config": {"path": str(config_path)}, "tasks": []}
    loaded: list[Path] = []
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.load_task_expert_config",
        lambda path: loaded.append(path) or {"authorities": {}},
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.authority_path",
        lambda _config, _name: tmp_path / "lora.json",
    )
    lora = object()
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.load_pi05_lora_contract", lambda _path: lora
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.inject_task_lora",
        lambda _policy, observed_lora: observed_lora is lora,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.task_lora_state_dict", lambda _policy: {}
    )
    adapter = FrozenTaskExpertAdapter(
        policy=torch.nn.Linear(1, 1),
        source={},
        evaluation_adapter=observed,
        task_keys=(),
        device=torch.device("cpu"),
        require_formal=False,
    )
    assert loaded == [config_path]
    assert adapter.lora is lora
