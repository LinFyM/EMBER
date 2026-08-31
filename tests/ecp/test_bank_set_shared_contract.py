from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSplit,
)
from ember.ecp.joint_program_primal.bank_set_shared_aggregate import (
    BANK_SET_SHARED_AGGREGATE_SCHEMA,
    aggregate_shared_evaluation,
)
from ember.ecp.joint_program_primal.bank_set_shared_contract import (
    BANK_SET_SHARED_GRADIENT_TASKS,
    BANK_SET_SHARED_RUN_SCHEMA,
    BANK_SET_SHARED_STAGE,
    BANK_SET_SHARED_TASK_PROFILES,
    BANK_SET_SHARED_TASKS,
    bank_set_shared_parameter_ownership,
    checkpoint_authority,
    load_bank_set_shared_config,
    task_cursor_counts,
    writer_trainable_inventory,
)
from ember.ecp.joint_program_primal.bank_set_shared_evaluation import (
    BANK_SET_SHARED_JOB_RESULT_SCHEMA,
    BANK_SET_SHARED_QUEUE_SCHEMA,
    BANK_SET_SHARED_WORKER_SCHEMA,
    _worker_queue_valid,
    build_job_queue,
)
from ember.ecp.joint_program_primal.bank_set_shared_runtime import _optimizer_cursor
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs/pi05_ecp_event_bank_set_s2_functional_polish_v1.json"
)


def test_shared_config_seals_loto_roles_rings_profiles_and_wall() -> None:
    config = load_bank_set_shared_config(CONFIG)
    shared = config["shared_training"]
    assert shared["gradient_task_ids"] == [8, 9, 32, 52, 72, 73, 75, 94]
    assert shared["held_interaction_task_ids"] == [1, 93]
    assert shared["wrong_task_by_task"] == {
        "8": 9, "9": 32, "32": 52, "52": 8,
        "72": 73, "73": 75, "75": 94, "94": 72,
    }
    assert shared["evaluation_wrong_task_by_task"] == {"1": 8, "93": 94}
    assert config["optimization"]["joint"]["global_tasks_per_optimizer_step"] == 8
    assert config["task_split"]["held_interaction_meta"] == [1]
    assert task_cursor_counts(70) == (70,) * 8
    assert task_cursor_counts(110) == (110,) * 8
    assert shared["task_profiles"] == {
        str(task): profile for task, profile in BANK_SET_SHARED_TASK_PROFILES.items()
    }
    assert config["information_wall"]["forbidden_task_ids"] == [2, 74]
    assert config["model"]["absent"][0] == "action_meta"
    assert config["model"]["generated_adapter"].endswith("rank16")
    direct = config["optimization"]["direct_functional"]
    assert direct["correct_backward_mass"] == 1.0
    assert direct["wrong_backward_mass"] == 1.0
    assert direct["task_gradient_combiner"] == (
        "scheduled_condition_unit_l2_mean_zero_for_inactive_no_mgda"
    )
    assert config["authorities"]["interaction_pretraining"]["optimizer_step"] == 110
    assert config["optimization"]["joint"]["optimizer"]["peak_lr"] == 0.0001
    assert config["evaluation"]["target_cache_scope"].endswith("never_training")


def test_shared_profile_allows_two_steps_but_formal_stops_remain_sealed() -> None:
    config = load_bank_set_shared_config(CONFIG)
    trainable = (torch.nn.Parameter(torch.tensor(0.0)),)
    args = SimpleNamespace(mode="profile", stop_after_step=2)
    assert _optimizer_cursor(args, config, trainable)[3] == 2
    args.stop_after_step = 3
    with pytest.raises(ValueError, match="not pre-registered"):
        _optimizer_cursor(args, config, trainable)
    args.mode, args.stop_after_step = "formal", 2
    with pytest.raises(ValueError, match="not pre-registered"):
        _optimizer_cursor(args, config, trainable)


def test_worker_queue_is_bound_to_runtime_paths(tmp_path: Path) -> None:
    args = SimpleNamespace(
        config=CONFIG.resolve(),
        base_config=(ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json").resolve(),
        asset_root=ROOT.resolve(),
        compiler_run=(tmp_path / "compiler").resolve(),
        worker_count=4,
        worker_index=2,
    )
    queue = {
        "schema_version": BANK_SET_SHARED_QUEUE_SCHEMA,
        "status": "ready",
        "worker_count": 4,
        "config": {"path": str(args.config), "bytes": args.config.stat().st_size},
        "base_config": str(args.base_config),
        "asset_root": str(args.asset_root),
        "compiler_run": str(args.compiler_run),
        "compiler_authority": {
            "run_contract_schema": BANK_SET_SHARED_RUN_SCHEMA,
            "training_commit": "c" * 40,
        },
        "checkpoints": [{"training_commit": "c" * 40}],
    }
    assert _worker_queue_valid(queue, args)
    queue["asset_root"] = str(tmp_path / "wrong")
    assert not _worker_queue_valid(queue, args)


class _TinyInteraction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.set_encoder = torch.nn.Linear(2, 2)
        self.input_candidate = torch.nn.Linear(2, 2)
        self.output_candidate = torch.nn.Linear(2, 2)
        self.input_condition = torch.nn.Linear(2, 2)
        self.output_condition = torch.nn.Linear(2, 2)


def test_shared_ownership_checkpoints_only_interaction_and_cursors() -> None:
    program = torch.nn.Linear(2, 2)
    compiler = torch.nn.Module()
    compiler.base = torch.nn.Linear(2, 2)
    compiler.bank_set_interaction = _TinyInteraction()
    writer, trainable, frozen = bank_set_shared_parameter_ownership(program, compiler)
    assert len(trainable) == len(tuple(compiler.bank_set_interaction.parameters()))
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in frozen)
    assert writer.task_arm_cursors.tolist() == [0] * 8
    assert "task_arm_cursors" in writer.state_dict()
    inventory = writer_trainable_inventory(writer)
    assert inventory["task_arm_cursor_task_order"] == list(
        BANK_SET_SHARED_GRADIENT_TASKS
    )
    assert inventory["interaction_shared_across_tasks"] is True
    assert inventory["interaction_initialization_owned_by_runtime_authority"] is True


def _mapping_split() -> SharedCompilerMappingSplit:
    fit, held = [], []
    for task in BANK_SET_SHARED_TASKS:
        role = "meta_fit" if task in {1, 8, 9, 32, 52} else "target_fit"
        frames = BANK_SET_SHARED_TASK_PROFILES[task]["correct_arm_sampled_frames"]
        fit.extend(
            (
                MappingCondition(task, role, task * 10, frames["correct_fit0"]),
                MappingCondition(task, role, task * 10 + 1, frames["correct_fit1"]),
            )
        )
        held.append(
            MappingCondition(task, role, task * 10 + 2, frames["correct_held"])
        )
    return SharedCompilerMappingSplit(tuple(fit), tuple(held), (), {})


def test_queue_is_costed_per_task_arm_checkpoint_and_never_selects_2_or_74(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.ecp.joint_program_primal.bank_set_shared_evaluation as evaluation

    monkeypatch.setattr(evaluation, "load_shared_compiler_config", lambda _: {})
    monkeypatch.setattr(evaluation, "load_mapping_split", lambda *_args, **_kw: _mapping_split())
    jobs = build_job_queue(
        config_path=CONFIG,
        base_config_path=ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
        asset_root=ROOT,
        checkpoints=(
            {"optimizer_step": 70, "path": "/run/checkpoints/macro_00000070"},
            {"optimizer_step": 110, "path": "/run/checkpoints/macro_00000110"},
        ),
    )
    assert len(jobs) == 100
    assert len({row["id"] for row in jobs}) == 100
    assert {row["task"] for row in jobs} == set(BANK_SET_SHARED_TASKS)
    assert not {2, 74}.intersection(row["bank_task"] for row in jobs)
    task1_wrong = next(
        row
        for row in jobs
        if row["checkpoint_optimizer_step"] == 70
        and row["task"] == 1
        and row["arm"] == "wrong_fit0"
    )
    assert task1_wrong["bank_task"] == 8
    assert task1_wrong["video_demo"] == 80
    assert all(row["receives_gradient"] is False for row in jobs)


def test_checkpoint_accepts_topology_and_cursor_extensions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "macro_00000070"
    checkpoint.mkdir(parents=True)
    save_file(
        {
            "task_arm_cursors": torch.tensor(task_cursor_counts(70)),
            "bank_set_interaction.weight": torch.ones(1),
        },
        checkpoint / "ecp.safetensors",
    )
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    for rank in range(2):
        (checkpoint / f"rank_{rank:02d}_state.pt").write_bytes(b"rank")
    topology = [{"rank": 0, "device": 0}, {"rank": 1, "device": 1}]
    files = {
        path.name: {"bytes": path.stat().st_size}
        for path in checkpoint.iterdir()
        if path.is_file()
    }
    write_json_atomic(run / "run_contract.json", {
        "schema_version": BANK_SET_SHARED_RUN_SCHEMA,
        "stage": BANK_SET_SHARED_STAGE,
        "phase": "shared_loto",
        "mode": "formal",
        "config": {"path": str(CONFIG), "bytes": CONFIG.stat().st_size},
        "git": {"commit": "a" * 40},
        "world_topology": topology,
        "task_cursors": [0] * 8,
    })
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {
        "schema_version": "ember_ecp_checkpoint_v1",
        "stage": BANK_SET_SHARED_STAGE,
        "next_macro": 70,
        "world_size": 2,
        "run_contract_schema": BANK_SET_SHARED_RUN_SCHEMA,
        "world_topology": topology,
        "task_cursors": list(task_cursor_counts(70)),
        "files": files,
    })
    authority = checkpoint_authority(
        config_path=CONFIG, compiler_run=run, checkpoint=checkpoint
    )
    assert authority["world_size"] == 2
    assert authority["task_cursors"] == list(task_cursor_counts(70))


def _job_metric(job: dict[str, object], value: float) -> dict[str, object]:
    target = (
        "each_bank_frozen_r5_base_residual"
        if str(job["arm"]).startswith("correct")
        else "task_wrong_fit0_one_round_functional_free_delta_suppressive_teacher"
    )
    wall = {
        "panel_b_backward_calls": 0,
        "held_interaction_task_backward_calls": 0,
        "same_task_held_backward_calls": 0,
        "wrong_fit1_backward_calls": 0,
        "result_or_action_gradient_calls": 0,
        "forbidden_task_reads": 0,
        "validation_or_test_reads": 0,
        "action_meta_installed": False,
        "shuffled_or_reversed_use": False,
        "single_complete_rank16": True,
        "adapter_rank": 16,
        "adapter_target_count": 38,
        "adapter_tensor_count": 76,
    }
    return {
        "task": job["task"],
        "arm": job["arm"],
        "functional_recovery": value,
        "panel_b": {
            "rows": [
                {"visit": visit, "carrier_loss": 1.0, "generated_loss": 0.1}
                for visit in range(16)
            ],
            "free_primal_benefit": 1.0,
        },
        "family_recovery": {name: value for name in ("q", "v", "action_in", "action_out")},
        "effective_rank4": {},
        "target_authority": {
            "effective_target": target,
            "family_denominator": "wrong_fit0_r5_base_to_suppressive_teacher_squared_distance",
            "cached_on_cpu": True,
            "real_bank_cached": False,
        },
        "information_wall": wall,
    }


def test_aggregate_applies_role_loto_ratio_and_adjacent_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ember.ecp.joint_program_primal.bank_set_shared_evaluation as evaluation

    monkeypatch.setattr(evaluation, "load_shared_compiler_config", lambda _: {})
    monkeypatch.setattr(evaluation, "load_mapping_split", lambda *_args, **_kw: _mapping_split())
    checkpoints = (
        {"optimizer_step": 70, "path": "/run/checkpoints/macro_00000070"},
        {"optimizer_step": 110, "path": "/run/checkpoints/macro_00000110"},
    )
    jobs = build_job_queue(
        config_path=CONFIG,
        base_config_path=ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
        asset_root=ROOT,
        checkpoints=checkpoints,
    )
    compiler_run = tmp_path / "compiler_run"
    compiler_run.mkdir()
    training_commit = "b" * 40
    write_json_atomic(compiler_run / "run_contract.json", {
        "schema_version": BANK_SET_SHARED_RUN_SCHEMA,
        "config": {"path": str(CONFIG.resolve()), "bytes": CONFIG.stat().st_size},
        "git": {"commit": training_commit},
    })
    for name in ("results", "workers"):
        (tmp_path / name).mkdir()
    write_json_atomic(tmp_path / "queue.json", {
        "schema_version": BANK_SET_SHARED_QUEUE_SCHEMA,
        "status": "ready",
        "worker_count": 2,
        "config": {"path": str(CONFIG.resolve()), "bytes": CONFIG.stat().st_size},
        "compiler_run": str(compiler_run),
        "compiler_authority": {
            "run_contract_schema": BANK_SET_SHARED_RUN_SCHEMA,
            "training_commit": training_commit,
        },
        "checkpoints": [
            {**row, "training_commit": training_commit} for row in checkpoints
        ],
        "queue_policy": "persistent_workers_atomic_dynamic_claim_long_first",
        "jobs": jobs,
    })
    completed = ([], [])
    for index, job in enumerate(jobs):
        arm = str(job["arm"])
        value = 0.2 if arm.startswith("wrong") else (0.85 if arm == "correct_held" else 0.9)
        if int(job["checkpoint_optimizer_step"]) == 110 and not arm.startswith("wrong"):
            value -= 0.01
        payload = {
            "schema_version": BANK_SET_SHARED_JOB_RESULT_SCHEMA,
            "status": "complete",
            "job": job,
            "checkpoint": {"optimizer_step": job["checkpoint_optimizer_step"]},
            "metrics": _job_metric(job, value),
            "bank_lifecycle": {"resident_real_bank_count_after_release": 0},
        }
        write_json_atomic(tmp_path / "results" / f"{job['id']}.json", payload)
        completed[index % 2].append(job["id"])
    for worker, ids in enumerate(completed):
        write_json_atomic(tmp_path / "workers" / f"worker_{worker:02d}.json", {
            "schema_version": BANK_SET_SHARED_WORKER_SCHEMA,
            "status": "complete",
            "worker_index": worker,
            "completed_job_ids": ids,
        })
    report = aggregate_shared_evaluation(output_dir=tmp_path, config_path=CONFIG)
    assert report["schema_version"] == BANK_SET_SHARED_AGGREGATE_SCHEMA
    assert report["primary_pass"] is True
    assert report["adjacent_checkpoint"]["pass"] is True
    assert report["gate_pass"] is True
    later_meta = report["checkpoint_reports"][1]["roles"]["meta"]
    assert later_meta["held_to_gradient_correct_fit"] == pytest.approx(1.0)
    queue = read_json(tmp_path / "queue.json")
    queue["config"]["bytes"] += 1
    write_json_atomic(tmp_path / "queue.json", queue)
    with pytest.raises(ValueError, match="queue schema changed"):
        aggregate_shared_evaluation(output_dir=tmp_path, config_path=CONFIG)
