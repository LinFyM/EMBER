from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, canonical_hash, sha256_file
from ember.pi05_source_contract import reconcile_metrics
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError
from ember.reward.protocol import RewardTask
from ember.rl_writer.checkpoint import (
    load_rl_writer_checkpoint,
    save_rl_writer_checkpoint,
    validate_rl_writer_checkpoint_files,
)
from ember.rl_writer.contract import (
    load_rl_writer_config,
    publish_contract,
    reward_tasks,
    resolve_runtime,
    schedule_summary,
    task_for_update,
    updates_per_cycle,
)
from ember.rl_writer.training import build_parser
from ember.rl_writer.loop import _episode_chunk_weights
from ember.rl_writer.runtime import build_runtime
from ember.writer.as_sampling import TeacherVideoSchedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_rl_writer_development_v1.json"
FINAL_CONFIG = ROOT / "configs/pi05_rl_writer_final_v1.json"


def test_rl_writer_config_seals_fresh_zero_and_micro_warmup_branches() -> None:
    config = load_rl_writer_config(CONFIG)
    assert config["sealed_stage"] == "development"
    assert config["branches"]["zero_as_warmup"]["teacher_action_queries"] == 0
    assert config["branches"]["micro_as_warmup"]["teacher_action_queries"] == 24
    assert config["algorithm"]["reward_replay_chunk_batch_size"] == 8
    assert (
        config["algorithm"]["gradient_synchronization"]
        == "ordered_manual_sum_after_local_backward"
    )
    assert "AS-Writer checkpoint" in config["branches"]["micro_as_warmup"][
        "writer_initialization"
    ]
    assert config["formal_run"]["status"] == "sealed"
    assert config["formal_run"]["initial_evidence_stop_update"] == 12


def test_rl_writer_chunk_weights_keep_successful_episodes_equal() -> None:
    weights = _episode_chunk_weights(torch.tensor([0, 0, 1]), global_successes=3)
    torch.testing.assert_close(weights, torch.tensor([1 / 6, 1 / 6, 1 / 3]))
    assert float(weights.sum()) == pytest.approx(2 / 3)


def test_rl_writer_metrics_reconcile_on_update_cursor(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(
            json.dumps({"next_update": update}) + "\n" for update in (1, 2, 3, 4)
        ),
        encoding="utf-8",
    )
    assert reconcile_metrics(path, 3, 3, cursor_key="next_update") == 3
    assert [json.loads(line)["next_update"] for line in path.read_text().splitlines()] == [
        1,
        2,
        3,
    ]


def test_rl_writer_roles_are_exact_24_development_and_32_final() -> None:
    config = load_rl_writer_config(CONFIG)
    development = reward_tasks(config, stage="development")
    final = reward_tasks(config, stage="final")
    assert len(development) == 24
    assert {task.split_role for task in development} == {"train"}
    assert len(final) == 32
    assert {task.split_role for task in final} == {"train", "validation"}
    assert not any(task.split_role == "test" for task in final)
    assert updates_per_cycle(development, 8) == 3
    assert updates_per_cycle(final, 8) == 4


def test_final_rl_writer_seals_twelve_32_task_cycles_without_test_reads() -> None:
    config = load_rl_writer_config(FINAL_CONFIG)
    tasks = reward_tasks(config, stage="final")
    assert config["sealed_stage"] == "final"
    assert len(tasks) == 32
    assert {task.split_role for task in tasks} == {"train", "validation"}
    assert config["branches"]["micro_as_warmup"]["teacher_action_queries"] == 32
    assert config["formal_run"]["total_updates"] == 48
    assert config["formal_run"]["checkpoint_updates"] == [4, 16, 32, 48]
    assert config["formal_run"]["development_selection"][
        "selected_rollouts_per_task"
    ] == 12
    assert config["information_wall"]["test_reward_reads"] == 0
    assert config["information_wall"]["test_action_reads"] == 0
    assert sha256_file(FINAL_CONFIG) == (
        ROOT / "configs/pi05_rl_writer_final_v1.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_final_rl_writer_runtime_accepts_only_complete_32_task_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.rl_writer.contract as contract

    config = load_rl_writer_config(FINAL_CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    monkeypatch.setattr(
        contract,
        "git_state",
        lambda _: {"dirty_paths": [], "commit": "pushed", "origin_main": "pushed"},
    )
    args = Namespace(
        stage="final",
        branch="zero_as_warmup",
        mode="formal",
        total_updates=None,
        checkpoint_updates=None,
        stop_after_update=None,
        resume=None,
    )
    assert resolve_runtime(args, config, context) == (48, (4, 16, 32, 48))
    assert args.stop_after_update == 48


def test_rl_writer_schedule_is_balanced_and_video_no_replacement() -> None:
    config = load_rl_writer_config(CONFIG)
    tasks = reward_tasks(config, stage="development")
    task_ids = tuple(task.global_task_id for task in tasks)
    videos = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=config["data"]["teacher_video_seed"],
    )
    first_cycle = [
        task_for_update(
            tasks,
            world_size=8,
            rank=rank,
            update=update,
            seed=config["data"]["task_schedule_seed"],
        )[0].global_task_id
        for update in range(3)
        for rank in range(8)
    ]
    assert len(first_cycle) == len(set(first_cycle)) == 24
    assert set(first_cycle) == set(task_ids)
    summary = schedule_summary(
        tasks,
        world_size=8,
        next_update=6,
        seed=config["data"]["task_schedule_seed"],
        rollouts_per_task_update=1,
        video_schedule=videos,
    )
    assert summary["completed_full_task_cycles"] == 2
    assert summary["cycle_slot_cursor"] == 0
    assert summary["min_rollouts_per_task"] == 2
    assert summary["max_rollouts_per_task"] == 2
    assert summary["min_unique_videos_per_task"] == 2
    assert summary["max_unique_videos_per_task"] == 2


def test_rl_writer_information_wall_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["branches"]["micro_as_warmup"]["teacher_action_queries"] = 1200
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RewardProtocolError, match="information"):
        load_rl_writer_config(path)


def test_rl_writer_runtime_has_no_as_checkpoint_and_micro_branch_is_blocked() -> None:
    destinations = {action.dest for action in build_parser()._actions}
    assert "writer_checkpoint" not in destinations
    config = load_rl_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    args = Namespace(
        stage="development",
        branch="micro_as_warmup",
        mode="profile",
        total_updates=None,
        checkpoint_updates=None,
        stop_after_update=None,
        resume=None,
    )
    with pytest.raises(RewardProtocolError, match="blocked"):
        resolve_runtime(args, config, context)


def test_rl_writer_runtime_fails_before_loading_retired_feature_interface() -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    args = Namespace(config=CONFIG, stage="development")
    with pytest.raises(RewardProtocolError, match="raw-video AP-ADR"):
        build_runtime(args, context)


def test_rl_writer_formal_can_stop_and_resume_at_sealed_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.rl_writer.contract as contract

    config = load_rl_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    monkeypatch.setattr(
        contract,
        "git_state",
        lambda _: {"dirty_paths": [], "commit": "pushed", "origin_main": "pushed"},
    )
    args = Namespace(
        stage="development",
        branch="zero_as_warmup",
        mode="formal",
        total_updates=None,
        checkpoint_updates=None,
        stop_after_update=12,
        resume=None,
    )
    total, checkpoints = resolve_runtime(args, config, context)
    assert total == 120
    assert checkpoints[-1] == 120
    assert args.stop_after_update == 12


def test_rl_writer_contract_is_single_owner_and_resume_bound(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    (tmp_path / "libero_config").mkdir()
    contract = {"schema_version": "test", "fresh_writer_initialization_seed": 7}
    digest = publish_contract(
        output_dir=tmp_path,
        contract=contract,
        resume=None,
        context=context,
    )
    assert digest == canonical_hash(contract)
    assert publish_contract(
        output_dir=tmp_path,
        contract=contract,
        resume=tmp_path / "checkpoints/update_00000003",
        context=context,
    ) == digest
    with pytest.raises(RewardProtocolError, match="contract changed"):
        publish_contract(
            output_dir=tmp_path,
            contract={**contract, "fresh_writer_initialization_seed": 8},
            resume=tmp_path / "checkpoints/update_00000003",
            context=context,
        )


def test_rl_writer_inference_recomputes_checkpoint_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.rl_writer.inference as inference

    config = load_rl_writer_config(CONFIG)
    tasks = reward_tasks(config, stage="development")
    videos = TeacherVideoSchedule(
        task_ids=tuple(task.global_task_id for task in tasks),
        demo_indices=range(50),
        seed=config["data"]["teacher_video_seed"],
    )
    consumed = schedule_summary(
        tasks,
        world_size=8,
        next_update=3,
        seed=config["data"]["task_schedule_seed"],
        rollouts_per_task_update=1,
        video_schedule=videos,
    )
    source = {"source_checkpoint": "sealed"}
    checkpoint = tmp_path / "run" / "checkpoints" / "update_00000003"
    checkpoint.mkdir(parents=True)
    training = {
        "schema_version": "ember_pi05_rl_writer_launch_v1",
        "mode": "profile",
        "stage": "development",
        "branch": "zero_as_warmup",
        "config_sha256": sha256_file(CONFIG),
        "source": source,
        "authorities": config["authorities"],
        "information_wall": config["information_wall"],
        "tasks": [
            {"global_task_id": task.global_task_id} for task in tasks
        ],
        "trainable": {"object": "shared_reward_trained_writer_only"},
        "runtime": {"world_size": 8, "checkpoint_updates": [3]},
    }
    (checkpoint.parent.parent / "run_contract.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    manifest = {"next_update": 3, "consumed": consumed}
    monkeypatch.setattr(
        inference,
        "validate_rl_writer_checkpoint_files",
        lambda *args, **kwargs: manifest,
    )
    observed, observed_manifest, cursor = inference._inspect_training_checkpoint(
        config_path=CONFIG,
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=False,
    )
    assert observed == training
    assert observed_manifest == manifest
    assert cursor == 3
    manifest["consumed"] = {**consumed, "schedule_sha256": "0" * 64}
    with pytest.raises(RewardProtocolError, match="authority changed"):
        inference._inspect_training_checkpoint(
            config_path=CONFIG,
            config=config,
            checkpoint=checkpoint,
            source=source,
            require_formal=False,
        )


def test_rl_writer_checkpoint_roundtrip_binds_ledger_before_pickle(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    writer = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    task = RewardTask(
        suite="libero_spatial",
        task_id=0,
        global_task_id=0,
        split_role="train",
        language="pick up the bowl",
        problem_folder="libero_spatial",
        bddl_file="task.bddl",
        bddl_bytes=1,
        bddl_sha256="a" * 64,
        horizon=220,
    )
    videos = TeacherVideoSchedule(task_ids=(0,), demo_indices=range(50), seed=3)
    contract = {"schema_version": "test", "branch": "zero_as_warmup"}
    ledger = {
        "rollout_cursor": 1,
        "environment_action_cursor": 7,
        "successes": 1,
        "reward_sum": 1.0,
        "ledger_prefix_sha256": "b" * 64,
    }
    expected = {name: value.detach().clone() for name, value in writer.state_dict().items()}
    checkpoint = save_rl_writer_checkpoint(
        output_dir=tmp_path,
        next_update=1,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks=(task,),
        task_schedule_seed=5,
        rollouts_per_task_update=1,
        video_schedule=videos,
        contract=contract,
        cursors=InteractionCursors(rollout=1, environment_actions=7, optimizer_updates=1),
        successes=1,
        reward_sum=1.0,
        wall_nanoseconds=11,
        ledger_summary=ledger,
        metrics_rows=1,
        formal=True,
    )
    for parameter in writer.parameters():
        parameter.data.add_(2)
    update, cursors, _, rows, counters = load_rl_writer_checkpoint(
        checkpoint=checkpoint,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract_sha256=canonical_hash(contract),
        tasks=(task,),
        task_schedule_seed=5,
        rollouts_per_task_update=1,
        video_schedule=videos,
        ledger_summary=ledger,
    )
    assert update == rows == 1
    assert cursors == InteractionCursors(1, 7, 1)
    assert counters == {"successes": 1, "reward_sum": 1.0, "wall_nanoseconds": 11}
    for name, value in writer.state_dict().items():
        torch.testing.assert_close(value, expected[name])

    trainer = checkpoint / "trainer_state.pt"
    trainer.write_bytes(trainer.read_bytes() + b"tamper")
    with pytest.raises(RewardProtocolError, match="file changed"):
        validate_rl_writer_checkpoint_files(
            checkpoint,
            world_size=1,
            contract_sha256=canonical_hash(contract),
            rank_ledgers=(ledger,),
        )
