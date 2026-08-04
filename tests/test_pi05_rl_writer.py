from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, canonical_hash, sha256_file
from ember.pi05_source_contract import reconcile_metrics
from ember.reward import rollout as reward_rollout
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError
from ember.rl_writer.checkpoint import (
    load_rl_writer_checkpoint,
    save_rl_writer_checkpoint,
    validate_rl_writer_checkpoint_files,
)
from ember.rl_writer.contract import (
    RL_WRITER_LAUNCH_SCHEMA,
    cycle_assignments,
    load_rl_writer_config,
    publish_contract,
    resolve_runtime,
    reward_tasks,
    schedule_summary,
)
from ember.rl_writer.training import build_parser
from ember.writer.as_sampling import TeacherVideoSchedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_rl_writer_development_v1.json"


def test_flow_credit_config_closes_actions_and_keeps_both_outcomes() -> None:
    config = load_rl_writer_config(CONFIG)
    assert config["sealed_stage"] == "development"
    assert config["algorithm"]["rollouts_per_task_condition"] == 4
    assert config["algorithm"]["flow_mc_samples"] == 4
    assert config["algorithm"]["retain_success_and_failure_prefixes"] is True
    assert config["algorithm"]["task_advantage"] == "leave_one_out_binary_return"
    assert config["information_wall"]["teacher_action_reads_after_coldstart"] == 0
    assert config["parallel"]["maximum_world_size"] == 6
    assert config["formal_run"]["status"].startswith("pending")


def test_full24_schedule_is_exact_and_horizon_balanced_for_bci_topologies() -> None:
    config = load_rl_writer_config(CONFIG)
    tasks = reward_tasks(config)
    expected = {task.global_task_id for task in tasks}
    for world_size in (1, 2, 3, 4, 6):
        assigned = cycle_assignments(
            tasks,
            world_size=world_size,
            cycle=0,
            seed=int(config["data"]["task_schedule_seed"]),
        )
        assert {task.global_task_id for rank in assigned for task in rank} == expected
        assert {len(rank) for rank in assigned} == {24 // world_size}
        if world_size == 6:
            assert {sum(task.horizon for task in rank) for rank in assigned} == {1320}

    videos = TeacherVideoSchedule(
        task_ids=tuple(sorted(expected)),
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    summary = schedule_summary(
        tasks,
        world_size=6,
        next_cycle=2,
        seed=int(config["data"]["task_schedule_seed"]),
        rollouts_per_task=4,
        video_schedule=videos,
    )
    assert summary["completed_full24_cycles"] == 2
    assert summary["tasks_with_interactions"] == 24
    assert summary["min_rollouts_per_task"] == summary["max_rollouts_per_task"] == 8
    assert summary["min_unique_videos_per_task"] == 2


def test_flow_credit_parser_requires_coldstart_and_raw_video_data() -> None:
    destinations = {action.dest for action in build_parser()._actions}
    assert "coldstart_checkpoint" in destinations
    assert "data_root" in destinations
    assert "feature_cache" not in destinations
    assert "branch" not in destinations


def test_random_reset_pool_binds_the_sealed_runtime_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bddl_root = tmp_path / "bddl"
    assets_root = tmp_path / "assets"
    bddl_root.mkdir()
    assets_root.mkdir()
    observed: list[Path] = []
    monkeypatch.setattr(
        reward_rollout,
        "configure_libero_runtime_assets",
        lambda path: observed.append(path.resolve()),
    )

    pool = reward_rollout.RandomResetEnvironmentPool(
        bddl_root=bddl_root,
        assets_root=assets_root,
        render_resolution=256,
    )

    assert pool.assets_root == assets_root.resolve()
    assert observed == [assets_root.resolve()]


def test_flow_credit_information_wall_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["information_wall"]["validation_reward_reads"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RewardProtocolError, match="information"):
        load_rl_writer_config(path)


def test_profile_runtime_uses_actual_divisor_world_size() -> None:
    config = load_rl_writer_config(CONFIG)
    context = DistributedContext(0, 0, 6, torch.device("cpu"), 0, (0,))
    args = Namespace(
        stage="development",
        mode="profile",
        total_cycles=None,
        checkpoint_cycles=None,
        stop_after_cycle=None,
        learning_epochs=None,
        resume=None,
    )
    assert resolve_runtime(args, config, context) == (1, (1,), 2)
    assert args.stop_after_cycle == 1
    invalid = DistributedContext(0, 0, 5, torch.device("cpu"), 0, (0,))
    with pytest.raises(RewardProtocolError, match="divide train24"):
        resolve_runtime(args, config, invalid)


def test_metrics_reconcile_on_complete_cycle_cursor(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps({"next_cycle": cycle}) + "\n" for cycle in (1, 2, 3)),
        encoding="utf-8",
    )
    assert reconcile_metrics(path, 2, 2, cursor_key="next_cycle") == 2
    assert [json.loads(line)["next_cycle"] for line in path.read_text().splitlines()] == [1, 2]


def test_flow_credit_contract_is_single_owner_and_resume_bound(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    (tmp_path / "libero_config").mkdir()
    contract = {"schema_version": "test", "coldstart": "sealed"}
    digest = publish_contract(
        output_dir=tmp_path, contract=contract, resume=None, context=context
    )
    assert digest == canonical_hash(contract)
    resume = tmp_path / "checkpoints/cycle_00000001"
    assert publish_contract(
        output_dir=tmp_path, contract=contract, resume=resume, context=context
    ) == digest
    with pytest.raises(RewardProtocolError, match="contract changed"):
        publish_contract(
            output_dir=tmp_path,
            contract={**contract, "coldstart": "different"},
            resume=resume,
            context=context,
        )


def test_inference_recomputes_full24_checkpoint_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.rl_writer.inference as inference

    config = load_rl_writer_config(CONFIG)
    tasks = reward_tasks(config)
    videos = TeacherVideoSchedule(
        task_ids=tuple(task.global_task_id for task in tasks),
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    consumed = schedule_summary(
        tasks,
        world_size=6,
        next_cycle=1,
        seed=int(config["data"]["task_schedule_seed"]),
        rollouts_per_task=4,
        video_schedule=videos,
    )
    source = {"source_checkpoint": "sealed"}
    checkpoint = tmp_path / "run" / "checkpoints" / "cycle_00000001"
    checkpoint.mkdir(parents=True)
    training = {
        "schema_version": RL_WRITER_LAUNCH_SCHEMA,
        "mode": "profile",
        "stage": "development",
        "config_sha256": sha256_file(CONFIG),
        "source": source,
        "authorities": config["authorities"],
        "information_wall": config["information_wall"],
        "tasks": [task.__dict__ for task in tasks],
        "trainable": {
            "object": "shared_task_relative_reward_trained_writer_only",
            "coldstart_teacher_action_phase_closed": True,
        },
        "runtime": {"world_size": 6, "checkpoint_cycles": [1]},
    }
    (checkpoint.parent.parent / "run_contract.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    manifest = {"next_cycle": 1, "consumed": consumed}
    monkeypatch.setattr(
        inference, "validate_rl_writer_checkpoint_files", lambda *args, **kwargs: manifest
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
    assert cursor == 1
    manifest["consumed"] = {**consumed, "schedule_sha256": "0" * 64}
    with pytest.raises(RewardProtocolError, match="authority changed"):
        inference._inspect_training_checkpoint(
            config_path=CONFIG,
            config=config,
            checkpoint=checkpoint,
            source=source,
            require_formal=False,
        )


def test_checkpoint_roundtrip_binds_full24_ledger_before_pickle(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    writer = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = load_rl_writer_config(CONFIG)
    tasks = reward_tasks(config)
    videos = TeacherVideoSchedule(
        task_ids=tuple(task.global_task_id for task in tasks),
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    ledger = {
        "rollout_cursor": 96,
        "environment_action_cursor": 192,
        "successes": 3,
        "reward_sum": 3.0,
        "ledger_prefix_sha256": "b" * 64,
    }
    contract = {"schema_version": "test", "method": "flow-credit"}
    expected = {name: value.detach().clone() for name, value in writer.state_dict().items()}
    checkpoint = save_rl_writer_checkpoint(
        output_dir=tmp_path,
        next_cycle=1,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        tasks=tasks,
        task_schedule_seed=int(config["data"]["task_schedule_seed"]),
        rollouts_per_task=4,
        video_schedule=videos,
        contract=contract,
        cursors=InteractionCursors(96, 192, 2),
        successes=3,
        reward_sum=3.0,
        wall_nanoseconds=11,
        ledger_summary=ledger,
        metrics_rows=1,
        learning_epochs=2,
    )
    for parameter in writer.parameters():
        parameter.data.add_(2)
    cycle, cursors, _, rows, counters = load_rl_writer_checkpoint(
        checkpoint=checkpoint,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract_sha256=canonical_hash(contract),
        tasks=tasks,
        task_schedule_seed=int(config["data"]["task_schedule_seed"]),
        rollouts_per_task=4,
        video_schedule=videos,
        ledger_summary=ledger,
        learning_epochs=2,
    )
    assert cycle == rows == 1
    assert cursors == InteractionCursors(96, 192, 2)
    assert counters == {"successes": 3, "reward_sum": 3.0, "wall_nanoseconds": 11}
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
