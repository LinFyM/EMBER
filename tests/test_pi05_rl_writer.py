from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, canonical_hash
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError
from ember.rl_writer import contract as rl_contract
from ember.rl_writer import loop as rl_loop
from ember.rl_writer import rendezvous as rl_rendezvous
from ember.rl_writer import runtime as rl_runtime
from ember.rl_writer.checkpoint import (
    load_rl_writer_checkpoint,
    save_rl_writer_checkpoint,
    validate_rl_writer_checkpoint_files,
)
from ember.rl_writer.contract import (
    cycle_assignments,
    load_coldstart_writer_config,
    load_rl_writer_config,
    publish_contract,
    resolve_runtime,
    reward_tasks,
    schedule_summary,
)
from ember.rl_writer.training import build_parser
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.topology import WriterModelError, visible_physical_cuda_index


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_rl_writer_development_v1.json"
)


def test_program_credit_config_closes_actions_and_freezes_public_decoder() -> None:
    config = load_rl_writer_config(CONFIG)
    algorithm = config["algorithm"]
    assert algorithm["name"] == "antithetic_program_credit_writer_v1"
    assert algorithm["rollouts_per_task_condition"] == 4
    assert algorithm["antithetic_pairs_per_task"] == 2
    assert algorithm["program_shape"] == [320, 256]
    assert algorithm["program_sigma"] == pytest.approx(0.05)
    assert algorithm["teacher_actions"] is False
    assert algorithm["functional_action_loss"] is False
    assert algorithm["executed_action_replay"] is False
    assert algorithm["semantic_encoder_frozen_after_coldstart"] is True
    assert algorithm["factor_head_decoder_frozen_after_coldstart"] is True
    assert config["coldstart_writer_runtime"]["max_frames_per_encoder_call"] == 16
    assert load_coldstart_writer_config(config)["writer"][
        "max_frames_per_encoder_call"
    ] == 16
    assert config["information_wall"]["teacher_action_reads_after_coldstart"] == 0
    assert config["parallel"]["maximum_world_size"] == 6
    assert config["formal_run"]["checkpoint_cycles"] == [1, 2, 4, 8]


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


def test_program_credit_parser_requires_coldstart_and_raw_video_data() -> None:
    destinations = {action.dest for action in build_parser()._actions}
    assert "coldstart_checkpoint" in destinations
    assert "data_root" in destinations
    assert "feature_cache" not in destinations
    assert "diagnostic" not in destinations


def test_only_program_upstream_is_trainable_after_coldstart() -> None:
    class TinyWriter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_encoder = torch.nn.Linear(2, 2)
            self.semantic_core = torch.nn.Linear(2, 2)
            self.visual_transition = torch.nn.Linear(2, 2)
            self.procedure = torch.nn.Linear(2, 2)
            self.compiler = torch.nn.Linear(2, 2)
            self.factor_heads = torch.nn.ModuleDict(
                {f"head_{index}": torch.nn.Linear(2, 2) for index in range(8)}
            )

    writer = TinyWriter()
    optimizer, _ = rl_runtime._prepare_program_optimizer(
        writer,
        {"mode": "writer_weight_warm_start"},
        load_rl_writer_config(CONFIG),
    )
    frozen = {
        name
        for name, parameter in writer.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen
    assert all(
        name.startswith(("semantic_encoder.", "factor_heads."))
        for name in frozen
    )
    trainable_prefixes = {
        name.split(".", 1)[0]
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_prefixes == {
        "semantic_core",
        "visual_transition",
        "procedure",
        "compiler",
    }
    assert sum(len(group["params"]) for group in optimizer.param_groups) == sum(
        parameter.requires_grad for parameter in writer.parameters()
    )


def test_torchrun_local_rank_maps_to_the_physical_egl_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,7")
    assert [visible_physical_cuda_index(rank) for rank in range(6)] == [1, 2, 3, 4, 5, 7]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    with pytest.raises(WriterModelError, match="numeric physical GPU"):
        visible_physical_cuda_index(0)


def test_credit_collective_waits_for_rank_local_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "unit-program-credit-rendezvous")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    session = rl_rendezvous._credit_rendezvous_session()
    ready = (
        tmp_path
        / ".rank-local-credit-ready"
        / session
        / "cycle-00000003-epoch-0000"
    )
    ready.mkdir(parents=True)
    (ready / "rank-01.json").write_text("{}\n", encoding="utf-8")
    original_write = rl_rendezvous.write_json_atomic
    monkeypatch.setattr(
        rl_rendezvous.torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: events.append("cuda_synchronize"),
    )
    monkeypatch.setattr(
        rl_rendezvous,
        "write_json_atomic",
        lambda path, value: (
            events.append(f"marker:{path.name}"),
            original_write(path, value),
        )[-1],
    )
    monkeypatch.setattr(
        rl_loop.dist,
        "all_reduce",
        lambda *_args, **_kwargs: events.append("all_reduce"),
    )
    writer = torch.nn.Linear(3, 2)
    for parameter in writer.parameters():
        parameter.grad = torch.ones_like(parameter)
    runtime = Namespace(
        args=Namespace(output_dir=tmp_path),
        context=DistributedContext(0, 0, 2, torch.device("cuda:0")),
        writer=writer,
    )

    rl_loop._all_reduce_writer_gradients(runtime, cycle=3)

    assert events == ["cuda_synchronize", "marker:rank-00.json", "all_reduce"]
    assert (ready / "rank-00.json").is_file()


def test_information_wall_and_runtime_gates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["information_wall"]["validation_reward_reads"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RewardProtocolError, match="information"):
        load_rl_writer_config(path)

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
    assert resolve_runtime(args, config, context) == (2, (1, 2), 1)
    blocked = json.loads(json.dumps(config))
    blocked["profile_defaults"]["status"] = "blocked"
    args.stop_after_cycle = None
    with pytest.raises(RewardProtocolError, match="awaits"):
        resolve_runtime(args, blocked, context)
    args.stop_after_cycle = None
    invalid = DistributedContext(0, 0, 5, torch.device("cpu"), 0, (0,))
    with pytest.raises(RewardProtocolError, match="divide train24"):
        resolve_runtime(args, config, invalid)
    config["formal_run"]["status"] = "sealed"
    args.mode = "formal"
    args.stop_after_cycle = None
    monkeypatch.setattr(
        rl_contract,
        "git_state",
        lambda _root: {"dirty_paths": [], "commit": "a", "origin_main": "a"},
    )
    assert resolve_runtime(args, config, context) == (8, (1, 2, 4, 8), 1)


def test_program_credit_contract_is_single_owner_and_resume_bound(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    (tmp_path / "libero_config").mkdir()
    contract = {"schema_version": "test", "coldstart": "sealed"}
    digest = publish_contract(
        output_dir=tmp_path, contract=contract, resume=None, context=context
    )
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


def test_checkpoint_roundtrip_binds_program_credit_ledger(tmp_path: Path) -> None:
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
        "program_credit_cursor": 24,
        "program_credit_prefix_sha256": "c" * 64,
    }
    contract = {"schema_version": "test", "method": "program-credit"}
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
        cursors=InteractionCursors(96, 192, 1),
        successes=3,
        reward_sum=3.0,
        wall_nanoseconds=11,
        ledger_summary=ledger,
        metrics_rows=1,
        learning_epochs=1,
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
        learning_epochs=1,
    )
    assert cycle == rows == 1
    assert cursors == InteractionCursors(96, 192, 1)
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
