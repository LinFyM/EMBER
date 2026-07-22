from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.pi05_source_checkpoint import sha256_file
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import environment_seed, policy_noise_seed
from ember.reward.protocol import RewardProtocolError
from ember.task_local.checkpoint import (
    load_task_local_checkpoint,
    save_task_local_checkpoint,
    validate_task_local_checkpoint_files,
    write_initialization_bundle,
)
from ember.task_local.contract import (
    cohort_video_demo,
    load_task_local_config,
    select_adaptation_checkpoint,
    task_local_units,
    test_tasks as sealed_test_tasks,
    unit_assignments,
)
from ember.task_local.initialization import prepare_unit_initialization


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_task_local_rl_test_v1.json"


def test_task_local_role_is_exactly_eight_test_tasks_and_never_source_sft() -> None:
    config = load_task_local_config(CONFIG)
    tasks = sealed_test_tasks(config)
    assert tuple(task.global_task_id for task in tasks) == (6, 8, 10, 17, 24, 27, 30, 33)
    assert {task.split_role for task in tasks} == {"test"}
    assert "source_sft" in config["forbidden_arms"]


def test_task_local_units_are_symmetric_with_or_without_rl_writer() -> None:
    config = load_task_local_config(CONFIG)
    three_arm = task_local_units(config, rl_writer_available=True)
    two_arm = task_local_units(config, rl_writer_available=False)
    assert len(three_arm) == 24
    assert tuple(map(len, unit_assignments(three_arm, 8))) == (3,) * 8
    assert {unit.arm for unit in three_arm} == {"identity", "as_writer", "rl_writer"}
    assert len(two_arm) == 16
    assert tuple(map(len, unit_assignments(two_arm, 8))) == (2,) * 8
    assert {unit.arm for unit in two_arm} == {"identity", "as_writer"}


def test_writer_arms_share_one_fixed_cohort_video_and_matched_seeds() -> None:
    config = load_task_local_config(CONFIG)
    units = task_local_units(config, rl_writer_available=True)
    grouped = {}
    for unit in units:
        grouped.setdefault((unit.global_task_id, unit.adaptation_seed), []).append(unit)
    for cohort in grouped.values():
        assert {cohort_video_demo(config, unit) for unit in cohort}.__len__() == 1
        assert {
            environment_seed(
                config["rng"]["environment_seed_root"],
                unit.suite,
                unit.task_id,
                unit.adaptation_seed,
                9,
            )
            for unit in cohort
        }.__len__() == 1
        assert {
            policy_noise_seed(
                config["rng"]["policy_noise_seed_root"],
                unit.suite,
                unit.task_id,
                unit.adaptation_seed,
                9,
                3,
            )
            for unit in cohort
        }.__len__() == 1


def test_task_local_selection_uses_random_reset_segment_rate_then_actions() -> None:
    candidates = (
        {"segment_successes": 1, "segment_rollouts": 8, "environment_action_cursor": 100},
        {"segment_successes": 2, "segment_rollouts": 8, "environment_action_cursor": 200},
        {"segment_successes": 1, "segment_rollouts": 4, "environment_action_cursor": 150},
    )
    assert select_adaptation_checkpoint(candidates) is candidates[2]


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)


def _toy_lora() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("projection", 3, 4),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=29,
    )


def test_initialization_bundle_and_checkpoint_bind_ledger_before_pickle(tmp_path: Path) -> None:
    policy = _ToyPolicy()
    contract = _toy_lora()
    inject_task_lora(policy, contract)
    unit_contract_sha = "a" * 64
    evidence = {
        "arm": "identity",
        "global_task_id": 24,
        "adaptation_seed": 7,
        "teacher_video_used": False,
        "teacher_demo_index": None,
        "source_checkpoint_manifest_sha256": "b" * 64,
        "stacked_source_sft": False,
    }
    initialization = write_initialization_bundle(
        unit_dir=tmp_path,
        state=task_lora_state_dict(policy, clone=True),
        contract=contract,
        unit_contract_sha256=unit_contract_sha,
        evidence=evidence,
    )
    initialization_sha = sha256_file(initialization / "initialization_manifest.json")
    optimizer = torch.optim.AdamW(task_lora_state_dict(policy).values(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ledger = {
        "rollout_cursor": 1,
        "environment_action_cursor": 8,
        "successes": 1,
        "reward_sum": 1.0,
        "ledger_prefix_sha256": "c" * 64,
    }
    expected = {
        name: value.detach().clone() for name, value in task_lora_state_dict(policy).items()
    }
    checkpoint = save_task_local_checkpoint(
        unit_dir=tmp_path,
        next_update=1,
        policy=policy,
        contract=contract,
        optimizer=optimizer,
        scheduler=scheduler,
        unit_contract_sha256=unit_contract_sha,
        initialization_manifest_sha256=initialization_sha,
        cursors=InteractionCursors(1, 8, 1),
        successes=1,
        reward_sum=1.0,
        wall_nanoseconds=13,
        rollouts_per_update=1,
        segment_successes=1,
        segment_rollouts=1,
        ledger_summary=ledger,
        device=torch.device("cpu"),
    )
    for value in task_lora_state_dict(policy).values():
        value.data.add_(2)
    update, cursors, _, counters = load_task_local_checkpoint(
        checkpoint=checkpoint,
        initialization_root=initialization,
        policy=policy,
        contract=contract,
        optimizer=optimizer,
        scheduler=scheduler,
        unit_contract_sha256=unit_contract_sha,
        rollouts_per_update=1,
        ledger_summary=ledger,
        device=torch.device("cpu"),
    )
    assert update == 1
    assert cursors == InteractionCursors(1, 8, 1)
    assert counters == {"successes": 1, "reward_sum": 1.0, "wall_nanoseconds": 13}
    for name, value in task_lora_state_dict(policy).items():
        torch.testing.assert_close(value, expected[name])

    rng = checkpoint / "rng_state.pt"
    rng.write_bytes(rng.read_bytes() + b"tamper")
    with pytest.raises(RewardProtocolError, match="file changed"):
        validate_task_local_checkpoint_files(
            checkpoint,
            unit_contract_sha256=unit_contract_sha,
            initialization_manifest_sha256=initialization_sha,
            ledger_summary=ledger,
        )


class _FakeWriterGenerator:
    def __init__(self, arm: str, state) -> None:
        self.arm = arm
        self.writer_state_sha256 = ("d" if arm == "as_writer" else "e") * 64
        self.state = state
        self.calls = []

    def generate(self, *, global_task_id: int, demo_index: int):
        self.calls.append((global_task_id, demo_index))
        return self.state, {"generator_arm": self.arm}


def test_as_and_rl_writer_initializations_use_same_fixed_video() -> None:
    config = load_task_local_config(CONFIG)
    units = [
        unit
        for unit in task_local_units(config, rl_writer_available=True)
        if unit.global_task_id == 24 and unit.arm != "identity"
    ]
    policy = _ToyPolicy()
    contract = _toy_lora()
    inject_task_lora(policy, contract)
    state = task_lora_state_dict(policy, clone=True)
    as_writer = _FakeWriterGenerator("as_writer", state)
    rl_writer = _FakeWriterGenerator("rl_writer", state)
    evidences = []
    for unit in units:
        _, evidence = prepare_unit_initialization(
            policy=policy,
            lora_contract=contract,
            config=config,
            unit=unit,
            source_checkpoint_manifest_sha256="f" * 64,
            as_writer=as_writer,
            rl_writer=rl_writer,
        )
        evidences.append(evidence)
    assert as_writer.calls == rl_writer.calls
    assert evidences[0]["teacher_demo_index"] == evidences[1]["teacher_demo_index"]
    assert all(evidence["teacher_video_used"] for evidence in evidences)
    assert all(evidence["stacked_source_sft"] is False for evidence in evidences)
