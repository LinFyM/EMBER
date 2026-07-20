from __future__ import annotations

from pathlib import Path

import torch

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.task_local_rl_checkpoint import (
    load_task_local_checkpoint,
    save_task_local_checkpoint,
    write_unit_ledger_once,
)


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("projection", 3, 4),),
        rank=2,
        alpha=1,
        dropout=0.0,
        identity_seed=29,
    )


def test_task_local_checkpoint_roundtrip_and_ledger(tmp_path: Path) -> None:
    policy = _ToyPolicy()
    contract = _contract()
    inject_task_lora(policy, contract)
    optimizer = torch.optim.AdamW(task_lora_state_dict(policy).values(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected = {
        name: value.detach().clone()
        for name, value in task_lora_state_dict(policy).items()
    }
    row = {"update": 0, "trajectories": []}
    assert write_unit_ledger_once(tmp_path, 0, row) == write_unit_ledger_once(
        tmp_path, 0, row
    )
    checkpoint = save_task_local_checkpoint(
        unit_dir=tmp_path,
        next_update=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        unit_contract_sha256="a" * 64,
        counters={
            "rollouts": 4,
            "successes": 1,
            "env_steps": 200,
            "optimizer_updates": 1,
        },
        rollouts_per_update=4,
        segment_successes=1,
        segment_rollouts=4,
        device=torch.device("cpu"),
    )
    for value in task_lora_state_dict(policy).values():
        value.data.add_(1)
    next_update, counters, _ = load_task_local_checkpoint(
        checkpoint=checkpoint,
        policy=policy,
        contract=contract,
        optimizer=optimizer,
        scheduler=scheduler,
        unit_contract_sha256="a" * 64,
        rollouts_per_update=4,
        device=torch.device("cpu"),
    )
    assert next_update == 1
    assert counters["rollouts"] == 4
    for name, value in task_lora_state_dict(policy).items():
        torch.testing.assert_close(value, expected[name])
