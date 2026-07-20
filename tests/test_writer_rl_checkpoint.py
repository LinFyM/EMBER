from __future__ import annotations

from pathlib import Path

import torch

from ember.source_base_checkpoint import DistributedContext, canonical_hash
from ember.writer_rl_checkpoint import (
    load_writer_rl_checkpoint,
    save_writer_rl_checkpoint,
    write_update_ledger_once,
)


def test_writer_rl_checkpoint_roundtrip_and_replay_ledger(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    writer = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected = {name: value.detach().clone() for name, value in writer.state_dict().items()}
    contract = {"schema_version": "test", "value": 3}
    ledger = {"rank": 0, "update": 0, "trajectories": []}
    first = write_update_ledger_once(tmp_path, 0, 0, ledger)
    assert write_update_ledger_once(tmp_path, 0, 0, ledger) == first
    checkpoint = save_writer_rl_checkpoint(
        output_dir=tmp_path,
        next_update=1,
        optimizer_updates=1,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        task_ids=(1,),
        rollouts_per_task=4,
        contract=contract,
        local_counters={
            "rollouts": 4,
            "successes": 1,
            "env_steps": 200,
            "wall_nanoseconds": 10,
        },
        formal=False,
    )
    with torch.no_grad():
        for value in writer.parameters():
            value.add_(1)
    update, optimizer_updates, counters, _ = load_writer_rl_checkpoint(
        checkpoint=checkpoint,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract_sha256=canonical_hash(contract),
        task_ids=(1,),
        rollouts_per_task=4,
    )
    assert update == optimizer_updates == 1
    assert counters == {
        "rollouts": 4,
        "successes": 1,
        "env_steps": 200,
        "wall_nanoseconds": 10,
    }
    for name, value in writer.state_dict().items():
        torch.testing.assert_close(value, expected[name])
