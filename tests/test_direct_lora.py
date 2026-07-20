from __future__ import annotations

from pathlib import Path

import torch

from ember.direct_lora_protocol import load_direct_lora_config, task_assignments
from ember.direct_lora_checkpoint import (
    load_direct_lora_checkpoint,
    save_direct_lora_checkpoint,
)
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    inject_task_lora,
    task_lora_state_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)


class _Sampler:
    per_rank_batch_size = 2
    seed = 31
    episodes_per_task = 2

    def coverage_for_steps(self, start: int, stop: int) -> dict[int, tuple[int, ...]]:
        assert (start, stop) == (0, 1)
        return {7: (0, 1)}

    def consumed_identity_summary(self, start: int, stop: int) -> dict[str, object]:
        assert (start, stop) == (0, 1)
        return {
            "start_step": 0,
            "stop_step": 1,
            "global_examples": 2,
            "unique_query_rows": 2,
            "min_examples_per_task": 2,
            "max_examples_per_task": 2,
            "identity_sha256": "a" * 64,
        }


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("projection", 3, 4),),
        rank=2,
        alpha=1,
        dropout=0.0,
        identity_seed=29,
    )


def test_direct_config_matches_writer_queries_and_validation_assignment() -> None:
    config = load_direct_lora_config(
        REPO_ROOT / "configs/direct_lora_sft_v1.json"
    )
    assert config["matching"]["consumed_queries_per_target_task"] == 69_120
    assert task_assignments((0, 8, 15, 28, 40, 56, 61, 71, 85, 88), 8) == (
        (0, 85),
        (8, 88),
        (15,),
        (28,),
        (40,),
        (56,),
        (61,),
        (71,),
    )


def test_direct_checkpoint_roundtrip_restores_only_lora(tmp_path: Path) -> None:
    policy = _ToyPolicy()
    contract = _contract()
    inject_task_lora(policy, contract)
    optimizer = torch.optim.AdamW(task_lora_state_dict(policy).values(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected = {
        name: value.detach().clone()
        for name, value in task_lora_state_dict(policy).items()
    }
    checkpoint = save_direct_lora_checkpoint(
        task_dir=tmp_path,
        task_id=7,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=_Sampler(),  # type: ignore[arg-type]
        task_contract_sha256="b" * 64,
        device=torch.device("cpu"),
        formal=True,
    )
    for value in task_lora_state_dict(policy).values():
        value.data.add_(1)
    loaded_step, _ = load_direct_lora_checkpoint(
        checkpoint=checkpoint,
        task_id=7,
        policy=policy,
        contract=contract,
        optimizer=optimizer,
        scheduler=scheduler,
        task_contract_sha256="b" * 64,
        per_rank_batch_size=2,
        sampler_seed=31,
        device=torch.device("cpu"),
    )
    assert loaded_step == 1
    for name, value in task_lora_state_dict(policy).items():
        torch.testing.assert_close(value, expected[name])
