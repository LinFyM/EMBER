import random
from pathlib import Path

import numpy as np
import torch

from ember.functional_adaptation.decoder_training import (
    ExpertAdapterRecord,
    balanced_task_order,
    decoder_task_split,
    load_functional_adapter_config,
)
from ember.functional_adaptation.decoder_flow_checkpoint import (
    load_decoder_flow_checkpoint,
    save_decoder_flow_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _records() -> tuple[ExpertAdapterRecord, ...]:
    return tuple(
        ExpertAdapterRecord(index, index, f"task {index}", Path(f"/{index}"))
        for index in range(24)
    )


def test_train24_decoder_split_is_result_independent_and_complete() -> None:
    split = decoder_task_split(_records(), fold_count=5, held_out_fold=0)

    assert len(split.fit) == 19
    assert tuple(row.ordinal for row in split.held) == (0, 5, 10, 15, 20)
    assert {row.ordinal for row in split.fit}.isdisjoint(
        row.ordinal for row in split.held
    )


def test_balanced_task_order_completes_each_full_macro_once() -> None:
    order = balanced_task_order(7, 15, seed=3)

    assert sorted(order[:7]) == list(range(7))
    assert sorted(order[7:14]) == list(range(7))
    assert len(order) == 15


def test_functional_adapter_profile_config_resolves_authorities() -> None:
    config = load_functional_adapter_config(
        REPO_ROOT / "configs/pi05_functional_adapter_v1.json", REPO_ROOT
    )

    assert config["decoder"]["production_code_width"] == 32
    assert config["code_inference"]["feature_encoder"]["text_meta_lora_rank"] == 0
    assert config["code_inference"]["feature_encoder"]["vl_meta_lora_rank"] == 0
    assert config["train24_mechanism"]["fit_task_count"] == 19
    formal = config["production_meta"]["flow_response"]["formal"]
    assert config["production_meta"]["formal_status"].startswith("frozen_56_15")
    assert formal["active_fit_tasks"] == 56
    assert formal["active_held_tasks"] == 15
    assert formal["checkpoint_steps"]["decoder"][-1] == formal["decoder_steps"]
    assert formal["checkpoint_steps"]["held_code"][-1] == formal["held_code_steps"]


def test_decoder_flow_checkpoint_restores_exact_training_state(tmp_path: Path) -> None:
    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(7)
    system = torch.nn.Linear(3, 2)
    held_codes = torch.nn.Parameter(torch.randn(2, 3))
    optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    system(torch.ones(1, 3)).square().sum().backward()
    optimizer.step()
    expected_system = {
        name: value.detach().clone() for name, value in system.state_dict().items()
    }
    expected_held = held_codes.detach().clone()

    checkpoint = save_decoder_flow_checkpoint(
        output_dir=tmp_path,
        phase="decoder",
        step=1,
        metrics_rows=1,
        visits=(1, 0),
        system=system,
        held_codes=held_codes,
        optimizer=optimizer,
    )
    expected_random = (random.random(), float(np.random.rand()), torch.rand(1))
    with torch.no_grad():
        for parameter in system.parameters():
            parameter.zero_()
        held_codes.zero_()

    cursor = load_decoder_flow_checkpoint(
        checkpoint=checkpoint,
        expected_phase="decoder",
        system=system,
        held_codes=held_codes,
        optimizer=optimizer,
    )

    assert cursor.step == 1
    assert cursor.visits == (1, 0)
    assert all(
        torch.equal(system.state_dict()[name], value)
        for name, value in expected_system.items()
    )
    assert torch.equal(held_codes, expected_held)
    assert random.random() == expected_random[0]
    assert float(np.random.rand()) == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])
