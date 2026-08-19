from pathlib import Path

from ember.functional_adaptation.decoder_training import (
    ExpertAdapterRecord,
    balanced_task_order,
    decoder_task_split,
    load_functional_adapter_config,
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
    assert config["train24_mechanism"]["fit_task_count"] == 19
    assert config["production_meta"]["formal_status"].startswith("unsealed")
