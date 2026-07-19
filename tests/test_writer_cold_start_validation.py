from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import torch

import ember.writer.validation as validation
from ember.writer.validation import (
    aggregate_validation_rows,
    load_validation_contract,
    validation_work_for_rank,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parallel_context_allows_slow_ranks_to_finish_before_gather(
    monkeypatch,
) -> None:
    spec = {"parallel": {"world_size": 8}}
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    observed: dict[str, object] = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: observed.setdefault("device", device))

    def init_process_group(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    context = validation._parallel(spec)
    assert context == validation.ParallelContext(rank=3, local_rank=3, world_size=8)
    assert observed["kwargs"]["timeout"] == timedelta(hours=3)


def test_validation_contract_binds_writer_and_keeps_held_closed() -> None:
    spec = load_validation_contract(
        ROOT / "configs/writer_cold_start_validation.toml",
        repo_root=ROOT,
    )
    assert spec["parallel"]["world_size"] == 8
    assert spec["evaluation"]["rollouts_per_task_arm"] == 64
    assert spec["evaluation"]["task_ids"] == [11, 21, 51, 70, 86]
    assert spec["authority"]["test_held_numeric_access"] is False
    assert spec["lora"]["expected_parameter_count"] == 1_485_312


def test_eight_rank_assignment_covers_each_validation_arm_once() -> None:
    spec = load_validation_contract(
        ROOT / "configs/writer_cold_start_validation.toml",
        repo_root=ROOT,
    )
    work = [validation_work_for_rank(spec, rank=rank, world_size=8) for rank in range(8)]
    fits = [item["direct_fit_task"] for item in work if item["direct_fit_task"] is not None]
    evals = [arm for item in work for arm in item["evaluation_arms"]]
    assert sorted(fits) == spec["evaluation"]["task_ids"]
    assert len(evals) == 15
    assert len(set(evals)) == 15
    assert {
        arm for _, arm in evals
    } == set(spec["evaluation"]["arms"])
    for task_id in spec["evaluation"]["task_ids"]:
        assert sum(pair == (task_id, "matched_direct_task_local_lora") for pair in evals) == 1


def _rows() -> list[dict[str, object]]:
    rows = []
    for task_id in (11, 21):
        for arm, wins in (
            ("frozen_base", 24),
            ("writer_cold_start", 36),
            ("matched_direct_task_local_lora", 40),
        ):
            for episode in range(64):
                rows.append(
                    {
                        "task_id": task_id,
                        "task_category": f"category_{task_id}",
                        "arm": arm,
                        "execution_horizon": 16,
                        "policy_rng_seed": 100 + episode // 8,
                        "evaluator_seed": 200 + episode % 8,
                        "physical_init_state_index": 16 + episode % 8,
                        "success": episode < wins,
                    }
                )
    return rows


def test_validation_aggregation_preserves_raw_denominators_and_pairs() -> None:
    result = aggregate_validation_rows(
        _rows(),
        task_ids=[11, 21],
        arms=["frozen_base", "writer_cold_start", "matched_direct_task_local_lora"],
        horizons=[16],
        expected_rollouts=64,
        bootstrap_seed=7,
        bootstrap_replicates=1000,
    )
    writer = result["per_task"]["11"]["writer_cold_start"]["16"]
    assert writer["successes"] == 36
    assert writer["episodes"] == 64
    assert writer["success_rate"] == 36 / 64
    comparison = result["paired_vs_frozen_base"]["11"]["writer_cold_start"]["16"]
    assert comparison["episodes"] == 64
    assert comparison["paired_success_rate_difference"] == 12 / 64
    assert result["overall"]["writer_cold_start"]["16"]["successes"] == 72
    assert result["overall"]["writer_cold_start"]["16"]["episodes"] == 128
