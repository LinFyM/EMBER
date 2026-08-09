from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.v6_prior_runtime import (
    _cursor_contract,
    _sampled_video_cost,
    _scheduler,
)
from ember.expert_manifold.v6_prior_training import (
    _mean_trainable_gradients,
    suggest_auxiliary_weight,
)
from ember.pi05_source_checkpoint import DistributedContext


def test_v6_prior_video_cost_includes_true_final_frame() -> None:
    assert _sampled_video_cost(1, 5) == 1
    assert _sampled_video_cost(6, 5) == 2
    assert _sampled_video_cost(7, 5) == 3
    assert _sampled_video_cost(105, 5) == 22


def test_v6_prior_auxiliary_weight_obeys_both_trainable_groups() -> None:
    positive = {"compiler": 2.0, "factor_heads": 4.0}
    auxiliary = {"compiler": 8.0, "factor_heads": 1.0}
    assert suggest_auxiliary_weight(
        positive,
        auxiliary,
        maximum_fraction=0.25,
    ) == pytest.approx(0.0625)
    assert suggest_auxiliary_weight(
        positive,
        {"compiler": 0.0, "factor_heads": 0.0},
        maximum_fraction=0.25,
    ) == 0.0


def test_v6_prior_scheduler_is_low_lr_warmup_then_fifty_macro_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW((parameter,), lr=3e-5)
    config = {
        "optimization": {
            "optimizer": {"peak_lr": 3e-5},
            "scheduler": {
                "warmup_macros": 2,
                "total_macros": 50,
                "decay_lr": 3e-6,
            },
        }
    }
    scheduler = _scheduler(optimizer, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    for _ in range(49):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)


def test_v6_prior_flat_gradient_reduction_is_one_global_task_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([3.0])

    def all_reduce(value: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert torch.equal(value, torch.tensor([1.0, 2.0, 3.0]))
        value.mul_(2).add_(torch.tensor([2.0, 4.0, 6.0]))

    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.dist.all_reduce",
        all_reduce,
    )
    runtime = SimpleNamespace(
        trainable_parameters=(first, second),
        context=DistributedContext(0, 0, 2, torch.device("cpu")),
    )
    _mean_trainable_gradients(runtime)
    assert torch.equal(first.grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(second.grad, torch.tensor([6.0]))


def test_v6_prior_cursor_records_all_stateless_schedules() -> None:
    config = {
        "data": {
            "sampler_seed": 11,
            "teacher_video_seed": 13,
            "counterfactual_seed": 17,
            "action_queries_per_task": 20,
        }
    }
    assert _cursor_contract(config, 25) == {
        "next_macro": 25,
        "task_visits_per_task": 25,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "counterfactual_phase": 1,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
    }
