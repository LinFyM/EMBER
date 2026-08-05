from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.writer import as_step
from ember.writer.condition_kernel import ProgramValueMemory
from ember.writer.task_gradient import parameter_layout


class _TinyWriter(torch.nn.Module):
    PROGRAM_SLOTS = 2
    program_width = 2

    def __init__(self) -> None:
        super().__init__()
        self.condition_feature = SimpleNamespace(feature_width=3)
        self.program_memory = ProgramValueMemory(
            feature_width=3,
            program_slots=2,
            program_width=2,
            initialization_seed=3,
        )
        self.program_memory.value.requires_grad_(False)
        self.factor_heads = torch.nn.Linear(1, 1, bias=False)
        self.factor_heads.weight.data.zero_()


def _runtime() -> SimpleNamespace:
    writer = _TinyWriter()
    optimizer = torch.optim.SGD(writer.factor_heads.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return SimpleNamespace(
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_layout=parameter_layout(writer),
        policy=torch.nn.Identity(),
        context=SimpleNamespace(
            device=torch.device("cpu"), world_size=1, rank=0
        ),
        task_ids=(3, 7),
        config={
            "conditioning_training": {
                "factor_decoder_train_through_macro": 50,
            },
            "optimization": {
                "factor_decoder_optimizer": {"gradient_clip_norm": 10.0},
                "program_memory_update": {
                    "step_size": 0.2,
                    "relative_damping": 0.01,
                    "induced_program_rms_cap": 1.0,
                },
            },
        },
    )


def _credit(runtime: SimpleNamespace) -> tuple[object, ...]:
    records = [
        {"task_id": 7, "loss": 0.7},
        {"task_id": 3, "loss": 0.3},
    ]
    return (
        records,
        0.0,
        torch.tensor([7, 3], dtype=torch.long),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ]
        ),
        torch.tensor([[1.0], [1.0]]),
    )


def test_factor_decoder_stops_at_50_but_program_memory_keeps_updating(
    monkeypatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        as_step,
        "_collect_local_credit",
        lambda selected, _step, factor_active: _credit(selected),
    )
    monkeypatch.setattr(
        as_step,
        "_step_metrics",
        lambda _runtime, **values: values,
    )
    initial_memory = runtime.writer.program_memory.value.detach().clone()
    active = as_step.run_writer_step(runtime, step=0, started=0.0)
    after_active_memory = runtime.writer.program_memory.value.detach().clone()
    after_active_factor = runtime.writer.factor_heads.weight.detach().clone()
    frozen = as_step.run_writer_step(runtime, step=50, started=0.0)
    assert active["factor_active"] is True
    assert frozen["factor_active"] is False
    assert not torch.equal(initial_memory, after_active_memory)
    assert not torch.equal(after_active_memory, runtime.writer.program_memory.value)
    assert torch.equal(after_active_factor, runtime.writer.factor_heads.weight)
    assert active["program_metrics"]["task_ids"] == [3, 7]
    assert frozen["factor_metrics"] is None
