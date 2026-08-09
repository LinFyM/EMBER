from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import (
    configure_v6_prior_trainability,
    v6_prior_trainable_parameters,
)
from ember.expert_manifold.v6_prior_checkpoint import (
    load_v6_prior_checkpoint,
    save_v6_prior_checkpoint,
)
from ember.pi05_source_checkpoint import DistributedContext


def _writer():
    path = Path(__file__).with_name("test_writer_model.py")
    spec = importlib.util.spec_from_file_location("v6_writer_test_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model()[0]


def _optimizer(writer):
    optimizer = torch.optim.AdamW(
        v6_prior_trainable_parameters(writer),
        lr=3e-5,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0 if step >= 0 else 0.0,
    )
    return optimizer, scheduler


def test_v6_prior_checkpoint_restores_writer_optimizer_scheduler_and_cursor(
    tmp_path: Path,
) -> None:
    writer = _writer()
    configure_v6_prior_trainability(writer)
    optimizer, scheduler = _optimizer(writer)
    for parameter in v6_prior_trainable_parameters(writer):
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {
        name: value.detach().clone() for name, value in writer.state_dict().items()
    }
    cursor = {
        "next_macro": 1,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "task_visits_per_task": 1,
    }
    contract = {
        "schema": "v1",
        "warm_start": "historical_v6_macro400_load_only",
        "frozen_blocks": 4,
        "trainable_blocks": 2,
    }
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    checkpoint = save_v6_prior_checkpoint(
        output_dir=tmp_path,
        macro=1,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        metrics_rows=1,
        cursor_contract=cursor,
        checkpoint_contract=contract,
    )
    for parameter in writer.parameters():
        parameter.data.zero_()
    loaded_macro, rows = load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        expected_cursor_contract=cursor,
        expected_checkpoint_contract=contract,
    )
    assert (loaded_macro, rows) == (1, 1)
    assert all(
        torch.equal(expected[name], value)
        for name, value in writer.state_dict().items()
    )
    assert scheduler.last_epoch == 1

    changed = {**cursor, "teacher_video_seed": 19}
    with pytest.raises(ExpertManifoldError, match="manifest changed"):
        load_v6_prior_checkpoint(
            checkpoint=checkpoint,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            expected_cursor_contract=changed,
            expected_checkpoint_contract=contract,
        )
