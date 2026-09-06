"""Stable checkpoint and sampling contracts, independent of large GPU assets."""

from pathlib import Path

import pytest
import torch

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_source_checkpoint import DistributedContext
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.writer.learning_data import JointTrainingData, load_learning_tasks


def test_fixed_validation_cannot_enter_gradient_loader():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="fixed development split"):
        load_learning_tasks(root, [1])
    with pytest.raises(ValueError, match="excludes Test"):
        load_learning_tasks(root, [6], role="test")


def test_video_sets_use_task_occurrence_and_real_cardinalities():
    # Exercise the sampler without opening data: identities never enter the model.
    sampler = object.__new__(JointTrainingData)
    sampler.seed, sampler.video_pool = 7, tuple(range(16))
    videos = [sampler.video_demos(7, visit) for visit in range(12)]
    assert {len(row) for row in videos} == {1, 2, 4}
    assert all(len(set(row)) == len(row) for row in videos)
    assert len(set(videos)) > 3
    assert videos[8] == sampler.video_demos(7, 8)


def test_checkpoint_restores_next_update_and_checks_sampler(tmp_path, monkeypatch):
    # CPU isolates the new sampler field; actual rank CUDA state is verified by
    # the GPU run. It does not replace the retained native RNG implementation.
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr("ember.ecp.checkpoint.restore_rng", lambda state, _: torch.set_rng_state(state))
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    update()
    sampler = {"next_step": 1, "task_occurrences": {7: 1}}
    checkpoint = save_ecp_checkpoint(
        output_dir=tmp_path, macro=1, stage="joint_test", context=context,
        model=model, optimizer=optimizer, scheduler=scheduler,
        run_contract_schema="joint_test_v1", metrics_rows=1, sampler_state=sampler,
    )
    expected_loss = update()
    expected_weight = model.weight.detach().clone()
    args = dict(checkpoint=checkpoint, stage="joint_test", context=context,
                model=model, optimizer=optimizer, scheduler=scheduler, run_contract_schema="joint_test_v1")
    with pytest.raises(ValueError, match="cursor changed"):
        load_ecp_checkpoint(**args, expected_sampler_state={"next_step": 2})
    assert load_ecp_checkpoint(**args, expected_sampler_state=sampler) == (1, 1)
    torch.testing.assert_close(update(), expected_loss)
    torch.testing.assert_close(model.weight, expected_weight)


def test_resume_retains_distinct_orphaned_exposure_and_step_evidence(tmp_path):
    for name in ("exposures", "metrics"):
        path = tmp_path / f"{name}.jsonl"
        for step in (1, 2):
            append_jsonl(path, {"step": step, "kind": name})
        assert reconcile_metrics(path, 1, 1, cursor_key="step", packet_label=name) == 1
    packets = list((tmp_path / "failure_packets").glob("*.jsonl"))
    assert len(packets) == 2
    assert all('"step": 2' in p.read_text() for p in packets)
