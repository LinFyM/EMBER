"""Dynamic Writer segment boundaries and nonzero continuing LR."""
from types import SimpleNamespace
import pytest
import torch
from ember.pi05_source_checkpoint import DistributedContext
from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.writer.continuation import LOW_LR_REPAIR, require_continuation_config
from ember.writer.training import (
    RUN_SCHEMA, STAGE, _activate_phase_schedule, _checkpoint_nodes,
    _learning_rate_multiplier, _optimization, _segment_limit,
)


def dynamic_config():
    return {"data": {"maximum_updates": None}, "training_control": {
        "kind": "validation_early_stopping", "checkpoint_interval": 100, "validation_interval": 200}}


def test_dynamic_segments_have_no_global_ceiling():
    config = dynamic_config()
    require_continuation_config(config)
    args = SimpleNamespace(mode="formal", stop_after_step=2400, checkpoint_updates=None)
    assert _segment_limit(args, config) == 2400
    assert _checkpoint_nodes(args, config) == tuple(range(100, 2401, 100))
    args.stop_after_step = 2300
    with pytest.raises(ValueError, match="validation boundary"):
        _segment_limit(args, config)
    args.mode, args.stop_after_step = "smoke", 2
    assert _segment_limit(args, config) == 2


def test_dynamic_lr_keeps_tail_floor_after_end():
    opt = dict(warmup_updates=150, tail_start_update=1350, tail_end_update=2250,
               tail_final_ratio=.1, decay_updates=18000, decay_lr=1e-5, lr=3e-4)
    floor = _learning_rate_multiplier(2250, opt)
    assert floor > 0
    assert floor == _learning_rate_multiplier(100000, opt)
    assert _learning_rate_multiplier(1350, opt) > floor


def test_dynamic_config_accepts_clock_and_preserves_loss(tmp_path):
    import json
    from pathlib import Path
    from ember.writer.training import _config, observer_mode_contract
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/pi05_writer.json').read_text())
    config.update(training_control=dynamic_config()['training_control'])
    config['data'].update(maximum_updates=None, protocol='coverage/protocol.json', task_ids=list(range(36)))
    config['model']['camera_view'] = 'agentview'
    config['observer'].update(observer_mode_contract(config['model']))
    config['optimization'].update(warmup_updates=150, tail_start_update=1350, tail_end_update=2250,
                                  tail_final_ratio=.1, decay_updates=18000)
    path = tmp_path / 'writer.json'
    path.write_text(json.dumps(config))
    assert _config(path)['data']['maximum_updates'] is None
    config['optimization']['teaching_weight'] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='scientific contract'):
        _config(path)


def test_low_lr_phase_uses_fixed_first_update_and_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr("ember.ecp.checkpoint.restore_rng", lambda state, _: torch.set_rng_state(state))
    config = {
        "optimization": {"lr": 3e-4, "betas": [.9, .95], "eps": 1e-8,
                         "weight_decay": 1e-4, "warmup_updates": 150,
                         "tail_start_update": 1350, "tail_end_update": 2250,
                         "tail_final_ratio": .1, "decay_updates": 18000, "decay_lr": 1e-5},
        "phase_continuation": dict(LOW_LR_REPAIR),
    }
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimization(model, config)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(1800)
    optimizer.param_groups[0]["lr"] = LOW_LR_REPAIR["parent_applied_lr"]
    scheduler.last_epoch = 1800
    scheduler._step_count = 1801
    scheduler._last_lr = [LOW_LR_REPAIR["parent_applied_lr"]]
    runtime = SimpleNamespace(state=model)

    _activate_phase_schedule(optimizer, scheduler, runtime, config, 1800, initial_transition=True)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(LOW_LR_REPAIR["fixed_lr"])
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1801
    assert optimizer.param_groups[0]["lr"] == pytest.approx(LOW_LR_REPAIR["fixed_lr"])

    checkpoint = save_ecp_checkpoint(
        output_dir=tmp_path, macro=1801, stage=STAGE, context=context, model=model,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA,
        metrics_rows=7204, sampler_state=None, training_state={"cursor": 1801},
    )
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer, restored_scheduler = _optimization(restored_model, config)
    restored = {}
    assert load_ecp_checkpoint(
        checkpoint=checkpoint, stage=STAGE, context=context, model=restored_model,
        optimizer=restored_optimizer, scheduler=restored_scheduler,
        run_contract_schema=RUN_SCHEMA, restored_state=restored,
    ) == (1801, 7204)
    _activate_phase_schedule(restored_optimizer, restored_scheduler,
                             SimpleNamespace(state=restored_model), config, 1801,
                             initial_transition=False)
    assert restored_scheduler.last_epoch == 1801
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(LOW_LR_REPAIR["fixed_lr"])
