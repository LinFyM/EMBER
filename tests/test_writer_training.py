"""Dynamic Writer segment boundaries and nonzero continuing LR."""
from types import SimpleNamespace
import pytest
from ember.writer.continuation import require_continuation_config
from ember.writer.training import _checkpoint_nodes, _segment_limit, _learning_rate_multiplier


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
