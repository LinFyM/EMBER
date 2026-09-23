"""Dynamic Writer segment boundaries and nonzero continuing LR."""
from types import SimpleNamespace
import pytest
from ember.writer.continuation import require_continuation_config
from ember.writer.training import (
    _checkpoint_nodes, _learning_rate_multiplier, _segment_limit,
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


def test_task_mixing_nodes_and_registered_six_rank_profile():
    import json
    from pathlib import Path
    from ember.writer.training import _config, _logical_batch
    root = Path(__file__).resolve().parents[1]
    config = _config(root / 'configs/libero_24_8_8_coverage_v1/writer_task_diversity.json')
    reference = json.loads((root / 'configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json').read_text())
    args = SimpleNamespace(mode='formal', stop_after_step=1200, checkpoint_updates=None)
    assert _segment_limit(args, config) == 1200
    assert _checkpoint_nodes(args, config) == tuple(range(100, 1201, 100))
    assert config['evidence']['qualification']['optimizer_updates'] == list(range(200, 1201, 200))
    profile = config['evidence']['profile_registration']
    assert profile['status'] == 'complete' and profile['world_size'] == 6
    assert profile['formal_uses_profile_weights'] is False
    assert config['optimization'] == reference['optimization']
    assert config['optimization']['warmup_updates'] == 150
    assert config['runtime']['policy_microbatch'] == 16 and config['observer']['frame_chunk'] == 8
    assert _logical_batch(config)['total_queries_per_update'] == 112


def test_dynamic_config_accepts_clock_and_preserves_loss(tmp_path):
    import json
    from pathlib import Path
    from ember.writer.training import _config, observer_mode_contract
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/libero_24_8_8_coverage_v1/writer_task_diversity.json').read_text())
    config.update(training_control=dynamic_config()['training_control'])
    config['data']['maximum_updates'] = None
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


def test_task_mixing_config_rejects_old_contract_and_keeps_authorities(tmp_path):
    import json
    from pathlib import Path
    from ember.writer.training import _config

    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root / 'configs/libero_24_8_8_coverage_v1/writer_task_diversity.json').read_text())
    path = tmp_path / 'writer.json'
    path.write_text(json.dumps(candidate))
    assert _config(path)['data']['tasks_per_update'] == 12
    candidate['data']['queries_per_task'] = 21
    candidate['data']['teaching_query_counts'] = [7] * 12
    path.write_text(json.dumps(candidate))
    assert _config(path)['data']['queries_per_task'] == 21
    candidate.pop('experiment')
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='scientific contract'):
        _config(path)
    old = root / 'configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json'
    with pytest.raises(ValueError, match='scientific contract'):
        _config(old)
    candidate = json.loads((root / 'configs/libero_24_8_8_coverage_v1/writer_task_diversity.json').read_text())
    candidate['data']['task_ids'][0] = 3  # fixed Validation task cannot replace a training task
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='authority'):
        _config(path)
