"""Frozen cache batching and asynchronous byte bounds used by real consumers."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Event

import pytest
import torch

from ember.experience_compiler.storage import FeatureCache, RecordWriter, tensor_bytes
from ember.experience_compiler import storage


def test_native_phi_batch_preserves_dtype_and_evicts_within_disk_budget(tmp_path):
    cache = FeatureCache(tmp_path, max_bytes=5 * 1024**2)
    first = torch.full((512, 2048), 1.25, dtype=torch.float32)
    second = first + 2
    cache.put_many({'first': {'phi': first}, 'second': {'phi': second}})
    result = cache.get_many(['first', 'second', 'absent'])
    assert set(result) == {'second'}
    torch.testing.assert_close(result['second']['phi'], second)
    cache.put('second', {'phi': first})
    torch.testing.assert_close(cache.get('second')['phi'], second)
    with sqlite3.connect(tmp_path / 'cache.sqlite3') as database:
        assert database.execute('SELECT SUM(bytes) FROM entries').fetchone()[0] <= cache.max_bytes
    assert not list(tmp_path.glob('*.partial'))
    assert tensor_bytes({'x': [first, second]}) == 8 * 1024**2


def test_writer_bounds_bytes_before_sixteen_futures_and_propagates_errors():
    release, entered, waited = Event(), Event(), Event()
    writer = RecordWriter(workers=1, max_bytes=4)
    def blocked():
        entered.set()
        assert release.wait(5)
    first = writer.submit(blocked, byte_cost=4)
    assert entered.wait(5)
    original_result = first.result
    def result(*args, **kwargs):
        waited.set()
        return original_result(*args, **kwargs)
    first.result = result
    try:
        with ThreadPoolExecutor(max_workers=1) as submitter:
            second = submitter.submit(writer.submit, lambda: 7, byte_cost=4)
            assert waited.wait(5)
            assert not second.done() and writer.pending_bytes == 4
            release.set()
            assert second.result(timeout=5).result(timeout=5) == 7
        writer.flush()
        assert writer.pending_bytes == 0
        writer.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            writer.flush()
    finally:
        release.set()
        writer.close()


def test_slow_serialization_does_not_hold_cache_publication_lock(tmp_path, monkeypatch):
    cache = FeatureCache(tmp_path)
    started, release = Event(), Event()
    original = storage.save_file
    def save(values, path):
        if '/slow.' in path:
            started.set()
            assert release.wait(5)
        return original(values, path)
    monkeypatch.setattr(storage, 'save_file', save)
    try:
        with ThreadPoolExecutor(2) as executor:
            slow = executor.submit(cache.put, 'slow', {'phi': torch.ones(2)})
            assert started.wait(5)
            fast = executor.submit(cache.put, 'fast', {'phi': torch.zeros(2)})
            fast.result(timeout=5)
            assert cache.get('slow') is None
            torch.testing.assert_close(cache.get('fast')['phi'], torch.zeros(2))
            release.set()
            slow.result(timeout=5)
    finally:
        release.set()
    assert not list(tmp_path.glob('*.partial'))


def test_actual_mt_layer_flag_retains_the_recorded_incoming_file(tmp_path):
    from ember.experience_compiler.interaction import Chain
    from safetensors.torch import load_file
    chain = Chain(states=[{'w': torch.ones(2)}] * 3,
        records=[dict(episode=0), dict(episode=1)], endpoints=[1, 2],
        behavior_versions=['MT', 'phi180:lambda001'],
        metrics={'edit_parameter_statistics': [dict(incoming_is_MT=True, outgoing_is_MT=True),
                                               dict(incoming_is_MT=True, outgoing_is_MT=False)]})
    record = storage.save_condition(tmp_path / 'condition', dict(condition_id='actual_mt'), chain)
    assert all(event['incoming_is_MT'] for event in record['events'])
    assert record['events'][1]['incoming'] == 'incoming_001.safetensors'
    torch.testing.assert_close(load_file(str(tmp_path / 'condition/incoming_001.safetensors'))['w'], torch.ones(2))
