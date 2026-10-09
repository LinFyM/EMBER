"""Actual event provenance, asynchronous raw facts and a bounded frozen cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import uuid

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import EVENT_SCHEMA, MT_PATH


class FeatureCache:
    """Bounded frozen teacher/native features within one host's cache shard.

    Cache entries are disposable derivatives, never raw facts or parameters.
    SQLite metadata and file mutations share the existing filesystem lock;
    there is no permanent duplicate pre/post Phi for one observation ID.
    """
    def __init__(self, root, *, max_bytes=64 * 1024**3):
        self.root, self.max_bytes = Path(root), max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / 'cache.lock'
        with self._locked() as database:
            database.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, bytes INTEGER, touched REAL)')

    @contextmanager
    def _locked(self):
        with self.lock.open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with closing(sqlite3.connect(self.root / 'cache.sqlite3', timeout=60)) as database:
                with database:
                    yield database

    def _path(self, key):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ' for c in key):
            raise ValueError('invalid frozen-cache observation identity')
        return self.root / (key + '.safetensors')

    def get(self, key):
        return self.get_many([key]).get(key)

    def get_many(self, keys):
        paths = {key: self._path(key) for key in keys}
        result = {}
        with self._locked() as database:
            for key, path in paths.items():
                if database.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone():
                    result[key] = load_file(str(path))
                    database.execute('UPDATE entries SET touched=? WHERE key=?', (time.time(), key))
        return result

    def put(self, key, values):
        self.put_many({key: values})

    def put_many(self, entries):
        # CPU conversion happens before taking the file/metadata lock. Native
        # batches use one future and one transaction, rather than one per Phi.
        entries = [(key, self._path(key), {k: v.detach().cpu().contiguous() for k, v in values.items()})
                   for key, values in entries.items() if tensor_bytes(values) + 4096 <= self.max_bytes]
        if not entries:
            return
        staged = []
        try:
            for key, path, values in entries:
                # File serialization overlaps other workers. Only publishing
                # and eviction need the shard metadata lock, not a 64MiB save.
                partial = path.with_suffix(f'.{uuid.uuid4().hex}.partial')
                staged.append((key, path, partial))
                save_file(values, str(partial))
            with self._locked() as database:
                used = database.execute('SELECT COALESCE(SUM(bytes),0) FROM entries').fetchone()[0]
                for key, path, partial in staged:
                    if database.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone():
                        continue
                    actual = partial.stat().st_size
                    if used + actual > self.max_bytes:
                        for old, old_size in database.execute('SELECT key,bytes FROM entries ORDER BY touched').fetchall():
                            if used + actual <= self.max_bytes:
                                break
                            self._path(old).unlink(missing_ok=True)
                            database.execute('DELETE FROM entries WHERE key=?', (old,))
                            used -= old_size
                    partial.replace(path)
                    database.execute('INSERT INTO entries VALUES (?,?,?)', (key, actual, time.time()))
                    used += actual
        finally:
            for _, _, partial in staged:
                partial.unlink(missing_ok=True)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.nbytes
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class RecordWriter:
    """CPU serialization overlaps other slots; failures propagate on flush."""
    def __init__(self, workers=2, *, max_bytes=512 * 1024**2):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='compiler-record')
        self.pending = []
        self.max_bytes, self.pending_bytes, self.wait_seconds = max_bytes, 0, 0.

    def _finish(self, index=0):
        future, size = self.pending.pop(index)
        tick = time.monotonic()
        try:
            return future.result()
        finally:
            self.wait_seconds += time.monotonic() - tick
            self.pending_bytes -= size

    def submit(self, function, *args, byte_cost=0, **kwargs):
        # Keep asynchronous temporary memory bounded instead of buffering a run.
        for index in reversed(range(len(self.pending))):
            if self.pending[index][0].done():
                self._finish(index)
        while self.pending and (len(self.pending) >= 16 or self.pending_bytes + byte_cost > self.max_bytes):
            self._finish()
        future = self.executor.submit(function, *args, **kwargs)
        self.pending.append((future, byte_cost))
        self.pending_bytes += byte_cost
        return future

    def flush(self):
        while self.pending:
            self._finish()

    def close(self):
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)


def save_condition(destination, condition, chain, *, fixed_behavior=False):
    """Every edit endpoint points to its real incoming complete factors."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'record.json').exists():
        raise ValueError('complete actual condition record already exists')
    started = time.monotonic()
    events = []
    for ordinal, endpoint in enumerate(chain.endpoints):
        if fixed_behavior or ordinal == 0:
            incoming = 'MT'
        else:
            incoming = f'incoming_{ordinal:03d}.safetensors'
            save_file({k: v.contiguous() for k, v in chain.states[ordinal].items()}, str(destination / incoming))
        behavior = chain.behavior_versions[ordinal]
        events.append(dict(endpoint=endpoint, incoming=incoming, behavior_version=behavior,
                           episode=chain.records[endpoint - 1]['episode']))
    # The initial MT is a read-only reference, never a duplicate model asset.
    if not fixed_behavior:
        save_file({k: v.contiguous() for k, v in chain.states[-1].items()}, str(destination / 'end.safetensors'))
    torch.save(chain.to_record(), destination / 'experience.pt')
    record = dict(**condition, schema_version=EVENT_SCHEMA, record_path=str(destination.resolve()),
        events=events, MT_reference=str(MT_PATH), behavior_actor_uses_teaching=not fixed_behavior,
        actual_incoming_parameters=True, metrics=chain.metrics, complete=True,
        serialization_seconds=time.monotonic() - started)
    write_json_atomic(destination / 'record.json', record)
    sizes = [path.stat() for path in destination.iterdir() if path.is_file()]
    # A save event provides actual growth/health evidence without polling the
    # queue, model or shared cache during a normal long run.
    print('EMBER_ARTIFACT ' + json.dumps(dict(condition_id=condition['condition_id'],
        path=str(destination.resolve()), bytes=sum(s.st_size for s in sizes),
        allocated_bytes=sum(s.st_blocks * 512 for s in sizes),
        environment_steps=chain.metrics.get('environment_steps'), recorded_unix=time.time())), flush=True)
    return record


def load_event(runtime, event, *, loaded=None):
    """Native H remains original; only missing frozen Phi is reconstructed."""
    from .interaction import Chain
    destination = Path(event.record_path)
    record = read_json(destination / 'record.json')
    authority = next((e for e in record['events'] if e['endpoint'] == event.endpoint), None)
    if (authority is None or authority['incoming'] != event.incoming
            or authority['behavior_version'] != event.behavior_version
            or record['teacher_demo'] != event.teacher_demo or record['task_id'] != event.task_id):
        raise ValueError('event incoming/experience behavior authority changed')
    chain = Chain.from_record(loaded if loaded is not None else
                              torch.load(destination / 'experience.pt', map_location='cpu', weights_only=False))
    incoming = runtime.mt if event.incoming == 'MT' else load_file(str(destination / event.incoming), device=str(runtime.device))
    experience = {} if event.masked_experience else chain.experience(event.endpoint, runtime=runtime)
    return incoming, experience, record
