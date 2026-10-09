"""Actual event provenance, asynchronous raw facts and a bounded frozen cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from pathlib import Path
import sqlite3
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import EVENT_SCHEMA, MT_PATH


class FeatureCache:
    """One disk budget shared by frozen teacher/native observation features.

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
            with sqlite3.connect(self.root / 'cache.sqlite3', timeout=60) as database:
                yield database

    def _path(self, key):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ' for c in key):
            raise ValueError('invalid frozen-cache observation identity')
        return self.root / (key + '.safetensors')

    def get(self, key):
        path = self._path(key)
        with self._locked() as database:
            if not database.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone():
                return None
            result = load_file(str(path))
            database.execute('UPDATE entries SET touched=? WHERE key=?', (time.time(), key))
            return result

    def put(self, key, values):
        size = sum(v.nbytes for v in values.values()) + 4096
        if size > self.max_bytes:
            return
        path = self._path(key)
        with self._locked() as database:
            if database.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone():
                return
            used = database.execute('SELECT COALESCE(SUM(bytes),0) FROM entries').fetchone()[0]
            for old, old_size in database.execute('SELECT key,bytes FROM entries ORDER BY touched').fetchall():
                if used + size <= self.max_bytes:
                    break
                self._path(old).unlink(missing_ok=True)
                database.execute('DELETE FROM entries WHERE key=?', (old,))
                used -= old_size
            partial = path.with_suffix('.partial')
            save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, str(partial))
            actual = partial.stat().st_size
            partial.replace(path)
            database.execute('INSERT INTO entries VALUES (?,?,?)', (key, actual, time.time()))


class RecordWriter:
    """CPU serialization overlaps other slots; failures propagate on flush."""
    def __init__(self, workers=2):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='compiler-record')
        self.pending = []

    def submit(self, function, *args, **kwargs):
        # Keep asynchronous temporary memory bounded instead of buffering a run.
        if len(self.pending) >= 16:
            self.pending.pop(0).result()
        future = self.executor.submit(function, *args, **kwargs)
        self.pending.append(future)
        return future

    def flush(self):
        for future in self.pending:
            future.result()
        self.pending.clear()

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
    return record


def complete_pool(output, contract):
    from ember.pi05_eval_queue import completed_jobs
    paths = {job['job_id']: read_json(Path(output) / job['rows_path'])['record_path']
             for job in completed_jobs(Path(output) / 'queue.sqlite3')}
    conditions = []
    for condition in contract['conditions']:
        record = read_json(Path(paths[condition['condition_id']]) / 'record.json')
        if not record.get('complete') or not record.get('actual_incoming_parameters'):
            raise ValueError('collection lost actual incoming provenance')
        conditions.append(record)
    manifest = dict(schema_version=EVENT_SCHEMA, pool=contract['pool'], version=contract['version'],
                    conditions=conditions, complete=True, collection_contract=str(Path(output) / 'collection_contract.json'))
    write_json_atomic(Path(output) / 'manifest.json', manifest)
    return manifest


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
