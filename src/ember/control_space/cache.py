"""One bounded immutable-M cache; SQLite and each payload own the same identity.

The fixed-inode index and payload binding retain the §17 cross-node repair.
Rollback journals, reservations and partial files share the registered capacity.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import sqlite3
import time

import torch

SCHEMA = 'ember_source_weight_memory_cache_v1'


@contextmanager
def _lock(path):
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


class _BoundedWriter:
    def __init__(self, stream, limit): self.stream, self.limit = stream, limit
    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise BufferError('M payload exceeded its reservation')
        return self.stream.write(data)
    def flush(self): self.stream.flush()


class MemoryCache:
    """Global LRU, concurrent per-key builds, no manifest-only tensor adoption."""
    def __init__(self, root, capacity=2**30):
        self.root, self.capacity = Path(root), int(capacity)
        if not 256*1024 <= self.capacity <= 2**30:
            raise ValueError('all M cache files must share at most 1GiB')
        self.root.mkdir(parents=True, exist_ok=True)
        self.control = min(8*2**20, max(128*1024, self.capacity//32))
        self.database = self.root/'index.sqlite'
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS tensors (id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT UNIQUE NOT NULL,metadata TEXT NOT NULL,status TEXT NOT NULL,bytes INTEGER NOT NULL,touched REAL NOT NULL)')
            for key, value in (('schema', SCHEMA), ('capacity', str(self.capacity))):
                db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', (key, value))
                if db.execute('SELECT value FROM settings WHERE name=?', (key,)).fetchone()[0] != value:
                    raise ValueError('M cache schema/capacity differs from this owner')
            if not db.execute('SELECT count(*) FROM tensors').fetchone()[0] and next(self.root.glob('entry_*.pt*'), None):
                raise RuntimeError('unindexed M payloads require explicit forensic cleanup')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.database, timeout=60, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=60000'); db.execute('PRAGMA temp_store=MEMORY')
        db.execute('PRAGMA synchronous=FULL')
        try:
            if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
                raise ValueError('shared M cache requires DELETE rollback journal, never WAL')
            db.execute('BEGIN IMMEDIATE'); yield db
            size = db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
            if 3*size > self.control:
                raise RuntimeError('M database/journal exceeds reserved control bytes')
            db.execute('COMMIT')
        except BaseException:
            if db.in_transaction: db.execute('ROLLBACK')
            raise
        finally: db.close()

    def _entry(self, key, metadata):
        encoded = json.dumps(metadata, sort_keys=True)
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO tensors(key,metadata,status,bytes,touched) VALUES (?,?,'absent',0,?)", (key, encoded, time.time()))
            entry = db.execute('SELECT * FROM tensors WHERE key=?', (key,)).fetchone()
            if entry['metadata'] != encoded:
                raise ValueError('requested M key changed immutable video/source/Reader identity')
            return f'entry_{entry["id"]:010d}.pt'

    def _reserve(self, db, key, needed):
        used = db.execute('SELECT coalesce(sum(bytes),0) FROM tensors').fetchone()[0]
        victims = db.execute("SELECT id,key,bytes FROM tensors WHERE key!=? AND status='ready' ORDER BY touched", (key,)).fetchall()
        for old in victims:
            if used + needed <= self.capacity-self.control: break
            (self.root/f'entry_{old["id"]:010d}.pt').unlink(missing_ok=True)
            db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?", (old['key'],))
            used -= old['bytes']
        available = used + needed <= self.capacity-self.control
        db.execute('UPDATE tensors SET status=?,bytes=?,touched=? WHERE key=?',
            ('reserved' if available else 'absent', needed if available else 0, time.time(), key))
        return available

    @staticmethod
    def _load(path, key, metadata):
        with path.open('rb') as handle:
            payload = torch.load(handle, map_location='cpu', weights_only=True)
        if (not isinstance(payload, dict) or payload.get('schema') != SCHEMA
                or payload.get('key') != key or payload.get('metadata') != metadata
                or not isinstance(payload.get('value'), torch.Tensor)):
            raise ValueError('M payload identity mismatch; preserve evidence and stop consumption')
        return payload['value']

    def _release(self, key, destination, temporary):
        with self.transaction() as db:
            temporary.unlink(missing_ok=True); destination.unlink(missing_ok=True)
            db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?", (key,))

    def get(self, key, metadata, required_bytes, build):
        tick = time.monotonic(); file = self._entry(key, metadata)
        destination, temporary = self.root/file, self.root/(file+'.partial')
        with _lock(self.root/(file+'.lock')):
            with self.transaction() as db:
                entry = db.execute('SELECT * FROM tensors WHERE key=?', (key,)).fetchone()
                if entry['status'] in ('ready', 'reserved') and destination.exists():
                    value = self._load(destination, key, metadata); size = destination.stat().st_size
                    if size > entry['bytes']: raise RuntimeError('M exceeds its indexed reservation')
                    db.execute("UPDATE tensors SET status='ready',bytes=?,touched=? WHERE key=?", (size, time.time(), key))
                    return value, dict(hit=True, cached=True, bytes=size, seconds=time.monotonic()-tick)
                temporary.unlink(missing_ok=True)
                db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?", (key,))
            needed = int(required_bytes)+65536+len(key.encode())+len(json.dumps(metadata).encode())
            with self.transaction() as db: cached = self._reserve(db, key, needed)
            try:
                value = build().detach().to(device='cpu', dtype=torch.float32).contiguous()
                if cached:
                    with temporary.open('wb') as handle:
                        torch.save(dict(schema=SCHEMA, key=key, metadata=metadata, value=value), _BoundedWriter(handle, needed))
                        handle.flush(); os.fsync(handle.fileno())
                    with self.transaction() as db:
                        size = temporary.stat().st_size; temporary.replace(destination)
                        db.execute("UPDATE tensors SET status='ready',bytes=?,touched=? WHERE key=?", (size, time.time(), key))
                return value, dict(hit=False, cached=cached, bytes=destination.stat().st_size if cached else value.numel()*4,
                    seconds=time.monotonic()-tick)
            except BufferError:
                self._release(key, destination, temporary)
                return value, dict(hit=False, cached=False, bytes=value.numel()*4, seconds=time.monotonic()-tick)
            except BaseException:
                self._release(key, destination, temporary); raise

    def usage(self):
        with self.transaction() as db:
            rows = [dict(row) for row in db.execute('SELECT status,count(*) entries,sum(bytes) bytes FROM tensors GROUP BY status')]
            return dict(capacity=self.capacity, control_allowance=self.control, rows=rows,
                charged_bytes=self.control+sum(row['bytes'] for row in rows))
