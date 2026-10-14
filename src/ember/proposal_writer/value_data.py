"""Frozen real-path inputs and bounded missing native responses for independent Phi."""
from collections import OrderedDict
from pathlib import Path
import fcntl
import time

import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import claim_next, complete_job, fail_job, publish_json_exclusive
from .learning import HistoryCache
from .path import restore_compilation, history_prefix


def publish_value_record(queue, claim, worker_id, file, row_count, *, successes=0):
    """The canonical queue requires an immutable shards/ receipt, scoped by role."""
    relative = Path('shards')/queue.stem/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
    size = publish_json_exclusive(queue.parent/relative, dict(record_file=str(file),
        record_bytes=file.stat().st_size, row_count=row_count))
    complete_job(queue, job_id=claim.shard.job_id, worker_id=worker_id, claim_token=claim.claim_token,
        rows_path=str(relative), rows_bytes=size, row_count=row_count, successes=successes)


def prefix_inputs(runtime, path, prefix, responses):
    """No labels or later episodes are arguments to the learned input computation."""
    candidates = prefix['candidates']
    history = history_prefix(path.history, prefix['records'], prefix['episodes'])
    if len(history.episodes) != prefix['episodes'] or not candidates:
        raise ValueError('value input requires the original completed real prefix')
    refs = {'MT300', *candidates, *(r['parameter_ref'] for r in history.records)}
    counts = {key: [0, 0] for key in candidates}
    for episode in history.episodes:
        key = episode['actual_parameter_ref']
        if key in counts:
            counts[key][0 if episode['success'] else 1] += 1
    rows = history.model_rows(runtime)
    path.history.image_features.update(history.image_features)
    return rows, {key: path.states[key] for key in refs}, responses, counts


class ValueInputs:
    """Only frozen source features are cached; Phi encodings are never cached."""
    def __init__(self, runtime):
        self.runtime, self.root = runtime, runtime.root
        manifest = read_json(self.root/'input_manifest.json')
        self.entries = {entry['id']: entry for entry in manifest['paths']}
        self.tasks = {(e['kind'], e['task']): e['id'] for e in manifest['paths']}
        self.cache = OrderedDict()

    def path(self, entry):
        key = entry['id']
        if key not in self.cache:
            self.cache[key] = restore_compilation(entry['path'], self.runtime)
        self.cache.move_to_end(key)
        while len(self.cache) > 2 or sum(HistoryCache.nbytes(vars(p.history))
                                      for p in self.cache.values()) > 8*1024**3:
            if len(self.cache) == 1:
                break
            self.cache.popitem(last=False)
        path = self.cache[key]
        if (path.task, path.demo, path.identity) != (entry['task'], entry['demo'], entry['identity']):
            raise ValueError('real path input identity changed')
        return path

    def example(self, task, update):
        kind = 'pair' if (update//18) % 2 == 0 else 'RL'
        entry = self.entries[self.tasks[kind, task]]
        path = self.path(entry)
        frozen = read_json(self.root/'frozen_native'/f"{entry['id']}.json")
        if not frozen['complete'] or frozen['path_source'] != entry['path']:
            raise ValueError('training requires its complete original frozen response inputs')
        labels = read_json(Path(entry['path'])/'query/complete.json')['rewards']
        return entry, path, frozen['prefixes'], labels

def reserve_native(root, identity, prefix, queries):
    """Failed calls retain their conservative reservation; no silent extra query budget."""
    ledger = root/'native_input_ledger.json'
    with (root/'native_input_ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = read_json(ledger) if ledger.exists() else dict(entries=[], upper_bound=0)
        key = f'{identity}:{prefix}'
        if any(row['key'] == key for row in record['entries']):
            raise ValueError('incomplete native input cannot be replayed without budget adjudication')
        if record['upper_bound'] + queries > 5536:
            raise ValueError('native input query upper bound would exceed5536')
        record['entries'].append(dict(key=key, queries=queries, complete=False))
        record['upper_bound'] += queries
        write_json_atomic(ledger, record)


def finish_native(root, identity, prefix, actual):
    with (root/'native_input_ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = read_json(root/'native_input_ledger.json')
        row = next(r for r in record['entries'] if r['key'] == f'{identity}:{prefix}')
        if actual != row['queries']:
            raise ValueError('native response consumer differs from its original support cardinality')
        row.update(complete=True, actual_queries=actual)
        write_json_atomic(root/'native_input_ledger.json', record)


def fill_native_prefix(runtime, path, entry, index, original):
    prefix = dict(original)
    saved = dict(prefix.get('native_response_predictions', {}))
    missing = [key for key in prefix['candidates'] if key not in saved]
    actual = 0
    if missing:
        history = history_prefix(path.history, prefix['records'], prefix['episodes'])
        support = min(16, len(history.records))
        reserve_native(runtime.root, entry['id'], index, support*len(missing))
        before = runtime.components['response_observations']
        output = runtime.responses({key: path.states[key] for key in missing}, history,
            runtime.tasks[path.task].authority.language)
        saved.update({key: value.detach().cpu().tolist() for key, value in output.items()})
        actual = runtime.components['response_observations'] - before
    if set(saved) != set(prefix['candidates']):
        raise ValueError('current candidates require exactly their own frozen native responses')
    for value in saved.values():
        tensor = torch.as_tensor(value)
        if tensor.shape != (min(16, prefix['records']), 70) or not tensor.isfinite().all():
            raise ValueError('response must be original native10 first5x7 plus zero final-relative35')
        if tensor[:, 35:].count_nonzero():
            raise ValueError('final-use responses cannot contain a practice parent-relative delta')
    return dict(prefix, native_response_predictions=saved, new_queries=actual)


def prepare_native(args, runtime):
    """Persistent long-first queue fills only missing final-use responses, once."""
    from .value_learning import initialize_value
    initialize_value(runtime, with_phi=False)
    inputs = ValueInputs(runtime)
    queue = args.root/'queues/native.sqlite'
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        entry = inputs.entries[claim.shard.job_id]
        destination = args.root/'frozen_native'/f"{entry['id']}.json"
        try:
            started = time.monotonic()
            previous = read_json(destination) if destination.exists() else None
            if previous is None or not previous['complete']:
                if previous is not None and previous['path_source'] != entry['path']:
                    raise ValueError('partial native inputs belong to another real path')
                path = inputs.path(entry)
                prefixes = [] if previous is None else previous['prefixes']
                for index, original in enumerate(entry['prefixes']):
                    if index < len(prefixes):
                        saved = prefixes[index]
                        if any(saved[key] != original[key] for key in ('records','episodes','candidates')):
                            raise ValueError('saved native prefix identity changed')
                        if saved['new_queries']:
                            finish_native(args.root, entry['id'], index, saved['new_queries'])
                        continue
                    completed = fill_native_prefix(runtime,path,entry,index,original)
                    prefixes.append(completed)
                    write_json_atomic(destination, dict(complete=False,path_source=entry['path'],prefixes=prefixes))
                    if completed['new_queries']:
                        finish_native(args.root,entry['id'],index,completed['new_queries'])
                write_json_atomic(destination, dict(complete=True, path_source=entry['path'], prefixes=prefixes,
                    seconds=time.monotonic()-started, new_queries=sum(p['new_queries'] for p in prefixes),
                    no_environment=True, query_labels_read=False))
            publish_value_record(queue,claim,args.worker_id,destination,len(entry['prefixes']))
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id,
                claim_token=claim.claim_token, error=repr(error))
            raise
