"""One shared immutable pure-visual cache and the registered 36-task query flow."""
from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from ember.pi05_eval_queue import claim_next, complete_job, fail_job, publish_json_exclusive
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import TASKS
from .image_compiler import SCHEMA, pure_features
from .learning import memory_peak


def publish_record(queue, claim, worker, file, rows, *, successes=0):
    relative = Path('shards') / queue.stem / claim.shard.job_id / f'attempt_{claim.attempt:03d}.json'
    size = publish_json_exclusive(queue.parent/relative, dict(record_file=str(file), row_count=rows))
    complete_job(queue, job_id=claim.shard.job_id, worker_id=worker, claim_token=claim.claim_token,
        rows_path=str(relative), rows_bytes=size, row_count=rows, successes=successes)


def feature_file(root, task, demo):
    return Path(root)/'image_features'/f'task_{task:04d}_demo_{demo:02d}.pt'


def feature_cache(args, runtime):
    manifest = read_json(args.root/'input_manifest.json')
    queue = args.root/'queues/features.sqlite'
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        try:
            entries = [e for e in manifest['training'] if e['task'] == claim.shard.task_id
                       and e['teacher_demo'] in claim.shard.init_state_ids]
            records = []
            for entry in entries:
                file = feature_file(args.root, entry['task'], entry['teacher_demo'])
                if not file.exists():
                    value = pure_features(runtime, entry['task'], entry['teacher_demo'], frame_chunk=args.frame_chunk)
                    if len(value['indices']) != entry['frames']:
                        raise ValueError('registered complete RGB length changed')
                    value.update(complete=True, task=entry['task'], demo=entry['teacher_demo'], schema_version=SCHEMA)
                    file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = file.with_suffix(f'.{args.worker_id}.tmp')
                    torch.save(value, temporary); temporary.replace(file)
                records.append(dict(task=entry['task'], demo=entry['teacher_demo'], file=str(file), frames=entry['frames']))
            file = args.root/'feature_receipts'/f'{claim.shard.job_id}_{claim.attempt:03d}.json'
            write_json_atomic(file, dict(complete=True, records=records, peak=memory_peak(runtime.device),
                Gemma_language=False, action_suffix=False, old_meta=False))
            publish_record(queue, claim, args.worker_id, file, len(records))
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token, error=repr(error))
            raise


def logical_conditions(root, update, panel=None):
    panel = read_json(Path(root)/'panel.json') if panel is None else panel
    rows = {row['task_id']:row for row in panel['tasks']}
    round_index, slot = divmod(update, 9)
    cycle, within = divmod(round_index, 4)
    order = np.random.default_rng(np.random.SeedSequence([20261014,17,0,cycle,within])).permutation(TASKS).tolist()
    output = []
    for task in order[4*slot:4*slot+4]:
        demos = np.random.default_rng(np.random.SeedSequence([20261014,17,1,cycle,task])).permutation(rows[task]['policy_videos'])
        output.append(dict(task=task, demo=int(demos[within])))
    return output


class ImageInputs:
    """Whole frozen CPU pool, bounded GPU cache; learned memory is never retained."""
    def __init__(self, root, device):
        self.root, self.device = Path(root), torch.device(device)
        self.host, self.gpu = {}, OrderedDict()
        self.bytes, self.hits, self.loads = 0, 0, 0
        self.transfer_seconds = 0.
        for row in read_json(self.root/'input_manifest.json')['training']:
            key = row['task'], row['teacher_demo']
            value = torch.load(feature_file(root,*key), map_location='cpu', weights_only=False)
            if (not value['complete'] or value['schema_version'] != SCHEMA or value['source']['L_in_Gemma']
                    or value['source']['action_suffix'] or value['source']['old_meta_loaded']
                    or value['z_img'].shape != (row['frames'],512,2048)
                    or not torch.isfinite(value['z_img']).all()):
                raise ValueError('actual pure frozen input cache changed')
            self.host[key] = value
        free = torch.cuda.mem_get_info(self.device)[0]
        # This compiler has no resident policy: 3GiB activations/Adam plus 2GiB headroom.
        self.limit = min(sum(v['z_img'].nbytes+v['language'].nbytes for v in self.host.values()), max(0,free-5*1024**3))

    def get(self, task, demo):
        key = task, demo
        if key in self.gpu:
            self.hits += 1; self.gpu.move_to_end(key)
            return self.gpu[key]
        started = time.monotonic()
        value = {name:self.host[key][name].to(self.device) for name in ('z_img','indices','language')}
        size = sum(v.nbytes for v in value.values())
        while self.gpu and self.bytes+size > self.limit:
            _, old = self.gpu.popitem(last=False); self.bytes -= sum(v.nbytes for v in old.values())
        if size <= self.limit:
            self.gpu[key] = value; self.bytes += size
        self.loads += 1; self.transfer_seconds += time.monotonic()-started
        return value

    def assignment(self, rows, world):
        loads, result = [0]*world, [[] for _ in range(world)]
        for row in sorted(rows, key=lambda r:len(self.host[r['task'],r['demo']]['indices']), reverse=True):
            rank = min(range(world),key=loads.__getitem__)
            result[rank].append(row); loads[rank] += len(self.host[row['task'],row['demo']]['indices'])
        return result
