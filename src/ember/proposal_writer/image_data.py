"""One shared immutable pure-visual cache and the registered 36-task query flow."""
from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from ember.pi05_eval_queue import complete_job, publish_json_exclusive
from ember.pi05_source_checkpoint import read_json
from .contract import TASKS
from .image_compiler import FEATURE_SCHEMA
from .image_history import history_file, model_history


def publish_record(queue, claim, worker, file, rows, *, successes=0):
    relative = Path('shards') / queue.stem / claim.shard.job_id / f'attempt_{claim.attempt:03d}.json'
    size = publish_json_exclusive(queue.parent/relative, dict(record_file=str(file), row_count=rows))
    complete_job(queue, job_id=claim.shard.job_id, worker_id=worker, claim_token=claim.claim_token,
        rows_path=str(relative), rows_bytes=size, row_count=rows, successes=successes)


def feature_file(root, task, demo):
    return Path(root)/'image_features'/f'task_{task:04d}_demo_{demo:02d}.pt'


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
            if (not value['complete'] or value['schema_version'] != FEATURE_SCHEMA or value['source']['L_in_Gemma']
                    or value['source']['action_suffix'] or value['source']['old_meta_loaded']
                    or value['z_img'].shape != (row['frames'],512,2048)
                    or not torch.isfinite(value['z_img']).all()):
                raise ValueError('actual pure frozen input cache changed')
            self.host[key] = value
        self.histories = {e['task']:model_history(torch.load(history_file(root,e['task']),
            map_location='cpu',weights_only=False)) for e in read_json(self.root/'history_manifest.json')['training']}
        free = torch.cuda.mem_get_info(self.device)[0]
        total = sum(v['z_img'].nbytes+v['language'].nbytes for v in self.host.values())
        total += sum(t.nbytes for h in self.histories.values() for t in h.values())
        # Whole unique frozen pool; H is shared across its four teaching videos.
        self.limit = min(total, max(0,free-6*1024**3))

    def get(self, task, demo):
        video = self._resident(('video',task,demo),{name:self.host[task,demo][name] for name in ('z_img','indices','language')})
        history = self._resident(('MT_H1',task),self.histories[task])
        return dict(**video,history=history)

    def _resident(self, key, source):
        if key in self.gpu:
            self.hits += 1; self.gpu.move_to_end(key)
            return self.gpu[key]
        started = time.monotonic()
        value = {name:t.to(self.device) for name,t in source.items()}
        size = sum(v.nbytes for v in value.values())
        while self.gpu and self.bytes+size > self.limit:
            _, old = self.gpu.popitem(last=False); self.bytes -= sum(v.nbytes for v in old.values())
        if size <= self.limit:
            self.gpu[key] = value; self.bytes += size
        self.loads += 1; self.transfer_seconds += time.monotonic()-started
        return value

    def assignment(self, rows, world):
        loads, result = [0]*world, [[] for _ in range(world)]
        def cost(row):
            h = self.histories[row['task']]
            return len(self.host[row['task'],row['demo']]['indices']) + len(h['images']) + len(h['facts'])/8
        for row in sorted(rows, key=cost, reverse=True):
            rank = min(range(world),key=loads.__getitem__)
            result[rank].append(row); loads[rank] += cost(row)
        return result
