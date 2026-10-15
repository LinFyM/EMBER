"""Frozen bank units and hardware-independent non-held query/point cursors."""
from collections import OrderedDict
from pathlib import Path
import re
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import default_collate
from ember.lora import identity_lora_state, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS,checkpoint
from ember.reference.training import QueryStream
from .contract import SCHEMA,TASKS,LIMITS,REFERENCE_ROOT


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def prepare(root):
    """Only a completed full bank can freeze labels and their 76 scalar units."""
    root=Path(root);previous=read_json(REFERENCE_ROOT/'completion.json')
    if not previous.get('reference_bank_ready') or previous.get('owner_failure'):
        raise ValueError('full 36 frozen references and independent reports are not ready')
    if read_json(REFERENCE_ROOT/'owner_exit.json')['exit_code']!=0:
        raise ValueError('reference owner has not exited successfully')
    bank=[freeze_task(task) for task in TASKS]
    root.mkdir(parents=True,exist_ok=True)
    value=dict(schema=SCHEMA,reference_root=str(REFERENCE_ROOT),tasks=bank)
    destination=root/'bank.json'
    if destination.exists() and read_json(destination)!=value:raise ValueError('frozen bank changed')
    write_json_atomic(destination,value)
    unit_path=root/'units.json'
    if not unit_path.exists():
        write_json_atomic(unit_path,compute_units(bank))
    contract=dict(schema=SCHEMA,limits=LIMITS,reference_root=str(REFERENCE_ROOT),bank=str(destination),
        environment=read_json(REFERENCE_ROOT/'run_contract.json')['environment'],seed=7,
        logical_batch=dict(tasks=4,queries=7,points=2),updates=144,checkpoint_nodes=[36,72,108,144])
    path=root/'run_contract.json'
    if path.exists() and read_json(path)!=contract:raise ValueError('immutable control-space run changed')
    write_json_atomic(path,contract)
    return contract


def freeze_task(task):
    selected=read_json(REFERENCE_ROOT/'selection'/f'task_{task:04d}'/'selected.json')
    if not selected['settled']:raise ValueError('unsettled teacher cannot enter E/D')
    points=[]
    for path in sorted((REFERENCE_ROOT/'training'/f'task_{task:04d}'/'checkpoints').glob('update_*/manifest.json')):
        if not re.fullmatch(r'update_\d{8}',path.parent.name):continue
        manifest=read_json(path)
        if manifest['complete']:points.append(dict(update=manifest['update'],path=str(path.parent/'lora.safetensors')))
    update=selected['selected_update'];chosen=None if update==0 else str(checkpoint(REFERENCE_ROOT,task,update)/'lora.safetensors')
    if chosen is not None and chosen not in {p['path'] for p in points}:raise ValueError('selected weights missing')
    for arm in ('source','selected'):
        for init in range(32,50):
            if not (REFERENCE_ROOT/'report'/f'task_{task:04d}'/arm/f'state_{init:03d}.json').is_file():
                raise ValueError('independent reference report is incomplete')
    return dict(task=task,selected_update=update,selected_path=chosen,checkpoints=points)


def compute_units(bank):
    lora=topology();identity=identity_lora_state(lora);sums={k:0. for k in identity}
    def accumulate(path,weight):
        factors=identity if path is None else load_file(path)
        validate_lora_state(factors,lora)
        for k in sums:sums[k]+=weight*float((factors[k]-identity[k]).double().square().mean())
    for task in bank:
        accumulate(task['selected_path'],.5/len(TASKS))
        other=[p for p in task['checkpoints'] if p['update']!=task['selected_update']]
        if other:
            for point in other:accumulate(point['path'],.5/len(TASKS)/len(other))
        else:accumulate(task['selected_path'],.5/len(TASKS))
    return {k:max(v**.5,1e-6) for k,v in sums.items()}


def logical_update(bank,update):
    if not 0<=update<144:raise ValueError('outside the single 144-update window')
    cycle,slot=divmod(update,9)
    order=np.random.default_rng(np.random.SeedSequence([7,1010,cycle])).permutation(TASKS)
    groups=[];by_task={r['task']:r for r in bank['tasks']}
    for task in map(int,order[slot*4:slot*4+4]):
        item=by_task[task];others=[p for p in item['checkpoints'] if p['update']!=item['selected_update']]
        if others:
            turn,index=divmod(cycle,len(others))
            perm=np.random.default_rng(np.random.SeedSequence([7,1011,task,turn])).permutation(len(others))
            other=others[int(perm[index])]
        else:other=dict(update=item['selected_update'],path=item['selected_path'])
        for point,path,node in [(0,item['selected_path'],item['selected_update']),(1,other['path'],other['update'])]:
            groups.append(dict(task=task,visit=cycle,point=point,path=path,reference_update=node))
    return groups


class QueryBank:
    """At most four active task streams; same x/noise for the paired references."""
    def __init__(self,runtime):self.runtime=runtime;self.streams=OrderedDict()

    def stream(self,task):
        if task not in self.streams:
            if len(self.streams)==4:self.streams.popitem(last=False)[1].data.close()
            self.streams[task]=QueryStream(self.runtime,task)
        self.streams.move_to_end(task);return self.streams[task]

    def batch(self,entries):
        rows=[];queries=[];noise=[]
        for entry in entries:
            task,visit,q=entry['task'],entry['visit'],entry['query']
            stream=self.stream(task);address=7*visit+q;item=stream.query(address)
            rows.append(stream.data[stream.rows[item['demo']][item['frame']]])
            seed=int(np.random.SeedSequence([7,1012,task,visit,q]).generate_state(1,dtype=np.uint64)[0])%(2**63-1)
            noise.append(torch.randn(50,32,generator=torch.Generator().manual_seed(seed)))
            queries.append(dict(task=task,visit=visit,query=q,ordinal=address,noise_seed=seed,**item))
        return self.runtime.processor.training_batch(default_collate(rows)),torch.stack(noise).to(self.runtime.device),queries

    def close(self):
        for stream in self.streams.values():stream.data.close()
        self.streams.clear()
