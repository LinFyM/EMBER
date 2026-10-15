"""Fixed operator units and the original hardware-independent task/point flow."""
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ember.lora import identity_lora_state,LORA_A_SUFFIX,LORA_B_SUFFIX
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS
from .contract import SCHEMA,TASKS,LIMITS,REFERENCE_ROOT,PARENT_ROOT,MAX_UPDATE


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def prepare(root):
    """Reuse the immutable parent bank/S; compute only the new 38 C_l once."""
    from .learning import pool_points
    from ember.proposal_writer.learning import operator_squared_error
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not read_json(PARENT_ROOT/'completion.json')['complete']:raise ValueError('parent144 not complete')
    for name in ('bank.json','units.json'):
        value=read_json(PARENT_ROOT/name);path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('parent bank/S changed')
        if not path.exists():write_json_atomic(path,value)
    path=root/'operator_constants.json'
    if not path.exists():
        lora=topology();identity=identity_lora_state(lora);energies={t.name:0. for t in lora.targets}
        for point in pool_points(read_json(root/'bank.json')):
            factors=identity if point['path'] is None else load_file(point['path'])
            for target in lora.targets:
                name=target.name;A=factors[name+LORA_A_SUFFIX].float();B=factors[name+LORA_B_SUFFIX].float()
                _,energy=operator_squared_error(A,B,A,B,lora.alpha/lora.rank)
                energies[name]+=point['weight']*float(energy)
        write_json_atomic(path,{k:max(v,1e-12) for k,v in energies.items()})
    contract=dict(schema=SCHEMA,phase='operator_from144',parent_root=str(PARENT_ROOT),parent_update=144,
        limits=LIMITS,reference_root=str(REFERENCE_ROOT),bank=str(root/'bank.json'),
        environment=read_json(PARENT_ROOT/'run_contract.json')['environment'],seed=7,
        logical_batch=dict(tasks=4,points=2),maximum_global_update=MAX_UPDATE,
        train_wall_seconds=dict(first_check=300,maximum=900),parent_query_cursor=112,phase_query_consumption=0)
    path=root/'run_contract.json'
    if path.exists():
        saved=read_json(path)
        if {k:v for k,v in saved.items() if k!='endpoint_update'}!=contract:raise ValueError('operator contract changed')
        return saved
    write_json_atomic(path,contract);return contract


def logical_update(bank,update):
    if not 0<=update<MAX_UPDATE:raise ValueError('outside registered logical task/point cursor')
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
