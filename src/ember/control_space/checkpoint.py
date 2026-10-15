"""A single atomic fitted basis; no optimizer or fictitious update cursor."""
from pathlib import Path
import os
import torch
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import SCHEMA


def save_basis(root,state):
    destination=Path(root)/'basis'
    if destination.exists():raise ValueError('completed basis must never be refitted or overwritten')
    temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}')
    temporary.mkdir(parents=True,exist_ok=False)
    torch.save(state,temporary/'state.pt')
    write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,complete=True,phase='linear64',
        code_dim=64,effective_rank=state['effective_rank'],optimizer_updates=0,method=state['method']))
    temporary.rename(destination)


def completed_basis(root):
    folder=Path(root)/'basis'
    if not folder.exists():
        complete=[p for p in Path(root).glob('basis.partial_*') if (p/'manifest.json').is_file() and (p/'state.pt').is_file()]
        if len(complete)>1:raise ValueError('more than one completed basis transaction requires reconciliation')
        if not complete:return False
        record=read_json(complete[0]/'manifest.json')
        if record['schema']!=SCHEMA or not record['complete']:raise ValueError('invalid partial basis transaction')
        complete[0].rename(folder)
    record=read_json(folder/'manifest.json')
    if record['schema']!=SCHEMA or not record['complete'] or not (folder/'state.pt').is_file():
        raise ValueError('basis transaction is not complete')
    return True
