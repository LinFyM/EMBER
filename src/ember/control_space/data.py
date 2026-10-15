"""Immutable authorized bank/units and the sole phase's source-centered inputs."""
from pathlib import Path
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS
from .contract import SCHEMA,LIMITS,REFERENCE_ROOT,PARENT_ROOT,OPERATOR_ROOT


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not read_json(OPERATOR_ROOT/'completion.json')['complete']:raise ValueError('prior1170 not complete')
    for name,source in [('bank.json',PARENT_ROOT),('units.json',PARENT_ROOT),('operator_constants.json',OPERATOR_ROOT)]:
        value=read_json(source/name);path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('frozen bank/S/C changed')
        if not path.exists():write_json_atomic(path,value)
    contract=dict(schema=SCHEMA,phase='linear64',basis_checkpoint=str(root/'basis/state.pt'),code_dim=64,
        parent_root=str(PARENT_ROOT),operator_root=str(OPERATOR_ROOT),reference_root=str(REFERENCE_ROOT),
        bank=str(root/'bank.json'),limits=LIMITS,seed=7,environment=read_json(PARENT_ROOT/'run_contract.json')['environment'],
        fit=dict(method='weighted_uncentered_small_Gram_FP64',mean_subtracted=False,
                 task_weight='equal36',selected_weight=.5,other_total_weight=.5,optimizer_updates=0))
    path=root/'run_contract.json'
    if path.exists() and read_json(path)!=contract:raise ValueError('immutable linear64 contract changed')
    if not path.exists():write_json_atomic(path,contract)
    return contract
