"""One immutable source-W/identity asset; no reference controller is an input."""
from pathlib import Path
import torch

from ember.pi05_source_checkpoint import read_json
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.reference.contract import ASSETS,SOURCE,ROOT as REFERENCE_ROOT
from .contract import SOURCE_INPUTS

SCHEMA='ember_source_weight_inputs_r8_seed7_v1'
CENTER_REFERENCE=REFERENCE_ROOT.parent/'control_linear64_seed7/basis/state.pt'
UNITS_FILE=REFERENCE_ROOT.parent/'control_space_seed7/units.json'
KINDS=('q_proj','v_proj','action_in_proj','action_out_proj')


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def module_kind(name):
    kind=name.rsplit('.',1)[-1]
    if kind not in KINDS:raise ValueError('unregistered source linear module kind')
    return kind


def load_source_inputs(root,device='cpu'):
    folder=SOURCE_INPUTS.parent;manifest=read_json(folder/'manifest.json')
    if manifest['schema']!=SCHEMA or not manifest['complete']:raise ValueError('source input transaction is incomplete')
    inputs=torch.load(folder/'state.pt',map_location=device,weights_only=False)
    lora=topology()
    if (inputs['schema'],inputs['rank'],inputs['alpha'],inputs['dropout'])!=(SCHEMA,8,8,0.):raise ValueError('source input schema changed')
    expected=[(t.name,t.in_features,t.out_features) for t in lora.targets]
    actual=[(t['name'],t['in_features'],t['out_features']) for t in inputs['targets']]
    if actual!=expected or inputs['metadata']!=manifest['metadata']:raise ValueError('source input target identity changed')
    identities=dict(source_checkpoint=str(ASSETS/SOURCE),source_center_checkpoint=str(CENTER_REFERENCE),units_path=str(UNITS_FILE))
    if any(inputs['metadata'].get(k)!=v for k,v in identities.items()):raise ValueError('source inputs belong to another base/center/unit authority')
    return inputs
