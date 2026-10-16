"""One immutable source-W/identity asset; no reference controller is an input."""
from pathlib import Path
import json,math,os
import torch
from safetensors import safe_open

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_eval_contract import FROZEN_SOURCE_POLICY_SUBDIR
from ember.reference.contract import ASSETS,SOURCE,ROOT as REFERENCE_ROOT
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX

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


def source_identity(center,layout):
    if (layout['rank'],layout['targets'],layout['valid_coordinates'],layout['padding_coordinates'])!=(8,38,643584,512):
        raise ValueError('source identity must retain the original full38/r8 layout')
    state={}
    for name,side,width,blocks,start,stop in layout['parts']:
        value=center[start:stop].reshape(8,blocks*64)[:,:width]
        state[name]=(value if side==0 else value.T).clone().contiguous()
    return state


def source_metadata():
    checkpoint=ASSETS/SOURCE;trainer=read_json(checkpoint/'trainer_state.json');manifest=read_json(checkpoint/'checkpoint_manifest.json')
    if trainer['optimizer_step']!=1000 or manifest['optimizer_step']!=1000:raise ValueError('only canonical raw source1000 is legal')
    relative=f'{FROZEN_SOURCE_POLICY_SUBDIR}/model.safetensors';file=checkpoint/relative
    entry=next(r for r in manifest['files'] if r['path']==relative)
    if not file.is_file() or file.stat().st_size!=entry['bytes']:raise ValueError('canonical source weight file missing or changed size')
    return dict(source_checkpoint=str(checkpoint),source_weights=dict(path=str(file),bytes=entry['bytes']),
        source_center_checkpoint=str(CENTER_REFERENCE),units_path=str(UNITS_FILE),source_frozen=True,MT_loaded=False,
        rank=8,alpha=8,dropout=0.,targets=38,valid_coordinates=643584,old_U_or_code_units_used=False)


def _target_key(handle,target):
    candidates=(target.name+'.weight',target.name.removeprefix('model.')+'.weight')
    matches=[key for key in candidates if key in handle.keys()]
    if len(matches)!=1:raise ValueError('actual source target needs one native weight key')
    key=matches[0]
    if tuple(handle.get_slice(key).get_shape())!=(target.out_features,target.in_features):raise ValueError('actual source linear weight shape changed')
    return key


def _materialize(metadata):
    lora=topology();units=read_json(UNITS_FILE)
    saved=torch.load(CENTER_REFERENCE,map_location='cpu',weights_only=False,mmap=True)
    center=saved['center'];layout=saved['layout'];identity=source_identity(center,layout);del saved
    names={t.name+s for t in lora.targets for s in (LORA_A_SUFFIX,LORA_B_SUFFIX)}
    if set(identity)!=names or set(units)!=names:raise ValueError('all76 exact source factors and shared units are required')
    targets=[]
    with safe_open(metadata['source_weights']['path'],framework='pt',device='cpu') as handle:
        for target in lora.targets:
            key=_target_key(handle,target);W=handle.get_tensor(key).float().contiguous();A0=identity[target.name+LORA_A_SUFFIX]
            B0=identity[target.name+LORA_B_SUFFIX]
            if (A0.shape!=(8,target.in_features) or B0.shape!=(target.out_features,8) or B0.count_nonzero()
                    or not bool(W.isfinite().all()) or not bool(A0.isfinite().all())):raise ValueError('invalid source W/A0/B0 identity')
            scales=[float(units[target.name+s]) for s in (LORA_A_SUFFIX,LORA_B_SUFFIX)]
            if any(not math.isfinite(v) or v<=0 for v in scales):raise ValueError('shared factor units must be finite positive')
            targets.append(dict(name=target.name,kind=module_kind(target.name),in_features=target.in_features,out_features=target.out_features,
                W=W,A0=A0,rho=max(float(W.square().mean().sqrt()),1e-6),rho_A=max(float(A0.square().mean().sqrt()),1e-6),
                S_A=scales[0],S_B=scales[1],source_key=key))
    metadata=dict(metadata,layout=layout,target_keys={t['name']:t['source_key'] for t in targets},
        tensor_inputs=[{k:t[k] for k in ('name','kind','in_features','out_features','source_key','rho','rho_A','S_A','S_B')} for t in targets])
    return dict(schema=SCHEMA,rank=8,alpha=8,dropout=0.,targets=targets,identity=identity,metadata=metadata)


def load_source_inputs(root,device='cpu'):
    folder=Path(root)/'source_inputs';manifest=read_json(folder/'manifest.json')
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


def _adopt_partial(root):
    complete=[p for p in root.glob('source_inputs.partial_*') if (p/'manifest.json').is_file() and (p/'state.pt').is_file()]
    if len(complete)>1:raise ValueError('multiple complete source-input transactions require reconciliation')
    if not complete:return False
    manifest=read_json(complete[0]/'manifest.json')
    if manifest['schema']!=SCHEMA or not manifest['complete']:raise ValueError('source-input partial transaction is invalid')
    complete[0].rename(root/'source_inputs');return True


def prepare_source_inputs(root):
    """Formal owner calls once from frozen code; completed assets are only adopted."""
    root=Path(root);folder=root/'source_inputs'
    if folder.exists() or _adopt_partial(root):return load_source_inputs(root)
    metadata=source_metadata();inputs=_materialize(metadata)
    inputs['metadata']=json.loads(json.dumps(inputs['metadata']))
    temporary=folder.with_name(folder.name+f'.partial_{os.getpid()}');temporary.mkdir(parents=True,exist_ok=False)
    torch.save(inputs,temporary/'state.pt')
    write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,complete=True,metadata=inputs['metadata'],
        actual_target_W_bytes=sum(t['W'].numel()*t['W'].element_size() for t in inputs['targets']),
        identity_factor_bytes=sum(v.numel()*v.element_size() for v in inputs['identity'].values())))
    temporary.rename(folder);return inputs
