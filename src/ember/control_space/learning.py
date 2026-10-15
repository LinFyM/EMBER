"""Full frozen-bank operator credit; no source-policy or query consumer."""
from collections import OrderedDict
import torch
from safetensors.torch import load_file
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX
from ember.proposal_writer.learning import operator_squared_error


class References:
    """Bounded immutable factor cache shared by training and full-pool reads."""
    def __init__(self,runtime,names):self.runtime,self.names=runtime,names;self.cache=OrderedDict()

    def get(self,path):
        if path not in self.cache:
            if len(self.cache)==271:self.cache.popitem(last=False)
            raw=self.runtime.identity if path is None else load_file(path,device=str(self.runtime.device))
            self.cache[path]={k:raw[k].detach().float() for k in self.names}
        self.cache.move_to_end(path);return self.cache[path]


class OperatorLoss:
    def __init__(self,lora,constants):
        self.layers=[(t.name,lora.alpha/lora.rank,constants[t.name]) for t in lora.targets]

    def __call__(self,generated,reference):
        errors=[];baseline=[]
        for name,scale,unit in self.layers:
            A,B,E_A,E_B=[torch.stack([r[name+suffix] for r in rows]) for rows,suffix in
                [(generated,LORA_A_SUFFIX),(generated,LORA_B_SUFFIX),(reference,LORA_A_SUFFIX),(reference,LORA_B_SUFFIX)]]
            error,energy=operator_squared_error(A,B,E_A,E_B,scale)
            errors.append(error/unit);baseline.append(energy/unit)
        return torch.stack(errors,-1),torch.stack(baseline,-1)


def local_credit(model,references,objective,groups,pack):
    total=0.;identities=0
    for start in range(0,len(groups),pack):
        entries=groups[start:start+pack]
        learned=[r for r in entries if r['path'] is not None]
        identities+=len(entries)-len(learned)
        if not learned:continue  # Its analytic zero retains its original /8 half weight.
        states=[references.get(r['path']) for r in learned]
        errors,_=objective(model.forward_pack(states),states)
        loss=errors.mean(-1).sum()/8
        loss.backward();total+=float(loss.detach())
    return total,identities


def pool_points(bank):
    """271 distinct points; selected/others weights retain source identity mass."""
    rows=[]
    for item in bank['tasks']:
        others=[p for p in item['checkpoints'] if p['update']!=item['selected_update']]
        rows.append(dict(task=item['task'],point=0,path=item['selected_path'],update=item['selected_update'],
                         weight=.5/36 if others else 1./36))
        for p in others:
            rows.append(dict(task=item['task'],point=1,path=p['path'],update=p['update'],weight=.5/36/len(others)))
    if len(rows)!=271:raise ValueError('fixed bank must contain all 271 distinct parameter points')
    return rows


@torch.no_grad()
def evaluate_pool(model,references,objective,bank,context,pack):
    from .checkpoint import gather
    rows=[];points=pool_points(bank)
    local=points[len(points)*context.rank//context.world_size:len(points)*(context.rank+1)//context.world_size]
    for start in range(0,len(local),pack):
        entries=local[start:start+pack]
        learned=[r for r in entries if r['path'] is not None]
        values={}
        if learned:
            states=[references.get(r['path']) for r in learned]
            error,baseline=objective(model.forward_pack(states),states)
            for p,e,b in zip(learned,error.cpu().tolist(),baseline.cpu().tolist(),strict=True):values[p['path']]=(e,b)
        for p in entries:
            error,baseline=values[p['path']] if p['path'] is not None else ([0.]*38,[0.]*38)
            rows.append(dict(p,error_by_layer=error,source_by_layer=baseline,error=sum(error)/38,source=sum(baseline)/38))
    all_rows=[r for part in gather(rows,context) for r in part]
    return dict(rows=all_rows,weighted_error=sum(r['weight']*r['error'] for r in all_rows),
        source_baseline=sum(r['weight']*r['source'] for r in all_rows),points=len(all_rows),
        selected=sum(r['weight']*r['error'] for r in all_rows if r['point']==0)*2,
        other=sum(r['weight']*r['error'] for r in all_rows if r['point']==1)*2)
