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
def evaluate_pool(model,references,objective,bank,*,vectors=None):
    """One full-pool projection/BA read, reusing the fit's in-memory coordinates."""
    points=pool_points(bank)
    if vectors is None:vectors=torch.stack([model.normalized(references.get(p['path'])) for p in points])
    projected=model.decode_vectors(vectors@model.basis)
    source=vectors.double().square().sum(-1)
    residual=(projected-vectors).double().square().sum(-1)
    rows=[]
    for start in range(0,len(points),16):
        entries=points[start:start+16]
        states=[references.get(p['path']) for p in entries]
        generated=[model.from_vector(v) for v in projected[start:start+16]]
        errors,baseline=objective(generated,states)
        for i,(p,e,b) in enumerate(zip(entries,errors.tolist(),baseline.tolist(),strict=True),start):
            rows.append(dict(p,error_by_layer=e,source_by_layer=b,error=sum(e)/38,source=sum(b)/38,
                factor_squared_error=float(residual[i]),factor_source_squared_norm=float(source[i]),
                factor_relative_error=float(residual[i]/source[i]) if source[i] else 0.))
    weighted_source=sum(r['weight']*r['factor_source_squared_norm'] for r in rows)
    weighted_error=sum(r['weight']*r['factor_squared_error'] for r in rows)
    return dict(rows=rows,weighted_error=sum(r['weight']*r['error'] for r in rows),
        source_baseline=sum(r['weight']*r['source'] for r in rows),points=len(rows),
        selected=sum(r['weight']*r['error'] for r in rows if r['point']==0)*2,
        other=sum(r['weight']*r['error'] for r in rows if r['point']==1)*2,
        weighted_factor_squared_error=weighted_error,weighted_factor_source_squared_norm=weighted_source,
        weighted_factor_relative_error=weighted_error/weighted_source if weighted_source else 0.)
