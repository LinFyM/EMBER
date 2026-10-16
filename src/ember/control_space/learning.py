"""Read-only factor sources and complete FP32 operator diagnostics after freezing."""
from collections import OrderedDict
import math
import torch
from safetensors.torch import load_file
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX
from ember.proposal_writer.learning import operator_squared_error


class References:
    """Bounded factor cache; None identifies the supplied exact saved source center."""
    def __init__(self,runtime,names,*,capacity=16):
        if type(capacity) is not int or capacity<1:raise ValueError('factor cache requires a positive capacity')
        self.runtime,self.names,self.capacity=runtime,tuple(names),capacity;self.cache=OrderedDict()

    def get(self,path):
        if path not in self.cache:
            raw=self.runtime.identity if path is None else load_file(str(path),device=str(self.runtime.device))
            if set(raw)!=set(self.names):raise ValueError('reference must contain the complete declared factors')
            value={name:raw[name].detach().float() for name in self.names}
            if any(not bool(v.isfinite().all()) for v in value.values()):raise ValueError('nonfinite reference factors')
            if len(self.cache)>=self.capacity:self.cache.popitem(last=False)
            self.cache[path]=value
        self.cache.move_to_end(path);return self.cache[path]


class OperatorLoss:
    """Full scaled BA residual via rank-sized Grams, with real FP32 gradients."""
    def __init__(self,lora,constants):
        names=[target.name for target in lora.targets]
        if set(constants)!=set(names) or any(not math.isfinite(float(constants[n])) or float(constants[n])<=0 for n in names):
            raise ValueError('operator diagnostics require all fixed positive layer constants')
        self.layers=[(name,lora.alpha/lora.rank,float(constants[name])) for name in names]

    def __call__(self,generated,reference):
        if not generated or len(generated)!=len(reference):raise ValueError('operator comparisons require paired complete states')
        errors=[];baseline=[]
        for name,scale,unit in self.layers:
            A,B,E_A,E_B=[torch.stack([r[name+suffix] for r in rows]) for rows,suffix in
                [(generated,LORA_A_SUFFIX),(generated,LORA_B_SUFFIX),(reference,LORA_A_SUFFIX),(reference,LORA_B_SUFFIX)]]
            error,energy=operator_squared_error(A,B,E_A,E_B,scale)
            errors.append(error/unit);baseline.append(energy/unit)
        return torch.stack(errors,-1),torch.stack(baseline,-1)


def pool_points(bank,expected_tasks,expected_points):
    """Explicit whole-task pool; source identities keep their original task mass."""
    tasks=tuple(expected_tasks)
    if not tasks or len(set(tasks))!=len(tasks):raise ValueError('pool tasks must be explicit and unique')
    selected=[item for item in bank['tasks'] if item['task'] in tasks]
    if {item['task'] for item in selected}!=set(tasks) or len(selected)!=len(tasks):raise ValueError('pool task coverage changed')
    rows=[]
    for item in selected:
        others=[p for p in item['checkpoints'] if p['update']!=item['selected_update']]
        rows.append(dict(task=item['task'],point=0,path=item['selected_path'],update=item['selected_update'],
            weight=.5/len(tasks) if others else 1./len(tasks)))
        rows.extend(dict(task=item['task'],point=1,path=p['path'],update=p['update'],weight=.5/len(tasks)/len(others)) for p in others)
    if len(rows)!=expected_points:raise ValueError('explicit whole-task pool point count changed')
    return rows
