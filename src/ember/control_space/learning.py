"""Selected36 operator supervision and native full-factor cotangent credit."""
from collections import OrderedDict
from pathlib import Path
import math
import torch
from torch.utils.data import default_collate
from safetensors.torch import load_file
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX
from ember.proposal_writer.learning import operator_squared_error
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset
from ember.writer.function_credit import FlowSample,NativeFlowPrediction,flow_sample,mean_velocity_loss
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.runtime import autocast
from .contract import OPERATOR_FILE


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

    def pair(self,generated,reference):
        if not generated or len(generated)!=len(reference):raise ValueError('pair loss requires complete paired states')
        errors_A=[];errors_B=[]
        for name,scale,unit in self.layers:
            A,B,E_A,E_B=[torch.stack([r[name+suffix] for r in rows]) for rows,suffix in
                [(generated,LORA_A_SUFFIX),(generated,LORA_B_SUFFIX),(reference,LORA_A_SUFFIX),(reference,LORA_B_SUFFIX)]]
            E_A,E_B=E_A.detach(),E_B.detach()
            error_A,_=operator_squared_error(A,E_B,E_A,E_B,scale)
            error_B,_=operator_squared_error(E_A,B,E_A,E_B,scale)
            errors_A.append(error_A/unit);errors_B.append(error_B/unit)
        return torch.stack(errors_A,-1),torch.stack(errors_B,-1)


def prepare_operator(root,inputs,lora):
    """Read the original fixed selected36 D; never recompute or copy labels."""
    record=read_json(OPERATOR_FILE)
    if record['sources']!=read_json(Path(root)/'run_contract.json')['references']:
        raise ValueError('original fixed operator labels changed')
    OperatorLoss(lora,record['denominators'])
    return record


class QueryBank:
    def __init__(self,runtime):self.runtime=runtime;self.datasets={}

    def sample(self,event,indices):
        task=event['task']
        if task not in self.datasets:
            self.datasets[task]=FunctionalQueryDataset((self.runtime.tasks[task].authority,),demo_indices=tuple(range(50)),
                action_chunk_size=50,action_start_offset=1,max_open_files_per_worker=1)
        data=self.datasets[task];rows=data.task_episode_rows[task]
        if any(demo==event['teacher_demo'] for demo in event['demos']):raise ValueError('FM episode must differ from current teacher')
        seed=task_logical_batch_policy_rng_seed(optimization_seed=event['query_seed'],task_id=task,task_visit=event['visit'],
            demo_indices=event['demos'],frame_indices=event['frames'])
        runs=[]
        for i in indices:
            if not runs or i!=runs[-1][-1]+1:runs.append([])
            runs[-1].append(i)
        samples=[]
        for run in runs:
            batch=default_collate([data[rows[event['demos'][i]][event['frames'][i]]] for i in run])
            samples.append(flow_sample(self.runtime.policy,self.runtime.processor.training_batch(batch),seed=seed,
                device=self.runtime.device,random_batch=28,offset=run[0]))
        return FlowSample(concatenate([s.arguments for s in samples]),torch.cat([s.target for s in samples]),7),seed

    def close(self):
        for dataset in self.datasets.values():dataset.close()


def sliced(value,indices):
    if isinstance(value,torch.Tensor):return value.index_select(0,indices)
    return type(value)(sliced(v,indices) for v in value)


def concatenate(values):
    if isinstance(values[0],torch.Tensor):return torch.cat(values)
    return type(values[0])(concatenate([v[i] for v in values]) for i in range(len(values[0])))


def assignments(world,rank):
    return [(t,[q for q in range(28) if ((t*4+q//7)*world)//16==rank]) for t in range(4)
        if any(((t*4+q//7)*world)//16==rank for q in range(28))]


def fm_credit(runtime,samples,states,teachers,objective,microbatch):
    """Native and BA cotangents meet on full A/B leaves, then enter generator once."""
    leaves=[{k:v.detach().requires_grad_(True) for k,v in state.items()} for state in states]
    sample=FlowSample(concatenate([s.arguments for s in samples]),torch.cat([s.target for s in samples]),7)
    counts=[len(s.target) for s in samples];which=torch.tensor([i for i,n in enumerate(counts) for _ in range(n)],device=runtime.device)
    owner=NativeFlowPrediction(runtime.policy);value=0.;batches=[]
    for first in range(0,len(which),microbatch):
        last=min(first+microbatch,len(which));index=torch.arange(first,last,device=runtime.device)
        part=FlowSample(sliced(sample.arguments,index),sample.target[first:last],7)
        with autocast(runtime.device),runtime.execution.activate(leaves,batch_indices=which[first:last]):
            loss=mean_velocity_loss(owner(part),part.target,7)*((last-first)/112)
        loss.backward();value+=float(loss.detach());batches.append(last-first)
    errors,_=objective(leaves,teachers)
    weights=errors.new_tensor(counts)/112
    BA=(errors.mean(-1)*weights).sum();(.1*BA).backward()
    outputs=[];cotangents=[]
    for state,leaf in zip(states,leaves,strict=True):
        for name in state:
            if leaf[name].grad is not None:outputs.append(state[name]);cotangents.append(leaf[name].grad)
    credit=sum(float(v.float().square().sum()) for v in cotangents)**.5
    torch.autograd.backward(outputs,cotangents)
    return dict(FM=value,BA=float(BA.detach()),LoRA_cotangent_norm=credit,physical_FM_batches=batches,query_rows=len(which))
