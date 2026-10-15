"""Exact native FM cotangents through the fixed D to the fresh image Reader."""
from pathlib import Path
import torch
from torch.utils.data import default_collate
from ember.writer.data import FunctionalQueryDataset
from ember.writer.function_credit import FlowSample,NativeFlowPrediction,flow_sample,mean_velocity_loss
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.runtime import autocast
from .features import load_cached


class QueryBank:
    def __init__(self,runtime):
        self.runtime=runtime;self.datasets={}

    def sample(self,event,indices):
        task=event['task']
        if task not in self.datasets:
            self.datasets[task]=FunctionalQueryDataset((self.runtime.tasks[task].authority,),demo_indices=tuple(range(50)),
                action_chunk_size=50,action_start_offset=1,max_open_files_per_worker=1)
        data=self.datasets[task];rows=data.task_episode_rows[task]
        seed=task_logical_batch_policy_rng_seed(optimization_seed=7,task_id=task,task_visit=event['visit'],
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


def selected_sample(samples,assignments,device):
    arguments=[];targets=[];which=[]
    for position,indices in assignments:
        index=torch.tensor(indices,device=device,dtype=torch.long);sample=samples[position]
        arguments.append(sliced(sample.arguments,index));targets.append(sample.target.index_select(0,index));which.extend([position]*len(indices))
    return FlowSample(concatenate(arguments),torch.cat(targets),7),torch.tensor(which,device=device,dtype=torch.long)


def assignments(world,rank):
    """Contiguous7-query blocks retain whole tasks where possible and one owner/query."""
    return [(t,[q for q in range(28) if ((t*4+q//7)*world)//16==rank]) for t in range(4)
        if any(((t*4+q//7)*world)//16==rank for q in range(28))]


def auxiliary_owner(position,world):return (position*4*world)//16


def gradient_norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def reader_output(reader,entry,runtime):
    value=load_cached(Path(entry['cache_path']),runtime.device)
    if Path(value['source']['model']).resolve()!=Path(runtime.source['model_path']).resolve():raise ValueError('frozen image source changed')
    with autocast(runtime.device):return reader(value['z_img'],value['indices'],checkpoint_layers=False).float()


def fm_credit(runtime,samples,states,owned,microbatch):
    """Detached leaves are an exact intermediate cotangent cache, never a lost edge."""
    leaves=[{k:v.detach().requires_grad_(True) for k,v in state.items()} for state in states]
    sample,which=selected_sample(samples,owned,runtime.device);owner=NativeFlowPrediction(runtime.policy);value=0.
    for first in range(0,len(which),microbatch):
        last=min(first+microbatch,len(which));index=torch.arange(first,last,device=runtime.device)
        part=FlowSample(sliced(sample.arguments,index),sample.target[first:last],sample.action_width)
        with autocast(runtime.device),runtime.execution.activate(leaves,batch_indices=which[first:last]):
            prediction=owner(part)
            # All112 queries have equal global mass; synchronization SUMs ranks.
            loss=mean_velocity_loss(prediction,part.target,part.action_width)*((last-first)/112)
        loss.backward();value+=float(loss.detach())
        if hasattr(runtime,'_FM_batches'):runtime._FM_batches.append(last-first)
    credit_norm=0.
    for state,leaf in zip(states,leaves,strict=True):
        names=[k for k in state if leaf[k].grad is not None]
        if names:
            gradients=[leaf[k].grad for k in names]
            credit_norm+=sum(float(v.float().square().sum()) for v in gradients)
            torch.autograd.backward([state[k] for k in names],gradients)
    return value,credit_norm**.5


def update_gradients(runtime,reader,decoder,units,labels,queries,events,context,microbatch,task_pack):
    owned=assignments(context.world_size,context.rank);fm=aux=credit=0.;seeds={};runtime._FM_batches=[]
    # Pack real task conditions through the existing per-sample LoRA owner.
    for first in range(0,len(owned),task_pack):
        group=owned[first:first+task_pack];states=[];samples=[];mapping=[]
        for local,(position,indices) in enumerate(group):
            y=reader_output(reader,events[position]['original'],runtime)
            states.append(decoder.decode(units*y));sample,seed=queries.sample(events[position],indices);samples.append(sample)
            seeds[events[position]['task']]=seed;mapping.append((local,list(range(len(indices)))))
        value,norm=fm_credit(runtime,samples,states,mapping,microbatch);fm+=value;credit+=norm
    fm_reader=gradient_norm(reader.parameters())
    for position,event in enumerate(events):
        if auxiliary_owner(position,context.world_size)!=context.rank:continue
        y=reader_output(reader,event['auxiliary'],runtime)
        loss=(y-labels[event['task']]).square().mean()*(.1/4)
        loss.backward();aux+=float(loss.detach())
    return dict(FM=fm,weighted_auxiliary=aux,LoRA_cotangent_norm=credit,FM_Reader_gradient_norm=fm_reader,
        head_gradient_norm=gradient_norm(reader.head.parameters()),body_gradient_norm=gradient_norm(
            p for n,p in reader.named_parameters() if not n.startswith('head.')),noise_time_roots=seeds,
        physical_FM_batches=runtime._FM_batches,query_rows=sum(len(q) for _,q in owned),auxiliary_rows=sum(auxiliary_owner(p,context.world_size)==context.rank for p in range(4)))
