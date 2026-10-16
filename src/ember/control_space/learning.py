"""Native cross-episode FM credit through fixed D to fresh H revision."""
from pathlib import Path
import torch
from torch.utils.data import default_collate
from ember.writer.data import FunctionalQueryDataset
from ember.writer.function_credit import FlowSample,NativeFlowPrediction,flow_sample,mean_velocity_loss
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.runtime import autocast
import time


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



def gradient_norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def fm_credit(runtime,samples,states,owned,microbatch):
    """Stage native cotangents, then propagate through the exact same D/psi graph."""
    leaves=[{k:v.detach().requires_grad_(True) for k,v in state.items()} for state in states]
    sample,which=selected_sample(samples,owned,runtime.device);owner=NativeFlowPrediction(runtime.policy);value=0.
    for first in range(0,len(which),microbatch):
        last=min(first+microbatch,len(which));index=torch.arange(first,last,device=runtime.device)
        part=FlowSample(sliced(sample.arguments,index),sample.target[first:last],sample.action_width)
        with autocast(runtime.device),runtime.execution.activate(leaves,batch_indices=which[first:last]):
            prediction=owner(part)
            loss=mean_velocity_loss(prediction,part.target,part.action_width)*((last-first)/112)
        loss.backward();value+=float(loss.detach())
        runtime._FM_batches.append(last-first)
    norm=0.
    for state,leaf in zip(states,leaves,strict=True):
        names=[k for k in state if leaf[k].grad is not None]
        if names:
            gradients=[leaf[k].grad for k in names]
            norm+=sum(float(g.float().square().sum()) for g in gradients)
            torch.autograd.backward([state[k] for k in names],gradients)
    return value,norm**.5


def fm_gradients(runtime,model,decoder,units,queries,features,events,context,microbatch,task_pack,event_chunk,image_chunk):
    owned=assignments(context.world_size,context.rank);loss=credit=H_seconds=FM_seconds=0.;seeds={};details=[]
    runtime._FM_batches=[]
    for first in range(0,len(owned),task_pack):
        group=owned[first:first+task_pack];states=[];samples=[];mapping=[]
        for local,(position,indices) in enumerate(group):
            entry=events[position];tick=time.monotonic()
            memory=features.memory(entry['original']).to(runtime.device)
            raw=features.events(entry['history_paths'],image_chunk=image_chunk)
            center=torch.tensor(entry['m'],device=runtime.device,dtype=torch.float32)
            with autocast(runtime.device):
                candidate=model.propose(memory,raw,center,entry['best_score'],entry['remaining_steps'],event_chunk=event_chunk,checkpointed=True)
            state=decoder.decode(units*candidate)
            states.append(state);H_seconds+=time.monotonic()-tick
            details.append(dict(task=entry['task'],events=len(raw),teacher_tokens=memory.shape[1],
                prefix=entry['prefix'],candidate=candidate.detach().float().cpu().tolist(),
                center=center.cpu().tolist(),remaining_steps=entry['remaining_steps']))
            sample,seed=queries.sample(entry,indices);samples.append(sample);seeds[entry['task']]=seed
            mapping.append((local,list(range(len(indices)))))
        tick=time.monotonic();value,norm=fm_credit(runtime,samples,states,mapping,microbatch)
        loss+=value;credit+=norm;FM_seconds+=time.monotonic()-tick
    return dict(FM=loss,LoRA_cotangent_norm=credit,noise_time_roots=seeds,physical_FM_batches=runtime._FM_batches,
        query_rows=sum(len(q) for _,q in owned),H_forward_seconds=H_seconds,native_and_revision_backward_seconds=FM_seconds,
        conditions=details,head_gradient_norm=gradient_norm(model.delta_head.parameters()),
        body_gradient_norm=gradient_norm(p for n,p in model.named_parameters() if not n.startswith('delta_head.')))
