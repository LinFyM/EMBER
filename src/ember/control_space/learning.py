"""Global-mean native10 functional credit with same-query frozen-prefix reuse."""
from collections import OrderedDict
import torch
from safetensors.torch import load_file
from ember.writer.runtime import autocast
from ember.writer.practice.execution import NativeVelocity,native_actions


class References:
    def __init__(self,runtime,names):self.runtime,self.names=runtime,names;self.cache=OrderedDict()

    def get(self,path):
        if path not in self.cache:
            if len(self.cache)==8:self.cache.popitem(last=False)
            raw=self.runtime.identity if path is None else load_file(path,device=str(self.runtime.device))
            self.cache[path]={k:raw[k].detach() for k in self.names}
        self.cache.move_to_end(path);return self.cache[path]


def compute_pack(runtime,model,references,query_bank,entries,counters):
    unique=OrderedDict();points=OrderedDict()
    for row in entries:
        unique.setdefault((row['task'],row['visit'],row['query']),row)
        points.setdefault((row['task'],row['point']),row)
    batch,noise,queries=query_bank.batch(list(unique.values()))
    qindex={k:i for i,k in enumerate(unique)};pindex={k:i for i,k in enumerate(points)}
    qids=torch.tensor([qindex[(e['task'],e['visit'],e['query'])] for e in entries],device=runtime.device)
    pids=torch.tensor([pindex[(e['task'],e['point'])] for e in entries],device=runtime.device)
    states=[references.get(p['path']) for p in points.values()]
    with autocast(runtime.device):
        prefix=NativeVelocity(runtime.policy,batch,capture_phi=False)
        prepared=prefix.select(qids);paired_noise=noise[qids]
        counters['prefix_queries']=counters.get('prefix_queries',0)+len(noise)
        with torch.no_grad():
            targets=native_actions(runtime,states,batch,paired_noise,batch_indices=pids,
                checkpointed=False,prepared_velocity=prepared,call_counter=counters)
        # The anchored source branch is identically its already computed target.
        # Its half weight remains in /56; it is never reassigned to other rows.
        learned=[i for i,p in enumerate(points.values()) if p['path'] is not None]
        remap={p:i for i,p in enumerate(learned)}
        selected=torch.tensor([i for i,p in enumerate(pids.tolist()) if p in remap],device=runtime.device,dtype=torch.long)
        errors=torch.zeros(len(entries),device=runtime.device)
        counters['identity_student_reuse']=counters.get('identity_student_reuse',0)+len(entries)-len(selected)
        if len(selected):
            generated=model.forward_many([states[i] for i in learned])
            student_pids=torch.tensor([remap[int(pids[i])] for i in selected],device=runtime.device)
            students=native_actions(runtime,generated,batch,paired_noise[selected],batch_indices=student_pids,
                prepared_velocity=prefix.select(qids[selected]),call_counter=counters)
            differences=(students.float()-targets[selected].float()).square().mean((1,2))
            errors=errors.index_copy(0,selected,differences)
        loss=errors.sum()/56
    if loss.requires_grad:loss.backward()
    responses=[dict(task=e['task'],point=e['point'],query=e['query'],target=t.cpu(),error=float(v))
               for e,t,v in zip(entries,targets.detach(),errors.detach(),strict=True)]
    return float(loss.detach()),responses,queries


def local_credit(runtime,model,references,query_bank,groups,microbatch,counters):
    rows=[dict(group,query=q) for group in groups for q in range(7)]
    total=0.;responses=[];queries=[]
    for first in range(0,len(rows),microbatch):
        value,part,addresses=compute_pack(runtime,model,references,query_bank,rows[first:first+microbatch],counters)
        total+=value;responses.extend(part);queries.extend(addresses)
    return total,responses,queries


def response_metrics(responses):
    if len(responses)!=56:raise ValueError('one logical update must preserve all 56 point/query rows')
    by_query={};errors={0:[],1:[]}
    for row in responses:
        by_query.setdefault((row['task'],row['query']),{})[row['point']]=row['target']
        errors[row['point']].append(row['error'])
    lower=sum(float((r[0]-r[1]).square().mean())/4 for r in by_query.values())/28
    actual=sum(sum(v) for v in errors.values())/56
    return dict(MSE=actual,selected_MSE=sum(errors[0])/28,other_MSE=sum(errors[1])/28,
        ignored_code_MSE_lower_bound=lower,MSE_to_lower_bound=actual/lower if lower else None)
