"""Equal-condition endpoint learning, final-use learning and on-policy credit."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import random
import time
import resource

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX

from .contract import TASKS, OPTIMIZER, seed, CHECKPOINT_SCHEMA, G_OBJECTIVE, OPERATOR_OBJECTIVE
from .path import decision_logits, history_prefix, score_log_probability, load_history, conditional_parent_log_probability
from .runtime import phase,finish_phases
from .shared_checkpoint import checkpoint, resume_checkpoint


def operator_squared_error(A, B, E_A, E_B, scaling):
    """Exact full s*BA residual norm using rank-sized Grams, with FP32 gradients."""
    with torch.autocast(device_type=A.device.type, enabled=False):
        A,B,E_A,E_B=(value.float() for value in (A,B,E_A,E_B))
        generated=((B.T@B)*(A@A.T)).sum()
        teacher=((E_B.T@E_B)*(E_A@E_A.T)).sum()
        cross=((B.T@E_B)*(A@E_A.T)).sum()
        return scaling**2*(generated+teacher-2*cross),scaling**2*teacher


class OperatorObjective:
    """Frozen all36 teacher energies; each condition averages the same 38 layers."""
    def __init__(self, runtime, root):
        record=read_json(Path(root)/'operator_scale.json')
        targets=[item.name for item in runtime.lora.targets]
        expected_scaling=runtime.lora.alpha/runtime.lora.rank
        if (record['objective']!=OPERATOR_OBJECTIVE or set(record['D_by_target'])!=set(targets)
                or set(record['scaling'])!=set(targets)
                or sorted(row['task'] for row in record['sources'])!=sorted(TASKS)
                or any(record['scaling'][name]!=expected_scaling for name in targets)
                or any(not np.isfinite(record['D_by_target'][name]) or record['D_by_target'][name]<=0 for name in targets)):
            raise ValueError('operator objective needs exact positive all36 teacher energies and deployed scaling')
        self.layers=[(name,record['scaling'][name],record['D_by_target'][name]) for name in targets]

    def __call__(self, generated, teacher):
        errors,baseline=[],[]
        for name,scaling,denominator in self.layers:
            A,B=generated[name+LORA_A_SUFFIX],generated[name+LORA_B_SUFFIX]
            E_A=teacher[name+LORA_A_SUFFIX].to(A.device);E_B=teacher[name+LORA_B_SUFFIX].to(A.device)
            error,energy=operator_squared_error(A,B,E_A,E_B,scaling)
            errors.append(error/denominator);baseline.append(energy/denominator)
        return torch.stack(errors).mean(),torch.stack(baseline).mean()


class Baseline(nn.Module):
    """Independent ω; frozen teaching/facts only, no actor or generator embedding."""
    def __init__(self):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(2048 + 5, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, features, rows):
        language = features['language'][features['language_mask']].detach().float().mean(0)
        facts = torch.stack([r['feedback'].to(language.device) for r in rows]).mean(0) if rows else language.new_zeros(5)
        inputs = torch.cat((language, facts))
        return self.value(inputs).squeeze(-1)


def synchronize_gradients(module, context):
    """Sum locally /4-scaled condition gradients; never average path decisions."""
    if context.world_size == 1:
        return
    parameters = list(module.parameters())
    used = torch.tensor([p.grad is not None for p in parameters], device=context.device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, flag in zip(parameters, used.tolist(), strict=True) if flag]
    if not active:
        return  # A complete missing-label batch still retains its logical opportunity.
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    cursor = 0
    for p in active:
        count = p.numel()
        if p.grad is None:
            p.grad = torch.empty_like(p)
        p.grad.copy_(flat[cursor:cursor + count].reshape_as(p))
        cursor += count



def memory_peak(device):
    return dict(allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30,
                host_max_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)


def metrics(root, stage, record, context):
    record['process_GPU_peak']=memory_peak(context.device)
    destination = Path(root) / 'metrics' / f'{stage}_rank_{context.rank}.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    import json
    with destination.open('a') as f:
        f.write(json.dumps(record) + '\n')


def logical_tasks(update,root=None):
    """The registered B4 task stream is independent of physical topology."""
    if root is not None:
        run=read_json(Path(root)/'run_contract.json');epoch,slot=divmod(update,18)
        seen=np.random.default_rng(np.random.SeedSequence([20261013,23,0,epoch])).permutation(run['G_seen_tasks']).tolist()
        novel=np.random.default_rng(np.random.SeedSequence([20261013,23,1,epoch])).permutation(run['G_novel_tasks']).tolist()
        return seen[2*slot:2*slot+2]+novel[2*slot:2*slot+2]
    round_index,slot=divmod(update,len(TASKS)//4)
    order=np.random.default_rng(seed(44,round_index)).permutation(TASKS).tolist()
    return order[4*slot:4*(slot+1)]


def assigned_tasks(context,update,root=None):
    return [t for i,t in enumerate(logical_tasks(update,root)) if i%context.world_size==context.rank]


class HistoryCache:
    """Decoded H and its frozen raw image features have a bounded worker lifetime."""
    def __init__(self,limit_bytes=8*1024**3,max_entries=2):
        self.limit_bytes,self.max_entries,self.items=limit_bytes,max_entries,OrderedDict()
        self.decode_seconds,self.cache_hits=0.,0

    @staticmethod
    def nbytes(value):
        if isinstance(value,(torch.Tensor,np.ndarray)):return value.nbytes
        if isinstance(value,dict):return sum(HistoryCache.nbytes(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(HistoryCache.nbytes(v) for v in value)
        return 0

    def trim(self):
        while self.items and (len(self.items)>self.max_entries or sum(self.nbytes(vars(h)) for h in self.items.values())>self.limit_bytes):
            self.items.popitem(last=False)

    def seed(self,path,history):
        """Borrow the actual current batch H/raw φ; learned encodings are absent."""
        path=str(path);self.items[path]=history;self.items.move_to_end(path);self.trim()

    def get(self,path):
        descriptor=path if isinstance(path,dict) else None
        path=descriptor['path'] if descriptor else path
        if path not in self.items:
            tick=time.monotonic();self.items[path]=load_history(path);self.decode_seconds+=time.monotonic()-tick
        else:self.cache_hits+=1
        self.items.move_to_end(path);result=self.items[path];self.trim()
        return history_prefix(result,descriptor['records'],descriptor['episodes']) if descriptor else result

    def remember_features(self,path,history):
        path=path['path'] if isinstance(path,dict) else path
        if path in self.items:self.items[path].image_features.update(history.image_features)
        self.trim()


def expected_query(logits, rewards):
    return (logits.softmax(-1)*logits.new_tensor(rewards)).sum()


def train_local(runtime,context,root,example_loader,*,stop,resume=None,baseline=None):
    """Mean real prefixes inside a condition, then four equal task conditions."""
    contract=read_json(Path(root)/'run_contract.json')
    if stop not in contract['updates']['local'] or stop%18:raise ValueError('registered equal seen/novel72 local rounds required')
    runtime.freeze_psi();runtime.actor.train()
    optimizer=torch.optim.AdamW(runtime.actor.parameters(),**OPTIMIZER)
    optimizer.load_state_dict(runtime.actor_optimizer_state)
    initialize_deferred_process_group(context,rendezvous_root=Path(root)/'rendezvous')
    start=resume_checkpoint(resume,'local',runtime,optimizer,context,baseline=baseline) if resume else 0
    for update in range(start,stop):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True);records=[]
        for task in assigned_tasks(context,update,root):
            example=example_loader(task,update);path=example['path'];prefixes=example['prefixes']
            if not prefixes:raise ValueError('local example lacks actual completed prefixes')
            features=runtime.teaching(task,path.demo,video_task=path.video_task);losses=[]
            for prefix in prefixes:
                candidates=prefix['practiced'];rewards=[prefix['reward'][name] for name in candidates]
                if len(candidates)==1:
                    losses.append(-rewards[0]);continue
                decision=prefix.setdefault('decision',dict(kind='final',candidates=candidates,
                    records=prefix['records'],episodes=prefix['episodes'],parent=None))
                logits=decision_logits(runtime,path,decision,features);loss=-expected_query(logits,rewards)
                prefix['training_probabilities']=logits.detach().softmax(-1).cpu().tolist()
                (loss/(4*len(prefixes))).backward();losses.append(float(loss.detach()))
                del logits,loss
            records.append(dict(task=task,condition=path.identity,prefixes=len(prefixes),
                actual_prefix_credit=[dict(candidates=p['practiced'],reward=p['reward'],probabilities=p.get('training_probabilities')) for p in prefixes],
                prefix_mean_loss=sum(losses)/len(losses),source=example['source'],query_excluded_from_H=True))
            path.native_response_cache.clear()  # Final predictions are frozen in prefix decisions; discard GPU KV.
            del example,path,prefixes,features
        synchronize_gradients(runtime.actor,context)
        norm=torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize(context.device)
        metrics(root,'local',dict(update=update+1,conditions=records,gradient_norm=float(norm),
            logical_task_batch=logical_tasks(update,root),seconds=time.monotonic()-tick,psi_frozen=True,
            actual_path_cache_hits=example_loader.cache_hits,H_decode_seconds=example_loader.decode_seconds),context)
        if update+1 in contract['updates']['local']:
            checkpoint(root,'local',update+1,runtime,optimizer,context,
                sampler=dict(next_update=update+1,task_order=contract['training_tasks'],condition_weight=.25,
                    prefix_reduction='mean within actual condition',data_version=contract['student_refresh']['version']),baseline=baseline,
                baseline_optimizer=runtime.training_baseline_optimizer)



def continuation_values(stop_probabilities,query_expectations,*,cost=.001):
    """All truncated-depth values and fixed-main-path adjoints, using actual probabilities."""
    rounds=len(stop_probabilities)
    if len(query_expectations)!=rounds+1 or not rounds:raise ValueError('incomplete continuation nodes')
    p,q=stop_probabilities,query_expectations
    values=[];q_coeff=[0.]*(rounds+1);p_coeff=[0.]*rounds
    for depth in range(1,rounds+1):
        suffix=[None]*(depth+1);suffix[depth]=q[depth]
        for i in range(depth-1,-1,-1):suffix[i]=p[i]*q[i]+(1-p[i])*(suffix[i+1]-cost)
        values.append(suffix[0]);survival=1.
        for i in range(depth):
            q_coeff[i]+=survival*p[i]/rounds
            p_coeff[i]+=survival*(q[i]-(suffix[i+1]-cost))/rounds
            survival*=1-p[i]
        q_coeff[depth]+=survival/rounds
    return values,q_coeff,p_coeff


def actual_path_return(path,rewards,cost):
    if path.sampling_identity!='conditional_continue':raise ValueError('RL requires conditional-continue collection')
    parents=[d for d in path.decisions if d['kind']=='parent']
    if len(parents)!=path.completed_rounds or len(path.outputs)!=len(parents)+1:
        raise ValueError('every actual round needs one complete parent, practice and output')
    p=[d['stop_probability'] for d in parents]
    q=[sum(v*rewards[ref] for v,ref in zip(out['probabilities'],out['candidates'],strict=True)) for out in path.outputs]
    return continuation_values(p,q,cost=cost),p,q


def on_policy_update(runtime,context,root,update,paths,query_rewards,optimizer,baseline,baseline_optimizer):
    """Conditional behavior score plus analytic STOP/final gradients, once per current batch."""
    tick=time.monotonic();runtime.freeze_psi();runtime.actor.train();baseline.eval()
    policy=read_json(Path(root)/'run_contract.json')['selector_training']
    cost=float(policy['continue_cost']);rounds=int(policy['rounds'])
    optimizer.zero_grad(set_to_none=True);baseline_optimizer.zero_grad(set_to_none=True);records=[]
    for path,rewards in zip(paths,query_rewards,strict=True):
        if path.completed_rounds!=rounds:raise ValueError('incomplete registered continuation depth')
        features=runtime.teaching(path.task,path.demo,video_task=path.video_task)
        (values,q_coeff,p_coeff),p,q=actual_path_return(path,rewards,cost)
        behavior=[d for d in path.decisions if d['kind'] in ('parent','practice')]
        if any(d['uniform_behavior'] for d in behavior):raise ValueError('teacher/uniform replay cannot receive RL score')
        scores=[];behavior_loss=0.;stop_loss=0.;final_loss=0.
        for decision in behavior:
            i=decision['round_index'];coefficient=sum(values[i:])/rounds
            coefficient-=(rounds-i)/rounds*decision['baseline_before_batch']
            logits=decision_logits(runtime,path,decision,features)
            if decision['kind']=='parent':
                if not decision['conditional_continue']:raise ValueError('unconditional parent score is retired')
                logp=conditional_parent_log_probability(logits,decision['selected_index'],decision['candidates'])
                direct=-p_coeff[i]*logits.softmax(-1)[decision['candidates'].index('STOP')]
                stop_loss+=float(direct.detach())
            else:
                logp=score_log_probability(logits,decision['selected_index'],forced=decision['forced'])
                direct=logits.sum()*0
            score=-coefficient*logp
            with phase(runtime,'theta_backward'):((score+direct)/4).backward()
            behavior_loss+=float(score.detach())
            history=history_prefix(path.history,decision['records'],decision['episodes'])
            prediction=baseline(features,history.records);target=sum(values[i:])/(rounds-i)
            with phase(runtime,'independent_baseline_backward'):((prediction-target)**2/(4*len(behavior))).backward()
            scores.append(dict(kind=decision['kind'],round_index=i,conditional_score=float(logp.detach()),
                behavior_score=decision['log_probability'],coefficient=coefficient,forced=decision['forced']))
            del logits,logp,score,direct,prediction,history
        for out,coefficient in zip(path.outputs,q_coeff,strict=True):
            if not coefficient or len(out['candidates'])==1:continue
            logits=decision_logits(runtime,path,out,features)
            loss=-coefficient*expected_query(logits,[rewards[ref] for ref in out['candidates']])
            with phase(runtime,'theta_backward'):(loss/4).backward()
            final_loss+=float(loss.detach());del logits,loss
        records.append(dict(condition=path.identity,depth_values=values,mean_depth_return=sum(values)/rounds,
            stop_probabilities=p,query_expectations=q,query_adjoint=q_coeff,stop_adjoint=p_coeff,
            candidate_query_rewards=rewards,conditional_behavior_loss=behavior_loss,
            direct_stop_loss=stop_loss,direct_final_loss=final_loss,decisions=scores,
            sampling_identity=path.sampling_identity,actual_rounds=path.completed_rounds,query_excluded_from_H=True))
    synchronize_gradients(runtime.actor,context)
    norm=torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    synchronize_gradients(baseline,context)
    torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.,error_if_nonfinite=True);baseline_optimizer.step()
    torch.cuda.synchronize(context.device);finish_phases(runtime)
    metrics(root,'RL',dict(update=update,conditions=records,actor_gradient_norm=float(norm),
        gradient='mean_d stopgrad(V0d)*SUM_{i<d} conditional behavior scores + grad V0d; mean B4',
        future_budget_model_input=False,continue_cost=cost,psi_frozen=True,one_actor_update=True,
        resample_required=True,theta_and_baseline_update_seconds=time.monotonic()-tick),context)
