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

from .contract import TASKS, OPTIMIZER, seed, CHECKPOINT_SCHEMA
from .path import decision_logits, history_prefix, score_log_probability, load_history, conditional_parent_log_probability
from .runtime import phase,finish_phases


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


def checkpoint(root, stage, update, runtime, optimizer, context, *, sampler, baseline=None, baseline_optimizer=None,
               generator_optimizer=None):
    root = Path(root) / 'checkpoints' / stage / f'update_{update:08d}'
    rng = capture_rng(context)
    ranks = [None] * context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(rng, ranks, dst=0)
    else:
        ranks = [rng]
    if context.is_main:
        root.mkdir(parents=True, exist_ok=False)
        state = dict(schema_version=CHECKPOINT_SCHEMA, stage=stage, next_update=update,
            G_objective='masked_endpoint_MSE',generation='deterministic_MT_centered_endpoint',
            psi=runtime.generator.state_dict(), theta=runtime.actor.state_dict(), optimizer=optimizer.state_dict(),
            baseline=baseline.state_dict() if baseline is not None else None,
            baseline_optimizer=baseline_optimizer.state_dict() if baseline_optimizer is not None else None,
            sampler=sampler, rank_rng=ranks, topology=dict(world_size=context.world_size,
                task_order=list(TASKS), logical_condition_batch=4), psi_version=runtime.version,
            coordinates=dict(rank=8,valid=runtime.lora.parameter_count,center=str(runtime.root/'initial.safetensors'),
                fixed_bank_scale=str(runtime.bank_scale_path)),
            fixed_kernel_identity=getattr(runtime,'fixed_kernel_identity',None),
            generator_optimizer=(generator_optimizer.state_dict() if generator_optimizer is not None else
                optimizer.state_dict() if stage=='G' else getattr(runtime,'generator_optimizer_state',None)),
            G_sampler=getattr(runtime,'G_sampler',dict(next_update=update if stage=='G' else 0)),
            supervision_version=getattr(runtime,'supervision_version',0),
            parameter_names=dict(psi=list(dict(runtime.generator.named_parameters())),
                                 theta=list(dict(runtime.actor.named_parameters()))))
        torch.save(state, root / 'state.pt')
        write_json_atomic(root / 'manifest.json', dict(schema_version=state['schema_version'], stage=stage,
            next_update=update, full_optimizer_rng_sampler=True, complete=True, topology=state['topology']))
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
    return root


def resume_checkpoint(path, stage, runtime, optimizer, context, *, baseline=None, baseline_optimizer=None,
                      generator_optimizer=None):
    state = torch.load(Path(path) / 'state.pt', map_location='cpu', weights_only=False)
    if (state['schema_version'] != CHECKPOINT_SCHEMA or state['stage'] != stage
            or state['topology']['world_size'] != context.world_size):
        raise ValueError('resume stage/topology changed; explicit migration needs a recorded new topology contract')
    runtime.generator.load_state_dict(state['psi']); runtime.actor.load_state_dict(state['theta'])
    optimizer.load_state_dict(state['optimizer']); runtime.version = state['psi_version']
    runtime.generator_optimizer_state=state['generator_optimizer'];runtime.G_sampler=state['G_sampler']
    runtime.supervision_version=state['supervision_version']
    if generator_optimizer is not None:generator_optimizer.load_state_dict(state['generator_optimizer'])
    restore_rng(state['rank_rng'][context.rank], context)
    if baseline is not None:
        baseline.load_state_dict(state['baseline'])
        if baseline_optimizer is not None:baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    return state['next_update']


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


def logical_tasks(update):
    """Nine B4 updates form one equal-weight full36 round, independent of topology."""
    round_index,slot=divmod(update,len(TASKS)//4)
    order=np.random.default_rng(seed(44,round_index)).permutation(TASKS).tolist()
    return order[4*slot:4*(slot+1)]


def assigned_tasks(context,update):
    return [t for i,t in enumerate(logical_tasks(update)) if i%context.world_size==context.rank]


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


class EndpointTrainer:
    """The same fixed-center endpoint update serves initial and joint supervision."""
    def __init__(self,runtime,context,root):
        self.runtime,self.context,self.root=runtime,context,Path(root)
        self.optimizer=torch.optim.AdamW(runtime.generator.parameters(),**OPTIMIZER)
        self.cache=HistoryCache();self.panel={r['task_id']:r for r in read_json(self.root/'panel.json')['tasks']}
        if not runtime.bank_scale_path.exists():raise ValueError('complete full36 bank and fixed S_G required')
        if getattr(runtime,'generator_optimizer_state',None) is not None:
            self.optimizer.load_state_dict(runtime.generator_optimizer_state)
        self.refresh_bank(getattr(runtime,'supervision_version',0))

    def refresh_bank(self,version):
        contract=read_json(self.root/'run_contract.json')
        mode=contract.get('G_bank_mode','actual_parents')
        if mode not in ('MT_only','actual_parents') or (mode=='MT_only' and version!=0):
            raise ValueError('registered MT-only endpoint bank requires supervision version0')
        self.bank={t:read_json(self.root/'teachers'/f'task_{t:04d}'/'E.json') for t in TASKS}
        for task in TASKS:
            bootstrap=read_json(self.root/'bootstrap'/f'task_{task:04d}'/'complete.json')
            if not bootstrap['complete'] or bootstrap['reference']!='MT300':
                raise ValueError('initial G needs registered nonempty MT self practice for all36')
            self.bank[task]['parents']=[dict(reference='MT300',weights=str(self.root/'initial.safetensors'),
                histories=[bootstrap['history']])]
        self.runtime.supervision_version=version
        if mode=='MT_only':return  # Never inspect student parent files or G_student_bank.
        version_name=contract.get('student_refresh',{}).get('version')
        if version_name is not None:
            files=[self.root/'student_data'/version_name/f'task_{task:04d}'/'parent.json' for task in TASKS]
            if any(file.exists() for file in files):
                if not all(file.exists() and read_json(file)['complete'] for file in files):
                    raise ValueError('all36 actual initial student parents must precede joint G updates')
                for task,file in zip(TASKS,files,strict=True):
                    self.bank[task]['parents'].append(read_json(file)['parent'])
        path=self.root/'G_student_bank.json'
        rows=read_json(path)['rows'] if path.exists() else []
        for row in rows:
            if row['joint_update']>version:continue
            parents=self.bank[row['task']]['parents'];parent=row['parent']
            existing=next((p for p in parents if p['reference']==parent['reference']),None)
            if existing is None:parents.append(parent)
            else:existing['histories'].extend(h for h in parent['histories'] if h not in existing['histories'])
        self.runtime.supervision_version=version

    def step(self,update,*,stage='G'):
        from safetensors.torch import load_file
        runtime,context,root=self.runtime,self.context,self.root
        tick=time.monotonic();before=dict(runtime.components);decode_before=self.cache.decode_seconds
        runtime.psi_frozen=False;runtime.generator.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True);records=[]
        for task in assigned_tasks(context,update):
            rng=np.random.default_rng(seed(40,task,update));entry=self.bank[task]
            # No class/category weights: parent cardinality and H cardinality are separate.
            parent_row=entry['parents'][int(rng.integers(len(entry['parents'])))]
            history_path=parent_row['histories'][int(rng.integers(len(parent_row['histories'])))]
            history=self.cache.get(history_path)
            if not history.records:raise ValueError('G receives nonempty real parent H')
            parent=load_file(parent_row['weights']);endpoint=load_file(entry['weights'])
            demo=self.panel[task]['policy_videos'][int(rng.integers(len(self.panel[task]['policy_videos'])))]
            features=runtime.teaching(task,demo)
            packed=runtime.generator.layout.pack(parent).to(runtime.device)
            target=(runtime.generator.layout.pack(endpoint).to(runtime.device)-runtime.generator.layout.center)/runtime.generator.layout.scale
            rows=history.model_rows(runtime)
            with phase(runtime,'endpoint_forward'),autocast(runtime.device):
                prediction=runtime.generator(packed,features,rows,history.states)
                loss=(prediction.float()[runtime.generator.layout.valid]-target[runtime.generator.layout.valid]).square().mean()
            with phase(runtime,'endpoint_backward'):(loss/4).backward()
            records.append(dict(task=task,parent=parent_row['reference'],E=entry['reference'],H=history_path,demo=demo,
                endpoint_loss=float(loss.detach()),MT_endpoint_loss=float(target[runtime.generator.layout.valid].square().mean()),
                frames=len(features['z']),H_records=len(history.records),
                H_episodes=len(history.episodes),H_producers=len(history.states),denominator=4,missing_label=False))
            self.cache.remember_features(history_path,history)
            del features,prediction,loss,target,endpoint,parent,history,rows
        synchronize_gradients(runtime.generator,context)
        norm=torch.nn.utils.clip_grad_norm_(runtime.generator.parameters(),1.,error_if_nonfinite=True)
        group_norms={g:sum(float(p.grad.float().square().sum()) for p,name in zip(runtime.generator.meta.values,runtime.generator.meta.groups,strict=True)
            if name==g and p.grad is not None)**.5 for g in ('gemma','action')}
        self.optimizer.step();runtime.mark_psi_update();torch.cuda.synchronize(context.device)
        finish_phases(runtime)
        runtime.G_sampler=dict(next_update=update+1,logical_batch=4,task_order=list(TASKS),
            task_weight=1/36,parent_weight='equal actual parent',H_weight='uniform within parent')
        metrics(root,stage,dict(update=update+1,events=records,logical_task_batch=logical_tasks(update),gradient_norm=float(norm),
            meta_frame_chunk=runtime.generator.meta.frame_chunk,
            meta_gradient_norm=group_norms,seconds=time.monotonic()-tick,decoded_H_cache_entries=len(self.cache.items),
            supervision_version=runtime.supervision_version,
            phases={k:v-before.get(k,0.) for k,v in runtime.components.items()},
            history_decode_seconds=self.cache.decode_seconds-decode_before,
            source_has_gradient=any(p.grad is not None for p in runtime.policy.parameters())),context)


def train_endpoint(runtime,context,root,*,stop,resume=None):
    contract=read_json(Path(root)/'run_contract.json')
    if stop not in contract['updates']['G'] or stop%9:raise ValueError('registered full36 endpoint rounds required')
    if contract.get('G_initial_resume') and resume is None:
        raise ValueError('registered continued endpoint learning cannot start fresh')
    trainer=EndpointTrainer(runtime,context,root)
    initialize_deferred_process_group(context,rendezvous_root=Path(root)/'rendezvous')
    start=resume_checkpoint(resume,'G',runtime,trainer.optimizer,context) if resume else 0
    if resume is not None:
        if (runtime.G_sampler['next_update']!=start or runtime.version!=start
                or start>stop or (contract.get('G_initial_resume') and start<contract['G_initial_resume']['update'])):
            raise ValueError('complete endpoint resume cursor/version does not match its learning window')
        steps=[int(s['step']) for s in trainer.optimizer.state.values() if 'step' in s]
        write_json_atomic(Path(root)/'code'/f'G_resume_rank_{context.rank:02d}.json',dict(
            complete=True,checkpoint=str(Path(resume).resolve()),next_update=start,psi_version=runtime.version,
            G_sampler=runtime.G_sampler,supervision_version=runtime.supervision_version,
            optimizer_state_count=len(trainer.optimizer.state),optimizer_step_min=min(steps),optimizer_step_max=max(steps),
            world_size=context.world_size,restored_rng_rank=context.rank,logical_condition_batch=4,
            G_bank_mode=contract.get('G_bank_mode','actual_parents'),fresh=False))
    trainer.refresh_bank(getattr(runtime,'supervision_version',0))
    if resume is None and context.is_main:
        torch.save(dict(schema_version=CHECKPOINT_SCHEMA,psi=runtime.generator.state_dict(),theta=runtime.actor.state_dict(),
            formal_fresh=True,coordinates=str(runtime.root/'coordinates.json'),bank_scale=str(runtime.bank_scale_path)),Path(root)/'initial_models.pt')
    for update in range(start,stop):
        trainer.step(update)
        if update+1 in contract.get('G_checkpoint_updates',contract['updates']['G']):
            checkpoint(root,'G',update+1,runtime,trainer.optimizer,context,sampler=dict(seed=20261011,next_update=update+1,
                logical_batch=4,task_order=list(TASKS),task_weight=1/36,parent_weight='equal actual parent',H_weight='uniform within parent',
                bank_version=contract.get('G_bank_version',0),history_cache='2GiB and at most2 decoded H per worker; includes raw features'))


def expected_query(logits, rewards):
    return (logits.softmax(-1)*logits.new_tensor(rewards)).sum()


def train_local(runtime,context,root,example_loader,*,stop,resume=None,baseline=None):
    """Mean real prefixes inside a condition, then four equal task conditions."""
    contract=read_json(Path(root)/'run_contract.json')
    if stop not in contract['updates']['local'] or stop%9:raise ValueError('registered full36 local rounds required')
    runtime.freeze_psi();runtime.actor.train()
    optimizer=torch.optim.AdamW(runtime.actor.parameters(),**OPTIMIZER)
    initialize_deferred_process_group(context,rendezvous_root=Path(root)/'rendezvous')
    start=resume_checkpoint(resume,'local',runtime,optimizer,context,baseline=baseline) if resume else 0
    for update in range(start,stop):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True);records=[]
        for task in assigned_tasks(context,update):
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
                (loss/(4*len(prefixes))).backward();losses.append(float(loss.detach()))
                del logits,loss
            records.append(dict(task=task,condition=path.identity,prefixes=len(prefixes),
                prefix_mean_loss=sum(losses)/len(losses),source=example['source'],query_excluded_from_H=True))
            del example,path,prefixes,features
        synchronize_gradients(runtime.actor,context)
        norm=torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize(context.device)
        metrics(root,'local',dict(update=update+1,conditions=records,gradient_norm=float(norm),
            logical_task_batch=logical_tasks(update),seconds=time.monotonic()-tick,psi_frozen=True),context)
        if update+1 in contract['updates']['local']:
            checkpoint(root,'local',update+1,runtime,optimizer,context,
                sampler=dict(next_update=update+1,task_order=list(TASKS),condition_weight=.25,
                    prefix_reduction='mean within actual condition',data_version=contract['student_refresh']['version']),baseline=baseline)


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
