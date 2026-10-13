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
    """The single F-only owner calibrates complete LoRAs on native FM."""
    def __init__(self,runtime,context,root,*,microbatch=28):
        from .endpoint_context import FrozenEndpointContexts
        from .endpoint_fm import NativeEndpointCredit
        self.runtime,self.context,self.root=runtime,context,Path(root)
        contract=read_json(self.root/'run_contract.json')
        if contract.get('G_gradient_owner')!='endpoint':
            raise ValueError('current shared G training requires the registered endpoint owner')
        runtime.generator.eval().requires_grad_(False)
        self.parameters=list(runtime.generator.endpoint.parameters())
        self.parameter_names=['endpoint.'+name for name,_ in runtime.generator.endpoint.named_parameters()]
        if len(self.parameters)!=166 or sum(p.numel() for p in self.parameters)!=8761152:
            raise ValueError('the original complete166-tensor endpoint owner changed')
        self.optimizer=torch.optim.AdamW(self.parameters,**OPTIMIZER)
        self.optimizer.owner='endpoint';self.optimizer.parameter_names=self.parameter_names
        if contract['G_objective']!=G_OBJECTIVE or contract['fixed_bank_version']!=9:
            raise ValueError('current endpoint owner requires the fixed-bank native FM contract')
        self.objective=NativeEndpointCredit(runtime,contract['native_FM'],microbatch)
        self.cache=HistoryCache();self.panel={r['task_id']:r for r in read_json(self.root/'panel.json')['tasks']}
        if not runtime.bank_scale_path.exists():raise ValueError('complete full36 bank and fixed S_G required')
        if getattr(runtime,'generator_optimizer_state',None) is not None:
            if getattr(runtime,'generator_optimizer_parameter_names',None)!=self.parameter_names:
                raise ValueError('saved endpoint Adam must map the actual166 named parameters')
            self.optimizer.load_state_dict(runtime.generator_optimizer_state)
        self.contexts=FrozenEndpointContexts(runtime,context,self.root,self.cache,self.panel)
        self.refresh_bank(getattr(runtime,'supervision_version',0))

    def prepare_bank(self):
        self.contexts.prepare(self.bank)

    def refresh_bank(self,version):
        contract=read_json(self.root/'run_contract.json')
        if contract['G_bank_mode']!='actual_parents':raise ValueError('native FM requires the actual fixed parent bank')
        if version!=9 or contract['fixed_bank_version']!=9:
            raise ValueError('native FM must not absorb a changed or student-expanded bank')
        self.bank={t:dict(task=t) for t in TASKS}
        for task in TASKS:
            bootstrap=read_json(self.root/'bootstrap'/f'task_{task:04d}'/'complete.json')
            if not bootstrap['complete'] or bootstrap['reference']!='MT300':
                raise ValueError('initial G needs registered nonempty MT self practice for all36')
            self.bank[task]['parents']=[dict(reference='MT300',weights=str(self.root/'initial.safetensors'),
                histories=[bootstrap['history']])]
        self.runtime.supervision_version=version
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

    def step(self,update,*,stage='G',profile=False):
        runtime,context,root=self.runtime,self.context,self.root
        tick=time.monotonic();before=dict(runtime.components)
        hits_before,loads_before=self.contexts.hits,self.contexts.loads
        load_before=self.contexts.load_seconds
        runtime.psi_frozen=False;runtime.generator.eval().requires_grad_(False)
        runtime.generator.endpoint.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True);records=[];profile_failure=None
        for task in assigned_tasks(context,update):
            rng=np.random.default_rng(seed(40,task,update));entry=self.bank[task]
            # No class/category weights: parent cardinality and H cardinality are separate.
            parent_index=int(rng.integers(len(entry['parents'])));parent_row=entry['parents'][parent_index]
            history_index=int(rng.integers(len(parent_row['histories'])));history_path=parent_row['histories'][history_index]
            demo=self.panel[task]['policy_videos'][int(rng.integers(len(self.panel[task]['policy_videos'])))]
            value,parent,_=self.contexts.get(task,parent_index,history_index,demo,parent_row)
            with phase(runtime,'endpoint_forward'),autocast(runtime.device):
                prediction=runtime.generator.endpoint(parent,memory=value['memory'],checkpoint_layers=False)
            layout=runtime.generator.layout
            generated=layout.unpack(layout.center+layout.scale*prediction.float())
            visit=runtime.G_sampler['native_FM']['task_visits'][str(task)]
            if not 0<=visit<48:raise ValueError('native FM query visit must be independent0..47')
            F_peak=torch.cuda.max_memory_allocated(runtime.device)/2**30
            try:
                credit=self.objective(task,demo,visit,generated)
            except torch.OutOfMemoryError as error:
                if not profile:raise
                profile_failure=str(error);del prediction,generated;torch.cuda.empty_cache();break
            records.append(dict(task=task,parent=parent_row['reference'],H=history_path,demo=demo,
                parent_index=parent_index,history_index=history_index,
                **credit,F_and_frozen_policy_peak_before_native_GiB=F_peak,
                frames=value['frames'],H_records=value['H_records'],H_episodes=value['H_episodes'],
                H_producers=value['producers'],context_source=value['source'],missing_label=False))
            del prediction,generated
        if profile:
            failures=[profile_failure]
            if context.world_size>1:
                failures=[None]*context.world_size;dist.all_gather_object(failures,profile_failure)
            if any(failures):
                self.optimizer.zero_grad(set_to_none=True)
                return dict(profile_failed=True,failures=failures,seconds=time.monotonic()-tick,
                            microbatch=self.objective.microbatch)
        synchronize_gradients(runtime.generator.endpoint,context)
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
        self.optimizer.step();runtime.mark_psi_update();torch.cuda.synchronize(context.device)
        steps=[int(value['step']) for value in self.optimizer.state.values()]
        if len(steps)!=166 or min(steps)!=update+1 or max(steps)!=update+1:
            raise ValueError('the complete166 endpoint Adam states must advance with the actual cursor')
        finish_phases(runtime)
        native_sampler=runtime.G_sampler['native_FM']
        for task in logical_tasks(update):native_sampler['task_visits'][str(task)]+=1
        runtime.G_sampler=dict(next_update=update+1,logical_batch=4,task_order=list(TASKS),
            task_weight=1/36,parent_weight='equal actual parent',H_weight='uniform within parent',gradient_owner='endpoint',
            native_FM=native_sampler)
        record=dict(update=update+1,events=records,logical_task_batch=logical_tasks(update),gradient_norm=float(norm),
            endpoint_parameter_tensors=len(self.parameters),Adam_step_min=min(steps),Adam_step_max=max(steps),
            seconds=time.monotonic()-tick,frozen_context_device_cache_entries=len(self.contexts.items),
            frozen_context_hits=self.contexts.hits-hits_before,frozen_context_loads=self.contexts.loads-loads_before,
            frozen_context_load_seconds=self.contexts.load_seconds-load_before,
            supervision_version=runtime.supervision_version,
            phases={k:v-before.get(k,0.) for k,v in runtime.components.items()},
            frontend_has_gradient=any(p.grad is not None for name,p in runtime.generator.named_parameters() if not name.startswith('endpoint.')),
            source_has_gradient=any(p.grad is not None for p in runtime.policy.parameters()))
        if record['frontend_has_gradient'] or record['source_has_gradient']:
            raise ValueError('native FM gradient escaped the sole endpoint owner')
        metrics(root,stage,record,context)
        return record


def train_endpoint(runtime,context,root,*,stop,resume=None,microbatch=28,profile_microbatches=None):
    contract=read_json(Path(root)/'run_contract.json')
    if stop not in contract['updates']['G'] or stop!=38781:raise ValueError('registered native FM terminal required')
    if contract.get('G_initial_resume') and resume is None:
        raise ValueError('registered continued endpoint learning cannot start fresh')
    trainer=EndpointTrainer(runtime,context,root,microbatch=microbatch)
    initialize_deferred_process_group(context,rendezvous_root=Path(root)/'rendezvous')
    start=resume_checkpoint(resume,'G',runtime,trainer.optimizer,context,
        baseline=runtime.training_baseline,baseline_optimizer=runtime.training_baseline_optimizer) if resume else runtime.G_sampler['next_update']
    if runtime.version!=start or start>stop:
        raise ValueError('component import must retain the endpoint sampler/version cursor')
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
            G_bank_mode=contract.get('G_bank_mode','actual_parents'),fresh=False,
            G_objective=G_OBJECTIVE,objective_migration=runtime.objective_migration))
    trainer.refresh_bank(getattr(runtime,'supervision_version',0))
    trainer.prepare_bank()
    expected_visits=(start-38349)//9
    if (start<38349 or (start-38349)%9 or runtime.supervision_version!=9
            or set(runtime.G_sampler['native_FM']['task_visits'].values())!={expected_visits}):
        raise ValueError('FM absolute cursor and independent per-task query visits changed')
    if profile_microbatches is not None:
        from .endpoint_fm import profile_physical_batches
        profile_physical_batches(trainer,runtime,context,root,start,resume,profile_microbatches)
        return
    for update in range(start,stop):
        trainer.step(update)
        if update+1 in contract.get('G_checkpoint_updates',contract['updates']['G']):
            checkpoint(root,'G',update+1,runtime,trainer.optimizer,context,sampler=dict(seed=20261011,next_update=update+1,
                logical_batch=4,task_order=list(TASKS),task_weight=1/36,parent_weight='equal actual parent',H_weight='uniform within parent',
                bank_version=runtime.supervision_version,gradient_owner='endpoint',
                context_cache='exact frozen frontend and real P/H prefix; shared data1 files and bounded device reuse'))
    trainer.objective.close()


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
            path.native_response_cache.clear()  # Final predictions are frozen in prefix decisions; discard GPU KV.
            del example,path,prefixes,features
        synchronize_gradients(runtime.actor,context)
        norm=torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize(context.device)
        metrics(root,'local',dict(update=update+1,conditions=records,gradient_norm=float(norm),
            logical_task_batch=logical_tasks(update),seconds=time.monotonic()-tick,psi_frozen=True,
            actual_path_cache_hits=example_loader.cache_hits,H_decode_seconds=example_loader.decode_seconds),context)
        if update+1 in contract['updates']['local']:
            checkpoint(root,'local',update+1,runtime,optimizer,context,
                sampler=dict(next_update=update+1,task_order=list(TASKS),condition_weight=.25,
                    prefix_reduction='mean within actual condition',data_version=contract['student_refresh']['version']),baseline=baseline,
                baseline_optimizer=runtime.training_baseline_optimizer)



@torch.no_grad()
def final_warm_readback(runtime,context,root,example_loader):
    runtime.freeze_psi();runtime.actor.eval();rows=[]
    for index,task in enumerate(TASKS):
        if index%context.world_size!=context.rank:continue
        example=example_loader(task,90);path=example['path'];prefix=example['prefixes'][-1]
        features=runtime.teaching(task,path.demo,video_task=path.video_task)
        decision=prefix.setdefault('decision',dict(kind='final',candidates=prefix['practiced'],
            records=prefix['records'],episodes=prefix['episodes'],parent=None))
        probabilities=decision_logits(runtime,path,decision,features).softmax(-1)
        reward=[prefix['reward'][name] for name in prefix['practiced']]
        rows.append(dict(task=task,probabilities=probabilities.cpu().tolist(),references=prefix['practiced'],
            expected_successes16=16*float((probabilities*probabilities.new_tensor(reward)).sum())))
        path.native_response_cache.clear()
    gathered=[None]*context.world_size if context.is_main else None
    if context.world_size>1:dist.gather_object(rows,gathered,dst=0)
    else:gathered=[rows]
    if context.is_main:
        rows=sorted([row for shard in gathered for row in shard],key=lambda row:row['task'])
        if [row['task'] for row in rows]!=list(TASKS):raise ValueError('warm expectation requires all36 actual pairs')
        write_json_atomic(Path(root)/'analysis/final_warm90_expected_query.json',dict(complete=True,rows=rows,
            expected_successes576=sum(row['expected_successes16'] for row in rows),uniform=382.5,always_G24849=425,
            posthoc_condition_pair_bound=430,new_environment_episodes=0,checkpoint_selection=False))

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
