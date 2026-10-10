"""Disposable mechanism and throughput consumers; no retained learned initial values."""
from __future__ import annotations
import time
import json
from pathlib import Path
import torch
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast
from ember.writer.model import DirectLoRAParameters
from ember.writer.function_credit import NativeFlowPrediction, mean_velocity_loss
from ember.lora import identity_lora_state, validate_lora_state, task_lora_state_dict
from .contract import OPTIMIZER, seed
from .teacher import QueryStream

def task_panel(root,task):
    return next(r for r in read_json(Path(root)/'panel.json')['tasks'] if r['task_id']==task)

def deep_components(runtime,runner):
    return dict(native=runner.components,environment_steps=runner.environment_steps,reader=runtime.components)

def _episode_request(runtime,task,state,reference,state_id,noise_root,identity,**fields):
    return dict(task=task,state=state,parameter_ref=reference,state_id=state_id,noise_root=noise_root,episode_id=identity,**fields)

def reader_profile(args,runtime,condition,raw_prefix,original,root):
    reader_profiles = []
    for chunk in args.frame_chunks:
        runtime.generator.load_state_dict(original)
        runtime.generator.meta.frame_chunk = chunk
        runtime.generator.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(runtime.device); torch.cuda.synchronize(runtime.device)
        started = time.monotonic()
        with autocast(runtime.device):
            features = runtime.generator.meta(condition,raw_prefix=raw_prefix)
            parent = runtime.generator.layout.pack(runtime.mt)
            x = torch.randn_like(parent) * runtime.generator.layout.valid
            output = runtime.generator(x, .5, parent, features, [], {'MT300': runtime.mt})
            # Disposable actual consumer loss; not an event label or retained update.
            loss = (output.float()[runtime.generator.layout.valid] + x[runtime.generator.layout.valid]).square().mean()
        loss.backward(); torch.cuda.synchronize(runtime.device)
        groups = {group: sum(float(p.grad.float().square().sum()) for p, g in
                            zip(runtime.generator.meta.values, runtime.generator.meta.groups, strict=True)
                            if g == group and p.grad is not None) ** .5 for group in ('gemma', 'action')}
        if any(value <= 0 for value in groups.values()) or any(p.grad is not None for p in runtime.policy.parameters()):
            raise ValueError('dual-meta CFM credit missing or source gradient leaked')
        reader_profiles.append(dict(frame_chunk=chunk, frames=len(condition[0]), loss=float(loss.detach()),
            seconds=time.monotonic() - started, meta_gradient_norm=groups,
            allocated_GiB=torch.cuda.max_memory_allocated(runtime.device) / 2**30,
            reserved_GiB=torch.cuda.max_memory_reserved(runtime.device) / 2**30))
        write_json_atomic(root / 'reader_profile.json', dict(reader=reader_profiles, fresh_init_restored_each_case=True,
            raw_cache_contains_contextual_Gemma=False))
        del features, output, loss
    return reader_profiles


def meta_learning_profile(runtime,task,demo,condition,raw_prefix,root):
    stream = QueryStream(runtime, task, demo, 0)
    sample, _ = stream.sample(runtime, 0)
    single = type(sample)(tuple(tree_slice_local(v, 0, 1) for v in sample.arguments), sample.target[:1], sample.action_width)
    task_factors = {'policy.' + k: v for k, v in runtime.mt.items()}
    with torch.no_grad(), autocast(runtime.device):
        function_before = torch.func.functional_call(NativeFlowPrediction(runtime.policy), task_factors, (single,), strict=False)
    meta_learning = []
    probe_optimizer = torch.optim.AdamW(runtime.generator.parameters(), **OPTIMIZER)
    for step in range(2):
        probe_optimizer.zero_grad(set_to_none=True)
        with autocast(runtime.device):
            features = runtime.generator.meta(condition,raw_prefix=raw_prefix)
            parent = runtime.generator.layout.pack(runtime.mt)
            x = torch.randn_like(parent) * runtime.generator.layout.valid
            out = runtime.generator(x, .5, parent, features, [], {'MT300': runtime.mt})
            value = (out.float()[runtime.generator.layout.valid] + x[runtime.generator.layout.valid]).square().mean()
        value.backward()
        group = {f'{g}_{side}': sum(float(p.grad.float().square().sum()) for i, (p, name) in
                    enumerate(zip(runtime.generator.meta.values, runtime.generator.meta.groups, strict=True))
                    if name == g and i % 2 == side and p.grad is not None) ** .5
                 for g in ('gemma', 'action') for side in (0, 1)}
        torch.nn.utils.clip_grad_norm_(runtime.generator.parameters(), 1., error_if_nonfinite=True)
        probe_optimizer.step(); meta_learning.append(group)
        del features, out, value
    with torch.no_grad(), autocast(runtime.device):
        function_after = torch.func.functional_call(NativeFlowPrediction(runtime.policy), task_factors, (single,), strict=False)
    isolation_error = float((function_before.float() - function_after.float()).square().mean().sqrt())
    if isolation_error > 1e-5 or any(meta_learning[1][f'{group}_0'] <= 0 for group in ('gemma', 'action')):
        raise ValueError('meta A after B update or read/execution isolation failed')
    write_json_atomic(root / 'meta_learning_profile.json', dict(two_steps=meta_learning,
        native_execution_RMSE_after_meta_update=isolation_error, source_frozen=True, disposable=True))
    del probe_optimizer
    return stream,single,sample


def teacher_profile(runtime,single,sample,root):
    owner = NativeFlowPrediction(runtime.policy)
    identity = identity_lora_state(runtime.lora, device=runtime.device)
    model = DirectLoRAParameters(identity).to(runtime.device)
    owner = NativeFlowPrediction(runtime.policy)
    with autocast(runtime.device):
        predicted = torch.func.functional_call(owner, {'policy.' + k: v for k, v in model().items()}, (single,), strict=False)
        loss = mean_velocity_loss(predicted, single.target, single.action_width)
    loss.backward()
    identity_B = sum(float(p.grad.square().sum()) for name, p in model().items() if '.lora_B.' in name) ** .5
    if identity_B <= 0:
        raise ValueError('legal nonzero-A/B0 task copy is not learnable through actual function loss')
    teacher_profiles = []
    for microbatch in (28, 56, 112):
        model = DirectLoRAParameters(runtime.mt).to(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device); torch.cuda.synchronize(runtime.device)
        tick = time.monotonic()
        try:
            for start in range(0, 112, microbatch):
                stop = min(start + microbatch, 112)
                part = type(sample)(tree_slice_local(sample.arguments, start, stop),
                                    sample.target[start:stop], sample.action_width)
                with autocast(runtime.device):
                    value = torch.func.functional_call(owner, {'policy.' + k: v for k, v in model().items()},
                                                       (part,), strict=False)
                    loss = mean_velocity_loss(value, part.target, part.action_width) * ((stop - start) / 112)
                loss.backward()
                del value, loss
            torch.cuda.synchronize(runtime.device)
            teacher_profiles.append(dict(microbatch=microbatch, seconds=time.monotonic() - tick,
                queries=112, peak_GiB=torch.cuda.max_memory_allocated(runtime.device) / 2**30, OOM=False))
        except torch.cuda.OutOfMemoryError:
            teacher_profiles.append(dict(microbatch=microbatch, seconds=time.monotonic() - tick, OOM=True))
            import gc
            del model
            gc.collect(); torch.cuda.empty_cache()
            break
        del model
    write_json_atomic(root / 'teacher_profile.json', dict(physical_batches=teacher_profiles,
                      identity_B_gradient=identity_B, source_frozen=True))
    return identity_B


def actual_consumers(args,runtime,runner,contract,task,demo,root,reader_profiles,identity_B):
    runtime.freeze_psi()
    # Parent-only real execution is safe. Fresh random G is never used as practice policy.
    task_info = next(t for t in contract['tasks'] if t['global_task_id'] == task)
    row = task_panel(args.root, task)
    request = _episode_request(runtime, task_info, runtime.mt, 'MT300', row['practice_states'][0],
                              seed(90, task), 'profile_MT_episode', remaining=task_info['horizon'] + 10,
                              snapshots=True, uses_teaching=False)
    if args.profile_history is not None:
        results=[dict(history=torch.load(args.profile_history,map_location='cpu',weights_only=False),
                      row=read_json(args.profile_history.parent/'profile.json')['actual_MT_episode'])]
    else:
        results = list(runner.run([request]))
    features = runtime.teaching(task, demo)
    before = time.monotonic()
    with autocast(runtime.device):
        candidate = runtime.generator.generate(runtime.mt, features, [], {'MT300': runtime.mt}, noise_seed=seed(91, task))
    validate_lora_state(candidate, runtime.lora)
    torch.cuda.synchronize(runtime.device)
    integration_seconds = time.monotonic() - before
    write_json_atomic(root / 'profile.json', dict(reader=reader_profiles, identity_B_gradient=identity_B,
        source_frozen=True, task_export_factor_count=len(task_lora_state_dict(runtime.policy)),
        complete_coordinate_output=len(candidate), parameter_integration_seconds=integration_seconds,
        fresh_G_not_practiced=True, actual_MT_episode=results[0]['row'], components=deep_components(runtime, runner)))
    torch.save(results[0]['history'], root / 'actual_MT_history.pt')
    return results,candidate,task_info,row


def snapshot_readout(args,runtime,runner,task_info,row,task,results):
    first = results[0]['history'].records[0]
    snapshot_result={}
    if args.profile_history is None:
        restored = _episode_request(runtime, task_info, runtime.mt, 'MT300', row['practice_states'][0],
            seed(90, task), 'profile_live_snapshot_replay', remaining=5, snapshots=False, uses_teaching=False,
            snapshot=first['snapshot'])
        replay = next(runner.run([restored]))
        actual = results[0]['history'].observations[first['post']]
        observed = replay['history'].observations[replay['history'].records[0]['post']]
        snapshot_result=dict(actual_RGB_replay_equal=torch.equal(observed['images'],actual['images']),
            proprio_RMSE=float((observed['proprio']-actual['proprio']).square().mean().sqrt()),
            replay_environment_steps=replay['row']['environment_steps'],replay_settling_steps=replay['row']['settling_steps'])
    else:
        snapshot_result=dict(existing_actual_H=str(args.profile_history), old_snapshot_not_used_for_new_labels=True,
                             live_snapshot_validation=str(args.root/'snapshot_smoke.json'))
    return snapshot_result


def head_readout(args,runtime,runner,task_info,row,task,demo,results,candidate):
    from .path import Compilation, categorical_record
    from .path import decision_logits
    identity_request=_episode_request(runtime,task_info,runtime.identity,'identity_actual',row['practice_states'][1],
        seed(92,task),'profile_identity_execution',remaining=11,snapshots=False,uses_teaching=False)
    identity_execution=next(runner.run([identity_request]))
    actual_history=results[0]['history'];actual_history.append(identity_execution['history'])
    sample_path = Compilation(task, demo, 'profile_actual_history', 0, seed(90, task),
        actual_history, {'MT300': {k: v.detach().cpu() for k, v in runtime.mt.items()},
                         'identity_actual':{k:v.detach().cpu() for k,v in runtime.identity.items()},
                         'fresh_G_prediction_only':{k:v.detach().cpu() for k,v in candidate.items()}})
    heads = {}
    for kind,candidates in [('parent',['MT300','identity_actual','STOP']),
                            ('practice',['MT300','fresh_G_prediction_only']),('final',['MT300','identity_actual'])]:
        decision=dict(kind=kind,candidates=candidates,records=len(actual_history.records),
            episodes=len(actual_history.episodes),budget=[.5,.5,.02,.06],parent='MT300' if kind=='practice' else None)
        logits=decision_logits(runtime,sample_path,decision,runtime.teaching(task,demo))
        if not bool(torch.isfinite(logits).all()):raise ValueError('three-head actual consumer is nonfinite')
        heads[kind]=dict(logits=logits.detach().cpu().tolist(),candidates=candidates,actual_state_and_history=True)
        del logits
    return heads,identity_execution


def native_credit(runtime,task_info,history):
    from ember.writer.practice import processed
    functional=DirectLoRAParameters(runtime.mt).to(runtime.device)
    first=history.records[0]
    batch=processed(runtime,history.observations[first['pre']],task_info['language'])
    noise=torch.randn(1,50,32,generator=torch.Generator().manual_seed(first['noise_seed'])).to(runtime.device)
    with torch.enable_grad():
        prediction=runtime.native_actions([functional()],batch,noise,
            batch_indices=torch.zeros(1,dtype=torch.long,device=runtime.device),checkpointed=True)
        loss=prediction.float().square().mean();loss.backward()
    native_gradient=sum(float(p.grad.square().sum()) for p in functional.parameters() if p.grad is not None)**.5
    if not native_gradient>0 or any(p.grad is not None for p in runtime.policy.parameters()):
        raise ValueError('complete ten-step checkpointed native function credit failed')
    return native_gradient


def profile(args,runtime,runner,contract):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    panel=read_json(Path(args.root)/'panel.json')
    choices=[(runtime.tasks[r['task_id']].episode_lengths[d],r['task_id'],d)
             for r in panel['tasks'] for d in r['teacher_videos']]
    _,task,demo=max(choices)
    condition=runtime.condition(task,demo)
    started=time.monotonic()
    with autocast(runtime.device):raw_prefix=runtime.generator.meta.encode_raw(condition)
    torch.cuda.synchronize(runtime.device)
    raw_seconds=time.monotonic()-started
    original={k:v.detach().cpu().clone() for k,v in runtime.generator.state_dict().items()}
    readers=reader_profile(args,runtime,condition,raw_prefix,original,root)
    data=read_json(root/'reader_profile.json')
    write_json_atomic(root/'reader_profile.json',{**data,'raw_prefix_seconds':raw_seconds})
    runtime.generator.load_state_dict(original);runtime.generator.zero_grad(set_to_none=True)
    stream,single,sample=meta_learning_profile(runtime,task,demo,condition,raw_prefix,root)
    runtime.generator.load_state_dict(original);runtime.generator.zero_grad(set_to_none=True)
    identity_B=teacher_profile(runtime,single,sample,root)
    results,candidate,task_info,row=actual_consumers(args,runtime,runner,contract,task,demo,root,readers,identity_B)
    snapshot=snapshot_readout(args,runtime,runner,task_info,row,task,results)
    heads,identity_execution=head_readout(args,runtime,runner,task_info,row,task,demo,results,candidate)
    gradient=native_credit(runtime,task_info,results[0]['history'])
    write_json_atomic(root/'snapshot_and_heads.json',dict(choices=heads,checkpointed_native_gradient=gradient,
        source_frozen=True,identity_execution=identity_execution['row'],fresh_G_not_executed=True,**snapshot))
    stream.close()
    print(json.dumps(dict(profile=str(root/'profile.json'),reader=readers)))


def tree_slice_local(value, first, last):
    if isinstance(value, torch.Tensor):
        return value[first:last]
    return type(value)(tree_slice_local(v, first, last) for v in value)


def functional_throughput(runtime,runner,event,history,task,root):
    from .teacher import functional_labels,function_credit
    labels=functional_labels(runtime,runner,event,history,task,None)
    items=labels['keep'][:4]
    if len(items)!=4:raise ValueError('existing real successful MT H must provide four keep consumers')
    results=[]
    for microbatch in (1,2,4):
        model=DirectLoRAParameters(runtime.mt).to(runtime.device)
        # Disposable small factor perturbation; never an execution policy or retained initial value.
        with torch.no_grad():model.values[0].mul_(1.05)
        torch.cuda.reset_peak_memory_stats(runtime.device);torch.cuda.synchronize(runtime.device)
        started=time.monotonic()
        loss=function_credit(runtime,model,items,microbatch=microbatch)
        torch.cuda.synchronize(runtime.device)
        norm=sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None)**.5
        if not norm>0 or any(p.grad is not None for p in runtime.policy.parameters()):
            raise ValueError('batched ten-step functional credit or source freeze failed')
        results.append(dict(microbatch=microbatch,items=4,seconds=time.monotonic()-started,
            loss=loss,gradient_norm=norm,peak_GiB=torch.cuda.max_memory_allocated(runtime.device)/2**30))
        del model
    write_json_atomic(root/'functional_credit_profile.json',dict(physical_batches=results,
        source_frozen=True,actual_parent_function_targets=True,only_disposable_factor_perturbation=True))
    return results


@torch.no_grad()
def stop_compilation_consumer(args,runtime,runner,task,panel,demo,root):
    from .path import Compilation,compile_condition,decision_logits,cpu_state
    from ember.writer.practice import History
    state=cpu_state(runtime.mt)
    empty=Compilation(task['global_task_id'],demo,'profile_STOP',0,0,History(states={'MT300':state}),{'MT300':state})
    decision=dict(kind='parent',candidates=['MT300','STOP'],records=0,episodes=0,budget=[1.,1.,0.,1/33])
    probability=decision_logits(runtime,empty,decision,runtime.teaching(empty.task,demo)).softmax(-1).detach().cpu()
    # Branch coverage from the genuine categorical head, without practicing an untrained G.
    for ordinal in range(1000):
        noise_root=seed(93,empty.task,ordinal)
        rng=torch.Generator().manual_seed(seed(30,noise_root,0))
        if int(torch.multinomial(probability,1,generator=rng))==1:break
    else:raise ValueError('engineering STOP branch was not reachable')
    path=compile_condition(runtime,runner,task,panel,demo,identity='profile_STOP',path_ordinal=0,noise_root=noise_root)
    if path.selected!='MT300' or path.environment_steps!=0 or path.attempts:
        raise ValueError('complete STOP compilation did not lock the legal full MT output')
    path.save(root/'complete_compilation')
    write_json_atomic(root/'complete_compilation_consumer.json',dict(complete=True,selected=path.selected,
        environment_steps=0,complete_task_factors=len(path.states[path.selected]),decisions=path.decisions,
        engineering_STOP_branch_only=True,no_outcome_used_for_seed=True,fresh_G_never_practiced=True))


def execution_profile(args,runtime,runner,contract):
    if args.profile_history is None:raise ValueError('focused profile needs the existing actual MT H')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    panel=read_json(args.root/'panel.json')
    choices=[(runtime.tasks[r['task_id']].episode_lengths[d],r['task_id'],d)
             for r in panel['tasks'] for d in r['teacher_videos']]
    _,task,demo=max(choices);condition=runtime.condition(task,demo)
    with autocast(runtime.device):raw=runtime.generator.meta.encode_raw(condition)
    original={k:v.detach().cpu().clone() for k,v in runtime.generator.state_dict().items()}
    readers=reader_profile(args,runtime,condition,raw,original,root)
    runtime.generator.load_state_dict(original);runtime.generator.zero_grad(set_to_none=True)
    results,_,task_info,row=actual_consumers(args,runtime,runner,contract,task,demo,root,readers,None)
    event=dict(task_id=task,event_ordinal=0,event_id='disposable_profile',parent_ref='MT300')
    functional=functional_throughput(runtime,runner,event,results[0]['history'],task_info,root)
    stop_compilation_consumer(args,runtime,runner,task_info,row,demo,root)
    write_json_atomic(root/'complete.json',dict(complete=True,reader=readers,functional=functional,
        disposable=True,retained_initial_state_changed=False))
    print(json.dumps(dict(complete=True,profile=str(root),functional=functional)))
