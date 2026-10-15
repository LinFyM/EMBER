"""One timed operator-learning phase, retaining complete state and true exit events."""
from pathlib import Path
from types import SimpleNamespace
import datetime,json,random,time
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.proposal_writer.learning import synchronize_gradients
from ember.lora import identity_lora_state
from .contract import OPTIMIZER,MAX_UPDATE
from .data import topology,logical_update
from .model import ControlSpace
from .learning import References,OperatorLoss,local_credit,evaluate_pool
from .checkpoint import save,restore,gather


def emit(**values):print('EMBER_WORK_EVENT '+json.dumps(values),flush=True)


def measure_credit(model,optimizer,references,objective,groups,context,active,pack):
    local=groups[len(groups)*context.rank//active:len(groups)*(context.rank+1)//active] if context.rank<active else []
    attempts=[];started=time.monotonic()
    while True:
        optimizer.zero_grad(set_to_none=True);failed=None;value=0.;identity=0
        try:value,identity=local_credit(model,references,objective,local,pack)
        except torch.cuda.OutOfMemoryError:failed='OOM'
        except Exception as error:failed=repr(error)
        statuses=gather(failed,context)
        if any(e and e!='OOM' for e in statuses):raise RuntimeError(f'operator credit failed: {statuses}')
        if not any(statuses):break
        attempts.append(dict(pack=pack,rank_failures=statuses))
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if pack==1:raise RuntimeError('operator credit still OOM at one point')
        pack=max(1,pack//2)
    synchronize_gradients(model,context)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize(context.device)
    payload=dict(loss=value,identity_points=identity,rank=context.rank,pack=pack,OOM=attempts,
        seconds=time.monotonic()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated(context.device)/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved(context.device)/2**20)
    measures=gather(payload,context)
    return dict(operator_loss=sum(m['loss'] for m in measures),gradient_norm=float(norm),rank_measures=measures,
        active_ranks=active,minimum_pack=min(m['pack'] for m in measures),
        seconds=max(m['seconds'] for m in measures),identity_points=sum(m['identity_points'] for m in measures))


def select_physical(physical):
    configs={}
    for row in physical['profiles']:configs.setdefault((row['active_ranks'],row['minimum_pack']),[]).append(row['seconds'])
    ranks,pack=min(configs,key=lambda c:float(np.median(configs[c])))
    physical.update(recommended_world_size=ranks,point_pack=pack,
        selection='median actual useful update wall; no scientific checkpoint choice',
        comparisons=[dict(active_ranks=r,point_pack=p,median_seconds=float(np.median(v)),points_per_second=8/float(np.median(v)))
                     for (r,p),v in configs.items()])


def pool_read(root,label,update,model,references,objective,bank,context,pack):
    path=root/'diagnostics'/f'{label}.json'
    if path.exists():
        result=read_json(path)
        if result['global_update']!=update:raise RuntimeError('a full-pool read survived without its matching complete checkpoint')
        return result
    started=time.monotonic();result=evaluate_pool(model,references,objective,bank,context,pack)
    result.update(global_update=update,read_wall_seconds=time.monotonic()-started)
    if context.is_main:write_json_atomic(path,result)
    if context.world_size>1:dist.barrier()
    return result


def recover_clock(root,clock):
    path=root/'training_clock.json'
    if not path.exists():return clock
    saved=read_json(path)
    clock['training_seconds']=max(clock['training_seconds'],saved['training_seconds'])
    pending=saved.get('pending_start_UTC')
    if pending:
        started=datetime.datetime.fromisoformat(pending).timestamp()
        exits=[read_json(p) for p in (root/'jobs').glob('operator_*/attempt_*/exit.json')]
        ends=[datetime.datetime.fromisoformat(r['end_utc']).timestamp() for r in exits
              if datetime.datetime.fromisoformat(r['start_utc']).timestamp()<=started<=datetime.datetime.fromisoformat(r['end_utc']).timestamp()]
        if not ends:raise RuntimeError('interrupted training has no defensible job exit bound yet')
        cost=min(ends)-started;clock['training_seconds']+=cost
        if not (root/'interrupted_clock_recovery.json').exists():
            write_json_atomic(root/'interrupted_clock_recovery.json',dict(pending_start_UTC=pending,
                added_training_seconds_upper_bound=cost,actual_exit_bound=True))
    return clock


def complete_training(root,update,clock,stop,physical):
    contract=read_json(root/'run_contract.json');contract['endpoint_update']=update
    write_json_atomic(root/'run_contract.json',contract)
    write_json_atomic(root/'training_completion.json',dict(complete=True,endpoint_update=update,
        phase_updates=update-144,clock=clock,stop=stop,physical=physical))


def record_measures(root,update,physical,clock,context,measures):
    offset=update-145
    if offset<8:physical['profiles'].append({k:measures[k] for k in ('active_ranks','minimum_pack','seconds','rank_measures')})
    if offset==7:
        select_physical(physical)
        if context.is_main:
            write_json_atomic(root/'physical_execution.json',physical);emit(kind='throughput_selected',physical=physical)
    if context.is_main:
        with (root/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(measures)+'\n')
        write_json_atomic(root/'training_clock.json',clock)
        if offset<8:emit(kind='useful_training_update',update=update,seconds=measures['seconds'],operator_loss=measures['operator_loss'])


def finish_cycle(root,update,clock,stop,physical,model,optimizer,scheduler,references,objective,bank,context,pack,initial):
    check=clock['first_check_update'] is None and clock['training_seconds']>=300
    if check:
        read=pool_read(root,'first_check',update,model,references,objective,bank,context,pack)
        clock['first_check_update']=update
        clock['first_check_improvement']=1-read['weighted_error']/initial['weighted_error']
        if clock['first_check_improvement']<.05:stop.update(complete=True,reason='first_check_improvement_below_5_percent')
    if clock['training_seconds']>=900:stop.update(complete=True,reason=stop['reason'] or 'training_wall_900_seconds')
    if update==MAX_UPDATE:stop.update(complete=True,reason='maximum_10800_new_updates')
    if stop['complete']:
        if check:
            if context.is_main:write_json_atomic(root/'diagnostics/endpoint.json',read)
        else:pool_read(root,'endpoint',update,model,references,objective,bank,context,pack)
    if stop['complete'] or check or (update-144)%90==0 or update==153:
        save(root,update,model,optimizer,scheduler,context,physical,clock,stop,keep=stop['complete'] or check)
    if stop['complete'] and context.is_main:complete_training(root,update,clock,stop,physical)


def discard_unsaved_metrics(root,start):
    metrics=root/'training_metrics.jsonl'
    if not metrics.exists():return
    rows=[json.loads(line) for line in metrics.read_text().splitlines()]
    discarded=[r for r in rows if r['update']>start]
    if discarded:
        write_json_atomic(root/f'discarded_unsaved_after_{start:08d}_{time.time_ns()}.json',discarded)
        metrics.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['update']<=start))


def train(args):
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    root=args.root;lora=topology()
    model=ControlSpace(lora,identity_lora_state(lora),read_json(root/'units.json')).to(context.device)
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    initialize_deferred_process_group(context,rendezvous_root=root/'ipc')
    state=restore(root,model,optimizer,scheduler,context);start=state['next_update']
    physical=state['physical'] if state.get('phase')=='operator_from144' else dict(profiles=[],point_pack=1,recommended_world_size=context.world_size)
    clock=state.get('clock',dict(training_seconds=0.,first_check_update=None,first_check_improvement=None))
    stop=state.get('stop',dict(complete=False,reason=None))
    if stop['complete']:
        if context.is_main:complete_training(root,start,clock,stop,physical)
        if context.world_size>1:dist.destroy_process_group()
        return
    clock=recover_clock(root,clock)
    references=References(SimpleNamespace(device=context.device,identity=model.layout.unpack(model.layout.center)),model.layout.names)
    objective=OperatorLoss(lora,read_json(root/'operator_constants.json'));bank=read_json(root/'bank.json')
    if context.is_main:discard_unsaved_metrics(root,start)
    from ember.reference.accounting import host_resources
    resources=gather(host_resources(),context)
    if context.is_main:write_json_atomic(root/f'training_resources_start_{start:08d}_world{context.world_size}.json',resources)
    initial=pool_read(root,'start144',144,model,references,objective,bank,context,1)
    try:
        for update in range(start,MAX_UPDATE):
            offset=update-144
            if offset<8:
                ranks,pack=((6,1),(4,2),(2,4),(1,8))[offset%4];active=min(context.world_size,ranks)
            else:active,pack=context.world_size,physical['point_pack']
            torch.cuda.reset_peak_memory_stats(context.device)
            if context.is_main:
                write_json_atomic(root/'training_clock.json',dict(clock,pending_start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            step_started=time.monotonic();groups=logical_update(bank,update)
            measures=measure_credit(model,optimizer,references,objective,groups,context,active,pack)
            scheduler.step();measures['seconds']=max(gather(time.monotonic()-step_started,context))
            clock['training_seconds']+=measures['seconds']
            measures.update(update=update+1,global_update=update+1,phase_update=update+1-144,
                groups=groups,logical_parameter_pairs=8,training_seconds=clock['training_seconds'])
            record_measures(root,update+1,physical,clock,context,measures)
            if (update+1)%9:continue
            finish_cycle(root,update+1,clock,stop,physical,model,optimizer,scheduler,references,objective,bank,context,pack,initial)
            if stop['complete'] or (args.profile_only and update+1==153):break
    finally:
        resources=gather(host_resources(),context)
        if context.is_main:write_json_atomic(root/f'training_resources_exit_{start:08d}_world{context.world_size}.json',resources)
        if context.world_size>1:dist.destroy_process_group()
