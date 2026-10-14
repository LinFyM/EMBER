"""Matched complete Reader/endpoint BA learning, checkpoint and node consumers."""
from pathlib import Path
from types import SimpleNamespace
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast
from .contract import OPTIMIZER
from .model import FactorLayout
from .image_compiler import ImageCompiler, SCHEMA, COMMON_SEED, LANGUAGE_SEED, HISTORY_SEED
from .image_data import ImageInputs, logical_conditions
from .learning import OperatorObjective, synchronize_gradients, memory_peak


def compiler(root, arm, device):
    root = Path(root)
    lora = load_pi05_lora_contract(root/'task_lora_contract.json')
    layout = FactorLayout(lora, load_file(str(root/'initial.safetensors')), read_json(root/'bank_scale.json')['S_G'])
    return ImageCompiler(layout, arm).to(device), lora


def load_compiler(root, arm, checkpoint, device):
    model, lora = compiler(root, arm, device)
    manifest = read_json(Path(checkpoint)/'manifest.json')
    state = torch.load(Path(checkpoint)/'state.pt', map_location='cpu', weights_only=False)
    if (not manifest['complete'] or state['schema_version'] != SCHEMA or state['arm'] != arm
            or state['next_update'] != manifest['next_update']
            or state['parameter_names'] != list(dict(model.named_parameters()))
            or state['initialization'] != dict(common_seed=COMMON_SEED, language_seed=LANGUAGE_SEED, history_seed=HISTORY_SEED, fresh=True)):
        raise ValueError('image compiler requires its complete matched initialization/schema')
    model.load_state_dict(state['compiler'])
    return model, lora, state


def gradient_credit(model):
    groups = {'Reader':model.reader, 'history':model.reader.history, 'decoder':model.decoder, 'output':model.decoder.output}
    result = {}
    for name,module in groups.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        result[name] = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients,2)))) if gradients else 0.
    return result


def save_checkpoint(root, arm, update, model, optimizer, context, run, migration):
    ranks = [None]*context.world_size if context.is_main else None
    rng = capture_rng(context)
    if context.world_size > 1: dist.gather_object(rng, ranks, dst=0)
    else: ranks = [rng]
    file = root/'arms'/arm/'checkpoints'/f'update_{update:08d}'
    if context.is_main:
        file.mkdir(parents=True, exist_ok=False)
        state = dict(schema_version=SCHEMA, arm=arm, next_update=update, compiler=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=None, scaler=None,
            sampler=dict(next_update=update, task_order=run['training_tasks'], logical_B4=4,
                task_video_stream='SeedSequence[20261014,17,stream,cycle,within/task]', task_weight=.25),
            rank_rng=ranks, topology=dict(world_size=context.world_size, logical_B4=4),
            parameter_names=list(dict(model.named_parameters())), optimizer_owner='complete Reader and six-layer decoder',
            initialization=dict(common_seed=COMMON_SEED, language_seed=LANGUAGE_SEED, history_seed=HISTORY_SEED, fresh=True),
            frozen_dependencies=run['frozen_dependencies'], physical_migration=migration)
        torch.save(state, file/'state.pt')
        write_json_atomic(file/'manifest.json', dict(schema_version=SCHEMA, arm=arm, next_update=update,
            complete=True, full_optimizer_sampler_RNG=True, parameter_count=sum(p.numel() for p in model.parameters()),
            topology=state['topology'], role='common_node' if update==2160 else 'recovery_only' if update<8640 else 'sole_normal_endpoint'))
    if context.world_size > 1: dist.barrier(device_ids=[context.local_rank])
    return file


@torch.no_grad()
def read_node(root, arm, update, model, inputs, teachers, objective, context):
    rows = []
    pool = read_json(root/'input_manifest.json')['training']
    for ordinal, entry in enumerate(pool):
        if ordinal % context.world_size != context.rank: continue
        value = inputs.get(entry['task'],entry['teacher_demo'])
        with autocast(context.device):
            generated = model(value['z_img'],value['indices'],value['history'],value['language'] if arm=='VL' else None)
            loss, baseline = objective(generated,teachers[entry['task']])
        rows.append(dict(task=entry['task'],demo=entry['teacher_demo'],frames=entry['frames'],BA=float(loss),MT_BA=float(baseline)))
    collected = [None]*context.world_size if context.is_main else None
    if context.world_size > 1: dist.gather_object(rows, collected, dst=0)
    else: collected = [rows]
    if context.is_main:
        rows = sorted([row for part in collected for row in part],key=lambda r:(r['task'],r['demo']))
        if len(rows)!=144 or len({(r['task'],r['demo']) for r in rows})!=144: raise ValueError('whole144 node denominator missing')
        write_json_atomic(root/'arms'/arm/'nodes'/f'update_{update:08d}.json',dict(complete=True,rows=rows,
            mean_BA=sum(r['BA'] for r in rows)/144,update=update,arm=arm,held_feedback=False,no_checkpoint_selection=True))
    if context.world_size > 1: dist.barrier(device_ids=[context.local_rank])


def learn(args, context):
    root, arm = args.root, args.compiler_arm
    run = read_json(root/'run_contract.json')
    if args.stop not in (2160,8640) or not 1<=context.world_size<=4: raise ValueError('bounded shared learning nodes/world required')
    model, lora = compiler(root,arm,context.device)
    optimizer = torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    initial, migration = 0, None
    if args.resume is not None:
        model, lora, saved = load_compiler(root,arm,args.resume,context.device)
        optimizer = torch.optim.AdamW(model.parameters(),**OPTIMIZER); optimizer.load_state_dict(saved['optimizer'])
        initial = saved['next_update']; old = saved['topology']['world_size']
        if context.rank < old: restore_rng(saved['rank_rng'][context.rank],context)
        else:
            value = int(np.random.SeedSequence([20261014,17,2,context.rank]).generate_state(1)[0])
            random.seed(value);np.random.seed(value);torch.manual_seed(value)
        migration = dict(checkpoint=str(args.resume),old_world=old,new_world=context.world_size,
            restored_RNG_ranks=list(range(min(old,context.world_size))),new_RNG_ranks=list(range(old,context.world_size)),bitwise_exact=False)
        del saved
    else:
        value = int(np.random.SeedSequence([20261014,17,2,context.rank]).generate_state(1)[0])
        random.seed(value);np.random.seed(value);torch.manual_seed(value)
    inputs = ImageInputs(root,context.device)
    teachers = {e['task']:load_file(e['weights'],device=str(context.device)) for e in run['teachers']}
    objective = OperatorObjective(SimpleNamespace(lora=lora),root)
    panel = read_json(root/'panel.json')
    initialize_deferred_process_group(context,rendezvous_root=root/'rendezvous'/arm)
    model.train();started=time.monotonic();first_credit=None;second_credit=None
    import json
    log = root/'metrics'/f'{arm}_rank_{context.rank}.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        for u in range(initial,args.stop):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            logical=logical_conditions(root,u,panel);assigned=inputs.assignment(logical,context.world_size)[context.rank]
            records=[]
            for entry in assigned:
                value=inputs.get(entry['task'],entry['demo'])
                with autocast(context.device):
                    generated=model(value['z_img'],value['indices'],value['history'],value['language'] if arm=='VL' else None,
                        checkpoint_layers=args.checkpoint_layers)
                    loss,baseline=objective(generated,teachers[entry['task']])
                if not torch.isfinite(loss): raise ValueError('nonfinite complete BA learning objective')
                (loss/4).backward()
                records.append(dict(**entry,BA=float(loss.detach()),frames=len(value['indices']),task_weight=.25,
                    H_records=len(value['history']['facts']),H_observations=len(value['history']['images']),
                    compiler_inputs=['z_img','indices','MT_H1_RGB_proprio_actions_executed_feedback_order']+(['exact_L_embedding'] if arm=='VL' else []),
                    execution_hidden_input=False))
            synchronize_gradients(model,context)
            credit=gradient_credit(model)
            if u==initial:first_credit=credit
            if u==initial+1:second_credit=credit
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite complete compiler gradient')
            optimizer.step();torch.cuda.synchronize(context.device)
            record=dict(update=u+1,arm=arm,rank=context.rank,logical_conditions=logical,conditions=records,
                seconds=time.monotonic()-tick,gradient_credit=credit,peak=memory_peak(context.device),
                source_frozen=True,old_meta=False,teacher_in_condition=False,learned_memory_recomputed=True)
            stream.write(json.dumps(record)+'\n');stream.flush()
            if u+1 in (9,2160,4320,6480,8640):
                save_checkpoint(root,arm,u+1,model,optimizer,context,run,migration)
            if u+1==9:
                periods=torch.tensor(time.monotonic()-started,device=context.device)
                if context.world_size>1:dist.all_reduce(periods,op=dist.ReduceOp.MAX)
                if context.is_main:
                    proof=dict(arm=arm,update=9,effective_B4=9,seconds=float(periods),world=context.world_size,
                        peak=memory_peak(context.device),first_credit=first_credit,second_credit=second_credit,
                        frozen_cache_GiB=inputs.bytes/2**30,cache_limit_GiB=inputs.limit/2**30,
                        discarded_updates=0,parameter_count=sum(p.numel() for p in model.parameters()))
                    proof.update(max_registered_H_records=max(len(h['facts']) for h in inputs.histories.values()),
                        exact_first36task_round_contains_longest_H=True,history_seed=HISTORY_SEED,
                        execution_hidden_input=False,whole_pool_includes_unique_H=True)
                    write_json_atomic(root/'arms'/arm/'first_task_round.json',proof)
                    print('EMBER_IMAGE_ROUND '+json.dumps(proof),flush=True)
    read_node(root,arm,args.stop,model,inputs,teachers,objective,context)
    if context.is_main:write_json_atomic(root/'arms'/arm/f'training_{args.stop:08d}_complete.json',dict(
        complete=True,initial=initial,next_update=args.stop,seconds=time.monotonic()-started,
        checkpoint=str(root/'arms'/arm/'checkpoints'/f'update_{args.stop:08d}'),cache_hits=inputs.hits,
        GPU_cache_loads=inputs.loads,transfer_seconds=inputs.transfer_seconds,peak=memory_peak(context.device)))
