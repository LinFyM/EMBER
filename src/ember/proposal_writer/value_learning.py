"""Independent Phi180 BCE learning on immutable real non-held path prefixes."""
from pathlib import Path
import json
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast
from .contract import OPTIMIZER, CHECKPOINT_SCHEMA, seed
from .model import IndependentValue
from .learning import assigned_tasks, logical_tasks, synchronize_gradients, metrics
from .runtime import phase
from .value_data import ValueInputs, prefix_inputs

VALUE_SCHEMA = 'ember_independent_value_fixed_paths_v1'


def initialize_value(runtime, *, with_phi=True, checkpoint=None):
    run = read_json(runtime.root/'run_contract.json')
    source = Path(run['source_beta_checkpoint'])
    manifest = read_json(source/'manifest.json')
    state = torch.load(source/'state.pt', map_location='cpu', weights_only=False)
    if (not manifest['complete'] or state['schema_version'] != CHECKPOINT_SCHEMA
            or state['stage'] != 'RL' or state['next_update'] != 18
            or state['psi_version'] != 38349 or state['G_sampler']['next_update'] != 38349
            or state['supervision_version'] != 9):
        raise ValueError('value learning requires the original complete fixed G38349/beta18')
    runtime.generator.load_state_dict(state['psi'])
    runtime.actor.load_state_dict(state['theta'])
    runtime.version = 38349
    runtime.freeze_psi()
    runtime.actor.eval().requires_grad_(False)
    runtime.policy.eval().requires_grad_(False)
    if with_phi:
        runtime.phi = IndependentValue(runtime.actor).to(runtime.device)
        if set(map(id, runtime.phi.parameters())) & set(map(id, runtime.actor.parameters())):
            raise ValueError('Phi and beta cannot share trainable parameters')
        if checkpoint is not None:
            saved = torch.load(Path(checkpoint)/'state.pt', map_location='cpu', weights_only=False)
            if saved['schema_version'] != VALUE_SCHEMA or saved['source_beta_checkpoint'] != str(source):
                raise ValueError('Phi checkpoint source or schema changed')
            runtime.phi.load_state_dict(saved['phi'])


def phi_logits(runtime, path, prefix, features):
    responses = {key: torch.as_tensor(value, dtype=torch.float32)
                 for key, value in prefix['native_response_predictions'].items()}
    rows, states, responses, counts = prefix_inputs(runtime, path, prefix, responses)
    with phase(runtime, 'phi_forward'), autocast(runtime.device):
        logits = runtime.phi(features, rows, states, prefix['candidates'], responses, counts).float()
    if not logits.isfinite().all():
        raise ValueError('non-finite independent candidate values')
    return logits, counts


def gradient_credit(runtime):
    groups = ('head', 'counts', 'reader', 'static', 'candidate_attention', 'response_attention',
              'set_attention', 'candidate_query', 'context', 'responses', 'practiced', 'norm')
    values = {group: 0. for group in groups}
    for name, parameter in runtime.phi.named_parameters():
        group = name.split('.')[0]
        if parameter.grad is not None:
            if not parameter.grad.isfinite().all():
                raise ValueError('Phi gradient is non-finite')
            values[group] += float(parameter.grad.float().square().sum())
    if any(p.requires_grad or p.grad is not None for module in
           (runtime.generator, runtime.actor, runtime.policy) for p in module.parameters()):
        raise ValueError('G, behavior beta and source must remain fully frozen')
    return {key: value**.5 for key, value in values.items()}


def save_checkpoint(root, update, runtime, optimizer, context):
    rng = capture_rng(context)
    ranks = [None]*context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(rng, ranks, dst=0)
    else:
        ranks = [rng]
    if context.is_main:
        run = read_json(root/'run_contract.json')
        destination = root/'checkpoints'/f'update_{update:08d}'
        destination.mkdir(parents=True, exist_ok=False)
        sampler = dict(next_update=update, task_order=run['training_tasks'], task_weight=.25,
            prefix_reduction='mean', candidate_reduction='mean',
            data_stream='even epoch pair2;odd epoch RL6;SeedSequence[20261013,23,stratum,epoch]')
        state = dict(schema_version=VALUE_SCHEMA, next_update=update, psi=runtime.generator.state_dict(),
            beta=runtime.actor.state_dict(), phi=runtime.phi.state_dict(), optimizer=optimizer.state_dict(),
            optimizer_owner='phi', scheduler=None, scaler=None, sampler=sampler, rank_rng=ranks,
            topology=dict(world_size=context.world_size, logical_condition_batch=4),
            source_beta_checkpoint=run['source_beta_checkpoint'], source_G_checkpoint=run['source_G_checkpoint'],
            psi_version=38349, beta_version=18, objective='task_prefix_candidate_equal_BCE_query16',
            parameter_names=list(dict(runtime.phi.named_parameters())),
            initialization='beta18 encoding copied;counts and scalar head zero;fresh same AdamW',
            frozen_behavior_optimizer_updated=False,
            physical_rng_migration=getattr(runtime, 'value_rng_migration', None))
        torch.save(state, destination/'state.pt')
        write_json_atomic(destination/'manifest.json', dict(complete=True, schema_version=VALUE_SCHEMA,
            next_update=update, topology=state['topology'], full_G_beta_Phi_optimizer_sampler_RNG=True,
            source_G_checkpoint=run['source_G_checkpoint'], source_beta_checkpoint=run['source_beta_checkpoint'],
            checkpoint_role='sole_scientific_endpoint' if update == 180 else 'recovery_only'))
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def resume_value(path, runtime, optimizer, context):
    state = torch.load(Path(path)/'state.pt', map_location='cpu', weights_only=False)
    manifest = read_json(Path(path)/'manifest.json')
    if (not manifest['complete'] or state['schema_version'] != VALUE_SCHEMA
            or state['next_update'] not in (60, 120) or state['optimizer_owner'] != 'phi'
            or state['source_beta_checkpoint'] != read_json(runtime.root/'run_contract.json')['source_beta_checkpoint']
            or state['parameter_names'] != list(dict(runtime.phi.named_parameters()))
            or state['sampler']['next_update'] != state['next_update']):
        raise ValueError('resume requires a complete original Phi60/120 boundary')
    runtime.phi.load_state_dict(state['phi'])
    optimizer.load_state_dict(state['optimizer'])
    old_world = state['topology']['world_size']
    if context.rank < old_world:
        restore_rng(state['rank_rng'][context.rank], context)
    else:
        torch.manual_seed(seed(160, context.rank))
    runtime.value_rng_migration = dict(source_world_size=old_world, current_world_size=context.world_size,
        original_rank_rng=state['rank_rng'], new_rank_seed=seed(160, context.rank) if context.rank >= old_world else None,
        logical_batch=4, bitwise_exact=False)
    return state['next_update']


def learn_value(args, runtime, context):
    if not 1 <= context.world_size <= 4:
        raise ValueError('Phi uses one to four ranks with the original logical B4')
    initialize_value(runtime)
    runtime.phi.train()
    torch.manual_seed(seed(160, context.rank))
    optimizer = torch.optim.AdamW(runtime.phi.parameters(), **OPTIMIZER)
    if optimizer.state:
        raise ValueError('Phi must start with fresh Adam state')
    initialize_deferred_process_group(context, rendezvous_root=args.root/'rendezvous')
    start = resume_value(args.resume, runtime, optimizer, context) if args.resume else 0
    inputs = ValueInputs(runtime)
    for update in range(start, 180):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(runtime.device)
        optimizer.zero_grad(set_to_none=True)
        records = []
        for task in assigned_tasks(context, update, args.root):
            entry, path, prefixes, labels = inputs.example(task, update)
            features = runtime.teaching(task, path.demo, video_task=path.video_task)
            credit = []
            for prefix in prefixes:
                logits, counts = phi_logits(runtime, path, prefix, features)
                y = logits.new_tensor([labels[key] for key in prefix['candidates']])
                if (y < 0).any() or (y > 1).any():
                    raise ValueError('query16 Bernoulli labels must be success fractions')
                loss = F.binary_cross_entropy_with_logits(logits, y)
                (loss/(4*len(prefixes))).backward()
                credit.append(dict(records=prefix['records'], episodes=prefix['episodes'],
                    candidates=prefix['candidates'], counts=counts, query_success_rates=y.detach().cpu().tolist(),
                    logits=logits.detach().cpu().tolist(), BCE=float(loss.detach())))
                del logits, loss
            records.append(dict(task=task, kind=entry['kind'], path_source=entry['path'],
                prefix_count=len(prefixes), actual_prefix_credit=credit,
                query_excluded_from_H_and_Phi_inputs=True))
            del features
        synchronize_gradients(runtime.phi, context)
        credit = gradient_credit(runtime)
        norm = torch.nn.utils.clip_grad_norm_(runtime.phi.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(runtime.device)
        record = dict(update=update+1, logical_task_batch=logical_tasks(update, args.root), conditions=records,
            gradient_norm=float(norm), module_gradient_credit=credit, seconds=time.monotonic()-started,
            G_beta_source_frozen=True, new_environment_episodes=0, new_training_queries=0)
        metrics(args.root, 'phi', record, context)
        if update in (start, start+1):
            from .learning import memory_peak
            first = dict(record, rank=context.rank, world_size=context.world_size, peak=memory_peak(runtime.device),
                         frame_chunk=runtime.frame_chunk, response_microbatch=runtime.response_microbatch)
            print('EMBER_VALUE_FIRST '+json.dumps(first), flush=True)
        if update+1 in (60, 120, 180):
            save_checkpoint(args.root, update+1, runtime, optimizer, context)
