"""Finite real-context diagnostic; reuse the canonical endpoint and BA objective."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast

from .contract import TASKS, CHECKPOINT_SCHEMA, G_OBJECTIVE, OPTIMIZER, seed, materialize_panel
from .learning import OperatorObjective, logical_tasks, synchronize_gradients, memory_peak
from .model import FactorLayout, FactorEncoder
from .path import load_history

SCHEMA = 'ember_frozen_real_context_endpoint_fit_v1'
COMMON_SOURCE = Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/'
    'operator_effect_learning1296_query16/analysis/main_scientific_readout/operator1296_common_conditional.json')


def source_state(root):
    contract = read_json(Path(root)/'run_contract.json')
    if contract['schema_version'] != SCHEMA:
        raise ValueError('frozen-context diagnostic needs its explicit gradient-owner contract')
    path = Path(contract['source_checkpoint'])
    manifest = read_json(path/'manifest.json')
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    if (manifest['schema_version'] != CHECKPOINT_SCHEMA or state['schema_version'] != CHECKPOINT_SCHEMA
            or not manifest['complete'] or not manifest['full_optimizer_rng_sampler']
            or manifest['stage'] != 'G' or state['stage'] != 'G'
            or manifest['next_update'] != 432 or state['next_update'] != 432
            or state['psi_version'] != 432 or state['G_sampler']['next_update'] != 432
            or state['sampler']['next_update'] != 432 or state['supervision_version'] != 0
            or state['G_objective'] != G_OBJECTIVE or manifest['topology'] != state['topology']
            or state['topology']['task_order'] != list(TASKS)
            or state['topology']['logical_condition_batch'] != 4
            or len(state['rank_rng']) != state['topology']['world_size']):
        raise ValueError('only the complete original v9 CP432 MT-only bank0 is importable')
    return contract, state


def prepare(args):
    root, previous = args.root, args.reuse_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    old = read_json(previous/'run_contract.json')
    if old.get('schema_version') == SCHEMA:
        return prepare_refinement(args, previous, checkpoint, old)
    if (root.exists() or checkpoint != previous/'checkpoints/G/update_00000432'
            or old['checkpoint_schema'] != CHECKPOINT_SCHEMA or old['G_bank_mode'] != 'MT_only'
            or old['G_bank_version'] != 0 or old['training_tasks'] != list(TASKS)):
        raise ValueError('new root and the original complete candidate CP432 are required')
    panel = materialize_panel(previous)
    if any(len(row['policy_videos']) != 4 for row in panel['tasks']):
        raise ValueError('all36 original four-video pools must be retained')
    links = ['initial.safetensors', 'task_lora_contract.json', 'bank_scale.json', 'operator_scale.json',
             'coordinates.json', 'panel.json', 'base/merged_targets.safetensors']
    for task in TASKS:
        for name in ('teachers', 'bootstrap'):
            leaf = 'E.json' if name == 'teachers' else 'complete.json'
            path = previous/name/f'task_{task:04d}'/leaf
            record = read_json(path)
            if not record['complete'] or (name == 'bootstrap' and record['reference'] != 'MT300'):
                raise ValueError('complete original E and real MT bootstrap required')
            if not Path(record['weights' if name == 'teachers' else 'history']).is_file():
                raise ValueError('original supervision/history asset missing')
            links.append(str(path.relative_to(previous)))
    root.mkdir(parents=True)
    for name in links:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((previous/name).resolve(strict=True))
    for name in ('cuda', 'triton', 'huggingface', 'tmp'):
        (root/'cache'/name).mkdir(parents=True)
    write_json_atomic(root/'run_contract.json', dict(schema_version=SCHEMA,
        source_checkpoint=str(checkpoint), source_root=str(previous), source_schema=CHECKPOINT_SCHEMA,
        design='docs/designs/video_guided_proposal_writer_20261010.md#16', training_tasks=list(TASKS),
        environment=old['environment'], condition_mapping=old['student_refresh']['condition_mapping'],
        frozen_components='CP432 meta/Reader/static/layout/theta; source/MT always frozen',
        trainable_owner='generator.endpoint only, all six original layers/input/addresses/output',
        optimizer=OPTIMIZER, G_objective=G_OBJECTIVE, logical_condition_batch=4, initial_cursor=432,
        gradient_owner_migration=True, exact_resume=False, bank_mode='MT_only', bank_version=0,
        condition_count=144, videos_per_task=4, new_environment_episodes=0,
        nodes=dict(first_seconds=300, maximum_seconds=900, round_updates=9,
                   first_stop_loss_at_least=.90, first_stop_loss_at_most=.10, substantial_fit=.50),
        terminal_selection='registered elapsed-time nodes only; no best historical selection',
        teacher_common_reference=str(COMMON_SOURCE), added_peak_GiB=8,
        limits=dict(added_peak_GiB=8,carry_in_GPU_hours=0.,GPU_hours=None,GPU_hours_hard_cap_active=False),
        bounded_stop='one cache/baseline/short fit; no control rollout, Actor, held or automatic continuation'))
    source_state(root)


def prepare_refinement(args, previous, checkpoint, old):
    complete = read_json(previous/'complete.json')
    manifest = read_json(checkpoint/'manifest.json')
    baseline = .5748958587646484
    if (args.root.exists() or checkpoint != previous/'checkpoints/update_00008883'
            or not complete['complete'] or complete['terminal_update'] != 8883
            or complete['terminal_node'] != 'terminal_15min'
            or Path(complete['checkpoint']).resolve() != checkpoint
            or complete['whole_pool']['mean_loss'] != baseline
            or manifest['schema_version'] != SCHEMA or not manifest['complete']
            or not manifest['full_endpoint_optimizer_rng_sampler']
            or manifest['topology'] != dict(world_size=4, logical_condition_batch=4)
            or not read_json(previous/'contexts/complete.json')['complete']):
        raise ValueError('one new bounded root requires the original complete8883 and144 cache')
    args.root.mkdir(parents=True)
    for name in ('initial.safetensors', 'task_lora_contract.json', 'bank_scale.json', 'operator_scale.json',
                 'coordinates.json', 'panel.json', 'base', 'teachers', 'bootstrap', 'contexts'):
        (args.root/name).symlink_to((previous/name).resolve(strict=True))
    for name in ('cuda', 'triton', 'huggingface', 'tmp'):
        (args.root/'cache'/name).mkdir(parents=True)
    contract = dict(old, design='docs/designs/video_guided_proposal_writer_20261010.md#18',
        resume_checkpoint=str(checkpoint), resume_schema=SCHEMA, previous_fit_root=str(previous), initial_cursor=8883,
        gradient_owner_migration=False, full_frozen_endpoint_resume=True, exact_resume=False,
        nodes=dict(first_seconds=600, maximum_seconds=1800, round_updates=9,
                   first_stop_relative_loss_at_least=.95, reference_loss=baseline, substantial_fit=.50),
        materialize_original36_only=True, initial_loss_reused=str(previous/'analysis/terminal_15min.json'),
        terminal_selection='actual elapsed-time stop only; no best node or loss-threshold pursuit',
        bounded_stop='one bounded refinement and one terminal paired576; no third cache block or automatic successor')
    write_json_atomic(args.root/'run_contract.json', contract)
    source_state(args.root)


def resume_endpoint_state(path, contract, names):
    path = Path(path); manifest = read_json(path/'manifest.json')
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    update = contract['initial_cursor']
    if (manifest['schema_version'] != SCHEMA or state['schema_version'] != SCHEMA
            or not manifest['complete'] or not manifest['full_endpoint_optimizer_rng_sampler']
            or manifest['next_update'] != update or state['next_update'] != update
            or Path(manifest['source_checkpoint']).resolve() != Path(state['frozen_checkpoint']).resolve()
            or manifest['topology'] != state['topology'] or len(state['rank_rng']) != state['topology']['world_size']
            or state['topology'] != dict(world_size=4, logical_condition_batch=4)
            or state['sampler'] != dict(next_update=update, logical_batch=4, task_order=list(TASKS))
            or Path(state['frozen_checkpoint']).resolve() != Path(contract['source_checkpoint']).resolve()
            or state['G_objective'] != G_OBJECTIVE or state['gradient_owner'] != 'endpoint only'
            or state['parameter_names'] != names or state['optimizer_scope'] != names
            or len(state['optimizer']['state']) != len(names)
            or len(state['optimizer']['param_groups']) != 1
            or len(state['optimizer']['param_groups'][0]['params']) != len(names)
            or any(state['optimizer']['param_groups'][0][key] != value for key, value in OPTIMIZER.items())
            or any(int(value['step']) != update for value in state['optimizer']['state'].values())):
        raise ValueError('full frozen-endpoint Adam/sampler/topology/RNG ownership changed')
    return state


def decoder_assets(root, device):
    root = Path(root)
    lora = load_pi05_lora_contract(root/'task_lora_contract.json')
    initial = load_file(str(root/'initial.safetensors'))
    validate_lora_state(initial, lora)
    layout = FactorLayout(lora, initial, read_json(root/'bank_scale.json')['S_G']).to(device)
    decoder = FactorEncoder(layout, 6, endpoint=True).to(device)
    return lora, layout, decoder


def import_endpoint(root, device):
    contract, state = source_state(root)
    lora, layout, decoder = decoder_assets(root, device)
    layout.load_state_dict({name.removeprefix('layout.'): value for name, value in state['psi'].items()
                            if name.startswith('layout.')}, strict=True)
    names = ['endpoint.'+name for name, _ in decoder.named_parameters()]
    if names != [name for name in state['parameter_names']['psi'] if name.startswith('endpoint.')]:
        raise ValueError('actual endpoint parameter names/order differ from the source owner')
    decoder.load_state_dict({name.removeprefix('endpoint.'): value for name, value in state['psi'].items()
                             if name.startswith('endpoint.')}, strict=True)
    old = state['generator_optimizer']; groups = old['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(state['parameter_names']['psi']):
        raise ValueError('source Adam parameter group must match the saved explicit names')
    if any(groups[0][key] != value for key, value in OPTIMIZER.items()):
        raise ValueError('source Adam configuration changed')
    source_ids = dict(zip(state['parameter_names']['psi'], groups[0]['params'], strict=True))
    optimizer = torch.optim.AdamW(decoder.parameters(), **OPTIMIZER)
    if contract.get('resume_checkpoint'):
        resumed = resume_endpoint_state(contract['resume_checkpoint'], contract, names)
        decoder.load_state_dict(resumed['endpoint'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        resumed['_frozen_tensor_count'] = len(state['parameter_names']['psi'])-len(names)
        return contract, resumed, lora, layout, decoder, optimizer, names
    new = optimizer.state_dict(); new_ids = new['param_groups'][0]['params']
    new['state'] = {new_id: old['state'][source_ids[name]] for new_id, name in zip(new_ids, names, strict=True)}
    new['param_groups'][0] = {**groups[0], 'params': new_ids}
    optimizer.load_state_dict(new)
    if any(int(value['step']) != 432 for value in optimizer.state.values()):
        raise ValueError('all endpoint Adam states must inherit step432')
    return contract, state, lora, layout, decoder, optimizer, names


def cache_contexts(args, context):
    from .runtime import Runtime, finish_phases
    from .run import frame_chunk_for_rank
    contract, state = source_state(args.root)
    runtime = Runtime(context.device, root=args.root, asset_root=args.asset_root,
        frame_chunk=frame_chunk_for_rank(args.frame_chunk, context))
    started = time.monotonic(); records = []
    try:
        if state['parameter_names']['psi'] != list(dict(runtime.generator.named_parameters())):
            raise ValueError('real context producer differs from the complete v9 source')
        runtime.generator.load_state_dict(state['psi']); runtime.actor.load_state_dict(state['theta'])
        runtime.version = state['psi_version']; runtime.freeze_psi(); runtime.actor.requires_grad_(False)
        del state
        panel = materialize_panel(args.root)['tasks']
        tasks = [row for i, row in enumerate(panel) if i % args.shard_count == args.shard_index]
        for row in tasks:
            task = row['task_id']; bootstrap = read_json(args.root/'bootstrap'/f'task_{task:04d}'/'complete.json')
            history = load_history(bootstrap['history'])
            if (not history.records or not history.episodes or set(history.states) != {'MT300'}
                    or any(record['parameter_ref'] != 'MT300' for record in history.records)):
                raise ValueError('cache must read the original nonempty MT-only real H')
            with torch.no_grad(), autocast(context.device):
                rows = history.model_rows(runtime)
                for demo in row['policy_videos']:
                    tick = time.monotonic(); features = runtime.teaching(task, demo)
                    memory = runtime.generator.context(features, rows, history.states)
                    if memory.ndim != 2 or memory.shape[1] != 256 or not torch.isfinite(memory).all():
                        raise ValueError('complete real context has invalid shape/value')
                    path = args.root/'contexts'/f'task_{task:04d}_demo_{demo:02d}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists(): raise ValueError('completed context must not be silently regenerated')
                    record = dict(task=task, demo=demo, history=bootstrap['history'], producers=list(history.states),
                        frames=len(features['z']), H_records=len(history.records), memory_shape=list(memory.shape),
                        memory_dtype=str(memory.dtype), source_checkpoint=contract['source_checkpoint'])
                    torch.save(dict(**record, memory=memory.detach().cpu().contiguous()), path)
                    torch.cuda.synchronize(context.device); record['seconds'] = time.monotonic()-tick
                    record['bytes'] = path.stat().st_size; records.append(record)
                    del features, memory
            runtime.feature_cache.clear(); runtime.raw_cache.clear()
            runtime.feature_cache_size = runtime.raw_cache_size = 0
            del history, rows
        finish_phases(runtime)
        write_json_atomic(args.root/'contexts'/f'shard_{args.shard_index:02d}_complete.json', dict(
            complete=True, rows=records, seconds=time.monotonic()-started, components=runtime.components,
            peak=memory_peak(context.device), frozen_all_frontend=True, new_environment_episodes=0))
    finally: runtime.close()


def load_pool(root, contract, device):
    pool = {}; teachers = {}
    for row in materialize_panel(root)['tasks']:
        task = row['task_id']; E = read_json(root/'teachers'/f'task_{task:04d}'/'E.json')
        teachers[task] = load_file(E['weights'], device=str(device))
        for demo in row['policy_videos']:
            value = torch.load(root/'contexts'/f'task_{task:04d}_demo_{demo:02d}.pt',
                               map_location='cpu', weights_only=False)
            if (value['task'] != task or value['demo'] != demo or value['source_checkpoint'] != contract['source_checkpoint']
                    or value['producers'] != ['MT300'] or value['H_records'] <= 0):
                raise ValueError('context identity or real MT-H provenance changed')
            pool[task, demo] = value['memory'].to(device)
    if len(pool) != 144: raise ValueError('all144 actual contexts required')
    return pool, teachers


@torch.no_grad()
def readout(root, label, decoder, layout, objective, pool, teachers, contract, context, *, save_endpoints,
            update=None, elapsed=None):
    tick = time.monotonic(); rows = []; decoder.eval()
    for i, task in enumerate(TASKS):
        if i % context.world_size != context.rank: continue
        for (owner, demo), memory in pool.items():
            if owner != task: continue
            with autocast(context.device): prediction = decoder(torch.zeros_like(layout.center), memory=memory, checkpoint_layers=False)
            generated = layout.unpack(layout.center+layout.scale*prediction.float())
            loss, baseline = objective(generated, teachers[task])
            if not torch.isfinite(loss): raise ValueError('nonfinite whole-pool BA readout')
            record = dict(task=task, demo=demo, loss=float(loss), MT_loss=float(baseline))
            original_demo = contract['condition_mapping'][str(task)]['teacher_demo']
            if save_endpoints and (not contract.get('materialize_original36_only') or demo == original_demo):
                path = root/'endpoints'/label/f'task_{task:04d}_demo_{demo:02d}.safetensors'
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file({name: value.cpu().contiguous() for name, value in generated.items()}, str(path))
                record['weights'] = str(path)
            rows.append(record)
    results = [None]*context.world_size if context.is_main else None
    if context.world_size > 1: dist.gather_object(rows, results, dst=0)
    else: results = [rows]
    mean = torch.zeros((), device=context.device)
    if context.is_main:
        rows = sorted([row for shard in results for row in shard], key=lambda row:(row['task'],row['demo']))
        expected = {(task, demo) for task, demo in pool}
        if len(rows) != 144 or {(r['task'],r['demo']) for r in rows} != expected:
            raise ValueError('whole-pool readout must contain exactly144 unique actual conditions')
        means = {str(task):float(np.mean([r['loss'] for r in rows if r['task']==task])) for task in TASKS}
        roles = {t['global_task_id']:t['suite'] for t in contract['environment']['tasks']}
        groups = {group:float(np.mean([r['loss'] for r in rows if roles[r['task']]==group])) for group in sorted(set(roles.values()))}
        mean.fill_(np.mean(list(means.values())))
        write_json_atomic(root/'analysis'/f'{label}.json', dict(complete=True, rows=rows, task_means=means,
            mean_loss=float(mean), train24=float(np.mean([means[str(t)] for t in TASKS if t<40])),
            support12=float(np.mean([means[str(t)] for t in TASKS if t>=40])), per_suite=groups,
            seconds=time.monotonic()-tick, source_checkpoint=contract['source_checkpoint'],
            four_videos_equal_weight=True, control_evidence=False, new_environment_episodes=0,
            update=update, training_elapsed_seconds=elapsed,
            added_updates=None if update is None else update-contract['initial_cursor']))
    if context.world_size > 1: dist.broadcast(mean, src=0)
    decoder.train()
    return float(mean)


def demo_for_update(task, update, panel):
    rng = np.random.default_rng(seed(40, task, update))
    rng.integers(1); rng.integers(1)  # Original one-MT-parent/one-H native consumer calls.
    return panel[task]['policy_videos'][int(rng.integers(4))]


def fit_step(update, decoder, optimizer, layout, objective, pool, teachers, panel, context):
    tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); records = []
    for i, task in enumerate(logical_tasks(update)):
        if i % context.world_size != context.rank: continue
        demo = demo_for_update(task, update, panel)
        with autocast(context.device):
            prediction = decoder(torch.zeros_like(layout.center), memory=pool[task,demo], checkpoint_layers=False)
        generated = layout.unpack(layout.center+layout.scale*prediction.float())
        loss, _ = objective(generated, teachers[task]); (loss/4).backward()
        records.append(dict(task=task, demo=demo, operator_loss=float(loss.detach())))
    synchronize_gradients(decoder, context)
    norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
    first_step = time.monotonic(); optimizer.step(); torch.cuda.synchronize(context.device)
    steps = [int(value['step']) for value in optimizer.state.values()]
    if len(steps) != 166 or min(steps) != update+1 or max(steps) != update+1:
        raise ValueError('all166 endpoint Adam states must advance with the actual logical cursor')
    record = dict(update=update+1, events=records, logical_tasks=logical_tasks(update), gradient_norm=float(norm),
                  seconds=time.monotonic()-tick, endpoint_only=True, peak=memory_peak(context.device),
                  Adam_step_min=min(steps),Adam_step_max=max(steps),trainable_parameters=len(steps))
    path = panel['_root']/'metrics'/f'endpoint_rank_{context.rank}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream: stream.write(json.dumps(record)+'\n')
    return first_step


def save_terminal(root, update, decoder, optimizer, names, contract, context, node, elapsed):
    rngs = [None]*context.world_size if context.is_main else None
    if context.world_size > 1: dist.gather_object(capture_rng(context), rngs, dst=0)
    else: rngs = [capture_rng(context)]
    if context.is_main:
        path = root/'checkpoints'/f'update_{update:08d}'; path.mkdir(parents=True, exist_ok=False)
        state = dict(schema_version=SCHEMA, endpoint=decoder.state_dict(), optimizer=optimizer.state_dict(),
            parameter_names=names, next_update=update, sampler=dict(next_update=update,logical_batch=4,task_order=list(TASKS)),
            rank_rng=rngs, topology=dict(world_size=context.world_size,logical_condition_batch=4),
            frozen_checkpoint=contract['source_checkpoint'], frozen_components=contract['frozen_components'],
            gradient_owner='endpoint only', optimizer_scope=names, exact_resume=False, G_objective=G_OBJECTIVE)
        torch.save(state, path/'state.pt')
        write_json_atomic(path/'manifest.json', dict(schema_version=SCHEMA,complete=True,next_update=update,
            full_endpoint_optimizer_rng_sampler=True,frozen_components_referenced=True,
            source_checkpoint=contract['source_checkpoint'],topology=state['topology']))
        result = read_json(root/'analysis'/f'{node}.json')
        mapping = contract['condition_mapping']
        final = [r for r in result['rows'] if r['demo']==mapping[str(r['task'])]['teacher_demo']]
        if len(final) != 36: raise ValueError('original36 fixed teaching conditions must each have one full terminal LoRA')
        initial = contract['initial_cursor']
        write_json_atomic(root/'complete.json', dict(complete=True,schema_version=SCHEMA,checkpoint=str(path),
            initial_update=initial,terminal_update=update,added_updates=update-initial,added_conditions=4*(update-initial),
            exposure_per_task=(update-initial)*4//36,terminal_node=node,training_elapsed_seconds=elapsed,
            whole_pool=result,original36_endpoints=final,new_environment_episodes=0,
            control_pass=False,substantial_teacher_fit=result['mean_loss']<=.50))


@torch.no_grad()
def terminal_operator_decomposition(root, teachers, objective, device):
    """Small Grams on the36 original panel endpoints, inside the terminal readout."""
    completion = read_json(root/'complete.json'); rows = completion['original36_endpoints']
    generated = [load_file(row['weights'], device=str(device)) for row in rows]
    reference = read_json(COMMON_SOURCE)
    if (Path(reference['root'])/'operator_scale.json').resolve() != (root/'operator_scale.json').resolve():
        raise ValueError('existing teacher-mean arithmetic requires the identical frozen D38 authority')
    mean_energy, mean_cross = [], []
    for name, scaling, denominator in objective.layers:
        A = torch.cat([state[name+LORA_A_SUFFIX] for state in generated])
        B = torch.cat([state[name+LORA_B_SUFFIX] for state in generated],1)
        E_A = torch.cat([teachers[row['task']][name+LORA_A_SUFFIX] for row in rows])
        E_B = torch.cat([teachers[row['task']][name+LORA_B_SUFFIX] for row in rows],1)
        unit = scaling**2/(len(rows)**2*denominator)
        mean_energy.append(float(((B.T@B)*(A@A.T)).sum()*unit))
        mean_cross.append(float(((B.T@E_B)*(A@E_A.T)).sum()*unit))
    previous = reference['aggregate']['all36']['G1296']
    energy = float(np.mean(mean_energy)); cross = float(np.mean(mean_cross))
    public_error = energy+previous['E_common_energy']-2*cross
    total = float(np.mean([row['loss'] for row in rows])); centered = total-public_error
    write_json_atomic(root/'analysis'/'terminal_operator_decomposition.json', dict(
        scope='36 original fixed teacher_demo endpoints; full144 losses remain the stopping criterion',
        actual_mean_BA=True,G_common_energy=energy,common_error=public_error,centered_error=centered,
        total_error=total,common_gain=previous['E_common_energy']-public_error,
        condition_gain=previous['E_variation']-centered,
        unconstrained_teacher_operator_mean_error=previous['E_variation'],
        reused_teacher_mean_arithmetic=str(COMMON_SOURCE),teacher_mean_not_deployed=True,
        no_extra_model_forward=True,new_environment_episodes=0,video_H_causality=False))


def fit(args, context):
    contract, state, lora, layout, decoder, optimizer, names = import_endpoint(args.root, context.device)
    old_world = state['topology']['world_size']
    resumed = bool(contract.get('resume_checkpoint'))
    if resumed and (context.world_size != old_world or (args.resume is not None
            and args.resume.resolve() != Path(contract['resume_checkpoint']).resolve())):
        raise ValueError('registered full endpoint resume requires its unchanged four-rank topology')
    if context.rank < old_world: restore_rng(state['rank_rng'][context.rank], context)
    else:
        value = seed(70,context.rank); torch.manual_seed(value)
        import random
        random.seed(value); np.random.seed(value)
    write_json_atomic(args.root/'code'/f'endpoint_import_rank_{context.rank}.json',dict(
        parameter_names=names,endpoint_Adam_states=len(optimizer.state),source_Adam_step=contract['initial_cursor'],
        source_world_size=old_world,new_world_size=context.world_size,
        RNG_source=f'CP{contract["initial_cursor"]} rank{context.rank}' if context.rank<old_world else f'seed(70,{context.rank})',
        frozen_generator_parameter_count=state['_frozen_tensor_count'] if resumed else len(state['parameter_names']['psi'])-len(names),
        gradient_owner_migration=not resumed,full_frozen_endpoint_resume=resumed,exact_resume=False,large_source_model_loaded=False,
        physical_activation_checkpointing=False,endpoint_parameter_count=sum(p.numel() for p in decoder.parameters())))
    del state
    pool, teachers = load_pool(args.root, contract, context.device)
    objective = OperatorObjective(SimpleNamespace(lora=lora),args.root)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    if not resumed:
        readout(args.root,'baseline_CP432',decoder,layout,objective,pool,teachers,contract,context,save_endpoints=False)
    panel = {r['task_id']:r for r in materialize_panel(args.root)['tasks']}; panel['_root'] = args.root
    initial = contract['initial_cursor']; nodes = contract['nodes']
    started = None; update = initial; tested_first = False
    while True:
        first = fit_step(update,decoder,optimizer,layout,objective,pool,teachers,panel,context)
        if started is None: started = first
        update += 1
        elapsed = torch.tensor(time.monotonic()-started,device=context.device)
        if context.world_size>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        if (update-initial)%nodes['round_updates']: continue
        if not tested_first and float(elapsed)>=nodes['first_seconds']:
            tested_first = True; node = 'first_10min' if resumed else 'first_5min'
        elif tested_first and float(elapsed)>=nodes['maximum_seconds']:
            node = 'terminal_30min' if resumed else 'terminal_15min'
        else: continue
        loss = readout(args.root,node,decoder,layout,objective,pool,teachers,contract,context,
                       save_endpoints=True,update=update,elapsed=float(elapsed))
        if node=='first_10min' and loss<nodes['first_stop_relative_loss_at_least']*nodes['reference_loss']: continue
        if node=='first_5min' and nodes['first_stop_loss_at_most']<loss<nodes['first_stop_loss_at_least']: continue
        save_terminal(args.root,update,decoder,optimizer,names,contract,context,node,float(elapsed))
        if context.is_main: terminal_operator_decomposition(args.root,teachers,objective,context.device)
        break


def run(args, context):
    if args.command == 'frozen-cache': cache_contexts(args, context)
    elif args.command == 'frozen-fit': fit(args, context)
    else: raise ValueError('unknown finite frozen-context consumer')
