"""Strict first-G readout using saved own-MT prefixes and the existing queue."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import fcntl
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import (EvaluationTask, EvaluationShard, build_cost_balanced_shards,
    initialize_queue, claim_next, complete_job, fail_job, publish_json_exclusive, queue_summary)
from ember.writer.runtime import autocast
from .function_data import CHECKPOINT_SCHEMA
from .learning import HistoryCache, memory_peak
from .analysis import paired


def prepare_panels(args):
    audit = read_json(args.input_audit)
    held = read_json(args.held_prefix_audit)
    original = read_json(Path(held['source_formal_panel']))
    if held['counts']['indexed_prefix_compatible'] != 400 or held['counts']['missing_paths_H_receipts_scenes']:
        raise ValueError('all400 compatible real own-MT prefixes required')
    rows = [dict(row, H=row['prefix'], condition_key=row['condition_id'],
                 MT_source=original['MT400_source']) for row in held['rows']]
    write_json_atomic(args.root/'readout_panels/validation.json', dict(complete=True,
        role='validation', rows=rows, environment=original['environment'],
        reference_first_G=str(Path(held['source_firstG_panel']).parent/'results.json'),
        inherited_initial_proprio_limitations=held['limitations']))
    run = read_json(args.root/'run_contract.json')
    seen = []
    for task in audit['fixed_288_panel']['rows']:
        receipt = read_json(Path(task['MT_G_query_receipt']))
        context = task['contexts'][0]
        for state in receipt['query_states']:
            seen.append(dict(task=task['task'], init_state_id=state,
                teacher_demo=task['G38349_designated_demo'], H=context['H'],
                condition_key=f"task_{task['task']:04d}", noise_root=receipt['policy_env_noise_root'],
                registered_scene=None, MT_source=task['MT_G_query_receipt']))
    write_json_atomic(args.root/'readout_panels/seen.json', dict(complete=True, role='train',
        rows=seen, environment=run['environment'], expected_rows=576,
        source_readout_is_not_independent_T_selection=True))
    write_json_atomic(args.root/'inputs/held_prefix_audit.json', held)


def make_queue(root, arm, role, *, workers, group):
    root = Path(root)
    panel = read_json(root/'readout_panels'/f'{role}.json')
    base = root/'arms'/arm/'readouts'/role
    base.mkdir(parents=True, exist_ok=True)
    checkpoint = str(root/'arms'/arm/'checkpoints/update_00000072')
    plan_file = base/'queue_plan.json'
    if plan_file.exists():
        plan = read_json(plan_file)
        if plan['checkpoint'] != checkpoint:
            raise ValueError('readout queue changed the sole endpoint')
        shards = [EvaluationShard(**dict(s, init_state_ids=tuple(s['init_state_ids']))) for s in plan['shards']]
    else:
        tasks = [EvaluationTask(t['suite'], t['task_id'], t['horizon'],
            tuple(r['init_state_id'] for r in panel['rows'] if r['task'] == t['global_task_id']))
            for t in panel['environment']['tasks']]
        balanced = build_cost_balanced_shards(tasks, env_batch_size=group, physical_gpu_count=workers)
        parts = [(s, s.init_state_ids[first:first+group]) for s in balanced
                 for first in range(0, len(s.init_state_ids), group)]
        shards = [EvaluationShard(f'{s.job_id}_{i:03d}', i, s.suite, s.task_id, s.horizon,
            ids, s.horizon*len(ids), s.preferred_gpu) for i, (s, ids) in enumerate(parts)]
        write_json_atomic(plan_file, dict(checkpoint=checkpoint, condition_batch=group,
            initial_physical_workers=workers, shards=[asdict(s) for s in shards]))
    initialize_queue(base/'jobs.sqlite', shards, contract_reference=str(root/'readout_panels'/f'{role}.json'))
    return base


def readout(args, runtime, runner):
    checkpoint = args.root/'arms'/args.arm/'checkpoints/update_00000072'
    manifest = read_json(checkpoint/'manifest.json')
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=False)
    if (not manifest['complete'] or manifest['schema_version'] != CHECKPOINT_SCHEMA
            or manifest['next_update'] != 72 or state['arm'] != args.arm):
        raise ValueError('readout only accepts the registered sole final G72')
    runtime.generator.load_state_dict(state['psi']); runtime.freeze_psi()
    panel = read_json(args.root/'readout_panels'/f'{args.readout_role}.json')
    tasks = {(t['suite'], t['task_id']): t for t in panel['environment']['tasks']}
    base = args.root/'arms'/args.arm/'readouts'/args.readout_role
    queue = base/'jobs.sqlite'
    histories = HistoryCache()
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        try:
            tick = time.monotonic()
            before = runner.environment_steps
            task = tasks[claim.task_key]
            entries = [r for r in panel['rows'] if r['task'] == task['global_task_id']
                       and r['init_state_id'] in claim.shard.init_state_ids]
            requests, sources = [], {}
            for entry in entries:
                destination = base/'conditions'/entry['condition_key']
                destination.mkdir(parents=True, exist_ok=True)
                row_file = destination/f"query_{entry['init_state_id']:03d}.json"
                if row_file.exists():
                    continue
                weights = generated_state(runtime, histories, destination, entry, checkpoint)
                identity = f"{args.arm}_{args.readout_role}_{entry['task']}_{entry['init_state_id']}"
                sources[identity] = entry
                requests.append(dict(episode_id=identity, task=task,
                    state=load_file(str(weights), device=str(runtime.device)), parameter_ref='first_G',
                    state_id=entry['init_state_id'], noise_root=entry['noise_root'],
                    registered_scene=entry['registered_scene'], capture_history=False, capture_hidden=False,
                    remaining=task['horizon']+10, uses_teaching=False))
            for output in runner.run(requests):
                entry = sources[output['request']['episode_id']]
                row = dict(output['row'], global_task_id=entry['task'], teacher_demo=entry['teacher_demo'],
                    checkpoint=str(checkpoint), arm=args.arm, condition=entry['condition_key'],
                    query_excluded_from_H=True, fixed_unique_LoRA=True, MT_source=entry['MT_source'])
                if output['history'].records:
                    first = output['history'].records[0]
                    row.update(first_native_normalized_actions=first['normalized_actions'].tolist(),
                               first_noise_seed=first['noise_seed'])
                write_json_atomic(base/'conditions'/entry['condition_key']/f"query_{entry['init_state_id']:03d}.json", row)
            rows = [read_json(base/'conditions'/r['condition_key']/f"query_{r['init_state_id']:03d}.json") for r in entries]
            relative = Path('shards')/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
            size = publish_json_exclusive(base/relative, dict(rows=rows, checkpoint=str(checkpoint)))
            complete_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token,
                rows_path=str(relative), rows_bytes=size, row_count=len(rows), successes=sum(bool(r['success']) for r in rows))
            with (base/f'performance_{args.worker_id}.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(seconds=time.monotonic()-tick, rows=len(rows),
                    environment_steps=runner.environment_steps-before, slots=args.slots,
                    peak=memory_peak(runtime.device), runner=runner.components))+'\n')
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id,
                     claim_token=claim.claim_token, error=repr(error))
            raise


def generated_state(runtime, histories, destination, entry, checkpoint):
    """Seen state shards share exactly one condition LoRA across queue workers."""
    weights = destination/'G.safetensors'
    with (destination/'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if weights.exists():
            generation = read_json(destination/'generation.json')
            if generation['checkpoint'] != str(checkpoint) or generation['teacher_demo'] != entry['teacher_demo']:
                raise ValueError('existing condition LoRA belongs to another endpoint or video')
            return weights
        history = histories.get(entry['H'])
        if (len(history.episodes) != 1 or any(r['parameter_ref'] != 'MT300' for r in history.records)
                or any(row['init_state_id'] == entry['init_state_id'] for row in history.episodes)):
            raise ValueError('first-G needs only the real MT episode, disjoint final state')
        with torch.no_grad(), autocast(runtime.device):
            generated = runtime.generator.generate(runtime.mt,
                runtime.teaching(entry['task'], entry['teacher_demo']),
                history.model_rows(runtime), {'MT300': runtime.mt})
        histories.remember_features(entry['H'], history)
        temporary = destination/'G.pending.safetensors'
        save_file({k: v.detach().float().cpu().contiguous() for k, v in generated.items()}, str(temporary))
        write_json_atomic(destination/'generation.json', dict(complete=True,
            checkpoint=str(checkpoint), teacher_demo=entry['teacher_demo'],
            own_MT_H=entry['H'], one_complete38_target_LoRA=True, no_Actor_or_practice=True))
        temporary.replace(weights)
    return weights


def summarize(root, arm, role):
    root = Path(root); panel = read_json(root/'readout_panels'/f'{role}.json')
    base = root/'arms'/arm/'readouts'/role
    rows = [read_json(base/'conditions'/r['condition_key']/f"query_{r['init_state_id']:03d}.json") for r in panel['rows']]
    if len(rows) != (400 if role == 'validation' else 576):
        raise ValueError('full registered endpoint readout required')
    mt, references = [], {}
    tasks = {t['global_task_id']: t for t in panel['environment']['tasks']}
    for entry in panel['rows']:
        source = entry['MT_source']
        if source not in references:
            references[source] = read_json(Path(source))['rows']
        task = tasks[entry['task']]
        matches = [row for row in references[source]
            if (row['suite'], row['task_id'], row['init_state_id']) ==
               (task['suite'], task['task_id'], entry['init_state_id'])
            and row.get('global_task_id', entry['task']) == entry['task']
            and (role != 'seen' or row.get('reference') == 'MT300')]
        if len(matches) != 1:
            raise ValueError(f"MT reference needs one suite/task/state row: {source}, "
                             f"{task['suite']}/{task['task_id']}/{entry['init_state_id']}; got {len(matches)}")
        mt.append(dict(matches[0], global_task_id=entry['task']))
    result = dict(complete=True, raw_rows=rows, MT_raw_rows=mt, paired=paired(mt, rows),
        strict_single_checkpoint400=role == 'validation', per_task={}, per_suite={}, queue=queue_summary(base/'jobs.sqlite'))
    for field, output in [('global_task_id', 'per_task'), ('suite', 'per_suite')]:
        for value in sorted({r[field] for r in rows}):
            selected = [i for i, r in enumerate(rows) if r[field] == value]
            result[output][str(value)] = paired([mt[i] for i in selected], [rows[i] for i in selected])
    write_json_atomic(base/'results.json', result)
    return result
