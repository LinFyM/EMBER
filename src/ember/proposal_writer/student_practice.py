"""Fixed MT/first-G practice pairs and query16 credit for final-use learning."""
from __future__ import annotations

from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from ember.lora import validate_lora_state
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_source_checkpoint import read_json, write_json_atomic

from .contract import TASKS, seed
from .learning import HistoryCache
from .path import (begin_condition, bootstrap, practice, restore_compilation, save_history,
                   generate_candidates, fixed_requests, fixed_row)
from .pipeline import task_spec
from .supervision import bootstrap_source, complete_query, control_row_compatibility, panels, query_panel

MODE = 'selector_fixed_pair'
REFERENCES = ['MT300', 'G_000']


def _link(destination, source):
    source = Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError('fixed pair source reference changed')
    else:
        destination.symlink_to(source)


def _episode_error(row, task, state, noise, reference):
    expected = dict(init_state_id=state, env_seed=noise, policy_seed_root=noise,
        actual_parameter_ref=reference, suite=task['suite'], task_id=task['task_id'],
        language=task['language'], settling_steps=10, restored_control_steps=0)
    if any(row.get(k) != v for k, v in expected.items()):
        return 'practice changed its frozen policy/state/official scene/RNG'
    if (row.get('model_failure') or row.get('budget_truncated') or row.get('failure_type')
            or not isinstance(row.get('success'), bool) or not 0 < row.get('steps', 0) <= task['horizon']
            or (not row['success'] and row['steps'] != task['horizon'])
            or row.get('environment_steps') != row['steps'] + 10):
        return 'practice ended before legitimate success-stop or the full official horizon'
    trace = row.get('policy_noise_seeds', [])
    if not trace or trace[:3] != [policy_noise_seed(noise, task['suite'], task['task_id'], state, i)
                                 for i in range(min(3, len(trace)))]:
        return 'practice native policy noise does not belong to its registered producer'
    return None


def _history_error(entry, count):
    path, task, panel = (entry[k] for k in ('path', 'task', 'panel'))
    if count not in (1, 2) or len(path.history.episodes) != count:
        return 'fixed pair needs the complete actual MT bootstrap and G practice'
    if any(row['init_state_id'] in entry['states'] for row in path.history.episodes):
        return 'query initial state entered practice history'
    for episode, row in enumerate(path.history.episodes):
        ref = REFERENCES[episode]
        noise = panel['bootstrap_noise_root'] if episode == 0 else entry['mapping']['env_seed']
        error = _episode_error(row, task, panel['practice_states'][episode], noise, ref)
        if error:
            return error
        records = [r for r in path.history.records if r['episode'] == episode]
        if (not records or any(r['parameter_ref'] != ref for r in records)
                or sum(int(r['executed'].sum()) for r in records) != row['steps']):
            return 'practice history lost its actual producer or executed actions'
        for record in records:
            shapes = dict(hidden=(2, 50, 1024), proprio=(2, 8), actions=(5, 7),
                          normalized_actions=(5, 7), executed=(5,), feedback=(5,))
            if any(tuple(record[k].shape) != shape for k, shape in shapes.items()):
                return 'actual RGB/proprio/native hidden/action/feedback capture is incomplete'
            for key in (record['pre'], record['post']):
                raw = path.history.observations[key]
                if raw['images'].shape[:2] != (2, 3) or tuple(raw['proprio'].shape) != (8,):
                    return 'actual two-camera RGB/proprio observation is incomplete'
    if any(r['episode'] not in range(count) for r in path.history.records):
        return 'fixed pair history contains an unregistered episode'
    if count == 2:
        if len(path.boundaries) != 1 or path.practiced != REFERENCES:
            return 'fixed pair requires exactly one forced actual G practice'
        boundary = path.boundaries[0]
        if (boundary['actual_parameter_ref'] != 'G_000' or boundary['records'] != len(path.history.records)
                or boundary['episodes'] != 2 or boundary['row'] != path.history.episodes[-1]):
            return 'G practice boundary does not describe its complete retained history'
    return None


def _query_error(row, entry, state, reference, paired=None):
    changed = control_row_compatibility(row, entry['task'], entry['path'].task, state,
        entry['query_noise'], reference=reference, paired_row=paired)
    if (changed or row.get('actual_parameter_ref') != reference or not row.get('query_excluded_from_H')
            or _episode_error(row, entry['task'], state, entry['query_noise'], reference)):
        return 'fixed query has incompatible actual policy/scene/RNG or incomplete episode'
    if reference == 'G_000' and row.get('teacher_demo', row.get('teaching_demo')) != entry['path'].demo:
        return 'fixed query uses a different teaching condition'
    return None


def _seen_credit(args, run, entry, source):
    """Validate actual F38349 weights/H and raw paired rows, never an older G path."""
    root = Path(source['root']); registered = read_json(Path(source['panel'])); results = read_json(Path(source['results']))
    previous = read_json(root/'run_contract.json'); generation = read_json(Path(source['generation']))
    weights = Path(source['weights']).resolve(strict=True); identity = generation['input_identity']
    checkpoint = Path(run['readout_checkpoint']).resolve(strict=True)
    task, path = entry['task'], entry['path']; task_id = path.task
    bootstrap_receipt = read_json(Path(source['bootstrap_complete']))
    current_bootstrap = read_json(args.root/'bootstrap'/f'task_{task_id:04d}'/'complete.json')
    metadata = [r for r in registered['rows'] if r['task'] == task_id]
    old_task = task_spec(registered['environment'], task_id)
    scene_keys = ('suite', 'task_id', 'language', 'horizon', 'bddl_file', 'init_states_file')
    if (not results.get('complete') or results.get('F_cursor') != 38349 or registered.get('F_cursor') != 38349
            or any(Path(value).resolve(strict=True) != checkpoint for value in
                   (registered['checkpoint'], results['checkpoint'], identity['checkpoint'], generation['kernel_identity']['checkpoint']))
            or identity['F_cursor'] != 38349 or identity['task'] != task_id or identity['teacher_demo'] != path.demo
            or identity['prefix_episodes'] != 1 or not generation.get('no_gradients')
            or generation.get('Actor_called') or generation.get('later_H_or_query_used')
            or previous['source'] != run['source'] or previous['mt_checkpoint'] != run['mt_checkpoint']
            or any(registered['environment'][key] != run['environment'][key] for key in ('environment', 'policy'))
            or any(old_task[key] != task[key] for key in scene_keys)
            or any((root/relative).resolve(strict=True) != (args.root/relative).resolve(strict=True)
                   for relative in ('initial.safetensors', 'base/merged_targets.safetensors'))
            or not bootstrap_receipt.get('complete') or not current_bootstrap.get('complete')
            or any(Path(value).resolve(strict=True) != Path(current_bootstrap['history']).resolve(strict=True)
                   for value in (bootstrap_receipt['history'], identity['history_source']))
            or bootstrap_receipt['original_episode_evidence'] != generation['MT_episode']
            or len(metadata) != 16 or {r['init_state_id'] for r in metadata} != set(entry['states'])):
        raise ValueError(f'task {task_id}: incompatible original F38349 endpoint/bootstrap/query source')
    directory = entry['destination']/'query'; directory.mkdir(parents=True, exist_ok=True)
    mt_rows = [r for r in results['MT_raw_rows'] if r['global_task_id'] == task_id]
    g_rows = [r for r in results['raw_rows'] if r['global_task_id'] == task_id]
    if any(len(rows) != 16 or {r['init_state_id'] for r in rows} != set(entry['states']) for rows in (mt_rows, g_rows)):
        raise ValueError('original seen source must contain every actual MT/G query16 row once')
    by_state = {row['init_state_id']: row for row in metadata}
    mt = {row['init_state_id']: row for row in mt_rows}
    for reference, rows, key in (('MT300', mt_rows, 'MT_raw_rows'), ('G_000', g_rows, 'raw_rows')):
        for row in rows:
            state = row['init_state_id']; condition = by_state[state]
            if (_query_error(row, entry, state, reference, mt[state])
                    or condition['MT_row'] != mt[state] or condition['teacher_demo'] != path.demo
                    or condition['noise_root'] != entry['query_noise'] or condition['prefix_episodes'] != 1
                    or condition['prefix_records'] != identity['prefix_records']
                    or Path(condition['history_source']).resolve(strict=True) != Path(identity['history_source']).resolve(strict=True)):
                raise ValueError(f'task {task_id}, state {state}: incompatible original query pair')
            provenance = dict(file=str(source['results']), key=key, global_task_id=task_id, init_state_id=state)
            if reference == 'G_000':
                raw_file = root/'conditions'/f'task_{task_id:04d}_state_{state:03d}'/'query.json'
                if (read_json(raw_file) != row or row['F_cursor'] != 38349
                        or Path(row['checkpoint']).resolve(strict=True) != checkpoint
                        or Path(row['fixed_G_weights']).resolve(strict=True) != weights):
                    raise ValueError('seen G credit belongs to another actual saved endpoint')
                provenance.update(raw_row=str(raw_file), weights=str(weights), generation=source['generation'])
            saved = dict(row, reused=True, source_provenance=provenance,
                         actual_new_episodes=0, actual_new_environment_steps=0)
            file = directory/f'{reference}_{state:03d}.json'
            if file.exists() and read_json(file) != saved:
                raise ValueError('readonly seen pair row source changed')
            if not file.exists():
                write_json_atomic(file, saved)
    return dict(weights=str(weights), generation=generation, source=source)


def _prepare(args, runtime, run, task, panel, cache):
    task_id = task['global_task_id']; spec = run['student_refresh']
    condition = spec['condition_mapping'][str(task_id)]
    states = query_panel(panel, 0); noise = seed(32, condition['noise_root'], 1)
    mapping = dict(init_state_id=panel['practice_states'][1], env_seed=noise, policy_seed_root=noise,
                   history_episode_ordinal=1, parameter_ref='G_000')
    if (condition['path_ordinal'] != 0 or condition['noise_root'] != seed(71, task_id, 0)
            or condition['teacher_demo'] != panel['policy_videos'][0] or len(states) != 16
            or panel['practice_states'] != panel['init_order'][:14]
            or states != panel['init_order'][36:46]+panel['init_order'][14:16]+panel['init_order'][32:36]
            or panel['bootstrap_state_id'] != panel['init_order'][0]
            or panel['bootstrap_noise_root'] != seed(4, task_id, panel['bootstrap_state_id'])
            or set(panel['practice_states']).intersection(states)
            or spec['practice_mapping'][str(task_id)] != mapping):
        raise ValueError('fixed pair teaching/practice/query/seed mapping changed')
    destination = args.root/'student_data'/spec['version']/f'task_{task_id:04d}'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'path.json').exists():
        path = restore_compilation(destination, runtime)
        if (path.sampling_identity != MODE or path.decisions or path.outputs
                or path.kernel_identity != runtime.fixed_kernel_identity
                or (path.demo, path.path_ordinal, path.noise_root) != tuple(condition[k]
                    for k in ('teacher_demo', 'path_ordinal', 'noise_root'))):
            raise ValueError('fixed pair path identity changed')
    else:
        path, _ = begin_condition(runtime, task, condition['teacher_demo'], destination.name,
            0, condition['noise_root'], 1, None, True, None, False, True)
        path.sampling_identity = MODE
    entry = dict(path=path, task=task, panel=panel, mapping=mapping, destination=destination,
        states=states, query_noise=seed(70, task_id, 0), seen=str(task_id) in run['fixed_pair_sources'])
    if entry['seen']:
        entry['credit_source'] = _seen_credit(args, run, entry, run['fixed_pair_sources'][str(task_id)])
    bootstrap_file = args.root/'bootstrap'/f'task_{task_id:04d}'/'complete.json'
    if not path.history.episodes and bootstrap_file.exists():
        history, source = bootstrap_source(args.root, task_id, panel, 0, cache)
        for _ in bootstrap(path, task, panel, history=history, provenance=source):
            raise ValueError('registered bootstrap unexpectedly requested another MT episode')
    if entry['seen'] and not path.history.episodes:
        raise ValueError('seen task must reuse its original complete MT bootstrap')
    if path.history.episodes and _history_error(entry, len(path.history.episodes)):
        raise ValueError('restored pair contains incomplete or incompatible actual practice')
    entry['already_complete'] = (destination/'parent.json').exists() and read_json(destination/'parent.json').get('complete', False)
    return entry


def _save_bootstrap(args, entry):
    path = entry['path']; error = _history_error(entry, 1)
    if error:
        raise ValueError(error)
    directory = args.root/'bootstrap'/f'task_{path.task:04d}'; directory.mkdir(parents=True, exist_ok=True)
    file = directory/'history.pt.gz'; temporary = directory/'.history.pt.gz.partial'
    save_history(temporary, path.history); temporary.replace(file)
    write_json_atomic(directory/'complete.json', dict(complete=True, task=path.task, reference='MT300',
        state_id=entry['panel']['bootstrap_state_id'], history=str(file),
        original_episode_evidence=path.history.episodes[0], new_MT_practice_episodes=1,
        actual_producer_weights=str((args.root/'initial.safetensors').resolve(strict=True)),
        query_excluded_from_H=True, readout_mode=MODE))
    path.bootstrap_provenance = dict(reused=True, source=str(file), new_environment_steps=0,
                                    collected_for_fixed_pair=True)


def _install_G(runtime, entry):
    path, directory = entry['path'], entry['destination']; file = directory/'G_000.safetensors'
    if path.attempts:
        if len(path.attempts) != 1 or not path.attempts[0]['valid'] or path.selected != 'G_000':
            raise ValueError('fixed pair cannot regenerate or replace its one endpoint')
        return
    if entry['seen']:
        source = entry['credit_source']; _link(file, source['weights'])
        path.states['G_000'] = load_file(str(file))
        attempt = dict(reference='G_000', parent='MT300', valid=True, decoder_calls=0,
            generation='saved_complete_G38349_endpoint', new_G_generation=False, saved_weights=source['weights'])
    else:
        receipt = directory/'generation.json'
        if receipt.exists():
            saved = read_json(receipt)
            if not saved.get('complete') or not file.exists() or saved['kernel_identity'] != path.kernel_identity:
                raise ValueError('incomplete first-G generation must be resolved without regenerating the condition')
            path.states['G_000'] = load_file(str(file)); attempt = saved['attempt']
        else:
            write_json_atomic(receipt, dict(complete=False, kernel_identity=path.kernel_identity,
                generation_started=True, new_G_generations=1, actual_MT_prefix_required=True))
            features = runtime.teaching(path.task, path.demo)
            if generate_candidates(runtime, path, 'MT300', features) != ['G_000']:
                raise ValueError('fixed pair needs one full deterministic first G')
            attempt = path.attempts.pop()
            temporary = directory/'.G_000.safetensors.partial'
            save_file(path.states['G_000'], str(temporary)); temporary.replace(file)
            write_json_atomic(receipt, dict(complete=True, kernel_identity=path.kernel_identity,
                attempt=attempt, new_G_generations=1, Actor_called=False, no_gradients=True,
                later_H_or_query_used=False, real_H_source=path.bootstrap_provenance,
                prefix_records=len(path.history.records), prefix_episodes=1, teacher_demo=path.demo))
    validate_lora_state(path.states['G_000'], runtime.lora)
    if not all(bool(torch.isfinite(v).all()) for v in path.states['G_000'].values()):
        raise ValueError('fixed pair endpoint has nonfinite parameters')
    path.attempts.append(attempt)
    if 'G_000' not in path.valid:
        path.valid.append('G_000')
    path.selected = 'G_000'


def _persist(args, entry, error=None):
    """Replace destination H atomically, without writing through an immutable source."""
    path, directory = entry['path'], entry['destination']; history = directory/'history.pt.gz'
    if len(path.history.episodes) == 1 and not path.boundaries and path.bootstrap_provenance.get('source'):
        if history.exists() and not history.is_symlink():
            history.unlink()  # Only this consumer's partial destination H can reach this case.
        _link(history, path.bootstrap_provenance['source'])
    else:
        temporary = directory/'.history.pt.gz.partial'; save_history(temporary, path.history); temporary.replace(history)
    descriptor = dict(path=str(history), records=len(path.history.records), episodes=len(path.history.episodes))
    bootstrap_records = sum(r['episode'] == 0 for r in path.history.records)
    segment = dict(path=str(history), record_start=bootstrap_records,
        records=len(path.history.records)-bootstrap_records, episode=1, parameter_ref='G_000')
    practiced = path.completed_rounds == 1 and error is None
    query_file = directory/'query/complete.json'; query = read_json(query_file) if query_file.exists() else {}
    complete = practiced and complete_query(query, entry['states'], entry['query_noise'], REFERENCES)
    weights = directory/'G_000.safetensors'; row = dict(path.boundaries[0]['row']) if path.boundaries else {}
    write_json_atomic(directory/'path.json', dict(**path.manifest(), retained_U=True,
        forced_practice_not_on_policy=True, final_warm_only=True, eligible_for_G_parent_bank=False))
    write_json_atomic(directory/'practice.json', dict(**row, complete=practiced, global_task_id=path.task,
        teaching_demo=path.demo, reference='G_000', condition=path.identity, readout_mode=MODE,
        generated_weights=str(weights), history=segment, forced_practice_not_on_policy=True, error=error))
    write_json_atomic(directory/'parent.json', dict(complete=complete, task=path.task,
        version=read_json(args.root/'run_contract.json')['student_refresh']['version'],
        parent=dict(reference=f'{path.identity}:G_000', weights=str(weights), histories=[descriptor]) if complete else None,
        eligible_for_G_parent_bank=False, final_warm_only=True, forced_practice_not_on_policy=True,
        readout_mode=MODE, actual_generated_parameter_ref='G_000', generated_weights=str(weights),
        actual_producer_refs=list(path.history.states), producer_refs=dict(MT300=str((args.root/'initial.safetensors').resolve()),
            G_000=str(weights.resolve()) if weights.exists() else None), real_H_source=path.bootstrap_provenance,
        practice_mapping=entry['mapping'], practice_segment=segment, boundaries=path.boundaries,
        practice_receipt=str(directory/'practice.json'), query_rewards=query.get('rewards'), query_count=16,
        query_receipt=str(query_file), query_excluded_from_H=True, query_credit_readonly=entry['seen'],
        fixed_pair_source=entry.get('credit_source', {}).get('source'), fixed_scale_reestimated=False,
        deterministic_endpoint_single_candidate=True, selection_did_not_use_reward_or_query=True,
        new_G_generations=sum(a['decoder_calls'] for a in path.attempts), new_G_practice_episodes=len(path.boundaries),
        new_MT_practice_episodes=int(not entry['seen'] and bool(path.history.episodes)),
        new_query_episodes=query.get('actual_new_episodes_total', 0),
        kernel_identity=path.kernel_identity, timings=path.timings, error=error))


def _pair_steps(args, runtime, entry):
    path, task, panel = (entry[k] for k in ('path', 'task', 'panel'))
    if not path.history.episodes:
        driver = bootstrap(path, task, panel); request = next(driver)
        request.update(noise_root=panel['bootstrap_noise_root'], capture_history=True,
                       capture_hidden=True, capture_phi=False, uses_teaching=False)
        output = yield request
        try:
            driver.send(output)
        except StopIteration:
            pass
        else:
            raise ValueError('fixed pair MT bootstrap requested more than one episode')
        _save_bootstrap(args, entry)
    if len(path.history.episodes) == 1:
        if _history_error(entry, 1):
            raise ValueError('first G requires a complete actual MT history before generation')
        _install_G(runtime, entry); _persist(args, entry)
        driver = practice(path, task, panel, 'G_000', snapshots=False); request = next(driver)
        if (request['state_id'] != entry['mapping']['init_state_id'] or request['noise_root'] != entry['mapping']['env_seed']
                or request['remaining'] < task['horizon']+10):
            raise ValueError('forced G practice changed its registered complete episode')
        request.update(capture_history=True, capture_hidden=True, capture_phi=False, uses_teaching=False)
        output = yield request
        try:
            driver.send(output)
        except StopIteration:
            pass
        else:
            raise ValueError('fixed pair G practice requested more than one episode')
    error = _history_error(entry, 2)
    path.completed_rounds = int(error is None); _persist(args, entry, error)
    if error:
        raise ValueError(error)


def _queries(runner, entries):
    owners = {}; requests = []
    for entry in entries:
        directory = entry['destination']/'query'; directory.mkdir(parents=True, exist_ok=True)
        existing = {(ref, state) for ref in REFERENCES for state in entry['states']
                    if (directory/f'{ref}_{state:03d}.json').exists()}
        if entry['seen'] and len(existing) != 32:
            raise ValueError('seen task may only reuse its compatible original query pair')
        for request in fixed_requests(entry['path'], entry['task'], entry['states'], entry['query_noise'],
                                      references=REFERENCES, existing_rows=existing):
            owners[request['episode_id']] = entry; requests.append(request)
    for output in runner.run(sorted(requests, key=lambda r:r['task']['horizon'], reverse=True)):
        entry = owners[output['request']['episode_id']]; row = fixed_row(entry['path'], output)
        write_json_atomic(entry['destination']/'query'/f"{row['reference']}_{row['init_state_id']:03d}.json", row)
    for entry in entries:
        directory = entry['destination']/'query'; rows = []
        for state in entry['states']:
            paired = read_json(directory/f'MT300_{state:03d}.json')
            for ref in REFERENCES:
                row = read_json(directory/f'{ref}_{state:03d}.json')
                error = _query_error(row, entry, state, ref, paired)
                if error:
                    raise ValueError(error)
                rows.append(row)
        fresh = [r for r in rows if not r.get('reused')]
        saved = dict(complete=True, rows=rows, query_states=entry['states'], query_count=16, logical_rows=32,
            rewards={ref:sum(r['success'] for r in rows if r['reference'] == ref)/16 for ref in REFERENCES},
            policy_env_noise_root=entry['query_noise'], query_excluded_from_H=True,
            each_parameter_query_once_across_prefixes=True, unpracticed_candidates_not_queried=True,
            forced_practice_not_on_policy=True, final_warm_only=True,
            actual_new_episodes_total=len(fresh),
            actual_new_environment_steps_total=sum(r['environment_steps'] for r in fresh),
            source_provenance=entry.get('credit_source', {}).get('source'))
        if not complete_query(saved, entry['states'], entry['query_noise'], REFERENCES):
            raise ValueError('fixed pair real16 credit is incomplete')
        write_json_atomic(directory/'complete.json', saved)


@torch.no_grad()
def pair_group(args, runtime, runner, environment, task_ids):
    """Interleave the bounded one MT/one G path; query never enters retained H."""
    run = read_json(args.root/'run_contract.json'); spec = run['student_refresh']; task_ids = tuple(task_ids)
    tasks = run['training_tasks']; seen = {int(task) for task in run['fixed_pair_sources']}
    if (spec.get('readout_mode') != MODE or spec.get('readonly_E_query') or not runtime.psi_frozen
            or runtime.version != 38349 or spec['G_update'] != 38349
            or len(tasks) != 72 or len(set(tasks)) != 72 or seen != set(TASKS)
            or set(spec['condition_mapping']) != {str(task) for task in tasks}
            or len(task_ids) != len(set(task_ids)) or not set(task_ids) <= set(tasks)):
        raise ValueError('fixed pair requires the registered72 tasks and frozen complete G38349')
    tick = time.monotonic(); env_start = runner.environment_steps; registered = panels(args.root); cache = HistoryCache()
    entries = [_prepare(args, runtime, run, task_spec(environment, task), registered[task], cache) for task in task_ids]
    active = {i:_pair_steps(args, runtime, entry) for i, entry in enumerate(entries)}; owners = {}
    def advance(index, output=None):
        try:
            request = next(active[index]) if output is None else active[index].send(output)
        except StopIteration:
            del active[index]; return []
        if request['episode_id'] in owners:
            raise ValueError('fixed pair actual episode identity repeated')
        owners[request['episode_id']] = index
        return [request]
    def receive(output):
        return advance(owners.pop(output['request']['episode_id']), output)
    try:
        requests = [request for i in sorted(active, key=lambda n:entries[n]['task']['horizon'], reverse=True)
                    for request in advance(i)]
        if requests:
            for _ in runner.run(requests, on_completion=receive):
                pass
        if active:
            raise ValueError('fixed pair driver exited with incomplete actual practice')
        _queries(runner, entries)
        for entry in entries:
            _persist(args, entry)
    except BaseException as error:
        for entry in entries:
            if not entry['already_complete']:
                _persist(args, entry, repr(error))
        raise
    shared = dict(seconds=time.monotonic()-tick, new_environment_steps=runner.environment_steps-env_start,
                  members=list(task_ids), final_warm_only=True, G_gradients=0, bank_writes=0)
    return [dict(task=e['path'].task, destination=str(e['destination']),
        parent_receipt=str(e['destination']/'parent.json'), practice_receipt=str(e['destination']/'practice.json'),
        query_receipt=str(e['destination']/'query/complete.json'), actual_generated_parameter_ref='G_000',
        query_count=16, query_rewards=read_json(e['destination']/'query/complete.json')['rewards'],
        already_complete=e['already_complete'], shared_pair_group=shared) for e in entries]
