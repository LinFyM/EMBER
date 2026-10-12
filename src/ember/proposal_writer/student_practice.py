"""One real practice of an immutable saved endpoint; no Writer or query execution."""
from __future__ import annotations

from pathlib import Path
import time

import torch

from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic

from .contract import TASKS, seed
from .learning import HistoryCache
from .path import begin_condition, bootstrap, practice, restore_compilation, save_history
from .pipeline import lock_saved_endpoint, task_spec
from .supervision import (MT_source_compatibility, bootstrap_source, complete_query,
                          control_row_compatibility, panels, query_panel, query_row)

MODE = 'saved_endpoint_practice_only'


def _link(destination, source):
    source = Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError('saved practice source reference changed')
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def _credit(args, runtime, run, task, panel):
    """Read the already measured MT/G pair, including its actual weight identity."""
    task_id = task['global_task_id']; states = query_panel(panel, 0)
    source_root = Path(run['previous_root']); version = run['previous_student_version']
    source_run = read_json(source_root/'run_contract.json')
    spec = source_run['student_refresh']; condition = run['student_refresh']['condition_mapping'][str(task_id)]
    original = spec['condition_mapping'][str(task_id)]
    if (args.root.resolve() == source_root.resolve() or len(states) != 16
            or spec['version'] != version or spec['G_update'] != runtime.version
            or spec.get('readout_mode') != 'endpoint_query_only'
            or MT_source_compatibility(args.root, source_root)
            or Path(source_run['saved_endpoint_complete']).resolve() != Path(run['saved_endpoint_complete']).resolve()
            or any(original[k] != condition[k] for k in ('teacher_demo', 'path_ordinal', 'noise_root'))):
        raise ValueError('practice needs the original compatible terminal query16 source')
    directory = source_root/'student_data'/version/f'task_{task_id:04d}'
    parent = read_json(directory/'parent.json'); receipt = directory/'query/complete.json'
    credit = read_json(receipt); noise_root = seed(70, task_id, 0)
    weights = Path(runtime.saved_endpoint_rows[task_id]['weights']).resolve(strict=True)
    if (not parent.get('complete') or parent.get('actual_generated_parameter_ref') != 'G_000'
            or parent.get('kernel_identity') != runtime.fixed_kernel_identity
            or Path(parent['generated_weights']).resolve(strict=True) != weights
            or (directory/'G_000.safetensors').resolve(strict=True) != weights
            or not credit.get('query_excluded_from_H')
            or not complete_query(credit, states, noise_root, ['MT300', 'G_000'])):
        raise ValueError('readonly credit does not belong to the actual saved complete endpoint')
    successes = dict(MT300=0, G_000=0)
    for state in states:
        paired = query_row(directory/'query'/f'MT300_{state:03d}.json')
        for reference in ('MT300', 'G_000'):
            row = query_row(directory/'query'/f'{reference}_{state:03d}.json')
            if (control_row_compatibility(row, task, task_id, state, noise_root,
                    reference=reference, paired_row=paired)
                    or row.get('actual_parameter_ref') != reference or not row.get('query_excluded_from_H')
                    or row.get('model_failure') or row.get('budget_truncated')
                    or (reference == 'G_000' and row.get('teaching_demo') != condition['teacher_demo'])):
                raise ValueError('readonly terminal query row has incompatible actual policy/scene/RNG')
            successes[reference] += bool(row['success'])
    if any(successes[ref]/16 != credit['rewards'][ref] for ref in successes):
        raise ValueError('readonly query rewards differ from their actual source rows')
    return dict(directory=directory, receipt=receipt, rewards=credit['rewards'], states=states,
                weights=weights, parent=parent)


def _practice_error(entry):
    path = entry['path']; mapping = entry['mapping']; task = entry['task']
    if len(path.history.episodes) != 2 or len(path.boundaries) != 1:
        return 'exactly one MT bootstrap and one actual G practice are required'
    boundary = path.boundaries[0]; row = boundary['row']
    if (boundary['actual_parameter_ref'] != 'G_000' or boundary['records'] != len(path.history.records)
            or boundary['episodes'] != 2 or row != path.history.episodes[-1]):
        return 'actual practice boundary does not describe its complete retained history'
    expected = {k: mapping[k] for k in ('init_state_id', 'env_seed', 'policy_seed_root')}
    expected.update(actual_parameter_ref='G_000', suite=task['suite'], task_id=task['task_id'],
                    language=task['language'], settling_steps=10, restored_control_steps=0)
    if any(row.get(k) != v for k, v in expected.items()):
        return 'actual G practice changed its frozen policy/state/official scene/RNG'
    if (row.get('model_failure') or row.get('budget_truncated') or row.get('failure_type')
            or not isinstance(row.get('success'), bool) or not 0 < row.get('steps', 0) <= task['horizon']
            or (not row['success'] and row['steps'] != task['horizon'])
            or row.get('environment_steps') != row['steps'] + 10):
        return 'G practice ended before a legitimate success-stop or full official horizon'
    if any(ep['init_state_id'] in entry['credit']['states'] for ep in path.history.episodes):
        return 'query initial state entered practice history'
    rows = [r for r in path.history.records if r['episode'] == 1]
    if (not rows or any(r['parameter_ref'] != ('MT300' if r['episode'] == 0 else 'G_000')
            or r['episode'] not in (0, 1) for r in path.history.records)
            or sum(int(r['executed'].sum()) for r in rows) != row['steps']):
        return 'actual practice history lost its producer or executed actions'
    for record in rows:
        shapes = dict(hidden=(2, 50, 1024), proprio=(2, 8), actions=(5, 7),
                      normalized_actions=(5, 7), executed=(5,), feedback=(5,))
        if any(tuple(record[k].shape) != shape for k, shape in shapes.items()):
            return 'actual RGB/proprio/two-flow hidden/action/feedback capture is incomplete'
        for key in (record['pre'], record['post']):
            raw = path.history.observations[key]
            if raw['images'].shape[:2] != (2, 3) or tuple(raw['proprio'].shape) != (8,):
                return 'actual two-camera RGB/proprio observation is incomplete'
    return None


def _persist(entry, version, error=None):
    """Replace only the destination H link; never write through a source link."""
    path, destination, credit = (entry[k] for k in ('path', 'destination', 'credit'))
    complete = error is None and path.completed_rounds == 1
    history = destination/'history.pt.gz'; temporary = destination/'.history.pt.gz.partial'
    save_history(temporary, path.history); temporary.replace(history)
    descriptor = dict(path=str(history), records=len(path.history.records), episodes=len(path.history.episodes))
    segment = dict(path=str(history), record_start=entry['bootstrap_records'],
                   records=len(path.history.records)-entry['bootstrap_records'], episode=1, parameter_ref='G_000')
    row = dict(path.boundaries[0]['row']) if path.boundaries else {}
    write_json_atomic(destination/'path.json', dict(**path.manifest(), retained_U=True,
        new_G_generation=False, new_G_practice_episodes=len(path.boundaries), saved_endpoint_source=str(credit['weights'])))
    write_json_atomic(destination/'practice.json', dict(**row, complete=complete,
        global_task_id=path.task, teaching_demo=path.demo, reference='G_000', condition=path.identity,
        readout_mode=MODE, generated_weights=str(destination/'G_000.safetensors'),
        actual_weights=str(credit['weights']), history=segment, error=error))
    producers = dict(MT300=str(entry['mt_weights']), G_000=str(credit['weights']))
    parent = dict(reference=f'{version}_task_{path.task:04d}_G_000',
                  weights=str(destination/'G_000.safetensors'), histories=[descriptor]) if complete else None
    write_json_atomic(destination/'parent.json', dict(complete=complete, task=path.task, version=version,
        parent=parent, eligible_for_G_parent_bank=complete, readout_mode=MODE,
        actual_generated_parameter_ref='G_000', generated_weights=str(destination/'G_000.safetensors'),
        saved_endpoint_source=str(credit['weights']), actual_producer_refs=list(path.history.states),
        producer_refs=producers, real_H_source=path.bootstrap_provenance, real_G_practice=bool(path.boundaries),
        new_G_practice_episodes=len(path.boundaries), practice_receipt=str(destination/'practice.json'),
        practice_mapping=entry['mapping'], practice_segment=segment, boundaries=path.boundaries,
        query_rewards=credit['rewards'], query_count=16, query_receipt=str(credit['receipt']),
        query_source_receipt=str(credit['receipt']), query_credit_readonly=True, query_excluded_from_H=True,
        common_E_reference=credit['parent']['common_E_reference'], common_E_weights=credit['parent']['common_E_weights'],
        readonly_E_receipt=credit['parent'].get('readonly_E_receipt'),
        fixed_scale_reestimated=False, deterministic_endpoint_single_candidate=True,
        selection_did_not_use_reward_or_query=True, new_G_generations=0, new_query_episodes=0,
        new_MT_practice_episodes=0, kernel_identity=path.kernel_identity, timings=path.timings, error=error))


def _prepare(args, runtime, run, task, panel, cache):
    task_id = task['global_task_id']; spec = run['student_refresh']; condition = spec['condition_mapping'][str(task_id)]
    destination = args.root/'student_data'/spec['version']/f'task_{task_id:04d}'
    credit = _credit(args, runtime, run, task, panel)
    state = panel['practice_states'][(condition['path_ordinal']+1) % len(panel['practice_states'])]
    noise = seed(32, condition['noise_root'], 1)
    mapping = dict(init_state_id=state, env_seed=noise, policy_seed_root=noise,
                   history_episode_ordinal=1, parameter_ref='G_000')
    if spec['practice_mapping'][str(task_id)] != mapping or state in credit['states']:
        raise ValueError('saved practice mapping must freeze the original next episode, disjoint from query16')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'path.json').exists():
        path = restore_compilation(destination, runtime)
        if (path.sampling_identity != MODE or path.selected != 'G_000' or path.decisions or path.outputs
                or path.kernel_identity != runtime.fixed_kernel_identity
                or (path.demo, path.path_ordinal, path.noise_root) != tuple(condition[k]
                    for k in ('teacher_demo', 'path_ordinal', 'noise_root'))):
            raise ValueError('saved practice path identity changed')
    else:
        path, _ = begin_condition(runtime, task, condition['teacher_demo'], destination.name,
            condition['path_ordinal'], condition['noise_root'], 1, None, True, None, False, True)
        path.sampling_identity = MODE
        history, source = bootstrap_source(args.root, task_id, panel, 0, cache)
        if history.episodes[0].get('budget_truncated') or history.episodes[0]['init_state_id'] in credit['states']:
            raise ValueError('saved endpoint requires the original complete MT bootstrap, outside query16')
        for _ in bootstrap(path, task, panel, history=history, provenance=source):
            raise ValueError('saved endpoint practice cannot collect a new MT bootstrap')
        lock_saved_endpoint(runtime, path, destination)
    _link(destination/'G_000.safetensors', credit['weights']); _link(destination/'query', credit['directory']/'query')
    validate_lora_state(path.states['G_000'], runtime.lora)
    source = read_json(args.root/'bootstrap'/f'task_{task_id:04d}'/'complete.json')['history']
    if path.bootstrap_provenance != dict(reused=True, source=source, new_environment_steps=0):
        raise ValueError('saved endpoint practice must retain the original real MT bootstrap source')
    if (len(path.attempts) != 1 or path.attempts[0].get('decoder_calls') != 0
            or path.attempts[0].get('new_G_generation') is not False):
        raise ValueError('saved practice cannot regenerate an endpoint')
    return dict(path=path, task=task, panel=panel, mapping=mapping, destination=destination, credit=credit,
        bootstrap_records=sum(r['episode'] == 0 for r in path.history.records),
        mt_weights=(args.root/'initial.safetensors').resolve(strict=True), already_complete=False)


@torch.no_grad()
def saved_practice_group(args, runtime, runner, environment, run, task_ids):
    """Interleave one actual G episode per condition, retaining readonly query credit."""
    spec = run['student_refresh']; task_ids = tuple(task_ids); tick = time.monotonic()
    if (spec.get('readout_mode') != MODE or not run.get('saved_endpoint_complete')
            or not runtime.psi_frozen or runtime.version != spec['G_update']
            or set(runtime.saved_endpoint_rows) != set(TASKS)
            or len(task_ids) != len(set(task_ids)) or not set(task_ids) <= set(TASKS)):
        raise ValueError('saved practice requires the validated fixed terminal full36 kernel')
    registered = panels(args.root); cache = HistoryCache(); entries = []; active = {}; failures = []
    env_start = runner.environment_steps
    for task_id in task_ids:
        entry = _prepare(args, runtime, run, task_spec(environment, task_id), registered[task_id], cache)
        entries.append(entry); path = entry['path']
        if path.boundaries:
            error = _practice_error(entry)
            parent_file = entry['destination']/'parent.json'; parent = read_json(parent_file) if parent_file.exists() else {}
            node = dict(reference=f"{spec['version']}_task_{task_id:04d}_G_000",
                weights=str(entry['destination']/'G_000.safetensors'), histories=[dict(path=str(entry['destination']/'history.pt.gz'),
                records=len(path.history.records), episodes=2)])
            entry['already_complete'] = (error is None and parent.get('complete') and parent.get('eligible_for_G_parent_bank')
                and parent.get('readout_mode') == MODE and parent.get('parent') == node
                and (entry['destination']/'practice.json').exists() and read_json(entry['destination']/'practice.json').get('complete'))
            path.completed_rounds = int(error is None)
            if not entry['already_complete']: _persist(entry, spec['version'], error)
            if error: failures.append((task_id, error))
            continue
        if len(path.history.episodes) != 1 or path.completed_rounds:
            raise ValueError('saved practice cannot continue an unregistered partial path')
        _persist(entry, spec['version'], 'actual G practice not yet complete')
        driver = practice(path, entry['task'], entry['panel'], 'G_000', snapshots=False)
        request = next(driver)
        if (request['state_id'] != entry['mapping']['init_state_id']
                or request['noise_root'] != entry['mapping']['env_seed']
                or request['remaining'] < entry['task']['horizon']+10):
            raise ValueError('actual practice coroutine changed the registered next full episode')
        request.update(capture_history=True, capture_hidden=True, capture_phi=False, uses_teaching=False)
        active[request['episode_id']] = (entry, driver, request)
    def receive(output):
        identity = output['request']['episode_id']; entry, driver, _ = active[identity]
        try: driver.send(output)
        except StopIteration: pass
        else: raise ValueError('saved practice unexpectedly requested another episode')
        error = _practice_error(entry); entry['path'].completed_rounds = int(error is None)
        entry['path'].timings['practice_group_residence_seconds'] = time.monotonic()-tick
        _persist(entry, spec['version'], error)
        del active[identity]
        if error: failures.append((entry['path'].task, error))
        return []
    try:
        requests = sorted((value[2] for value in active.values()), key=lambda r:r['task']['horizon'], reverse=True)
        if requests:
            for _ in runner.run(requests, on_completion=receive): pass
        if active: raise ValueError('practice driver exited without complete actual episode receipts')
    except BaseException as error:
        for entry, _, _ in active.values(): _persist(entry, spec['version'], repr(error))
        raise
    if failures: raise ValueError(f'incomplete saved endpoint practice: {failures}')
    shared = dict(seconds=time.monotonic()-tick, new_environment_steps=runner.environment_steps-env_start,
                  members=list(task_ids), new_tasks=[e['path'].task for e in entries if not e['already_complete']])
    return [dict(task=e['path'].task, destination=str(e['destination']), parent_receipt=str(e['destination']/'parent.json'),
        practice_receipt=str(e['destination']/'practice.json'), query_receipt=str(e['credit']['receipt']),
        actual_generated_parameter_ref='G_000', query_count=16, query_rewards=e['credit']['rewards'],
        already_complete=e['already_complete'], shared_practice_group=shared) for e in entries]
