"""Frozen180/360 train readouts and the single registered360 paired400."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import json
from pathlib import Path
import socket
import time
import traceback

from ember.expert_manifold.video_schedule import condition_demo_index
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_queue import (EvaluationShard, claim_next, complete_job, completed_jobs,
    fail_job, initialize_queue, publish_json_exclusive, queue_summary, read_json_with_size)
from ember.pi05_source_checkpoint import read_json
from .contract import SCHEMA


PARENT = Path('/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009')
FULL360 = Path('/data1/user/ymdai/ember_runs/parameter_compiler_fullreadout_20261009')
KINDS = {'train180': 180, 'train360': 360, 'formal360': 360}


def _seal(path, value):
    if Path(path).exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f'immutable evaluation record changed: {path}')
    else:
        publish_json_exclusive(Path(path), value)


def condition_shards(conditions, environment_contract, *, step_budget=1024):
    """One registered condition per claim; cost ordering has no GPU affinity.

    Collection's single output uses its manifest ordinal as the queue row ID;
    actual practice states always come from the condition's registered stream.
    """
    horizons = environment_contract['environment']['horizons']
    return tuple(EvaluationShard(job_id=c['condition_id'], ordinal=i, suite=c['suite'],
        task_id=int(c['suite_task_id']), horizon=int(horizons[c['suite']]),
        init_state_ids=tuple(c.get('final_state_ids', [i])),
        estimated_cost=step_budget + max(1, len(c.get('final_state_ids', []))) * int(horizons[c['suite']]))
        for i, c in enumerate(conditions))


def initialize_condition_queue(path, conditions, environment_contract, *, contract_reference,
                               step_budget=1024, recover_claims=False, retry_failed=False):
    initialize_queue(Path(path), condition_shards(conditions, environment_contract, step_budget=step_budget),
        contract_reference=str(contract_reference), recover_claims=recover_claims, retry_failed=retry_failed)
    return Path(path)


def prepare_evaluation(root, checkpoint, kind, *, recover_claims=False, retry_failed=False):
    """Seal original explicit identities before learning; weights may not exist yet."""
    if kind not in KINDS:
        raise ValueError('only train180/train360 and the single formal360 are registered')
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    source = PARENT / 'evaluation' / ('formal180' if kind == 'formal360' else 'train180')
    original = read_json(source / 'evaluation_contract.json')
    contract = deepcopy(original)
    contract.update(schema_version=SCHEMA, stage=kind, checkpoint=str(checkpoint), arms=['end'],
        inherited_contract=str(source / 'evaluation_contract.json'), no_evaluation_gradients=True,
        final_actor_only=True, Test_opened=False,
        MT_reuse=None if kind == 'formal360' else str(source / 'results.json'))
    if kind == 'formal360':
        contract['references'].update(parameter180=str(source / 'results.json'),
            parameter360=str(FULL360 / 'evaluation/formal360/results.json'))
    else:
        contract['references'] = {'MT': str(source / 'results.json')}
        if kind == 'train360':
            contract['references']['adjacent180'] = str(root / 'evaluation/train180/results.json')
    _validate_panel(contract)
    output = root / 'evaluation' / kind
    path = output / 'evaluation_contract.json'
    _seal(path, contract)
    initialize_condition_queue(output / 'queue.sqlite3', contract['conditions'],
        contract['environment_contract'], contract_reference=path,
        step_budget=contract['adaptation_step_budget'], recover_claims=recover_claims, retry_failed=retry_failed)
    return dict(contract=str(path), queue=str(output / 'queue.sqlite3'), output=str(output))


def _task(contract, condition):
    matches = [t for t in contract['environment_contract']['tasks']
               if t['global_task_id'] == condition['task_id']]
    if len(matches) != 1 or any(matches[0][k] != condition[v]
        for k, v in [('suite', 'suite'), ('task_id', 'suite_task_id'), ('language', 'language')]):
        raise ValueError('condition differs from its registered task authority')
    return matches[0]


def _validate_panel(contract):
    formal = contract['stage'] == 'formal360'
    conditions, expected = contract['conditions'], 400 if formal else 16
    if len(conditions) != expected or len({c['condition_id'] for c in conditions}) != expected:
        raise ValueError('registered panel has missing or duplicate conditions')
    if contract['adaptation_step_budget'] != 1024 or contract['expected_rows_per_arm'] != (400 if formal else 48):
        raise ValueError('registered adaptation/final budgets changed')
    tasks = defaultdict(list)
    for c in conditions:
        _validate_condition(contract, c, formal)
        tasks[c['task_id']].append(c)
    if len(tasks) != 8 or any(len(rows) != (50 if formal else 2) or
        len({c['teacher_demo'] for c in rows}) != len(rows) or
        (formal and {c['init_state_id'] for c in rows} != set(range(50))) for rows in tasks.values()):
        raise ValueError('panel requires eight tasks and all registered videos without replacement')


def _validate_condition(contract, c, formal):
    if not c['condition_id'] or '/' in c['condition_id'] or '..' in c['condition_id']:
        raise ValueError('unsafe condition identity')
    task, states = _task(contract, c), c['final_state_ids']
    if task['split_role'] != ('validation' if formal else 'train') or task['installed_init_state_count'] != 50:
        raise ValueError('readout changed the registered split/full50 pool')
    if not 0 <= c['teacher_demo'] < 50 or (not formal and states != [32, 33, 34]):
        raise ValueError('registered teacher/final-state panel changed')
    if formal:
        state = c['init_state_id']
        if states != [state] or c['video_ordinal'] != state or not 0 <= state < 50:
            raise ValueError('formal state/video ordinal changed')
        for arm, field in [('correct', 'paired_correct_demos'), ('same_task_other', 'paired_other_demos')]:
            demo = condition_demo_index(c['video_schedule_seed'], c['suite'], c['suite_task_id'], state,
                condition=arm, demo_count=50, sampling_mode='without_replacement')
            if c[field] != [demo] or (arm == 'correct' and c['teacher_demo'] != demo):
                raise ValueError('explicit formal condition differs from canonical video_schedule')


def _validate_row(row, condition, contract):
    task, root = _task(contract, condition), contract['policy_seed_root']
    expected = dict(suite=condition['suite'], task_id=condition['suite_task_id'],
        condition_id=condition['condition_id'], teacher_demo=condition['teacher_demo'], arm='end',
        language=condition['language'], split_role=task['split_role'], env_seed=root, policy_seed_root=root,
        compiler_checkpoint=contract['checkpoint'])
    if any(row.get(k) != v for k, v in expected.items()) or row['init_state_id'] not in condition['final_state_ids']:
        raise ValueError('raw row changed registered condition/scene/RNG authority')
    seeds = row['policy_noise_seeds']
    if not isinstance(row['success'], bool) or not seeds or seeds != [policy_noise_seed(root,
        row['suite'], row['task_id'], row['init_state_id'], i) for i in range(len(seeds))]:
        raise ValueError('raw row does not preserve actual canonical policy-noise evidence')
    scene = contract['environment_contract']['operator_read_write_scene']['root']
    expected_scene = str(Path(scene) / f"{row['suite']}_task_{row['task_id']:02d}_state_{row['init_state_id']:03d}.npz")
    if row.get('scene_reference', {}).get('path') != expected_scene or (
        'video_ordinal' in condition and row.get('video_ordinal') != condition['video_ordinal']):
        raise ValueError('raw row changed the registered final scene/video ordinal')


class _Cases:
    """Claims become independent live slots; completed adaptation is never repeated."""
    def __init__(self, runtime, contract, output, worker, capacity, code_git):
        self.runtime, self.contract, self.output = runtime, contract, output
        self.worker, self.capacity, self.code_git = worker, capacity, code_git
        self.cases, self.finals, self.completed = {}, deque(), []
        self.conditions = {c['condition_id']: c for c in contract['conditions']}

    def next_request(self):
        self.publish_ready()
        while True:
            if self.finals:
                return self.finals.popleft()
            if len(self.cases) >= self.capacity:
                return None
            claim = claim_next(self.output / 'queue.sqlite3', worker_id=self.worker)
            if claim is None:
                return None
            condition = self.conditions[claim.shard.job_id]
            destination = self.output / 'conditions' / claim.shard.job_id / claim.claim_token
            case = dict(claim=claim, condition=condition, destination=destination, rows=[], future=None)
            self.cases[claim.shard.job_id] = case
            if self._recover(case):
                self.publish_ready()
                continue
            return dict(kind='adapt', case_id=claim.shard.job_id, condition=condition,
                task=_task(self.contract, condition), role=self.contract['adaptation_environment_contract']['role'],
                step_budget=self.contract['adaptation_step_budget'], behavior_version=self.contract['checkpoint'],
                environment_contract=self.contract['adaptation_environment_contract'])

    def _recover(self, case):
        from safetensors.torch import load_file
        for path in sorted(case['destination'].parent.glob('*/record.json')):
            record = read_json(path)
            if not record.get('complete') or record.get('compiler_checkpoint') != self.contract['checkpoint']:
                continue
            if any(record.get(k) != v for k, v in case['condition'].items()):
                raise ValueError('completed adaptation identity changed')
            if not (path.parent / 'end.safetensors').is_file() or not (path.parent / 'experience.pt').is_file():
                raise ValueError('completed adaptation lost its actual parameters/experience')
            if set(record['practice_state_ids']) & set(case['condition']['final_state_ids']):
                raise ValueError('final initial state participated in this condition practice')
            case.update(destination=path.parent, metrics=record['metrics'], recovered_from=str(path))
            self._final_requests(case, load_file(str(path.parent / 'end.safetensors')))
            return True
        return False

    def adapted(self, result):
        from .storage import save_condition, tensor_bytes
        chain, case = result['chain'], self.cases[result['request']['case_id']]
        practiced = sorted({e['init_state_id'] for e in chain.episodes})
        if set(practiced) & set(case['condition']['final_state_ids']):
            raise ValueError('final initial state participated in this condition practice')
        case['metrics'] = chain.metrics
        metadata = dict(**case['condition'], compiler_checkpoint=self.contract['checkpoint'],
            compilation_code_git=self.code_git, practice_state_ids=practiced)
        case['future'] = self.runtime.io.submit(save_condition, case['destination'], metadata, chain,
            byte_cost=tensor_bytes(chain.to_record()) + tensor_bytes(chain.states))
        self._final_requests(case, chain.states[-1])

    def _final_requests(self, case, state):
        for state_id in case['condition']['final_state_ids']:
            path = case['destination'] / 'finals' / f'final_end_{state_id:02d}.json'
            if path.exists():
                row = read_json(path)
                _validate_row(row, case['condition'], self.contract)
                case['rows'].append(row)
            else:
                self.finals.append(dict(kind='final', case_id=case['claim'].shard.job_id,
                    task=_task(self.contract, case['condition']), state_id=state_id, state=state,
                    noise_root=self.contract['policy_seed_root'],
                    environment_contract=self.contract['environment_contract']))

    def final(self, result):
        case, row = self.cases[result['request']['case_id']], result['row']
        c = case['condition']
        row.update(arm='end', condition_id=c['condition_id'], teacher_demo=c['teacher_demo'],
            compiler_checkpoint=self.contract['checkpoint'], evaluation_code_git=self.code_git)
        if 'video_ordinal' in c:
            row['video_ordinal'] = c['video_ordinal']
        _validate_row(row, c, self.contract)
        # save_condition measures top-level facts while this atomic write runs.
        _seal(case['destination'] / 'finals' / f"final_end_{row['init_state_id']:02d}.json", row)
        case['rows'].append(row)

    def publish_ready(self, *, flush=False):
        for job, case in list(self.cases.items()):
            if len(case['rows']) != len(case['condition']['final_state_ids']):
                continue
            future = case['future']
            if future is not None:
                if not flush and not future.done():
                    continue
                future.result()
            rows, claim = sorted(case['rows'], key=lambda r: r['init_state_id']), case['claim']
            if [r['init_state_id'] for r in rows] != sorted(case['condition']['final_state_ids']):
                raise ValueError('completed condition changed final state coverage')
            path = f'shards/{job}-{claim.claim_token}.json'
            size = publish_json_exclusive(self.output / path, dict(condition=case['condition'], rows=rows,
                metrics=case['metrics'], artifact_root=str(case['destination'].relative_to(self.output)),
                recovered_from=case.get('recovered_from'), chain='experience.pt', weights='end.safetensors'))
            complete_job(self.output / 'queue.sqlite3', job_id=job, worker_id=self.worker,
                claim_token=claim.claim_token, rows_path=path, rows_bytes=size,
                row_count=len(rows), successes=sum(r['success'] for r in rows))
            self.completed.append(job)
            del self.cases[job]


def run_evaluation(runtime, args):
    """Drain this worker's dynamic claims; caller owns Runtime/GPU/cost lifecycle."""
    from .interaction import Runner
    output = Path(args.run_root).resolve() / 'evaluation' / args.evaluation
    contract = read_json(output / 'evaluation_contract.json')
    _validate_panel(contract)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = read_json(checkpoint / 'manifest.json')
    if (str(checkpoint) != contract['checkpoint'] or
        manifest['macro_update'] != KINDS[args.evaluation] or manifest['schema_version'] != SCHEMA):
        raise ValueError('worker changed its registered frozen shared checkpoint')
    runtime.load_checkpoint(checkpoint)
    runtime.compiler.eval()
    capacity = getattr(args, 'claim_batch', None) or args.slot_batch
    if not 1 <= capacity <= args.slot_batch:
        raise ValueError('claim capacity must fit actual persistent slots')
    identity = f'{args.worker_id}-{time.time_ns()}'
    cases = _Cases(runtime, contract, output, identity, capacity, getattr(args, 'code_git', None))
    runner = Runner(runtime, contract['adaptation_environment_contract'], args.physical_gpu, slot_batch=args.slot_batch)
    started, error = time.time(), None
    try:
        while True:
            for result in runner.run(cases.next_request):
                cases.adapted(result) if 'chain' in result else cases.final(result)
                cases.publish_ready()
            cases.publish_ready(flush=True)
            if cases.cases:
                raise ValueError('worker exited with incomplete claimed conditions')
            if not queue_summary(output / 'queue.sqlite3')['status_counts'].get('pending', 0):
                break
    except BaseException:
        error = traceback.format_exc()
        for job, case in cases.cases.items():
            fail_job(output / 'queue.sqlite3', job_id=job, worker_id=identity,
                     claim_token=case['claim'].claim_token, error=error)
        raise
    finally:
        try:
            runner.close()
        except BaseException:
            error = traceback.format_exc()
            raise
        finally:
            if error is not None:
                runner.preserve_partial(output / 'failures' / identity)
            receipt = dict(worker_id=identity, host=socket.gethostname(), physical_gpu=args.physical_gpu,
                started_unix=started, finished_unix=time.time(), error=error, complete=error is None,
                checkpoint=str(checkpoint), code_git=cases.code_git, completed_jobs=cases.completed,
                environment_steps=runner.total_environment_steps, components=runner.components,
                model_loading_cost_in_parent_lifecycle=True)
            _seal(output / 'workers' / f'{identity}.json', receipt)
    return receipt


def _reference_rows(path, arm='end'):
    value = read_json(Path(path))
    rows = value['arms'][arm]['rows'] if 'arms' in value else value['rows']
    result = []
    for row in rows:
        teacher = row.get('operator_read_write_lora', {})
        result.append({**row, **{k: teacher[k] for k in ('condition_id', 'teacher_demo', 'video_ordinal') if k in teacher}})
    return result


def compare_rows(reference, candidate):
    """Actual success sets include condition identity, preserving train's two videos."""
    key = lambda r: (r['suite'], r['task_id'], r['condition_id'], r['init_state_id'])
    left, right = {key(r): r for r in reference}, {key(r): r for r in candidate}
    if not left or len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError('paired comparison changed/duplicated complete condition-state coverage')
    for k, before in left.items():
        after = right[k]
        common = min(len(before['policy_noise_seeds']), len(after['policy_noise_seeds']))
        fields = ('language', 'env_seed', 'policy_seed_root', 'split_role', 'teacher_demo', 'scene_reference', 'video_ordinal')
        if any(before.get(f) != after.get(f) for f in fields) or not common or (
            before['policy_noise_seeds'][:common] != after['policy_noise_seeds'][:common]):
            raise ValueError('paired comparison changed actual scene/teacher/language/RNG')
    def counts(keys):
        a, b = {k for k in keys if left[k]['success']}, {k for k in keys if right[k]['success']}
        return dict(rows=len(keys), reference_successes=len(a), candidate_successes=len(b),
            retained=len(a & b), gained=len(b - a), lost=len(a - b), churn_count=len(a ^ b),
            churn_fraction=len(a ^ b) / len(keys), success_set_jaccard=len(a & b) / len(a | b) if a | b else None,
            retained_keys=sorted(a & b), gained_keys=sorted(b - a), lost_keys=sorted(a - b))
    tasks = [{**counts([k for k in left if k[:2] == t]), 'suite': t[0], 'task_id': t[1]} for t in sorted({k[:2] for k in left})]
    return dict(**counts(left), per_task=tasks,
        per_suite=[dict(suite=s, **counts([k for k in left if k[0] == s])) for s in sorted({k[0] for k in left})],
        reference_breadth=sum(t['reference_successes'] > 0 for t in tasks), candidate_breadth=sum(t['candidate_successes'] > 0 for t in tasks))


def _completed_rows(output, contract):
    conditions, rows = [], []
    for job in completed_jobs(output / 'queue.sqlite3'):
        result, size = read_json_with_size(output / job['rows_path'])
        condition = contract['conditions'][job['ordinal']]
        if result['condition'] != condition or size != job['rows_bytes']:
            raise ValueError('completed shard changed its sealed condition/raw-row publication')
        if (len(result['rows']) != job['row_count'] or
            sorted(r['init_state_id'] for r in result['rows']) != sorted(condition['final_state_ids']) or
            sum(r['success'] for r in result['rows']) != job['successes']):
            raise ValueError('completed shard changed its complete final-row coverage/successes')
        for row in result['rows']:
            _validate_row(row, condition, contract)
        conditions.append(result)
        rows.extend(result['rows'])
    return conditions, rows


def aggregate_evaluation(root, kind):
    output = Path(root).resolve() / 'evaluation' / kind
    contract, summary = read_json(output / 'evaluation_contract.json'), queue_summary(output / 'queue.sqlite3')
    expected = contract['expected_rows_per_arm']
    if summary['status_counts'] != {'complete': len(contract['conditions'])} or summary['completed_rows'] != expected:
        raise ValueError('complete registered evaluation is required; retain failed/partial evidence')
    conditions, rows = _completed_rows(output, contract)
    self_comparison = compare_rows(rows, rows)
    comparisons = {name: compare_rows(_reference_rows(path, 'MT' if name == 'MT' and kind != 'formal360' else 'end'), rows)
                   for name, path in contract['references'].items()}
    arm = dict(rows=rows, row_count=len(rows), successes=summary['successes'], breadth=self_comparison['candidate_breadth'],
        success_keys=[(r['suite'], r['task_id'], r['condition_id'], r['init_state_id']) for r in rows if r['success']],
        per_task=self_comparison['per_task'], per_suite=self_comparison['per_suite'])
    receipts = [read_json(p) for p in sorted((output / 'workers').glob('*.json'))]
    fields = ('actual_J', 'initial_reads', 'rereads', 'read_frames', 'environment_steps', 'resets',
              'failure_environment_steps', 'tail_environment_steps', 'edit_seconds', 'wall_seconds')
    distributions = {k: [c['metrics'][k] for c in conditions] for k in fields}
    cost = dict(condition_cost_values=distributions, condition_cost_totals={k: sum(v) for k, v in distributions.items()},
        practice_successes=sum(c['metrics']['practice_success'] for c in conditions),
        budget_stops=sum(c['metrics']['stop_reason'] == 'step_budget' for c in conditions),
        final_environment_steps=sum(r['environment_steps'] for r in rows), worker_receipts=receipts,
        attempted_confirmed_environment_steps=sum(r['environment_steps'] for r in receipts),
        worker_consumer_GPUh=sum(r['finished_unix'] - r['started_unix'] for r in receipts) / 3600,
        parent_cost_ledger=str(Path(root).resolve() / 'costs.jsonl'),
        failures=[str(p.relative_to(output)) for p in sorted((output / 'failures').glob('*'))])
    result = dict(schema_version=SCHEMA, stage=kind, checkpoint=contract['checkpoint'], aggregated_complete=True,
        arms={'end': arm}, comparisons=comparisons, conditions=conditions, cost=cost, queue=summary)
    _seal(output / 'results.json', result)
    _seal(output / 'complete.json', dict(complete=True, stage=kind, checkpoint=contract['checkpoint'],
        rows=expected, successes=arm['successes'], results=str(output / 'results.json')))
    return result
