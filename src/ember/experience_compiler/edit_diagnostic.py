"""Frozen one-edit diagnostic using the canonical queue and native Runner."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import os
from pathlib import Path
import socket
import time
import traceback

from ember.pi05_eval_queue import (EvaluationShard, claim_next, complete_job, completed_jobs,
    fail_job, initialize_queue, publish_json_exclusive, queue_summary)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .evaluation import immutable, _summary, _task, compare_train


SCHEMA = 'ember_parameter_edit_credit_diagnostic_v1'
PARENT = Path('/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009')
PANEL = ((0, 43), (0, 29), (13, 16), (13, 4), (20, 15), (20, 6), (32, 16), (32, 44))
ARMS = ('I', 'E+', 'E0')


def prepare(root, *, code_git, recover_claims=False, retry_failed=False):
    from .contract import MT_PATH
    root = Path(root).resolve()
    parent = PARENT / 'evaluation/train360'
    authority = read_json(parent / 'evaluation_contract.json')
    original = read_json(parent / 'results.json')
    by_key = {(c['condition']['task_id'], c['condition']['teacher_demo']): c for c in original['conditions']}
    conditions = []
    for key in PANEL:
        meta = by_key[key]
        condition = deepcopy(meta['condition'])
        source = parent / meta['artifact_root']
        record = read_json(source / 'record.json')
        event = record['events'][-1]
        if condition['final_state_ids'] != [32, 33, 34] or record['compiler_checkpoint'] != authority['checkpoint']:
            raise ValueError('diagnostic panel or recorded compiler changed')
        incoming = MT_PATH if event['incoming'] == 'MT' else source / event['incoming']
        condition.update(actual_J=meta['actual_J'], source_record=str(source / 'record.json'),
            source_experience=str(source / 'experience.pt'), last_event=event,
            original_metrics=meta['metrics'], weights={'I': str(incoming), 'E+': str(source / 'end.safetensors'),
                'E0': str(root / 'materialized' / condition['condition_id'] / 'E0.safetensors')})
        if not all(Path(condition['weights'][arm]).is_file() for arm in ('I', 'E+')):
            raise ValueError('diagnostic lost its actual complete incoming/outgoing')
        conditions.append(condition)
    contract = dict(schema_version=SCHEMA, stage='parameter_edit_credit_diagnostic_20261009',
        parent=str(PARENT), asset_root=authority['asset_root'], checkpoint=authority['checkpoint'],
        environment_contract=authority['environment_contract'], conditions=conditions, arms=list(ARMS),
        expected_rows_per_arm=24, no_optimizer=True, no_new_adaptation=True, Test_opened=False,
        limits=dict(GPU_hours=6, science_elapsed_hours=3, artifact_GiB=24,
                    environment_steps=24480, offline_ten_step_predictions=576, FM_predictions=672),
        sampling=dict(query_seed_domain=0xED17, points=6, queries=[7, 4]), parent_read_only=True)
    immutable(root / 'run_contract.json', contract)
    write_json_atomic(root / f'preparation_{os.getpid()}.json', dict(code_git=code_git, recorded_unix=time.time()))
    shards = [EvaluationShard(f"{c['condition_id']}-{arm}", ordinal, c['suite'], c['suite_task_id'],
        authority['environment_contract']['environment']['horizons'][c['suite']], (32, 33, 34),
        3 * authority['environment_contract']['environment']['horizons'][c['suite']])
        for ordinal, (c, arm) in enumerate((c, a) for c in conditions for a in ARMS)]
    initialize_queue(root / 'evaluation/queue.sqlite3', shards, contract_reference=str(root / 'run_contract.json'),
        recover_claims=recover_claims, retry_failed=retry_failed)
    return dict(conditions=len(conditions), rows=72, queue=queue_summary(root / 'evaluation/queue.sqlite3'))


def _budget(root):
    from .run import check_budget
    return check_budget(root, gpu_hours=6, science_hours=3, artifact_gib=24)


def materialize(runtime, root, contract, code_git):
    import torch
    from safetensors.torch import load_file, save_file
    results = []
    for condition in contract['conditions']:
        _budget(root)
        destination = Path(condition['weights']['E0']).parent
        receipt = destination / 'record.json'
        if receipt.exists():
            result = read_json(receipt)
            if not result['complete'] or result['condition'] != condition or not Path(condition['weights']['E0']).is_file():
                raise ValueError('E0 recovery changed the actual fixed incoming')
        else:
            started = time.monotonic()
            incoming = load_file(condition['weights']['I'], device=str(runtime.device))
            teacher = runtime.teacher(condition['task_id'], condition['teacher_demo'], 'train')
            cost = dict(runtime.last_teacher_cost)
            state = runtime.edit(incoming, teacher, {})
            if state.keys() != incoming.keys() or any(v.shape != incoming[k].shape or not torch.isfinite(v).all()
                                                       for k, v in state.items()):
                raise ValueError('same-incoming empty-E edit lost the complete finite factors')
            destination.mkdir(parents=True, exist_ok=True)
            path = Path(condition['weights']['E0'])
            partial = path.with_suffix('.partial')
            save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(partial))
            partial.replace(path)
            result = dict(schema_version=SCHEMA, condition=condition, checkpoint=contract['checkpoint'],
                code_git=code_git, experience_input='empty_dictionary_same_actual_incoming',
                full_video_equivalent_reads=1, teacher_frames=len(teacher['indices']),
                native_feature_cost=cost, wall_seconds=time.monotonic() - started, complete=True)
            write_json_atomic(receipt, result)
            print('EMBER_E0 ' + condition['condition_id'], flush=True)
        results.append(result)
    write_json_atomic(root / 'materialized/results.json', dict(conditions=results, complete=True))
    return dict(conditions=8, new_environment_steps=0)


def _capture_contract(contract, root, condition, arm):
    value = deepcopy(contract['environment_contract'])
    destination = root / 'evaluation/capture' / condition['condition_id'] / arm
    value['diagnostic_stage_predicates'] = dict(full_conditions_only=False)
    value['diagnostic_occupancy_capture'] = dict(mode='compact', full_conditions=[],
        trajectory_root=str(destination / 'compact'), passive_trace=dict(trace_root=str(destination / 'passive')))
    value['output_dir'] = str(root / 'evaluation')
    return value


def _save_result(root, condition, arm, result, code_git, checkpoint):
    import torch
    state_id, row = result['row']['init_state_id'], result['row']
    row.update(condition_id=condition['condition_id'], teacher_demo=condition['teacher_demo'], arm=arm,
        actual_J=condition['actual_J'], evaluation_code_git=code_git, compiler_checkpoint=checkpoint)
    for name in ('occupancy_trajectory', 'continuous_control_trace'):
        if name in row:
            row[name]['diagnostic_identity'] = dict(condition_id=condition['condition_id'], arm=arm)
    trace = root / 'evaluation/traces' / condition['condition_id'] / arm / f'state{state_id:02d}.pt'
    trace.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=SCHEMA, condition_id=condition['condition_id'], arm=arm,
        init_state_id=state_id, row=row, records=result['trajectory'], terminal_raw=result['terminal_raw'],
        units=dict(normalized_actions='normalized native first7 of50x32 latent',
                   commands='unnormalized environment commands', actions='actually executed environment commands'),
        complete=True)
    if sum(int(r['executed'].sum()) for r in payload['records']) != row['steps']:
        raise ValueError('diagnostic trace lost actual environment prefix steps')
    partial = trace.with_suffix('.partial')
    torch.save(payload, partial)
    partial.replace(trace)
    row.update(raw_trace=str(trace), complete=True)
    immutable(trace.with_suffix('.json'), row)
    print(f"EMBER_DIAGNOSTIC_ROW {condition['condition_id']} {arm} {state_id} {int(row['success'])}", flush=True)
    return row


class Cases:
    """One dynamic queue claim owns one arm's three registered initial states."""
    def __init__(self, runtime, root, contract, identity, gpu, code_git):
        self.runtime, self.root, self.contract = runtime, root, contract
        self.identity, self.gpu, self.code_git = identity, gpu, code_git
        self.cases, self.requests, self.completed = {}, deque(), []
        self.by_id = {f"{c['condition_id']}-{a}": (c, a) for c in contract['conditions'] for a in ARMS}

    def next_request(self):
        from safetensors.torch import load_file
        self.publish_ready()
        while not self.requests:
            _budget(self.root)
            claim = claim_next(self.root / 'evaluation/queue.sqlite3', worker_id=self.identity, physical_gpu=self.gpu)
            if claim is None:
                return None
            condition, arm = self.by_id[claim.shard.job_id]
            state = load_file(condition['weights'][arm])
            case = dict(claim=claim, condition=condition, arm=arm, rows=[], futures=[])
            self.cases[claim.shard.job_id] = case
            for state_id in condition['final_state_ids']:
                trace = self.root / 'evaluation/traces' / condition['condition_id'] / arm / f'state{state_id:02d}.pt'
                if trace.with_suffix('.json').exists():
                    row = read_json(trace.with_suffix('.json'))
                    if not trace.is_file() or not row['complete'] or row['arm'] != arm or row['condition_id'] != condition['condition_id']:
                        raise ValueError('diagnostic recovery lost completed trace identity')
                    case['rows'].append(row)
                else:
                    self.requests.append(dict(kind='final', task=_task(self.contract, condition), state=state,
                        state_id=state_id, noise_root=7, capture=True, condition=condition, arm=arm,
                        case_id=claim.shard.job_id,
                        environment_contract=_capture_contract(self.contract, self.root, condition, arm)))
            self.publish_ready()
        return self.requests.popleft()

    def accept(self, result):
        from .storage import tensor_bytes
        request = result['request']
        case = self.cases[request['case_id']]
        future = self.runtime.io.submit(_save_result, self.root, case['condition'], case['arm'], result,
            self.code_git, self.contract['checkpoint'], byte_cost=tensor_bytes(result['trajectory']))
        case['futures'].append(future)
        self.publish_ready()

    def publish_ready(self, *, flush=False):
        for job_id, case in list(self.cases.items()):
            for future in list(case['futures']):
                if flush or future.done():
                    case['rows'].append(future.result())
                    case['futures'].remove(future)
            if len(case['rows']) != 3:
                continue
            rows = sorted(case['rows'], key=lambda r: r['init_state_id'])
            if [r['init_state_id'] for r in rows] != [32, 33, 34]:
                raise ValueError('diagnostic arm changed its state coverage')
            claim = case['claim']
            path = f'shards/{job_id}-{claim.claim_token}.json'
            size = publish_json_exclusive(self.root / 'evaluation' / path,
                dict(condition=case['condition'], arm=case['arm'], rows=rows))
            complete_job(self.root / 'evaluation/queue.sqlite3', job_id=job_id, worker_id=self.identity,
                claim_token=claim.claim_token, rows_path=path, rows_bytes=size, row_count=3,
                successes=sum(r['success'] for r in rows))
            self.completed.append(job_id)
            del self.cases[job_id]


def worker(args):
    import torch
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .runtime import Runtime
    from .interaction import Runner
    root = Path(args.run_root).resolve()
    contract = read_json(root / 'run_contract.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu) or torch.cuda.device_count() != 1:
        raise ValueError('diagnostic worker requires one admitted physical GPU')
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if affinity is None:
        raise ValueError('diagnostic worker needs GPU-local NUMA affinity')
    torch.set_grad_enabled(False)
    identity = f'{socket.gethostname()}-{args.physical_gpu}-{args.diagnostic_phase}-{os.getpid()}'
    started, runtime, runner, cases, result = time.time(), None, None, None, None
    error = None
    try:
        runtime = Runtime(contract['asset_root'], args.device, frame_chunk=args.frame_chunk,
            native_frame_chunk=args.native_frame_chunk, decoder_chunk=args.decoder_chunk,
            cache_root=root / 'frozen_features', feature_cache_bytes=128 * 1024**2)
        runtime.load_checkpoint(contract['checkpoint'])
        runtime.compiler.eval()
        if args.diagnostic_phase == 'materialize':
            result = materialize(runtime, root, contract, args.code_git)
        elif args.diagnostic_phase == 'readout':
            from .edit_readouts import read_condition
            conditions = contract['conditions']
            if args.condition_ids:
                chosen = set(args.condition_ids.split(','))
                conditions = [c for c in conditions if c['condition_id'] in chosen]
                if len(conditions) != len(chosen):
                    raise ValueError('offline worker requested unregistered conditions')
            result = []
            for condition in conditions:
                _budget(root)
                result.append(read_condition(runtime, root, condition, microbatch=args.microbatch))
        else:
            runner = Runner(runtime, contract['environment_contract'], args.physical_gpu, slot_batch=args.slot_batch)
            cases = Cases(runtime, root, contract, identity, args.physical_gpu, args.code_git)
            for output in runner.run(cases.next_request):
                cases.accept(output)
            cases.publish_ready(flush=True)
            if cases.cases:
                raise ValueError('diagnostic worker exited before its valid claimed rows completed')
            result = dict(completed_jobs=cases.completed)
    except BaseException:
        error = traceback.format_exc()
        if cases is not None:
            for job_id, case in cases.cases.items():
                fail_job(root / 'evaluation/queue.sqlite3', job_id=job_id, worker_id=identity,
                         claim_token=case['claim'].claim_token, error=error)
        raise
    finally:
        if runner is not None:
            runner.close()
            if error is not None:
                runner.preserve_partial(root / 'failures' / identity)
        if runtime is not None:
            runtime.close()
        write_json_atomic(root / 'workers' / f'{identity}.json', dict(worker_id=identity, code_git=args.code_git,
            phase=args.diagnostic_phase, started_unix=started, finished_unix=time.time(), cpu_affinity=affinity,
            physical_gpu=args.physical_gpu, host=socket.gethostname(), result=result, error=error,
            status='complete' if error is None else 'failed',
            environment_steps=0 if runner is None else runner.total_environment_steps,
            components=None if runner is None else runner.components,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    return result


def aggregate(root):
    root = Path(root).resolve()
    contract = read_json(root / 'run_contract.json')
    jobs = completed_jobs(root / 'evaluation/queue.sqlite3')
    if len(jobs) != 24:
        raise ValueError('diagnostic requires all72 rows including negatives')
    rows = {arm: [] for arm in ARMS}
    for job in jobs:
        shard = read_json(root / 'evaluation' / job['rows_path'])
        rows[shard['arm']].extend(shard['rows'])
    expected = {(c['condition_id'], state) for c in contract['conditions'] for state in (32, 33, 34)}
    if any(len(v) != 24 or {(r['condition_id'], r['init_state_id']) for r in v} != expected for v in rows.values()):
        raise ValueError('diagnostic aggregate changed the registered panel')
    comparisons = {f'{a}_to_{b}': compare_train(rows[a], rows[b]) for a, b in (('I', 'E+'), ('E0', 'E+'), ('I', 'E0'))}
    result = dict(schema_version=SCHEMA, arms={a: dict(rows=v, **_summary(v)) for a, v in rows.items()},
        comparisons=comparisons, conditions=contract['conditions'], aggregated_complete=True,
        environment_steps=sum(r['environment_steps'] for v in rows.values() for r in v))
    if result['environment_steps'] > contract['limits']['environment_steps']:
        raise ValueError('actual environment steps exceed the registered diagnostic')
    write_json_atomic(root / 'evaluation/results.json', result)
    return result
