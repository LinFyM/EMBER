"""Predeclared fixed-MT and chi180 actual parameter/experience collection."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import time
import traceback

from ember.pi05_eval_queue import (EvaluationShard, claim_next, complete_job, fail_job,
    initialize_queue, publish_json_exclusive, queue_summary)
from ember.pi05_source_checkpoint import read_json
from .contract import ASSET_ROOT, EVENT_SCHEMA, MT_PATH, SCHEMA, SOURCE, STAGE, learning_environment, state_stream
from .data import collection_conditions
from .storage import complete_pool, save_condition


def prepare(output, pool, *, checkpoint=None, asset_root=ASSET_ROOT, code_git=None,
            recover_claims=False, retry_failed=False):
    output, asset_root = Path(output), Path(asset_root)
    if pool not in {'pool0', 'refresh180'}:
        raise ValueError('unknown registered collection pool')
    if pool == 'pool0' and checkpoint is not None:
        raise ValueError('fixed behavior pool must use the unedited canonical MT')
    if pool == 'refresh180':
        if checkpoint is None:
            raise ValueError('refresh requires the frozen180 checkpoint')
        manifest = read_json(Path(checkpoint) / 'checkpoint_manifest.json')
        if (manifest['stage'] != STAGE or manifest['next_macro'] != 180
                or manifest['run_contract_schema'] != SCHEMA):
            raise ValueError('refresh policy changed its registered optimizer node')
    conditions = collection_conditions(pool, asset_root=asset_root)
    environment = learning_environment(asset_root=asset_root)
    contract = dict(schema_version=EVENT_SCHEMA, pool=pool, version=f'{pool}_20261009_v1',
        asset_root=str(asset_root), code_git=code_git, conditions=conditions,
        checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()),
        source=SOURCE, MT_reference=str(MT_PATH), environment_contract=environment,
        practice_step_budget=1024, behavior_actor_uses_teaching=pool != 'pool0',
        excluded_shared_final_states=[32, 33, 34], expected_condition_count=len(conditions),
        queue_result='one actual condition summary; shard init ID is its true first practice state')
    path = output / 'collection_contract.json'
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = read_json(path)
        # Repair code identity is separate; scientific input provenance remains.
        current = {k: v for k, v in contract.items() if k != 'code_git'}
        if current != {k: v for k, v in previous.items() if k != 'code_git'}:
            raise ValueError('immutable collection authority changed')
    else:
        publish_json_exclusive(path, contract)
    horizons = environment['environment']['horizons']
    shards = tuple(EvaluationShard(job_id=c['condition_id'], ordinal=index, suite=c['suite'],
        task_id=c['suite_task_id'], horizon=horizons[c['suite']],
        init_state_ids=(next(state_stream(c, c['excluded_states'])),), estimated_cost=1024)
        for index, c in enumerate(conditions))
    initialize_queue(output / 'queue.sqlite3', shards, contract_reference=str(path.resolve()),
                     recover_claims=recover_claims, retry_failed=retry_failed)
    return path


def aggregate(output):
    output = Path(output)
    contract = read_json(output / 'collection_contract.json')
    if queue_summary(output / 'queue.sqlite3')['status_counts'] != {'complete': len(contract['conditions'])}:
        raise ValueError('behavior pool is not fully collected')
    return complete_pool(output, contract)


def worker(args):
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .runtime import Runtime
    from .interaction import Runner
    from .run import check_budget
    import torch

    output = Path(args.output)
    contract = read_json(output / 'collection_contract.json')
    gpu = args.physical_gpu
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu) or torch.cuda.device_count() != 1:
        raise ValueError('collection must expose exactly its admitted physical GPU')
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if affinity is None:
        raise ValueError('collection requires verified GPU-local NUMA affinity')
    torch.set_grad_enabled(False)
    identity = f'{socket.gethostname()}-{gpu}-{os.getpid()}'
    runtime, runner, claimed, pending, completed = None, None, {}, [], []
    started, status, error = time.time(), 'failed', None
    try:
        runtime = Runtime(contract['asset_root'], args.device, frame_chunk=args.frame_chunk,
            native_frame_chunk=args.native_frame_chunk, decoder_chunk=args.decoder_chunk,
            experience_chunk=args.experience_chunk, cache_root=args.run_root / 'frozen_features')
        if contract['checkpoint'] is not None:
            runtime.load_checkpoint(contract['checkpoint'])
        runtime.compiler.eval()
        runner = Runner(runtime, contract['environment_contract'], gpu, slot_batch=args.slot_batch)
        tasks = {t['global_task_id']: t for t in contract['environment_contract']['tasks']}
        conditions = {c['condition_id']: c for c in contract['conditions']}
        fixed = contract['pool'] == 'pool0'
        def requests():
            while not (args.run_root / 'early_stop_requested.json').is_file():
                check_budget(args.run_root)
                claim = claim_next(output / 'queue.sqlite3', worker_id=identity, physical_gpu=gpu)
                if claim is None:
                    return
                claimed[claim.shard.job_id] = claim
                condition = conditions[claim.shard.job_id]
                roots = list((output / 'conditions').glob(condition['condition_id'] + '*'))
                recovered = False
                for root in roots:
                    path = root / 'record.json'
                    if not path.is_file():
                        continue
                    record = read_json(path)
                    if (record.get('complete') and record.get('schema_version') == EVENT_SCHEMA
                            and record.get('behavior_checkpoint') == contract['checkpoint']
                            and all(record.get(k) == v for k, v in condition.items())):
                        from .interaction import Chain
                        chain = Chain.from_record(torch.load(root / 'experience.pt', map_location='cpu', weights_only=False))
                        publish(condition, chain, claim, record)
                        recovered = True
                        break
                if recovered:
                    continue
                yield dict(kind='fixed' if fixed else 'adapt', condition=condition,
                    task=tasks[condition['task_id']], role='train',
                    behavior_version='MT' if fixed else contract['version'])
        def publish(condition, chain, claim, record):
            if chain.episodes[0]['init_state_id'] != claim.shard.init_state_ids[0]:
                raise ValueError('actual first practice state changed its registered stream')
            path = f'shards/{claim.shard.job_id}-{claim.claim_token}.json'
            size = publish_json_exclusive(output / path, dict(condition=condition,
                record_path=record['record_path'], metrics=chain.metrics,
                first_practice_state=chain.episodes[0]['init_state_id']))
            complete_job(output / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=identity,
                claim_token=claim.claim_token, rows_path=path, rows_bytes=size,
                row_count=1, successes=int(chain.metrics['practice_success']))
            completed.append(claim.shard.job_id)
            claimed.pop(claim.shard.job_id)
        def consume_finished(*, all_pending=False):
            remaining = []
            for condition, chain, claim, future in pending:
                if all_pending or future.done():
                    publish(condition, chain, claim, future.result())
                else:
                    remaining.append((condition, chain, claim, future))
            pending[:] = remaining
        for result in runner.run(requests()):
            condition, chain = result['request']['condition'], result['chain']
            claim = claimed[condition['condition_id']]
            destination = output / 'conditions' / condition['condition_id']
            # Failed attempts retain their files; a retry has its own root.
            if destination.exists():
                destination = output / 'conditions' / f"{condition['condition_id']}_{claim.claim_token}"
            from .storage import tensor_bytes
            future = runtime.io.submit(save_condition, destination,
                dict(**condition, behavior_checkpoint=contract['checkpoint'], collection_code_git=args.code_git),
                chain, fixed_behavior=fixed, byte_cost=tensor_bytes(vars(chain)))
            pending.append((condition, chain, claim, future))
            consume_finished()
        consume_finished(all_pending=True)
        status = 'scientific_stopped' if (args.run_root / 'early_stop_requested.json').is_file() else 'complete'
    except BaseException:
        error = traceback.format_exc()
        for claim in claimed.values():
            fail_job(output / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=identity,
                     claim_token=claim.claim_token, error=error)
        raise
    finally:
        if runner is not None:
            runner.close()
            if status == 'failed':
                runner.preserve_partial(output / 'failures' / identity)
        if runtime is not None:
            runtime.close()
            if runner is not None:
                runner.components['record_io_wait_seconds'] = runtime.io.wait_seconds
        receipt = dict(worker_id=identity, host=socket.gethostname(), physical_gpu=gpu,
            code_git=args.code_git, checkpoint=contract['checkpoint'], started_unix=started,
            finished_unix=time.time(), status=status, completed_jobs=completed, error=error,
            environment_steps=0 if runner is None else runner.total_environment_steps,
            components=None if runner is None else runner.components, cpu_affinity=affinity)
        publish_json_exclusive(output / 'workers' / f'{identity}.json', receipt)
    print(dict(scientific_stopped=status == 'scientific_stopped',
               queue_pending=queue_summary(output / 'queue.sqlite3')['status_counts'].get('pending', 0), status=status), flush=True)
    return receipt
