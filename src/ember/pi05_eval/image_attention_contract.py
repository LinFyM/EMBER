"""Frozen 48-case registration using existing banks and the canonical evaluator."""
from __future__ import annotations

import copy
import functools
import importlib.util
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

ROOT = Path('/data1/user/ymdai/ember_runs/self_image_attention_transfer_20261003')
REPO = Path(__file__).resolve().parents[3]
TAG = 'ember_self_image_attention_transfer_v1'
ARMS = ('C', 'N', 'C_from_N', 'N_from_C')
TASKS = {4: ('libero_spatial', 4, [28, 8, 0, 30], 220),
         13: ('libero_object', 3, [15, 8, 31, 30], 280),
         56: ('libero_90', 16, [4, 47, 30, 24], 400)}
STATES = (32, 33, 34, 35)
BANKS = {
    'C': Path('/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001/conditional_read_write_seen/banks/450/manifest.json'),
    'N': Path('/data1/user/ymdai/ember_runs/control_calibrated_read_write_20261003/control_calibrated_read_write_seen/banks/450/manifest.json'),
}


def route(arm):
    if arm not in ARMS:
        raise Pi05EvaluationError('unregistered image-attention arm')
    return arm[0], arm[-1] if '_from_' in arm else None


@functools.lru_cache(maxsize=2)
def inspected_bank(name):
    from ember.operator_writer.bank import inspect_bank
    from ember.operator_writer import joint_readout

    path = BANKS[name]
    raw = read_json(path)
    tasks = raw['tasks']
    original_record = joint_readout.source_record

    def historical_record(*args, **kwargs):
        spec, training, current = original_record(*args, **kwargs)
        historical = Path(raw['spec']['path'])
        if file_record(historical) != raw['spec'] or read_json(current) != read_json(historical):
            raise Pi05EvaluationError('historical bank spec semantics differ from current inspector')
        return spec, training, historical

    # Restore the exact old provenance path, after comparing the small specs.
    # The original inspector still checks every source, factor and scope field.
    joint_readout.source_record = historical_record
    try:
        bank = inspect_bank(manifest_path=path, source=raw['source'],
                            task_keys=tuple((t['suite'], t['task_id']) for t in tasks),
                            evaluation_role='operator_seen_training36', require_formal=True,
                            task_init_state_ids={(t['suite'], t['task_id']): STATES for t in tasks})
    finally:
        joint_readout.source_record = original_record
    selected = [t for t in tasks if t['global_task_id'] in TASKS]
    if len(selected) != 3 or raw.get('condition_factors') != 'complete_A0_plus_S_B0_plus_M':
        raise Pi05EvaluationError('original complete conditional bank changed')
    for task in selected:
        suite, local, teachers, _ = TASKS[task['global_task_id']]
        episodes = task['episodes']
        if ((task['suite'], task['task_id']) != (suite, local)
                or [e['init_state_id'] for e in episodes] != list(STATES)
                or [e['teacher_demo_indices'] for e in episodes] != [[d] for d in teachers]
                or [e['video_ordinal'] for e in episodes] != list(STATES)):
            raise Pi05EvaluationError('frozen teacher/state/video mapping changed')
    keys = {e['condition_id'] for t in selected for e in t['episodes']}
    bank = {**bank, 'tasks': selected,
            'conditions': [c for c in bank['conditions'] if c['condition_id'] in keys]}
    if len(bank['conditions']) != 12:
        raise Pi05EvaluationError('twelve source conditions missing')
    return bank


def registration(arm):
    recipient, donor = route(arm)
    return {'schema_version': TAG, 'study_id': ROOT.name, 'arm': arm,
            'recipient': recipient, 'donor': donor,
            'banks': {k: file_record(v) for k, v in BANKS.items()},
            'global_tasks': list(TASKS), 'states': list(STATES),
            'layers': 18, 'slots': 50, 'flow_steps': 10,
            'image_mass': 'recipient', 'non_image_probabilities': 'recipient',
            'donor_input': 'recipient_current_x_tau_tau_same_prefix',
            'donor_integrations': 0, 'diagnostic_only': donor is not None,
            'teacher_runtime_reads': 0, 'training': False}


def inspect_adapter(arm, source):
    recipient, _ = route(arm)
    bank = copy.deepcopy(inspected_bank(recipient))
    other = inspected_bank('N' if recipient == 'C' else 'C')
    if bank['source'] != source or other['source'] != source or bank['tasks'] != other['tasks']:
        raise Pi05EvaluationError('source or paired task identity differs between banks')
    if bank['scene_root'] != other['scene_root']:
        raise Pi05EvaluationError('C/N scene roots differ')
    bank['self_image_attention_transfer'] = registration(arm)
    return bank


def capture_contract(output):
    full = [{'suite': t[0], 'task_id': t[1], 'init_state_id': 32} for t in TASKS.values()]
    selector = ROOT / 'launch' / f'capture_{output.name}.json'
    capture = {'schema_version': 'ember_pi05_registered_trajectory_capture_v1',
               'mode': 'compact', 'full_conditions': full,
               'selection_path': str(selector), 'selection_bytes': selector.stat().st_size,
               'trajectory_root': str(output / 'trajectories'),
               'passive_trace': {'schema_version': 'ember_operator_read_write_passive_capture_v1',
                                 'trace_root': str(output / 'continuous_traces')},
               'training_gradient_use': False, 'checkpoint_selection_use': False,
               'validation_use': False, 'test_use': False}
    stage = {'schema_version': 'ember_pi05_stage_predicate_capture_v1',
             'capture': 'all_rows_post_settling_then_every_executed_control_step',
             'predicate_source': 'installed_LIBERO_BDDL_goal_conjunction',
             'full_conditions_only': False, 'training_gradient_use': False,
             'checkpoint_selection_use': False, 'validation_action_reads': 0,
             'validation_reward_reads': 0, 'held_data_use': False,
             'claim_boundary': 'BDDL predicates are partial progress signals'}
    return capture, stage


def prepare(devices, replicas, *, env_batch):
    from ember.pi05_eval_contract import (TargetTaskContract, load_evaluation_authorities,
                                         inspect_source_checkpoint, inspect_tokenizer)
    from ember.pi05_eval.run_contract import build_run_contract
    from ember.pi05_eval.preparation import shards_from_contract
    from ember.pi05_eval_queue import initialize_queue

    template_root = BANKS['C'].parents[2] / 'evaluation/correct144'
    template = read_json(template_root / 'run_contract.json')
    authorities = load_evaluation_authorities(REPO / 'configs/libero_24_8_8_coverage_v1/evaluation.json', REPO)
    model = inspect_source_checkpoint(authorities, Path(template['model']['source_run']),
                                      Path(template['model']['checkpoint']), evaluation_mode='formal')
    tokenizer = inspect_tokenizer(authorities, Path(template['tokenizer']['path']))
    keys = {(v[0], v[1]): v[3] for v in TASKS.values()}
    tasks = [TargetTaskContract(**t) for t in template['tasks'] if (t['suite'], t['task_id']) in keys]
    if len(tasks) != 3 or any(tuple(t.init_state_ids) != STATES or t.horizon != keys[t.suite, t.task_id] for t in tasks):
        raise Pi05EvaluationError('original runtime task/horizon identity changed')
    for arm in ARMS:
        output = ROOT / 'evaluation' / arm
        output.mkdir(parents=True, exist_ok=False)
        selector = ROOT / 'launch' / f'capture_{arm}.json'
        write_json_atomic(selector, {'schema_version': TAG, 'arm': arm,
                                    'full_conditions': [{'suite': t.suite, 'task_id': t.task_id,
                                                         'init_state_id': 32} for t in tasks]})
        contract = build_run_contract(authorities=authorities, tasks=tasks,
                                      libero_paths=template['libero_paths'], model=model, tokenizer=tokenizer,
                                      output_dir=output, role='operator_seen_training36', mode='formal',
                                      replicas_per_gpu=replicas, command=sys.argv,
                                      adapter=inspect_adapter(arm, model), physical_gpu_ids=devices)
        contract['parallel']['envs_per_replica'] = env_batch
        # Keep the twelve legitimate panel cases in full task batches. Runtime
        # workers still claim dynamically, and may enter later ready panels.
        contract['parallel']['queue_sharding'] = {
            'envs_per_replica': env_batch, 'shard_target_cost': 4160,
            'physical_gpu_count': 1, 'replicas_per_gpu': 1}
        contract['self_image_attention_transfer'] = registration(arm)
        contract['diagnostic_task_subset'] = None
        contract['diagnostic_occupancy_capture'], contract['diagnostic_stage_predicates'] = capture_contract(output)
        contract['operator_read_write_scene'] = template['operator_read_write_scene']
        contract['passive_capture_provenance'] = {'schema_version': TAG,
                                                 'registration': registration(arm),
                                                 'evaluation_commit': contract['git']['commit']}
        shutil.copytree(template_root / 'libero_config', output / 'libero_config')
        write_json_atomic(output / 'run_contract.json', contract)
        validate_contract(contract, REPO)
        initialize_queue(output / 'queue.sqlite3', shards_from_contract(contract),
                         contract_reference=contract['contract_reference'])
    return {'arms': list(ARMS), 'cases': 48, 'full': 12, 'compact': 36}


def validate_contract(contract, repo_root):
    del repo_root
    tag = contract['self_image_attention_transfer']
    arm = tag['arm']
    expected = registration(arm)
    output = ROOT / 'evaluation' / arm
    if (tag != expected or contract['adapter'] != inspect_adapter(arm, contract['model'])
            or contract['output_dir'] != str(output) or contract['mode'] != 'formal'
            or contract['role'] != 'operator_seen_training36'):
        raise Pi05EvaluationError('frozen attention contract/source/arm changed')
    keys = {(v[0], v[1]): v[3] for v in TASKS.values()}
    if (len(contract['tasks']) != 3 or set(keys) != {(t['suite'], t['task_id']) for t in contract['tasks']}
            or any(t['init_state_ids'] != list(STATES) or t['horizon'] != keys[t['suite'], t['task_id']] for t in contract['tasks'])
            or contract['policy']['num_inference_steps'] != 10 or contract['policy']['replan_steps'] != 5
            or contract['rng']['inference_seed'] != 7):
        raise Pi05EvaluationError('frozen finite case or official policy contract changed')
    capture, stage = capture_contract(output)
    if contract['diagnostic_occupancy_capture'] != capture or contract['diagnostic_stage_predicates'] != stage:
        raise Pi05EvaluationError('twelve full and thirty-six compact scope changed')
    bank = contract['adapter']
    if contract['operator_read_write_scene']['manifest'] != bank['scene_manifest']:
        raise Pi05EvaluationError('canonical scene registry differs from bank')


def validate_episode(adapter, row, suite, task_id, state):
    from ember.operator_writer.bank import validate_episode as bank_validate

    evidence = row.get('operator_read_write_lora')
    if not isinstance(evidence, dict):
        return False
    evidence = dict(evidence)
    tag = evidence.pop('self_image_attention_transfer', {})
    if tag.get('registration') != adapter['self_image_attention_transfer']:
        return False
    effect = tag.get('effect', {})
    path = Path(effect.get('path', ''))
    return (bank_validate(adapter, evidence, suite=suite, task_id=task_id, init_state_id=state)
            and path.is_relative_to(ROOT / 'evaluation') and path.is_file()
            and path.stat().st_size == effect.get('bytes')
            and not any(row.get(k) for k in ('task_expert', 'static_task_lora', 'conditional_velocity_lora')))


def run_linked_worker(output, worker_id):
    from ember import pi05_evaluation as evaluator
    from ember.pi05_eval_contract import load_run_contract

    if output != ROOT / 'evaluation' / ARMS[0]:
        raise Pi05EvaluationError('linked worker must start at the first registered panel')
    runtime = None

    def provider(path, identity):
        nonlocal runtime
        if runtime is None:
            runtime = evaluator._initialize_worker(path, identity)
        else:
            runtime.output_dir, runtime.queue_path = path, path / 'queue.sqlite3'
            runtime.contract = load_run_contract(path / 'run_contract.json')
            runtime.tasks = evaluator.task_lookup(runtime.contract)
            runtime.event_path = path / 'workers' / f'{identity}.jsonl'
        runtime.task_adapter.select_panel(runtime.contract)
        return runtime

    try:
        for arm in ARMS:
            evaluator.run_worker(output_dir=ROOT / 'evaluation' / arm, worker_id=worker_id,
                                 runtime_provider=provider, close_pool=False)
        import torch

        write_json_atomic(ROOT / 'analysis' / f'worker_{worker_id}_packing.json',
                          {'pid': os.getpid(), 'source_loads': 1,
                           'packing': runtime.task_adapter.packing,
                           'cuda_peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                           'cuda_peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30})
    finally:
        if runtime is not None:
            runtime.task_adapter.close()
            runtime.pool.close()


def launch_linked(script):
    """One canonical launcher lease; shared source workers drain all four queues."""
    from ember.pi05_eval import recovery, launcher
    from ember.pi05_eval_queue import queue_summary
    from ember.pi05_eval_contract import load_run_contract

    module_spec = importlib.util.spec_from_file_location('canonical_pi05_launch', REPO / 'scripts/evaluate_pi05.py')
    owner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(owner)
    outputs = [ROOT / 'evaluation' / arm for arm in ARMS]
    contracts = [load_run_contract(p / 'run_contract.json') for p in outputs]
    for contract in contracts:
        recovery.validate_resume_inputs(contract)
    first = contracts[0]
    devices, replicas = first['parallel']['physical_gpu_ids'], first['parallel']['replicas_per_gpu']
    preflight = launcher.gpu_preflight(devices, materialized_lora_replicas=replicas)
    if not launcher.evaluator_gpus_are_eligible(preflight):
        raise Pi05EvaluationError('live GPUs fail canonical evaluator admission')
    workers = recovery.worker_ids(replicas, devices)
    lease, started = uuid.uuid4().hex, time.time()
    for output, contract in zip(outputs, contracts, strict=True):
        owner._append_jsonl(output / 'invocations.jsonl',
                           {'event': 'started', 'unix': started, 'invocation_id': lease,
                            'argv': sys.argv, 'contract_reference': contract['contract_reference'],
                            'worker_ids': workers, 'preflight': preflight,
                            'linked_same_source_panels': list(ARMS)})
    processes, codes, error = launcher.spawn_worker_processes(outputs[0], first, workers,
                                                              invocation_id=lease, repo_root=REPO,
                                                              script_path=script)
    for output, contract in zip(outputs, contracts, strict=True):
        queue = queue_summary(output / 'queue.sqlite3')
        if error is not None or any(codes.get(w) != 0 for w in workers) or set(queue['status_counts']) != {'complete'}:
            owner._fail_launcher_invocation(output, invocation_id=lease, started_unix=started,
                                             processes=processes, return_codes=codes, queue=queue, launch_error=error)
        owner._publish_launcher_completion(output, contract=contract, invocation_id=lease, started_unix=started,
                                            worker_ids=workers, processes=processes, return_codes=codes,
                                            queue=queue, preflight=preflight)
        owner._finalize_aggregate(output)
