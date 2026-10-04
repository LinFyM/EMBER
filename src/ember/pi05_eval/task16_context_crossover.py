"""Fixed eight-case context crossover; retire after this registered readback."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import time

import torch
import ember.pi05_evaluation as evaluation
from ember.eval_adapters import load_evaluation_adapter
from ember.operator_writer.bank import PreparedOperatorLoRA
from ember.pi05_assets import prepare_libero_config, write_json_atomic
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.writer.topology import bind_current_process_to_cuda_numa

ROOT = Path('/data1/user/ymdai/ember_runs/task16_condition_context_crossover_20261004')
ORIGINAL = Path('/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400')
SCHEMA = 'ember_task16_condition_context_crossover_v1'


def panel() -> list[dict]:
    return [dict(case_id=f'scene{i}_teacher{j}_noise{k}', panel_ordinal=n,
                 physical_init_state_id=i, teacher_demo=j,
                 condition_id=f'task_16_demos_{j}', noise_stream_init_state_id=k,
                 source_condition_init_state_id=0 if j == 47 else 46,
                 original_diagonal=(i, j, k) in ((0, 47, 0), (46, 40, 46)))
            for n, (i, j, k) in enumerate(product((0, 46), (47, 40), (0, 46)))]


def contract() -> dict:
    value = json.loads((ORIGINAL / 'run_contract.json').read_text())
    value['original_run_contract'] = str(ORIGINAL / 'run_contract.json')
    value['original_git'] = value['git']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    value.update(schema_version=SCHEMA, role='task16_context_diagnostic',
                 mode='frozen_diagnostic', output_dir=str(ROOT),
                 git={'commit': commit, 'dirty': False},
                 contract_reference={'path': str(ROOT / 'run_contract.json')})
    task = next(t for t in value['tasks'] if (t['suite'], t['task_id']) == ('libero_object', 6))
    value['tasks'] = [task]
    value['adapter']['tasks'] = [next(t for t in value['adapter']['tasks']
        if (t['suite'], t['task_id']) == ('libero_object', 6))]
    keys = {c['condition_id'] for c in panel()}
    value['adapter']['conditions'] = [c for c in value['adapter']['conditions'] if c['condition_id'] in keys]
    assert {c['condition_id'] for c in value['adapter']['conditions']} == keys
    assert value['adapter']['condition_factors'] == 'complete_A0_plus_S_B0_plus_M'
    value['parallel'].update(envs_per_replica=8, worker_count=1, replicas_per_gpu=1,
        physical_gpu_count=1, physical_gpu_ids=[int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))])
    value['diagnostic_occupancy_capture'].update(mode='full', full_conditions=[],
        trajectory_root=str(ROOT / 'cases'),
        passive_trace={'schema_version': 'ember_operator_read_write_passive_capture_v1',
                       'trace_root': str(ROOT / 'cases')})
    value['task16_condition_context_crossover'] = dict(schema_version=SCHEMA,
        cases=panel(), qualification=False, original_mapping_unchanged=True,
        panel_ordinals_are_scheduler_keys_not_physical_states=True,
        original_results=str(ORIGINAL / 'results.json'),
        teacher_state_action_reward_reads=0, writer_native_calls=0,
        complete_condition_factors=True, shared_added_again=False,
        source_training='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed',
        C900_training_materialization='85919994aef11c17b49b7d0e70a2c110158bff61',
        C450_parent='a0e0248d96568e42b24a3d1c4102e2ca6e35a40c',
        reading_git=commit)
    value['command'] = ['scripts/task16_context_crossover.py']
    return value


@contextmanager
def panel_context(adapter, run_contract):
    """Bound only condition selection, stream identity and collision-free capture."""
    cases = panel()
    task = run_contract['tasks'][0]
    prepared = {j: adapter.prepare_episode(suite=task['suite'], task_id=6,
        init_state_id=0 if j == 47 else 46) for j in (47, 40)}
    streams = json.loads((ROOT / 'analysis/preparation/noise_stream_registration.json').read_text())['streams']
    start, noise, finish = (evaluation.start_fixed_episode,
                           evaluation.make_policy_noise, evaluation.finish_episode_row)

    def start_case(**arguments):
        case = cases[int(arguments['init_state_id'])]
        donor = prepared[case['teacher_demo']]
        assert donor.key == case['condition_id'] and donor.evidence['teacher_demo'] == case['teacher_demo']
        evidence = dict(schema_version=SCHEMA, **case,
            init_state_id=case['physical_init_state_id'],
            source_condition_evidence=deepcopy(donor.evidence), qualification=False)
        selected = PreparedOperatorLoRA(donor.key, evidence)
        arguments.update(init_state_id=case['physical_init_state_id'],
            task_adapter=SimpleNamespace(prepare_episode=lambda **kw: selected))
        slot = start(**arguments)
        slot['task16_case'] = case
        return slot

    def paired_noise(slots, **arguments):
        copies = [dict(slot, init_state_id=slot['task16_case']['noise_stream_init_state_id'])
                  for slot in slots]
        result, seeds = noise(copies, **arguments)
        for slot, seed in zip(slots, seeds, strict=True):
            expected = streams[str(slot['task16_case']['noise_stream_init_state_id'])]['seeds']
            assert seed == expected[slot['replan_index']]
        return result, seeds

    def finish_case(**arguments):
        case = arguments['slot']['task16_case']
        local = deepcopy(arguments['contract'])
        case_root = ROOT / 'cases' / case['case_id']
        local['diagnostic_occupancy_capture']['trajectory_root'] = str(case_root / 'trajectory')
        local['diagnostic_occupancy_capture']['passive_trace']['trace_root'] = str(case_root / 'continuous')
        arguments['contract'] = local
        row = finish(**arguments)
        row.update(case)
        row['task16_condition_context_crossover'] = dict(schema_version=SCHEMA, **case)
        assert row['init_state_id'] == case['physical_init_state_id']
        expected = streams[str(case['noise_stream_init_state_id'])]['seeds']
        assert row['policy_noise_seeds'] == expected[:len(row['policy_noise_seeds'])]
        write_json_atomic(case_root / 'results.json', row)
        return row

    evaluation.start_fixed_episode, evaluation.make_policy_noise, evaluation.finish_episode_row = (
        start_case, paired_noise, finish_case)
    try:
        yield
    finally:
        evaluation.start_fixed_episode, evaluation.make_policy_noise, evaluation.finish_episode_row = start, noise, finish


def main() -> None:
    assert not (ROOT / 'consumer_completion.json').exists()
    assert subprocess.check_output(['git', 'status', '--porcelain'], text=True) == ''
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == ''
    run = contract()
    write_json_atomic(ROOT / 'run_contract.json', run)
    torch.set_num_threads(8)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    affinity = bind_current_process_to_cuda_numa(0)
    assert affinity is not None, 'canonical consumer requires GPU-local CPU affinity'
    torch.set_grad_enabled(False)
    prepare_libero_config(ROOT / 'cache/libero_config')
    policy = adapter = pool = None
    started = time.monotonic()
    try:
        model, normalization, tokenizer = validate_worker_assets(run)
        policy, preprocess, postprocess = load_policy(model, normalization['stats'], tokenizer, run['policy'])
        adapter = load_evaluation_adapter(policy, run, device=torch.device('cuda:0'))
        assert len(adapter.lora.targets) == 38
        for key in {c['condition_id'] for c in panel()}:
            state = adapter._state(key)
            assert len(state) == 76 and all(torch.isfinite(v).all() for v in state.values())
        pool = PersistentTaskEnvironmentPool(run,
            physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
        envs, init_states = pool.switch(run['tasks'][0])
        assert len(envs) == 8
        rollout_started = time.monotonic()
        with panel_context(adapter, run):
            rows = evaluation.rollout_shard(envs=envs, init_states=init_states,
                task=run['tasks'][0], state_ids=list(range(8)), contract=run,
                policy=policy, preprocess=preprocess, postprocess=postprocess, task_adapter=adapter)
        rows.sort(key=lambda row: row['panel_ordinal'])
        assert [r['case_id'] for r in rows] == [c['case_id'] for c in panel()]
        assert all(r['occupancy_trajectory']['capture_level'] == 'full' and
                   r['continuous_control_trace'] for r in rows)
        write_json_atomic(ROOT / 'results.json', dict(schema_version=SCHEMA, episodes=rows))
        elapsed = time.monotonic() - rollout_started
        write_json_atomic(ROOT / 'consumer_completion.json', dict(complete=True,
            reading_git=run['git']['commit'], rows=8, full=8, model_loads=1,
            max_actual_inference_batch=adapter.max_inference_batch,
            rollout_seconds=elapsed, environment_steps=sum(r['steps'] for r in rows),
            environment_steps_per_second=sum(r['steps'] for r in rows)/elapsed,
            source_loading_and_setup_seconds=rollout_started-started,
            reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30,
            allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            GPU_local_CPU_affinity=affinity, logical_cases=panel(),
            packing='all eight authorized cases in one resident batch; no further legal cases to enlarge it',
            physical_effective_gradient='no learning', formal_mapping_modified=False,
            formal_guards_modified=False, extra_environment_smoke=0))
    finally:
        if adapter is not None:
            adapter.close()
        if pool is not None:
            pool.close()
        del policy
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
