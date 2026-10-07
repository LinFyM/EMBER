"""Finite complete-factor/text handoff over sealed task23 prefix consumers."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import resource
import sqlite3
import time

ROOT = Path('/data1/user/ymdai/ember_runs/cross_task_post_open_transfer_20261007')
SEALED = Path('/data1/user/ymdai/ember_runs/task23_post_open_language_20261006/frozen/src/ember/pi05_eval/post_open_language.py')
TEXTS = {'L23': 'open the top drawer and put the bowl inside',
         'L42': 'put the black bowl in the top drawer of the cabinet'}
DONORS = ('M23', 'M42_demo30', 'M42_demo14', 'M42_demo43', 'M42_demo09')


def consumers():
    """Only archived prefix/plan/capture functions; no archived dispatcher runs."""
    spec = importlib.util.spec_from_file_location('sealed_language_functions', SEALED)
    language = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(language)
    language.ROOT = ROOT
    language.TEXTS = {f'{donor}_{text}': value for donor in DONORS
                      for text, value in TEXTS.items()}
    return language, language.prefix_owner()


def factor_states(cohort):
    """Reuse shared A0 and each complete B0+M, separate from environment IDs."""
    from safetensors.torch import load_file
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract

    contract = json.loads(Path(cohort['rows'][0]['source']['source_run_contract']).read_text())
    target = contract['adapter']
    donor_contract = json.loads(Path(cohort['donor_run_contract']).read_text())
    donor_bank = donor_contract['adapter']
    if any(target[k] != donor_bank[k] for k in ('checkpoint', 'shared', 'lora')):
        raise ValueError('donor is not the same complete T2340/shared topology')
    spec = json.loads(Path(target['spec']['path']).read_text())
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(target['asset_root']) / spec['source']['lora_contract']), rank=128)
    if lora.to_dict() != target['lora'] or len(lora.targets) != 38:
        raise ValueError('original complete LoRA topology changed')
    shared = load_file(target['shared']['path'], device='cpu')
    if len(shared) != 38 or not all(k.endswith(LORA_A_SUFFIX) for k in shared):
        raise ValueError('shared bank must be complete A0')
    records = {r['condition_id']: r for bank in (target, donor_bank)
               for r in bank['conditions']}
    keys = {r['source']['source_row']['operator_read_write_lora']['condition_id']
            for r in cohort['rows']} | {r['condition_id'] for r in cohort['donors']}
    states, provenance = {}, {}
    for key in sorted(keys):
        record = records[key]
        path = Path(record['factors']['path'])
        if path.stat().st_size != record['factors']['bytes']:
            raise ValueError('required original condition factor is unavailable')
        condition = load_file(str(path), device='cpu')
        if len(condition) != 38 or not all(k.endswith(LORA_B_SUFFIX) for k in condition):
            raise ValueError('condition bank must be complete B0+M')
        states[key] = {**shared, **condition}
        validate_lora_state(states[key], lora)
        provenance[key] = record
    return contract, lora, states, dict(checkpoint=target['checkpoint'],
        shared=target['shared'], complete_targets=38, rank=128,
        condition_records=provenance, target_global_task=23,
        donor_global_task=42, source_model=contract['model'])


def claim(capacity, worker_id, phase):
    with sqlite3.connect(ROOT/'launch/jobs.sqlite', timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        indices = [r[0] for r in db.execute(
            'SELECT pair_id FROM pairs WHERE phase=? AND status="pending" '
            'ORDER BY remaining DESC,pair_id LIMIT ?', (phase, capacity))]
        for index in indices:
            db.execute('UPDATE pairs SET status="running",worker=? WHERE pair_id=?',
                       (worker_id, index))
    return indices


def finish(slot, language, *, partial=False):
    result = language.finish(slot, slot['prefix_consumer'], partial=partial)
    result.update(evaluation_global_task_id=23, factor_label=slot['factor_label'],
        parameter_global_task_id=23 if slot['factor_label'] == 'M23' else 42,
        condition_id=slot['factor_key'], compile_language=slot['compile_language'],
        language_label=slot['language_label'], parameter_source=slot['parameter_source'],
        complete_factor_count=76, complete_targets=38, total_horizon=300,
        old_Full_used_only_as_branch_reference=True)
    name = 'partial_row.json' if partial else 'row.json'
    language.write(slot['directory']/name, result)
    if not partial and slot['directory'] != slot['canonical']:
        language.write(slot['canonical']/'row.json', result)
    return result


def prepare_group(job, cohort, environments, active, contract, language, prefix,
                  owner, worker_id, identity):
    row = cohort['rows'][job['cohort_index']]
    exact = language.stored_branch(row, 'exact_full', exact=True)
    if exact is None or not exact['admitted']:
        raise ValueError('registered actual oldFull branch reference is unavailable')
    pair = []
    for label in TEXTS:
        arm = f"{job['factor_label']}_{label}"
        saved = language.stored_branch(row, arm)
        if saved is not None:
            pair.append(saved)
            continue
        slot = prefix.begin(environments[len(active)], row, arm, contract, owner, worker_id)
        prefix.pair_check(exact, slot)
        slot.update(paired_exact_full=slot['pairing'].copy(),
            exact_full_reference=exact['reference_path'], prefix_consumer=prefix,
            factor_key=job['condition_id'], factor_label=job['factor_label'],
            language_label=label, compile_language=TEXTS['L23' if job['factor_label']=='M23' else 'L42'],
            parameter_source=identity['condition_records'][job['condition_id']])
        active.append(slot)
        pair.append(slot)
    prefix.pair_check(*pair)


def run_worker(capacity, worker_id, gpu_id, phase):
    import torch
    import ember.pi05_eval_contract  # initialize canonical facade before sealed leaves
    from ember.batched_lora import BatchedLoRAInference
    from ember.lora import inject_task_lora
    from ember.pi05_assets import configure_libero_runtime_assets
    from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
    from ember.pi05_eval.episode import stage_predicate_snapshot
    from ember.pi05_eval.trajectory_capture import record_passive_step
    from ember.writer.topology import bind_current_process_to_cuda_numa

    if (ROOT/'retired_entrypoints.json').exists() or (ROOT/'completion.json').exists():
        raise ValueError('finite transfer study is sealed')
    started, cpu_started = time.time(), time.process_time()
    torch.set_num_threads(2)
    bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    cohort = json.loads((ROOT/'cohort.json').read_text())
    jobs = json.loads((ROOT/'jobs.json').read_text())['pairs']
    language, prefix = consumers()
    owner = prefix.replay_owner()
    contract, lora, states, identity = factor_states(cohort)
    model_path, normalization, tokenizer = validate_worker_assets(contract)
    policy, processor, _ = load_policy(model_path, normalization['stats'], tokenizer, contract['policy'])
    policy.requires_grad_(False)
    inject_task_lora(policy, lora)
    policy.requires_grad_(False)
    batched = BatchedLoRAInference(policy, lora)
    configure_libero_runtime_assets(Path(contract['libero_paths']['assets']))
    from libero.libero.envs import OffScreenRenderEnv
    task = cohort['rows'][0]['source']['task']
    bddl = Path(contract['libero_paths']['bddl_files'])/task['problem_folder']/task['bddl_file']
    environments, results, measurements, active, capacities = [], [], [], [], []
    try:
        while indices := claim(capacity, worker_id, phase):
            group = [jobs[i] for i in indices]
            needed = sum(language.stored_branch(cohort['rows'][j['cohort_index']],
                f"{j['factor_label']}_{text}") is None for j in group for text in TEXTS)
            while len(environments) < needed:
                environments.append(OffScreenRenderEnv(bddl_file_name=str(bddl),
                    camera_heights=256, camera_widths=256))
            active = []
            for job in group:
                prepare_group(job, cohort, environments, active, contract, language,
                              prefix, owner, worker_id, identity)
            for slot in list(active):
                if not slot['admitted']:
                    results.append(finish(slot, language))
                    active = [other for other in active if other is not slot]
            while active:
                language.plan(active, policy, processor, batched, states, measurements)
                for slot in list(active):
                    for action in slot['action_plan']:
                        observation, _, done, _ = slot['env'].step(action)
                        _, predicates = stage_predicate_snapshot(slot['env'], slot['stage_predicate_states'])
                        slot.update(obs=observation, steps=slot['steps']+1,
                                    stage_predicate_last=predicates, success=bool(done))
                        record_passive_step(slot['env'], slot, action,
                            {'passive_trace': {'trace_root': str(ROOT/'rows')}})
                        slot['drawer_capture'].sample(slot['steps'])
                        if done or slot['branch']+slot['steps'] == 300:
                            break
                    if slot['success'] or slot['branch']+slot['steps'] == 300:
                        results.append(finish(slot, language))
                        active = [other for other in active if other is not slot]
            with sqlite3.connect(ROOT/'launch/jobs.sqlite') as db:
                for index in indices:
                    db.execute('UPDATE pairs SET status="done" WHERE pair_id=?', (index,))
            # Larger physical batches consume only uncompleted fixed scientific rows.
            peak = torch.cuda.max_memory_reserved()
            next_capacity = min(capacity*2, 2 if phase=='pilot' else 24)
            if peak < 24*1024**3 and next_capacity > capacity:
                capacities.append(dict(previous=capacity, next=next_capacity,
                    reason='real authorized rows; measured reserved headroom', peak_reserved_bytes=peak))
                capacity = next_capacity
    except BaseException as error:
        partial_errors = []
        for slot in active:
            if not (slot['canonical']/'row.json').exists():
                try:
                    finish(slot, language, partial=True)
                except BaseException as save_error:
                    partial_errors.append(dict(row=str(slot['directory']), error=repr(save_error)))
        language.write(ROOT/'launch'/f'{worker_id}_failure.json',
            dict(error=repr(error), partial_errors=partial_errors))
        raise
    finally:
        for environment in environments:
            environment.close()
        batched.close()
        language.write(ROOT/'launch'/f'{worker_id}_worker_exit.json',
            dict(worker_id=worker_id, pid=os.getpid(), gpu_id=gpu_id, started_unix=started,
                finished_unix=time.time(), wall_seconds=time.time()-started,
                process_cpu_seconds=time.process_time()-cpu_started,
                maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                rows=len(results), identities=identity, inference_measurements=measurements,
                capacity_changes=capacities, stop_increasing='finite remaining pairs or measured24GiB reserved threshold; at most48 active authorized rows, no additional scientific cases',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                frozen_repo=str(Path(__file__).resolve().parents[3]), cuda_release='process_exit'))
    return results
