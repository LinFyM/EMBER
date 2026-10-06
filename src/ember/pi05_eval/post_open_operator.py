"""Temporary fixed task23 Full/Common continuation consumer; retire at closure."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import resource
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path('/data1/user/ymdai/ember_runs/task23_post_open_operator_20261006')
REPLAY = Path('/data1/user/ymdai/ember_runs/task23_drawer_state_replay_20261006/frozen_remaining/src/ember/pi05_eval/drawer_state_replay.py')
FULL_CASES = {('T2340', 8), ('T2340', 19), ('C900', 8), ('C900', 16)}
TRACE_KEYS = ('actions', 'body_positions', 'eef_pos', 'eef_quat', 'gripper_qpos', 'predicates')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def replay_owner():
    spec = importlib.util.spec_from_file_location('sealed_drawer_consumer', REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def factors(cohort, model):
    """Read public tensors and already-compiled complete conditions, never Writer."""
    from safetensors.torch import load_file
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
    from ember.operator_writer.public_beta import public_state, factor_map
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract

    selected = [r for r in cohort['rows'] if r['model'] == model]
    source = selected[0]['source']
    contract = json.loads(Path(source['source_run_contract']).read_text())
    bank = contract['adapter']
    spec = json.loads(Path(bank['spec']['path']).read_text())
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank['asset_root']) / spec['source']['lora_contract']), rank=128)
    if lora.to_dict() != bank['lora'] or len(lora.targets) != 38:
        raise ValueError('original complete LoRA topology changed')
    shared = load_file(bank['shared']['path'], device='cpu')
    common = public_state(Path(bank['checkpoint']), lora)
    if set(shared) != {k for k in common if k.endswith(LORA_A_SUFFIX)}:
        raise ValueError('original shared bank is not complete public A0')
    # Identity of these two necessary A0 sources is an interface check only.
    for name, value in shared.items():
        if not __import__('torch').equal(value, common[name]):
            raise ValueError('bank A0 is not this checkpoint public A0')
    common.update(shared)
    conditions = {r['condition_id']: r for r in bank['conditions']}
    states = {model + ':Common': common}
    for row in selected:
        key = row['source']['source_row']['operator_read_write_lora']['condition_id']
        record = conditions[key]['factors']
        path = Path(record['path'])
        if path.stat().st_size != record['bytes']:
            raise ValueError('frozen condition factor asset changed')
        state = load_file(str(path), device='cpu')
        if model == 'T2340':
            if set(state) != {k for k in common if k.endswith(LORA_B_SUFFIX)}:
                raise ValueError('T condition must retain complete B0+M')
            state = {**shared, **state}
        elif bank.get('condition_factors') != 'complete_A0_plus_S_B0_plus_M':
            raise ValueError('C condition is not its original complete A/B bank')
        validate_lora_state(state, lora)
        states[model + ':' + key] = state
    return contract, lora, states, dict(checkpoint=bank['checkpoint'],
        shared=bank['shared'], public_B0_source=str(Path(bank['checkpoint']) / 'ecp.safetensors'),
        public_factor_map=factor_map(lora), conditions=[conditions[r['source']['source_row']['operator_read_write_lora']['condition_id']] for r in selected])


def trace_arrays(slot):
    trace = slot['passive_trace']
    return {k: np.stack(trace[k]) if trace[k] else np.empty((0, 7), np.float32)
            for k in TRACE_KEYS}


def runtime_state(env):
    robot, sim = env.env.robots[0], env.sim
    controller = robot.controller
    return dict(sim_state=np.asarray(env.get_sim_state()).copy(), ctrl=np.asarray(sim.data.ctrl).copy(),
        qacc_warmstart=np.asarray(sim.data.qacc_warmstart).copy(),
        controller_goal_pos=np.asarray(controller.goal_pos).copy(),
        controller_goal_ori=np.asarray(controller.goal_ori).copy(),
        gripper_current_action=np.asarray(robot.gripper.current_action).copy())


def begin(env, row, arm, contract, owner, worker_id):
    from ember.pi05_eval.episode import stage_predicate_snapshot
    from ember.pi05_eval.trajectory_capture import start_passive_trace, record_passive_step

    source = row['source']['source_row']
    with np.load(source['continuous_control_trace']['trace']['path'], allow_pickle=False) as f:
        original = {k: f[k] for k in f.files}
    with np.load(source['scene_reference']['path'], allow_pickle=False) as f:
        snapshot = {k: f[k] for k in f.files if k != 'initial_rgb_canonical180'}
    branch = row['branch_control_step']
    if branch % 5 or original['actions'].shape != (source['steps'], 7):
        raise ValueError('original action stream or registered branch changed')
    env.seed(source['env_seed']); env.reset()
    observation, names = owner.restore_original_start(env, snapshot, contract)
    if names != original['body_names'].tolist():
        raise ValueError('original passive body registry changed')
    registry, pair_group = owner.geometry_registry(env)
    states, predicates = stage_predicate_snapshot(env)
    capture = {'passive_trace': {'trace_root': str(ROOT/'rows')}}
    slot = dict(obs=observation, steps=0, stage_predicate_states=states, stage_predicate_last=predicates)
    start_passive_trace(env, slot, capture)
    drawers = owner.DrawerCapture(env, registry, pair_group); drawers.sample(0)
    for step, action in enumerate(original['actions'][:branch], 1):
        observation, _, done, _ = env.step(action)
        _, predicates = stage_predicate_snapshot(env, states)
        slot.update(obs=observation, steps=step, stage_predicate_last=predicates)
        record_passive_step(env, slot, action, capture); drawers.sample(step)
        if done:
            raise ValueError('original non-terminal prefix became terminal')
    prefix = {**trace_arrays(slot), **drawers.arrays()}
    original_prefix = {k: original[k][:branch if k == 'actions' else branch + 1] for k in TRACE_KEYS}
    deviation = owner.replay_deviation({**prefix, 'body_names': original['body_names']}, original_prefix)
    with np.load(row['matched_replay'], allow_pickle=False) as f:
        qerr = np.abs(prefix['drawer_qpos'] - f['drawer_qpos'][:branch + 1])
        verr = np.abs(prefix['drawer_qvel'] - f['drawer_qvel'][:branch + 1])
    deviation.update(drawer_max_qpos_error_m=qerr.max(axis=0).tolist(),
                     drawer_max_qvel_error_m_per_s=verr.max(axis=0).tolist(),
                     branch_top_Open=bool(prefix['drawer_Open'][-1, 0]))
    admitted = deviation['interpretable_parent'] and deviation['branch_top_Open']
    canonical = ROOT / 'rows' / f"{row['model']}_state{row['state']:03d}_{arm}"
    if (canonical/'row.json').exists(): raise ValueError('completed row must not run again')
    directory = canonical / f'attempt_{worker_id}' if canonical.exists() else canonical
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'prefix.npz', body_names=np.asarray(names), **prefix,
                        **{'branch_' + k: v for k, v in runtime_state(env).items()})
    write(directory / 'prefix.json', dict(deviation=deviation, admitted=bool(admitted), registry=registry))
    # A new trace starts at the actual branch. No state restore or settling here.
    slot.update(steps=0)
    start_passive_trace(env, slot, capture)
    drawers = owner.DrawerCapture(env, registry, pair_group); drawers.sample(0)
    condition = source['operator_read_write_lora']['condition_id']
    slot.update(env=env, row=row, arm=arm, directory=directory, canonical=canonical, worker_id=worker_id, drawer_capture=drawers,
        replan_index=branch // 5, branch=branch, admitted=bool(admitted), deviation=deviation,
        factor_key=row['model'] + (':Common' if arm == 'Common' else ':' + condition),
        full=(row['model'], row['state']) in FULL_CASES, rgb=[], rgb_steps=[],
        policy_noise_seeds=[], first_normalized=None, first_physical=None,
        original_branch_actions=original['actions'][branch:branch + 5],
        body_names=names, registry=registry, branch_runtime=runtime_state(env), success=False)
    return slot


def saved_partner(row, arm):
    """Read a completed arm's captured branch; never restore a mid-state or rerun it."""
    path=ROOT/'rows'/f"{row['model']}_state{row['state']:03d}_{arm}"/'row.json'
    if not path.exists(): return None
    result=json.loads(path.read_text())
    with np.load(result['prefix'], allow_pickle=False) as f:
        data={key:f[key] for key in f.files}
    return dict(passive_trace={k:[data[k][-1]] for k in ('body_positions','eef_pos','eef_quat','gripper_qpos')},
        drawer_capture=SimpleNamespace(qpos=[data['drawer_qpos'][-1]],qvel=[data['drawer_qvel'][-1]]),
        branch_runtime={k[7:]:v for k,v in data.items() if k.startswith('branch_')},
        stage_predicate_last=tuple(data['predicates'][-1].tolist()), admitted=result['admitted'])


def pair_check(left, right):
    """Compare actual repeated-prefix physics and controller runtime before inference."""
    deviations = {}
    for key in ('body_positions', 'eef_pos', 'eef_quat', 'gripper_qpos'):
        deviations[key] = float(np.max(np.abs(left['passive_trace'][key][0] - right['passive_trace'][key][0])))
    for key in ('qpos', 'qvel'):
        deviations['drawer_' + key] = float(np.max(np.abs(
            getattr(left['drawer_capture'], key)[0] - getattr(right['drawer_capture'], key)[0])))
    for key, value in left['branch_runtime'].items():
        deviations[key] = float(np.max(np.abs(value - right['branch_runtime'][key])))
    predicates = left['stage_predicate_last'] == right['stage_predicate_last']
    paired = predicates and all(v <= 1e-8 for v in deviations.values())
    for slot in (left, right):
        slot['pairing'] = dict(paired=bool(paired), max_abs_differences=deviations,
            predicates_identical=bool(predicates), tolerance=1e-8,
            mechanism='separate original-command prefixes; no mid-state clone')
        slot['admitted'] = slot['admitted'] and paired


def plan(slots, policy, processor, batched, factor_states, measurements):
    import torch
    from ember.pi05_processing import libero_policy_input

    started = time.monotonic()
    inputs = [processor(libero_policy_input(s['obs'], s['row']['source']['task']['language'])) for s in slots]
    batch = {k: torch.cat([r[k] for r in inputs]) for k in inputs[0]}
    seeds = [s['row']['source']['source_row']['policy_noise_seeds'][s['replan_index']] for s in slots]
    noise = torch.stack([torch.randn((50, policy.config.max_action_dim), dtype=torch.float32,
        generator=torch.Generator(device='cpu').manual_seed(seed), device='cpu') for seed in seeds]).to('cuda:0')
    with torch.inference_mode(), batched.activate([factor_states[s['factor_key']] for s in slots]):
        normalized = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
        physical = processor.unnormalize_action(normalized).detach().cpu().numpy()
        normalized = normalized.detach().float().cpu().numpy()
    if normalized.shape != (len(slots), 50, 7) or not np.isfinite(physical).all():
        raise ValueError('actual full50x7 plan is invalid')
    for i, s in enumerate(slots):
        if s['full']:
            s['rgb'].append(np.stack([s['obs'][k][::-1, ::-1].copy() for k in ('agentview_image', 'robot0_eye_in_hand_image')]))
            s['rgb_steps'].append(s['branch'] + s['steps'])
        if s['first_normalized'] is None:
            s['first_normalized'], s['first_physical'] = normalized[i].copy(), physical[i, :5].copy()
        s['action_plan'] = physical[i, :5].copy()
        s['policy_noise_seeds'].append(seeds[i]); s['replan_index'] += 1
    measurements.append(dict(batch=len(slots), seconds=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def finish(s, partial=False):
    arrays = {**trace_arrays(s), **s['drawer_capture'].arrays()}
    arrays.update(body_names=np.asarray(s['body_names']), control_steps=np.arange(s['branch'], s['branch'] + s['steps'] + 1),
                  first_normalized_plan=s['first_normalized'] if s['first_normalized'] is not None else np.empty((0, 7)),
                  first_physical_5=s['first_physical'] if s['first_physical'] is not None else np.empty((0, 7)),
                  original_branch_actions=s['original_branch_actions'])
    trace_name='partial_continuation.npz' if partial else 'continuation.npz'
    np.savez_compressed(s['directory'] / trace_name, **arrays)
    rgb_path = None
    if s['full']:
        rgb_path = str(s['directory'] / ('partial_full_rgb.npz' if partial else 'full_rgb.npz'))
        np.savez_compressed(rgb_path, rgb=np.asarray(s['rgb'], dtype=np.uint8),
                            control_steps=np.asarray(s['rgb_steps']), cameras=np.asarray(['agentview', 'eye_in_hand']), rotation_degrees=np.asarray(180))
    result = dict(model=s['row']['model'], state=s['row']['state'], arm=s['arm'],
        branch=s['branch'], remaining_horizon=300-s['branch'], continuation_steps=s['steps'],
        success=bool(s['success']), admitted=s['admitted'], prefix_deviation=s['deviation'],
        capture='full' if s['full'] else 'compact', full_rgb=rgb_path,
        source=s['row']['source'], policy_noise_seeds=s['policy_noise_seeds'],
        first_noise_index=s['branch']//5, trace=str(s['directory']/trace_name), worker_id=s['worker_id'],
        prefix=str(s['directory']/'prefix.npz'), registry=s['registry'], pairing=s['pairing'],
        exit='worker_error_partial' if partial else ('complete' if s['admitted'] else 'prefix_not_admitted'))
    write(s['directory'] / ('partial_row.json' if partial else 'row.json'), result)
    if not partial and s['directory']!=s['canonical']: write(s['canonical']/'row.json',result)
    return result


def claim_pairs(max_pairs, worker_id):
    with sqlite3.connect(ROOT/'launch/pairs.sqlite', timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        indices = [r[0] for r in db.execute('SELECT cohort_index FROM pairs WHERE status="pending" ORDER BY remaining DESC, cohort_index LIMIT ?', (max_pairs,))]
        for index in indices:
            db.execute('UPDATE pairs SET status="running", worker=? WHERE cohort_index=?', (worker_id, index))
    return indices


def run_worker(max_pairs, worker_id, gpu_id):
    import torch
    from ember.batched_lora import BatchedLoRAInference
    from ember.lora import inject_task_lora
    from ember.pi05_assets import configure_libero_runtime_assets
    import ember.pi05_eval_contract  # initialize the canonical facade before its submodules
    from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
    from ember.pi05_eval.episode import stage_predicate_snapshot
    from ember.pi05_eval.trajectory_capture import record_passive_step
    from ember.writer.topology import bind_current_process_to_cuda_numa

    start, cpu_start = time.time(), time.process_time()
    torch.set_num_threads(2); bind_current_process_to_cuda_numa(0)
    if (ROOT/'launch/retired.json').exists(): raise ValueError('sealed batch cannot be relaunched')
    cohort = json.loads((ROOT/'cohort.json').read_text()); owner = replay_owner()
    # Both frozen checkpoints share the same canonical source/topology.
    selected = cohort['rows']
    factor_states, identities = {}, {}
    for model in sorted({r['model'] for r in selected}):
        contract, lora, states, identity = factors(cohort, model)
        factor_states.update(states); identities[model] = identity
    torch.manual_seed(7); torch.cuda.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    model_path, normalization, tokenizer_path = validate_worker_assets(contract)
    policy, processor, _ = load_policy(model_path, normalization['stats'], tokenizer_path, contract['policy'])
    policy.requires_grad_(False); inject_task_lora(policy, lora); policy.requires_grad_(False)
    batched = BatchedLoRAInference(policy, lora)
    configure_libero_runtime_assets(Path(contract['libero_paths']['assets']))
    from libero.libero.envs import OffScreenRenderEnv
    task = selected[0]['source']['task']
    bddl = Path(contract['libero_paths']['bddl_files'])/task['problem_folder']/task['bddl_file']
    envs, results, measurements, active = [], [], [], []
    try:
        capacity = max_pairs
        while indices := claim_pairs(capacity, worker_id):
            group = [cohort['rows'][i] for i in indices]
            needed=sum(saved_partner(r,a) is None for r in group for a in ('Full','Common'))
            while len(envs) < needed:
                envs.append(OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256))
            active=[]
            for row in group:
                pair=[]
                for arm in ('Full','Common'):
                    saved=saved_partner(row,arm)
                    if saved is not None: pair.append(saved)
                    else:
                        slot=begin(envs[len(active)],row,arm,contract,owner,worker_id)
                        active.append(slot); pair.append(slot)
                pair_check(*pair)
            for s in list(active):
                if not s['admitted']: results.append(finish(s)); active=[v for v in active if v is not s]
            while active:
                plan(active, policy, processor, batched, factor_states, measurements)
                for s in list(active):
                    for action in s['action_plan']:
                        obs, _, done, _ = s['env'].step(action)
                        _, predicates = stage_predicate_snapshot(s['env'], s['stage_predicate_states'])
                        s.update(obs=obs, steps=s['steps']+1, stage_predicate_last=predicates, success=bool(done))
                        record_passive_step(s['env'], s, action, {'passive_trace': {'trace_root': str(ROOT/'rows')}})
                        s['drawer_capture'].sample(s['steps'])
                        if done or s['branch'] + s['steps'] == 300: break
                    if s['success'] or s['branch'] + s['steps'] == 300:
                        results.append(finish(s)); active=[v for v in active if v is not s]
            with sqlite3.connect(ROOT/'launch/pairs.sqlite') as db:
                for index in indices: db.execute('UPDATE pairs SET status="done" WHERE cohort_index=?', (index,))
            # Use the authorized rows themselves to test physical 8 -> 16 batches.
            if capacity == 4 and torch.cuda.max_memory_reserved() < 32*1024**3:
                capacity = 8
    except BaseException as error:
        for s in active:
            if not (s['canonical']/'row.json').exists(): finish(s, partial=True)
        write(ROOT/'launch'/f'{worker_id}_failure.json',dict(error=repr(error),
            unfinished=[str(s['directory']) for s in active if not (s['canonical']/'row.json').exists()]))
        raise
    finally:
        for env in envs: env.close()
        batched.close()
        write(ROOT/'launch'/f'{worker_id}_worker_exit.json', dict(worker_id=worker_id, gpu_id=gpu_id,
            started_unix=start, finished_unix=time.time(), wall_seconds=time.time()-start,
            process_cpu_seconds=time.process_time()-cpu_start, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rows=len(results), identities=identities, inference_measurements=measurements,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_release='process_exit', frozen_repo=str(Path(__file__).resolve().parents[3])))
    return results
