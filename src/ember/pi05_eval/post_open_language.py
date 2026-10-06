"""One temporary fixed-language consumer over sealed physical prefix functions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import resource
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path('/data1/user/ymdai/ember_runs/task23_post_open_language_20261006')
OLD = Path('/data1/user/ymdai/ember_runs/task23_post_open_operator_20261006')
LEAF = OLD/'frozen_fix2/src/ember/pi05_eval/post_open_operator.py'
TEXTS = {'explicit_full': 'open the top drawer and put the bowl in the top drawer',
         'remaining_goal': 'put the bowl in the top drawer'}
ORIGINAL_TEXT = 'open the top drawer and put the bowl inside'


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def prefix_owner():
    spec = importlib.util.spec_from_file_location('sealed_operator_prefix_leaf', LEAF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Output context for leaf functions only. No old dispatcher/worker/plan runs.
    module.ROOT = ROOT
    return module


def stored_branch(row, arm, *, exact=False):
    root = OLD if exact else ROOT
    path = root/'rows'/f"{row['model']}_state{row['state']:03d}_{'Full' if exact else arm}"/'row.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['source'] != row['source'] or result['branch'] != row['branch_control_step']:
        raise ValueError('saved branch identity/source mismatch')
    if not exact and result['policy_language'] != TEXTS[arm]:
        raise ValueError('completed new arm has a different execution text')
    with np.load(result['prefix'], allow_pickle=False) as f:
        data = {key:f[key] for key in f.files}
    return dict(passive_trace={k:[data[k][-1]] for k in ('body_positions','eef_pos','eef_quat','gripper_qpos')},
        drawer_capture=SimpleNamespace(qpos=[data['drawer_qpos'][-1]],qvel=[data['drawer_qvel'][-1]]),
        branch_runtime={k[7:]:v for k,v in data.items() if k.startswith('branch_')},
        stage_predicate_last=tuple(data['predicates'][-1].tolist()), admitted=result['admitted'],
        saved_result=result, reference_path=str(path))


def plan(slots, policy, processor, batched, factor_states, measurements):
    import torch
    from ember.pi05_processing import libero_policy_input
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    started = time.monotonic()
    inputs = [processor(libero_policy_input(s['obs'],TEXTS[s['arm']])) for s in slots]
    # The canonical processor supplies actual fixed-length token padding/masks.
    batch = {k:torch.cat([value[k] for value in inputs]) for k in inputs[0]}
    seeds = [s['row']['source']['source_row']['policy_noise_seeds'][s['replan_index']] for s in slots]
    noise = torch.stack([torch.randn((50,policy.config.max_action_dim),dtype=torch.float32,
        generator=torch.Generator(device='cpu').manual_seed(seed),device='cpu') for seed in seeds]).to('cuda:0')
    with torch.inference_mode(),batched.activate([factor_states[s['factor_key']] for s in slots]):
        normalized = policy.predict_action_chunk(batch,noise=noise,num_steps=10)
        physical = processor.unnormalize_action(normalized).detach().cpu().numpy()
        normalized = normalized.detach().float().cpu().numpy()
    if normalized.shape != (len(slots),50,7) or not np.isfinite(physical).all():
        raise ValueError('actual full50x7 language-conditioned plan is invalid')
    for index,slot in enumerate(slots):
        if slot['full']:
            slot['rgb'].append(np.stack([slot['obs'][k][::-1,::-1].copy()
                for k in ('agentview_image','robot0_eye_in_hand_image')]))
            slot['rgb_steps'].append(slot['branch']+slot['steps'])
        if slot['first_normalized'] is None:
            slot['first_normalized'],slot['first_physical'] = normalized[index].copy(),physical[index,:5].copy()
            slot['first_language_tokens'] = batch[OBS_LANGUAGE_TOKENS][index].detach().cpu().numpy()
            slot['first_language_mask'] = batch[OBS_LANGUAGE_ATTENTION_MASK][index].detach().cpu().numpy()
        slot['action_plan'] = physical[index,:5].copy()
        slot['policy_noise_seeds'].append(seeds[index]);slot['replan_index'] += 1
    measurements.append(dict(batch=len(slots),seconds=time.monotonic()-started,
        token_lengths=batch[OBS_LANGUAGE_ATTENTION_MASK].sum(1).cpu().tolist(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def finish(slot, leaf, *, partial=False):
    arrays = {**leaf.trace_arrays(slot),**slot['drawer_capture'].arrays()}
    arrays.update(body_names=np.asarray(slot['body_names']),
        control_steps=np.arange(slot['branch'],slot['branch']+slot['steps']+1),
        first_normalized_plan=slot['first_normalized'] if slot['first_normalized'] is not None else np.empty((0,7)),
        first_physical_5=slot['first_physical'] if slot['first_physical'] is not None else np.empty((0,7)),
        first_language_tokens=slot.get('first_language_tokens',np.empty(0,np.int64)),
        first_language_mask=slot.get('first_language_mask',np.empty(0,bool)),
        original_branch_actions=slot['original_branch_actions'])
    name = 'partial_continuation.npz' if partial else 'continuation.npz'
    np.savez_compressed(slot['directory']/name,**arrays)
    rgb = None
    if slot['full']:
        rgb = str(slot['directory']/('partial_full_rgb.npz' if partial else 'full_rgb.npz'))
        np.savez_compressed(rgb,rgb=np.asarray(slot['rgb'],dtype=np.uint8),
            control_steps=np.asarray(slot['rgb_steps']),cameras=np.asarray(['agentview','eye_in_hand']),rotation_degrees=np.asarray(180))
    result = dict(model=slot['row']['model'],state=slot['row']['state'],arm=slot['arm'],
        policy_language=TEXTS[slot['arm']],compile_language=ORIGINAL_TEXT,
        exact_full_reference=slot['exact_full_reference'],paired_exact_full=slot['paired_exact_full'],
        branch=slot['branch'],remaining_horizon=300-slot['branch'],continuation_steps=slot['steps'],
        success=bool(slot['success']),admitted=bool(slot['admitted']),prefix_deviation=slot['deviation'],
        capture='full' if slot['full'] else 'compact',full_rgb=rgb,source=slot['row']['source'],
        policy_noise_seeds=slot['policy_noise_seeds'],first_noise_index=slot['branch']//5,
        trace=str(slot['directory']/name),prefix=str(slot['directory']/'prefix.npz'),
        registry=slot['registry'],pairing=slot['pairing'],worker_id=slot['worker_id'],
        exit='worker_error_partial' if partial else ('complete' if slot['admitted'] else 'prefix_not_admitted'))
    write(slot['directory']/('partial_row.json' if partial else 'row.json'),result)
    if not partial and slot['directory'] != slot['canonical']:
        write(slot['canonical']/'row.json',result)
    return result


def claim_pairs(capacity,worker_id):
    with sqlite3.connect(ROOT/'launch/pairs.sqlite',timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        indices = [r[0] for r in db.execute('SELECT cohort_index FROM pairs WHERE status="pending" ORDER BY remaining DESC,cohort_index LIMIT ?',(capacity,))]
        for index in indices:
            db.execute('UPDATE pairs SET status="running",worker=? WHERE cohort_index=?',(worker_id,index))
    return indices


def run_worker(max_pairs,worker_id,gpu_id):
    import torch
    import ember.pi05_eval_contract
    from ember.batched_lora import BatchedLoRAInference
    from ember.lora import inject_task_lora
    from ember.pi05_assets import configure_libero_runtime_assets
    from ember.pi05_eval.worker_setup import load_policy,validate_worker_assets
    from ember.pi05_eval.episode import stage_predicate_snapshot
    from ember.pi05_eval.trajectory_capture import record_passive_step
    from ember.writer.topology import bind_current_process_to_cuda_numa

    started,cpu_start = time.time(),time.process_time()
    if (ROOT/'launch/retired.json').exists():
        raise ValueError('sealed language diagnostic cannot restart')
    torch.set_num_threads(2);bind_current_process_to_cuda_numa(0)
    cohort = json.loads((ROOT/'cohort.json').read_text());leaf = prefix_owner();owner = leaf.replay_owner()
    factor_states,identities = {},{}
    for model in sorted({r['model'] for r in cohort['rows']}):
        contract,lora,states,identity = leaf.factors(cohort,model)
        # Public tensors are read by the verified leaf loader, but only complete
        # original video conditions are ever active in this language diagnostic.
        factor_states.update({k:v for k,v in states.items() if not k.endswith(':Common')})
        identities[model] = identity
    torch.manual_seed(7);torch.cuda.manual_seed(7);torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    model_path,normalization,tokenizer = validate_worker_assets(contract)
    policy,processor,_ = load_policy(model_path,normalization['stats'],tokenizer,contract['policy'])
    policy.requires_grad_(False);inject_task_lora(policy,lora);policy.requires_grad_(False)
    batched = BatchedLoRAInference(policy,lora)
    configure_libero_runtime_assets(Path(contract['libero_paths']['assets']))
    from libero.libero.envs import OffScreenRenderEnv
    task = cohort['rows'][0]['source']['task']
    bddl = Path(contract['libero_paths']['bddl_files'])/task['problem_folder']/task['bddl_file']
    envs,results,measurements,active = [],[],[],[]
    try:
        capacity = max_pairs
        while indices := claim_pairs(capacity,worker_id):
            group = [cohort['rows'][i] for i in indices]
            needed = sum(stored_branch(row,arm) is None for row in group for arm in TEXTS)
            while len(envs) < needed:
                envs.append(OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256))
            active = []
            for row in group:
                pair = []
                exact = stored_branch(row,'exact_full',exact=True)
                if exact is None or not exact['admitted']:
                    raise ValueError('registered old actualFull reference is unavailable')
                for arm in TEXTS:
                    saved = stored_branch(row,arm)
                    if saved is not None:
                        pair.append(saved)
                        continue
                    slot = leaf.begin(envs[len(active)],row,arm,contract,owner,worker_id)
                    leaf.pair_check(exact,slot)
                    slot['paired_exact_full'] = slot['pairing'].copy()
                    slot['exact_full_reference'] = exact['reference_path']
                    active.append(slot);pair.append(slot)
                leaf.pair_check(*pair)
            for slot in list(active):
                if not slot['admitted']:
                    results.append(finish(slot,leaf));active = [s for s in active if s is not slot]
            while active:
                plan(active,policy,processor,batched,factor_states,measurements)
                for slot in list(active):
                    for action in slot['action_plan']:
                        obs,_,done,_ = slot['env'].step(action)
                        _,predicates = stage_predicate_snapshot(slot['env'],slot['stage_predicate_states'])
                        slot.update(obs=obs,steps=slot['steps']+1,stage_predicate_last=predicates,success=bool(done))
                        record_passive_step(slot['env'],slot,action,{'passive_trace':{'trace_root':str(ROOT/'rows')}})
                        slot['drawer_capture'].sample(slot['steps'])
                        if done or slot['branch']+slot['steps'] == 300:
                            break
                    if slot['success'] or slot['branch']+slot['steps'] == 300:
                        results.append(finish(slot,leaf));active = [s for s in active if s is not slot]
            with sqlite3.connect(ROOT/'launch/pairs.sqlite') as db:
                for index in indices:
                    db.execute('UPDATE pairs SET status="done" WHERE cohort_index=?',(index,))
            # Real fixed rows test larger physical batching; no extra profile row.
            if capacity == 3 and torch.cuda.max_memory_reserved() < 30*1024**3:
                capacity = 12
    except BaseException as error:
        partial_errors = []
        for slot in active:
            if not (slot['canonical']/'row.json').exists():
                try:
                    finish(slot,leaf,partial=True)
                except BaseException as save_error:
                    partial_errors.append({'row':str(slot['directory']),'save_error':repr(save_error)})
        write(ROOT/'launch'/f'{worker_id}_failure.json',dict(error=repr(error),partial_save_errors=partial_errors,
            unfinished=[str(s['directory']) for s in active if not (s['canonical']/'row.json').exists()]))
        raise
    finally:
        for env in envs:
            env.close()
        batched.close()
        write(ROOT/'launch'/f'{worker_id}_worker_exit.json',dict(worker_id=worker_id,gpu_id=gpu_id,
            started_unix=started,finished_unix=time.time(),wall_seconds=time.time()-started,
            process_cpu_seconds=time.process_time()-cpu_start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rows=len(results),identities=identities,inference_measurements=measurements,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_release='process_exit',frozen_repo=str(Path(__file__).resolve().parents[3])))
    return results
