"""Sealed historical baseline VJPs through the same-version complete Writer.

Temporary owner for state_baseline_full_writer_credit_20261008. The immutable
1c90e7d5 native/compiler/velocity implementations remain the actual consumers.
No optimizer, environment, action dataset, or baseline fitting is instantiated.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OLD = Path('/data1/user/ymdai/ember_runs/denoising_return_writer_20261006')
ROOT = Path('/data1/user/ymdai/ember_runs/state_baseline_full_writer_credit_20261008')
ASSETS = Path('/data1/user/ymdai/projects/EMBER')
BASELINES = ('LOO', 'Context', 'State')
sys.path.insert(0, str(OLD / 'frozen/src'))

from ember.operator_writer.data import FormalData
from ember.operator_writer.denoising_collection import restore_raw
from ember.operator_writer.denoising_credit import score_cotangent
from ember.operator_writer.denoising_policy import NativeVelocity
from ember.operator_writer.run import build_runtime
from ember.operator_writer.credit import native_credit
from ember.writer.runtime import autocast


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def code_identity():
    tree = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tree), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    if git('branch', '--show-current') or git('status', '--porcelain'):
        raise ValueError('consumer requires a clean detached worktree')
    subprocess.run(['git', '-C', str(tree), 'merge-base', '--is-ancestor', commit, 'origin/main'], check=True)
    return {'consumer_commit': commit, 'consumer_file': str(Path(__file__).resolve()),
            'native_compiler_commit': '1c90e7d5681bf18c99f9dc4084225d03e76ee929',
            'native_compiler_path': str(OLD / 'frozen'), 'parameter_updates': 0}


def summaries(vectors, groups):
    """FP64 scalar accumulation from saved FP32 coordinates, no epsilon ratios."""
    result = {}
    for group, names in groups.items():
        gram = torch.zeros(3, 3, dtype=torch.float64)
        zeros, count = [0, 0, 0], 0
        for name in names:
            values = [v[name].double().reshape(-1) for v in vectors]
            count += values[0].numel()
            for i in range(3):
                zeros[i] += int(torch.count_nonzero(values[i]) == 0)
                for j in range(i, 3):
                    gram[i, j] += torch.dot(values[i], values[j])
        gram = gram + gram.T - torch.diag(gram.diag())
        norms2 = gram.diag().tolist()
        pairs = {}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            diff = max(0., float(gram[i, i] + gram[j, j] - 2 * gram[i, j]))
            denom = (norms2[i] * norms2[j]) ** .5
            pairs[f'{BASELINES[j]}_vs_{BASELINES[i]}'] = {
                'squared_norm_ratio': norms2[j] / norms2[i] if norms2[i] else None,
                'squared_norm_difference': norms2[j] - norms2[i],
                'difference_squared_norm': diff, 'difference_norm': diff ** .5,
                'cosine': float(gram[i, j]) / denom if denom else None}
        result[group] = {'Gram': gram.tolist(), 'squared_norm': dict(zip(BASELINES, norms2)),
                         'norm': {b: n ** .5 for b, n in zip(BASELINES, norms2)},
                         'zero_tensor_counts': dict(zip(BASELINES, zeros)),
                         'parameter_tensors': len(names), 'coordinates': count, 'pairs': pairs}
    return result


def groups_for(writer):
    names = tuple(name for name, _ in writer.named_parameters())
    groups = {'full': names}
    for suffix, label in (('.lora_A.default.weight', 'public_A'), ('.lora_B.default.weight', 'public_B0')):
        groups[label] = tuple(f'common.values.{i}' for i, name in enumerate(writer.common.names) if name.endswith(suffix))
    for lower in ('p', 'c', 'd', 'o'):
        groups[lower.upper()] = tuple(name for name in names if name.endswith(f'.{lower}.weight'))
    groups['video_read_write'] = tuple(name for name in names if name.startswith('writes.'))
    if len(names) != 228 or set(names) != set(groups['public_A'] + groups['public_B0'] + groups['video_read_write']):
        raise ValueError('complete original T parameter coverage changed')
    return groups


def entries_for(item, predictions):
    group = item['group']
    entries, mapped, totals = [], [], {b: 0. for b in BASELINES}
    for row, advantage in zip(group['rows'], group['advantages'], strict=True):
        saved = torch.load(row['score_records']['path'], map_location='cpu', weights_only=False)
        if len(saved['decisions']) != row['saved_decisions']:
            raise ValueError('original reservoir count mismatch')
        for decision in saved['decisions']:
            key = (row['macro'], row['global_task'], row['replica'], decision['replan'])
            prediction = predictions[key]
            transitions = decision['transitions']
            if (prediction['t'] != decision['control_step'] or prediction['Q'] != row['actual_replans']
                    or prediction['M'] != row['saved_decisions'] or prediction['R'] != int(row['success'])
                    or prediction['selected_steps'] != decision['selected_steps']
                    or prediction['selected_steps'] != [t['step'] for t in transitions]
                    or prediction['score_source'] != row['score_records']['path']
                    or saved['row']['parameter_version'] != item['macro'] - 1):
                raise ValueError('baseline/event/collection-version mapping changed')
            bs = prediction['predictions']
            if abs(row['success'] - bs['LOO'] - advantage) > 1e-12:
                raise ValueError('frozen LOO differs from original advantage')
            for k, transition in enumerate(transitions):
                if any(transition[n].requires_grad or tuple(transition[n].shape) != (50, 32)
                       for n in ('z', 'z_next', 'm')):
                    raise ValueError('captured sampler tensors must be stopped full50x32')
                credits = [score_cotangent(transition, int(row['success']) - bs[b],
                                          row['actual_replans'], row['saved_decisions']) for b in BASELINES]
                base = score_cotangent(transition, 1., row['actual_replans'], row['saved_decisions'])
                if not torch.isfinite(base).all():
                    raise ValueError('nonfinite original score capture')
                local = float(base.double().square().sum())
                expected = prediction['C_squared_norm'][k] / 256
                if abs(local - expected) > 3e-6 * max(1., expected):
                    raise ValueError('full-coordinate cotangent scale differs from frozen CPU baseline')
                for b, credit in zip(BASELINES, credits):
                    totals[b] += float(credit.double().square().sum())
                entries.append((decision['observation'], transition, credits))
            mapped.append({'macro':row['macro'], 'task':row['global_task'], 'replica':row['replica'],
                'replan':decision['replan'], 't':decision['control_step'], 'selected_steps':decision['selected_steps'],
                'Q':row['actual_replans'], 'M':row['saved_decisions'], 'R':int(row['success']),
                'baselines':{b:bs[b] for b in BASELINES}, 'window':prediction['window'],
                'score_source':row['score_records']['path'], 'parameter_version':saved['row']['parameter_version']})
    if len(entries) != 2 * item['decisions']:
        raise ValueError('original transition cohort changed')
    return entries, mapped, totals


def lora_credit(runtime, state, entries, microbatch):
    credits = [{n: torch.zeros_like(v, dtype=torch.float32) for n, v in state.items()} for _ in BASELINES]
    leaves = {n: v.detach().requires_grad_(True) for n, v in state.items()}
    if len(leaves) != 76:
        raise ValueError('full38 A/B boundary changed')
    for start in range(0, len(entries), microbatch):
        block = entries[start:start + microbatch]
        processed = [runtime.processor(restore_raw(e[0])) for e in block]
        batch = {key: torch.cat([p[key] for p in processed]) for key in processed[0]}
        z = torch.stack([e[1]['z'] for e in block]).to(runtime.device)
        tau = torch.tensor([e[1]['tau'] for e in block], device=runtime.device)
        with torch.enable_grad(), autocast(runtime.device):
            velocity = torch.func.functional_call(NativeVelocity(runtime.policy, batch),
                {'policy.'+n:v for n, v in leaves.items()}, (z, tau), strict=False)
            if velocity.shape != (len(block), 50, 32) or not torch.isfinite(velocity).all():
                raise ValueError('actual native velocity shape/finite boundary failed')
            for j in range(3):
                q = torch.stack([e[2][j] for e in block]).to(velocity)
                gradients = torch.autograd.grad(velocity, tuple(leaves.values()), grad_outputs=q, retain_graph=j < 2)
                for n, value in zip(leaves, gradients, strict=True):
                    credits[j][n].add_(value.detach().float())
        del velocity, gradients, batch, processed
    if any(not torch.isfinite(v).all() for c in credits for v in c.values()):
        raise ValueError('nonfinite complete-LoRA cotangent')
    return credits


def writer_credit(runtime, condition, credits, frame_chunk):
    vectors, natives, coverage = [], [], []
    with torch.enable_grad():
        replay, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
        constant_x = [n for n,x in native['x'].items() if not x.requires_grad]
        # The action_in projection receives the fixed probe itself. Its X is
        # legitimately constant; later X and final H depend on public A/B0.
        if (len(replay) != 76 or not native['h'].requires_grad
                or constant_x != ['model.action_in_proj']):
            raise ValueError('complete live native/compiler credit was detached')
        for j in range(3):
            runtime.writer.zero_grad(set_to_none=True)
            for p in native['passes']:
                for x in (p['h'], *p['x'].values(), *p['state'].values()):
                    x.grad = None
            torch.autograd.backward(tuple(replay.values()), tuple(credits[j][n].to(v) for n, v in replay.items()),
                                    retain_graph=j < 2)
            present = [n for n, p in runtime.writer.named_parameters() if p.grad is not None]
            if len(present) != 228:
                raise ValueError('a complete original Writer parameter lost its VJP')
            coverage.append(len(present))
            vectors.append({n:p.grad.detach().float().cpu().clone() for n, p in runtime.writer.named_parameters()})
            natives.append(native_credit(native))
    runtime.writer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(v).all() for c in vectors for v in c.values()):
        raise ValueError('nonfinite complete G gradient')
    return vectors, natives, coverage, {'h':True,'X_live':len(native['x'])-len(constant_x),
                                      'constant_X':constant_x,'complete_X':len(native['x'])}


def evaluate(runtime, condition, state, entries, microbatch, frame_chunk):
    start = time.perf_counter()
    credits = lora_credit(runtime, state, entries, microbatch)
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - start
    boundary = summaries([{n:v.cpu() for n, v in c.items()} for c in credits], {'full':tuple(state)})
    vectors, natives, coverage, graph = writer_credit(runtime, condition, credits, frame_chunk)
    torch.cuda.synchronize()
    return vectors, {'lora_boundary':boundary, 'native_cotangent':dict(zip(BASELINES,natives)),
        'trainable_coverage':dict(zip(BASELINES,coverage)), 'native_graph':graph, 'score_seconds':score_seconds,
        'replay_seconds':time.perf_counter()-start-score_seconds}


def profile(runtime, condition, state, entries, root, macro):
    measurements = []
    for microbatch, frame_chunk in ((32,16),(64,32),(128,32)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        failure, vectors = None, None
        try:
            vectors, info = evaluate(runtime, condition, state, entries, microbatch, frame_chunk)
            measurement = info
        except torch.cuda.OutOfMemoryError:
            failure, measurement = traceback.format_exc(), {}
        measurements.append({**measurement, 'microbatch':microbatch, 'frame_chunk':frame_chunk,
            'seconds':time.perf_counter()-started, 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'failure':failure, 'transitions':len(entries)})
        del vectors
        gc.collect()
        write(root / 'profiles' / f'macro_{macro:03d}.json', {'measurements':measurements, 'parameter_updates':0})
    valid = [m for m in measurements if not m['failure']]
    if not valid:
        raise ValueError('authorized profile has no feasible physical layout')
    chosen = min(valid, key=lambda m:m['seconds'])
    write(root / 'profiles' / f'macro_{macro:03d}.json', {'measurements':measurements, 'chosen':chosen,
        'stop_reason':'128 covers all selected transitions of the condition; no new case or update', 'parameter_updates':0})
    return chosen['microbatch'], chosen['frame_chunk']


def execute(args):
    identity = code_identity()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    old_contract = json.loads((OLD / 'train/run_contract.json').read_text())
    spec = json.loads(Path(old_contract['source_spec']['path']).read_text())
    runtime = build_runtime(ASSETS, spec, device, 'T')
    if any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError('source parameters must remain frozen')
    groups = groups_for(runtime.writer)
    data = FormalData(ASSETS, spec, query_labels=False)
    if data.queries is not None:
        raise ValueError('no teacher actions/query labels may be loaded')
    predictions = { (p['macro'],p['task'],p['replica'],p['replan']):p for p in
        map(json.loads, (ROOT/'analysis/frozen_baseline_predictions.jsonl').read_text().splitlines()) }
    cohort = json.loads((ROOT/'analysis/cohort.json').read_text())
    microbatch, frame_chunk = args.microbatch, args.frame_chunk
    for macro in args.macros:
        output = ROOT/'analysis'/f'macro_{macro:03d}.json'
        if output.exists():
            raise ValueError('completed macro must never be rerun or overwritten')
        items = sorted([c for c in cohort if c['macro']==macro], key=lambda c:c['group']['event']['task'])
        runtime.writer.load_state_dict(load_file(items[0]['checkpoint']+'/ecp.safetensors',device=str(device)),strict=True)
        totals = [{n:torch.zeros_like(p,device='cpu',dtype=torch.float32) for n,p in runtime.writer.named_parameters()} for _ in BASELINES]
        conditions = []
        for index, item in enumerate(items):
            event = item['group']['event']
            entries, mapping, local_energy = entries_for(item,predictions)
            condition, raw, sampled = data.condition(runtime,event['task'],event['teacher_demo'])
            if (raw,sampled)!=(item['group']['raw_teacher_frames'],item['group']['sampled_teacher_frames']):
                raise ValueError('actual original teacher frame stream changed')
            with torch.no_grad():
                state,_ = runtime.compile(condition,frame_chunk=frame_chunk)
            if args.profile and index==0 and macro==args.macros[0]:
                microbatch, frame_chunk = profile(runtime,condition,state,entries,ROOT,macro)
            torch.cuda.reset_peak_memory_stats()
            vectors, info = evaluate(runtime,condition,state,entries,microbatch,frame_chunk)
            stats = summaries(vectors,groups)
            conditions.append({'task':event['task'],'teacher_demo':event['teacher_demo'],'macro':macro,
                'parameter_version':macro-1,'checkpoint':item['checkpoint'],'group_kind':item['group_kind'],
                'returns':item['group']['returns'],'decision_mapping':mapping,'local_energy':local_energy,
                'Brier_episode_equal':item['Brier_episode_equal'],'writer':stats,**info,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            for total, vector in zip(totals,vectors):
                for n,v in vector.items():total[n].add_(v)
            del entries, vectors, state, condition
            gc.collect()
        macro_stats = summaries(totals,groups)
        crosses = {g:{b:macro_stats[g]['squared_norm'][b]-sum(c['writer'][g]['squared_norm'][b] for c in conditions)
            for b in BASELINES} for g in groups}
        for b, vector in zip(BASELINES,totals):
            save_file(vector,str(ROOT/'gradients'/f'macro_{macro:03d}_{b}.safetensors'),
                      metadata={'macro':str(macro),'checkpoint':items[0]['checkpoint'],'baseline':b,**{k:str(v) for k,v in identity.items()}})
        write(output,{'macro':macro,'checkpoint_version':macro-1,'conditions':conditions,'writer':macro_stats,
            'cross_condition_terms':crosses,'parameter_order':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)}
                for n,p in runtime.writer.named_parameters()], 'source_trainable':0,'finite':True,
            'independent_S':'absent/nonactive','physical':{'microbatch':microbatch,'frame_chunk':frame_chunk},
            'old_LOO_macro_norm':items[0]['old_macro_preclip_norm'],**identity})
        del totals
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--macros',type=int,nargs='+',required=True,choices=(19,28,37,46,55,64))
    parser.add_argument('--microbatch',type=int,default=64)
    parser.add_argument('--frame-chunk',type=int,default=32)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--attempt',default='initial')
    args = parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('unsafe attempt identity')
    path = ROOT/'launch'/('worker_'+'_'.join(map(str,args.macros))+'_'+args.attempt+'.json')
    if path.exists():raise ValueError('worker receipt already exists')
    started = time.time()
    receipt = {'started_UTC':datetime.now(timezone.utc).isoformat(),'macros':args.macros,'command':sys.argv,
               'host':socket.gethostname(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'PID':os.getpid()}
    write(path,receipt)
    try:
        execute(args)
        receipt.update(exit=0,status='complete')
    except Exception:
        receipt.update(exit=1,status='failed',failure=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.time()-started,finished_UTC=datetime.now(timezone.utc).isoformat())
        write(path,receipt)


if __name__ == '__main__':
    main()
