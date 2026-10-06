"""One train-only frozen signal readback; imports the original sealed G consumer.

No optimizer step, environment, held data, or alternative scientific cases.
This entry and its two statistics helpers retire when this bounded batch closes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ASSETS = Path('/data1/user/ymdai/projects/EMBER')
OLD = Path('/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006')
CROOT = Path('/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001/conditional_read_write')
GCP = OLD / 'train/attempts/fresh/checkpoints/macro_00000450'
CCP = CROOT / 'train/attempts/resume360_native_packing/checkpoints/macro_00000450'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def summary(value, *, temporal=False, grid=False):
    original_dtype = str(value.dtype)
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite actual signal')
    result = {'shape': list(value.shape), 'dtype': original_dtype, 'statistics_dtype': 'torch.float32',
              'rms': float(value.square().mean().sqrt()), 'norm': float(value.norm()),
              'max_abs': float(value.abs().max())}
    rms = value.square().mean(-1).sqrt().flatten()
    result['token_rms_quantiles'] = torch.quantile(rms, rms.new_tensor([0, .05, .5, .95, 1])).cpu().tolist()
    if temporal:
        result['frame_rms'] = value.reshape(len(value), -1).square().mean(-1).sqrt().cpu().tolist()
    if grid:
        result['frame_token_rms'] = value.square().mean(-1).sqrt().cpu().tolist()
    return result


class ChainObserver:
    """Observe existing operations without an additional native forward."""
    def __init__(self, writer):
        self.writer, self.handles, self.targets, self.w = writer, [], {}, None

    def __enter__(self):
        if hasattr(self.writer, 'omega'):
            self.handles.append(self.writer.omega.register_forward_hook(
                lambda _m, _a, out: setattr(self, 'w', summary(out, temporal=True, grid=True))))
        for i, unit in enumerate(self.writer.conditional_targets):
            row = self.targets.setdefault(str(i), {})
            for field in ('a_dynamic', 'b_dynamic'):
                self.handles.append(getattr(unit, field).register_forward_hook(
                    lambda _m, _a, out, row=row, field=field: row.update({field: summary(out, temporal=True)})))
            for field in ('a_out', 'b_out'):
                self.handles.append(getattr(unit, field).register_forward_pre_hook(
                    lambda _m, args, row=row, field=field: row.update({field + '_input_product': summary(args[0], temporal=True)})))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()

    def report(self, native):
        mechanism = native['mechanism']
        for i, name in enumerate(self.writer.names):
            row = self.targets[str(i)]
            row.update(name=name, S=summary(mechanism['targets'][name]['S']),
                       M=summary(mechanism['targets'][name]['M']))
        return {'H': summary(native['h'], temporal=True, grid=True), 'w': self.w,
                'context': summary(mechanism['c'], temporal=True, grid=True),
                'd': summary(mechanism['d'], temporal=True, grid=True), 'targets': self.targets}


@contextmanager
def normalize_d(writer):
    """The sole intervention is at the actual ConditionalTarget argument."""
    from ember.operator_writer.conditional_read_write import rms_zero
    handles = [unit.register_forward_pre_hook(
        lambda _m, args: (*args[:5], rms_zero(args[5]), *args[6:]))
        for unit in writer.conditional_targets]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def manifest(data):
    rows = json.loads((OLD / 'readouts/G/450/seen144/evaluation/results.json').read_text())['rows']
    representation = []
    for row in rows:
        if row['init_state_id'] == 32:
            item = row['operator_read_write_lora']
            representation.append({'task': item['global_task_id'], 'teacher_demo': item['teacher_demo'],
                                   'condition_id': item['condition_id'], 'source_row':
                                   str(OLD / 'readouts/G/450/seen144/evaluation/results.json')})
    assert len(representation) == 36 and {r['task'] for r in representation} == set(data.tasks)
    metrics = {r['update']: r for r in map(json.loads, (OLD / 'train/attempts/fresh/metrics.jsonl').open())
               if r['update'] in (1, 225, 450)}
    events = []
    for macro in (1, 225, 450):
        for task in data.tasks_for_step(macro - 1):
            event = data.event(macro - 1, task)
            old = next(j for j in metrics[macro]['jobs'] if j['task'] == task)
            assert event['teacher_demo'] == old['teacher_demo'] and event['flow_seed'] == old['flow_seed']
            assert [q['demo'] for q in event['queries']] == old['query_demos']
            assert [q['frame'] for q in event['queries']] == old['query_frames']
            events.append(event)
    assert len(events) == 12 and sum(len(e['queries']) for e in events) == 336
    for item in [*representation, *events]:
        assert (OLD / f"labels/task{item['task']}_demo{item['teacher_demo']}.npz").is_file()
    return {'representation': sorted(representation, key=lambda r: r['task']), 'events': events,
            'sampling': 'originalGseen_init32_and_originalGmacro1_225_450_actual_metrics',
            'GT_usage': 'offline matching/loss and F only, never G condition',
            'G_loss_definition': 'frozen450 FM/.25KD/.1relation; each condition .25,28query mean'}


def read_only_labels(data, spec):
    from ember.relation_writer.labels import LabelStore
    class ExistingLabels(LabelStore):
        def _episode(self, task, demo, requested):
            assert (self.cache_root / f'task{task}_demo{demo}.npz').is_file(), 'missing original cache; no synthesis permitted'
            return super()._episode(task, demo, requested)
    store = ExistingLabels(data, ASSETS, cache_root=OLD / 'labels')
    store.set_semantics(json.loads((OLD / 'labels/semantic_vectors.json').read_text())['vectors'])
    return store


def metadata(writer):
    return [dict(name=n, shape=list(p.shape), numel=p.numel(), dtype=str(p.dtype), requires_grad=p.requires_grad)
            for n, p in writer.named_parameters()]


def matrix_differences(states, writer):
    public = writer.public_state()
    result = {}
    for name in states['G']:
        base, modified = states['G'][name].float(), states['normalized'][name].float()
        s0, s1 = base - public[name].detach().float(), modified - public[name].detach().float()
        denominator = float(s0.norm() * s1.norm())
        result[name] = {'actual_factor_difference': summary(modified - base),
                        'edit_cosine': float((s0 * s1).sum()) / denominator if denominator else None}
    return result


def compiled(runtime, condition, chunk, normalized=False):
    from contextlib import nullcontext
    with normalize_d(runtime.writer) if normalized else nullcontext():
        with torch.no_grad(), ChainObserver(runtime.writer) as observer:
            state, native = runtime.compile(condition, frame_chunk=chunk, capture_mechanism=True)
            report = observer.report(native)
            if normalized:
                from ember.operator_writer.conditional_read_write import rms_zero
                report['actual_ConditionalTarget_d'] = summary(rms_zero(native['mechanism']['d']), temporal=True)
    return state, report


def query_credit(runtime, feedback, batch, teacher, current, indices, states, event, microbatch, output):
    from ember.relation_writer.credit import _slice_batch, _add
    from ember.relation_writer.runtime import source_query
    from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
    from ember.writer.runtime import autocast
    owner = NativeFlowPrediction(runtime.policy)
    cotangents = {key: {} for key in ('FM', 'KD', 'normalized', 'C')}
    losses = {key: 0. for key in cotangents}
    arrays = {key: [] for key in ('G', 'normalized', 'C', 'F_target', 'target', 'tau')}
    for start in range(0, 28, microbatch):
        stop = min(start + microbatch, 28)
        weight = (stop - start) / 28
        with autocast(runtime.device):
            sample = flow_sample(runtime.policy, _slice_batch(batch, start, stop, 28),
                seed=event['flow_seed'], device=runtime.device, random_batch=28, offset=start)
            prepared = owner.prepare(sample)
            with torch.no_grad():
                h0, v0 = source_query(runtime.policy, owner, sample, prepared)
                f_target = feedback(teacher, {k: v[start:stop] for k, v in current.items()}, h0, v0, indices).detach()
            for key, state in states.items():
                leaves = {n: v.detach().requires_grad_(True) for n, v in state.items()}
                prediction = torch.func.functional_call(owner, {'policy.' + n: v for n, v in leaves.items()},
                                                        (sample, prepared), strict=False)
                fm = mean_velocity_loss(prediction, sample.target, sample.action_width)
                term = 'FM' if key == 'G' else key
                gradients = torch.autograd.grad(fm, tuple(leaves.values()), retain_graph=key == 'G')
                _add(cotangents[term], leaves, gradients, .25 * weight)
                losses[term] += float(fm.detach()) * weight
                if key == 'G':
                    kd = mean_velocity_loss(prediction, f_target, sample.action_width)
                    gradients = torch.autograd.grad(kd, tuple(leaves.values()))
                    _add(cotangents['KD'], leaves, gradients, .25 * .25 * weight)
                    losses['KD'] += float(kd.detach()) * weight
                arrays[key].append(prediction.detach()[..., :7].float().cpu().numpy())
                del prediction, leaves, gradients
            arrays['F_target'].append(f_target[..., :7].float().cpu().numpy())
            arrays['target'].append(sample.target[..., :7].float().cpu().numpy())
            arrays['tau'].append(sample.arguments[-1].float().cpu().numpy())
            del prepared, sample, f_target, h0, v0
    payload = {k: np.concatenate(v) for k, v in arrays.items()}
    np.savez_compressed(output, **payload)
    differences = {key: {'velocity_RMS_difference_from_G': float(np.sqrt(np.mean((payload[key] - payload['G']) ** 2))),
                         'per_query_FM': ((payload[key] - payload['target']) ** 2).mean((1, 2)).tolist(),
                         'per_query_FM_delta_from_G': (((payload[key] - payload['target']) ** 2).mean((1, 2)) -
                                                    ((payload['G'] - payload['target']) ** 2).mean((1, 2))).tolist()}
                   for key in ('normalized', 'C')}
    return cotangents, {'losses': losses, 'differences': differences, 'same_sample_object_all_models': True,
                       'same_z_tau_y': True, 'F_target_stopgrad': True, 'action_width': 7, 'horizon': 50}


def replay_gradients(runtime, condition, teacher, credits, frame_chunk, *, normalized=False):
    from contextlib import nullcontext
    from ember.relation_writer.relation_loss import relation_loss
    from ember.writer.runtime import autocast
    parameters = tuple(runtime.writer.parameters())
    names = tuple(n for n, _ in runtime.writer.named_parameters())
    terms, native_records = {}, {}
    with normalize_d(runtime.writer) if normalized else nullcontext():
        state, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
        objectives = list(credits.items())
        if teacher is not None:
            with autocast(runtime.device):
                rel, rel_numbers = relation_loss(runtime.writer.last_prediction, teacher)
                weighted_relation = .25 * .1 * rel
            objectives.append(('rel', None))
        for number, (term, credit) in enumerate(objectives):
            for value in (native['h'], *native['x'].values()):
                value.grad = None
            if credit is None:
                gradients = torch.autograd.grad(weighted_relation, parameters, allow_unused=True,
                                                retain_graph=number + 1 < len(objectives))
            else:
                assert set(credit) == set(state) and all(bool(torch.isfinite(v).all()) for v in credit.values())
                gradients = torch.autograd.grad(tuple(state.values()), parameters,
                    grad_outputs=tuple(credit[n].to(state[n]) for n in state), allow_unused=True,
                    retain_graph=number + 1 < len(objectives))
            terms[term] = {n: None if g is None else g.detach().cpu() for n, g in zip(names, gradients, strict=True)}
            native_records[term] = {field: float(torch.stack([v.grad.float().norm() for v in values
                if v.grad is not None]).norm()) if any(v.grad is not None for v in values) else 0.
                for field, values in [('H', [native['h']]), ('X', list(native['x'].values()))]}
        if teacher is not None:
            native_records['relation_loss'] = float(rel.detach())
            native_records['relation_metrics'] = rel_numbers
    if hasattr(runtime.writer, 'last_prediction'):
        runtime.writer.last_prediction = None
    return terms, native_records


def run(args):
    from safetensors.torch import load_file
    from ember.operator_writer.data import FormalData
    from ember.operator_writer.model import OperatorReadWrite
    from ember.relation_writer.runtime import build_runtime, to_device
    from ember.relation_writer.materialization import model_weights
    from ember.relation_writer.feedback import FeedbackFunction
    from relation_signal_geometry import physical_report
    from relation_signal_gradient_stats import credit_report, accumulate_
    from ember.pi05_assets import prepare_libero_config
    prepare_libero_config(args.output / 'runtime')
    spec = json.loads((OLD / 'frozen_PEFTfix/configs/relation_grounded_writer_v1/spec.json').read_text())
    c_spec_path = Path('/data1/user/ymdai/projects/EMBER-conditional-read-write-mlp-formal/configs/operator_read_write_v1/conditional_read_write_fresh_spec.json')
    c_spec = json.loads(c_spec_path.read_text())
    assert spec['source'] == c_spec['source'], 'shared source/prefix identity differs'
    data = FormalData(ASSETS, spec)
    plan = manifest(data)
    save(args.output / 'manifest.json', plan)
    if args.manifest_only:
        data.close()
        return
    from ember.writer.materialization_workers import _configure_device
    _configure_device(torch.device(args.device), 8)
    runtime = build_runtime(ASSETS, spec, torch.device(args.device))
    runtime.writer.load_state_dict(model_weights(GCP, 'G', device=runtime.device), strict=True)
    runtime.writer.eval()
    c_writer = OperatorReadWrite(runtime.lora, runtime.identity, 'conditional_read_write').to(runtime.device)
    c_writer.load_state_dict(load_file(str(CCP / 'ecp.safetensors'), device=str(runtime.device)), strict=True)
    c_writer.eval()
    c_runtime = replace(runtime, writer=c_writer)
    feedback = FeedbackFunction().to(runtime.device).eval().requires_grad_(False)
    feedback.load_state_dict(model_weights(GCP, 'F', device=runtime.device), strict=True)
    assert not any(p.requires_grad for p in runtime.policy.parameters())
    labels = read_only_labels(data, spec)
    original_g = torch.load(GCP / 'trainer_state.pt', map_location='cpu', weights_only=False)['optimizer']['G']
    original_c = torch.load(CCP / 'trainer_state.pt', map_location='cpu', weights_only=False)['optimizer']
    optimizer = {'G': original_g, 'C': original_c}
    meta = {'G': metadata(runtime.writer), 'C': metadata(c_writer)}
    save(args.output / 'parameter_metadata.json', meta)
    save(args.output / 'source_identity.json', {'G_training_git': '85614d9c',
        'G_original_frozen': str(OLD / 'frozen_PEFTfix'), 'G_F_checkpoint': str(GCP),
        'C_training_git': 'a0e0248d96568e42b24a3d1c4102e2ca6e35a40c', 'C_checkpoint': str(CCP),
        'C_source_spec': str(c_spec_path), 'shared_source': runtime.source,
        'native_reads': 'each original writer uses its own shared beta; no cross-model H transfer',
        'optimizer_moments': 'original actual dtypes, no dtype correction, no optimizer step'})
    records, started = [], time.monotonic()
    try:
        for item in plan['representation']:
            guard(args)
            condition, raw, sampled = data.condition(runtime, item['task'], item['teacher_demo'])
            teacher = labels.teacher(item['task'], item['teacher_demo'], condition[1].cpu().numpy())
            provenance = labels.provenance(item['task'], item['teacher_demo'])
            for model, owner in [('G', runtime), ('C', c_runtime)]:
                t = time.monotonic()
                torch.cuda.reset_peak_memory_stats(runtime.device)
                state, chain = compiled(owner, condition, args.inference_frame_chunk)
                row = {'panel': 'representation', 'model': model, **item, 'source': provenance,
                       'frame_indices': condition[1].cpu().tolist(), 'raw_frames': raw, 'sampled_frames': sampled,
                       'chain': chain, 'seconds': time.monotonic() - t,
                       'inference_frame_chunk': args.inference_frame_chunk,
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(runtime.device)}
                if model == 'G':
                    prediction = {k: v.detach().cpu() for k, v in owner.writer.last_prediction.items()}
                    row['physical'], arrays = physical_report(prediction, {k: torch.as_tensor(v) for k, v in teacher.items()},
                        provenance, condition[1].cpu(), return_arrays=True)
                    np.savez_compressed(args.output / f'representation_G_task{item["task"]}.npz', **arrays)
                save(args.output / f'representation_{model}_task{item["task"]}.json', row)
                records.append({'kind': 'representation', 'task': item['task'], 'model': model, 'seconds': row['seconds']})
                clear(owner)
                del state, chain
            del condition, teacher
        totals = {}
        for event in plan['events']:
            guard(args)
            t = time.monotonic()
            torch.cuda.reset_peak_memory_stats(runtime.device)
            condition, raw, sampled = data.condition(runtime, event['task'], event['teacher_demo'])
            teacher = to_device(labels.teacher(event['task'], event['teacher_demo'], condition[1].cpu().numpy()), runtime.device)
            current = to_device(labels.queries(event['task'], event['queries']), runtime.device)
            states, chains = {}, {}
            states['G'], chains['G'] = compiled(runtime, condition, args.inference_frame_chunk)
            states['normalized'], chains['normalized'] = compiled(runtime, condition, args.inference_frame_chunk, True)
            states['C'], chains['C'] = compiled(c_runtime, condition, args.inference_frame_chunk)
            edits = matrix_differences(states, runtime.writer)
            batch = runtime.processor.training_batch(data.batch(event))
            stem = f'credit_macro{event["update"]}_task{event["task"]}'
            credits, query = query_credit(runtime, feedback, batch, teacher, current, condition[1], states, event,
                                          args.microbatch, args.output / (stem + '_velocity.npz'))
            del states
            parts = {}
            parts['G'], g_native = replay_gradients(runtime, condition, teacher,
                {k: credits[k] for k in ('FM', 'KD')}, args.frame_chunk)
            parts['normalized'], n_native = replay_gradients(runtime, condition, None,
                {'FM': credits['normalized']}, args.frame_chunk, normalized=True)
            parts['C'], c_native = replay_gradients(c_runtime, condition, None,
                {'FM': credits['C']}, args.frame_chunk)
            row = {'panel': 'functional_credit', 'event': event, 'raw_frames': raw, 'sampled_frames': sampled,
                   'query': query, 'chain': chains, 'normalized_G_factor_effect': edits,
                   'native_credit': {'G': g_native, 'normalized': n_native, 'C': c_native},
                   'credit': {k: credit_report(v, meta['G' if k == 'normalized' else k], optimizer['G' if k == 'normalized' else k],
                                             'G' if k == 'normalized' else k) for k, v in parts.items()}}
            for model, terms in parts.items():
                destination = totals.setdefault((event['update'], model), {})
                for term, gradients in terms.items():
                    accumulate_(destination.setdefault(term, {}), gradients)
            row.update(seconds=time.monotonic() - t, frame_chunk=args.frame_chunk,
                       microbatch=args.microbatch, peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device))
            save(args.output / (stem + '.json'), row)
            records.append({'kind': 'credit', 'macro': event['update'], 'task': event['task'],
                            'seconds': row['seconds'], 'frame_chunk': args.frame_chunk,
                            'peak_reserved_bytes': row['peak_reserved_bytes']})
            for owner in (runtime, c_runtime):
                clear(owner)
            del parts, credits, chains, batch, condition, teacher, current
            # Enlarge only remaining fixed cases; all28 queries already fill the logical condition.
            if args.frame_chunk == 16 and row['peak_reserved_bytes'] < 20 * 1024**3:
                free, _ = torch.cuda.mem_get_info(runtime.device)
                if free > 34 * 1024**3:
                    args.frame_chunk = 32
        for (macro, model), terms in totals.items():
            save(args.output / f'macro{macro}_{model}_credit.json', credit_report(terms,
                meta['G' if model == 'normalized' else model], optimizer['G' if model == 'normalized' else model],
                'G' if model == 'normalized' else model))
        save(args.output / 'completion.json', {'status': 'complete', 'representation_conditions_per_model': 36,
            'functional_conditions': 12, 'queries': 336, 'records': records, 'seconds_after_load': time.monotonic() - started,
            'optimizer_steps': 0, 'source_trainable': 0, 'new_environments': 0, 'held_Test': 0})
    finally:
        data.close()
        labels.close()


def clear(runtime):
    for field in ('last_prediction', 'last_mechanism'):
        if hasattr(runtime.writer, field):
            setattr(runtime.writer, field, None)
    runtime.writer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def guard(args):
    if time.time() >= args.deadline_epoch:
        raise RuntimeError('hard wall deadline reached; no scope reduction or continuation')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=OLD / 'frozen_PEFTfix')
    p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--microbatch', type=int, default=28)
    p.add_argument('--frame-chunk', type=int, default=16)
    p.add_argument('--inference-frame-chunk', type=int, default=64)
    p.add_argument('--deadline-epoch', type=float, required=True)
    args = p.parse_args()
    assert args.output.resolve().is_relative_to(Path('/data1/user/ymdai/ember_runs/relation_signal_readback_20261007'))
    assert args.microbatch in (7, 14, 28)
    assert args.source_root.resolve() == (OLD / 'frozen_PEFTfix').resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root / 'src'))
    started = time.time()
    try:
        run(args)
    except Exception:
        save(args.output / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc(),
             'start_epoch': started, 'exit_epoch': time.time(), 'optimizer_steps': 0})
        raise


if __name__ == '__main__':
    main()
