"""Frozen one-edit cross-control and native FM readouts; no learning or rollout."""
from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample
from ember.writer.runtime import autocast
from .contract import condition_seed
from .credit import _map_tensors, per_query_loss
from .data import QueryData
from .execution import NativeVelocity, action_chunk
from .interaction import processed


SCHEMA = 'ember_parameter_edit_readouts_v1'
ARMS = ('I', 'E+', 'E0')


def _points(length):
    return np.unique(np.floor(np.linspace(0, length - 1, min(6, length))).astype(int)).tolist()


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def _raw(raw):
    if (raw['images'].shape != (2, 3, 256, 256) or raw['images'].dtype != torch.uint8
            or raw['proprio'].shape != (8,)):
        raise ValueError('readout must consume actual rotated dual RGB256 and proprio8')
    return raw


def _actual(record):
    if (record['actions'].shape != (5, 7) or record['executed'].shape != (5,)
            or record['executed'].dtype != torch.bool):
        raise ValueError('actual execution must retain environment prefix5x7 and its boolean mask')
    return {key: record[key] for key in ('actions', 'executed', 'reward', 'done')}


def _stats(value):
    """Signed and squared channel diagnostics; tail predictions are not executions."""
    result = {}
    for name, chunk in (('front5', value[:5]), ('full50', value)):
        chunk = chunk.float()
        mse = chunk.square().mean(0)
        result[name] = dict(mean=chunk.mean().item(), mse=mse.mean().item(),
            channel_mean=chunk.mean(0).tolist(), channel_mse=mse.tolist(),
            channel_rms=mse.sqrt().tolist(), channel_mae=chunk.abs().mean(0).tolist())
    result['front5_gripper'] = value[:5, 6].float().tolist()
    return result


def _difference(left, right):
    return {unit: _stats(left[unit] - right[unit]) for unit in ('normalized_actions', 'commands')}


def _prediction(record):
    result = {name: record[name].detach().float().cpu() for name in ('normalized_actions', 'commands')}
    if any(v.shape != (50, 7) or not torch.isfinite(v).all() for v in result.values()):
        raise ValueError('native ten-step prediction must retain finite full50x7 output')
    return result


def _identity(contract, condition):
    return dict(schema_version=SCHEMA, checkpoint=contract['checkpoint'],
                asset_root=contract['asset_root'], condition=condition)


def _validate_source(condition, root):
    source = read_json(Path(condition['source_record']))
    if not source.get('complete') or not source['events']:
        raise ValueError('original actual-edit event record is incomplete')
    keys = ('condition_id', 'task_id', 'suite', 'suite_task_id', 'teacher_demo', 'language', 'final_state_ids')
    if any(source[key] != condition[key] for key in keys):
        raise ValueError('diagnostic condition differs from its actual parent condition')
    if (source['events'][-1] != condition['last_event']
            or source['metrics']['actual_J'] != condition['actual_J']):
        raise ValueError('readout must use the final actual edit, without event selection')
    directory = Path(condition['source_record']).parent
    incoming = (Path(source['MT_reference']) if condition['last_event']['incoming'] == 'MT'
                else directory / condition['last_event']['incoming'])
    expected = {'I': incoming, 'E+': directory / 'end.safetensors',
                'E0': root / 'materialized' / condition['condition_id'] / 'E0.safetensors'}
    if any(Path(condition['weights'][arm]).resolve() != path.resolve() for arm, path in expected.items()):
        raise ValueError('I/E+/E0 do not refer to the frozen actual one-edit weights')
    if Path(condition['source_experience']).resolve() != (directory / 'experience.pt').resolve():
        raise ValueError('practice readout must use the original cumulative experience')


def _cross(root, condition, jobs):
    points, traces = [], []
    for state in condition['final_state_ids']:
        paths = {arm: root / 'evaluation' / 'traces' / condition['condition_id'] / arm / f'state{state:02d}.pt'
                 for arm in ('I', 'E+')}
        rows = {arm: _load(path) for arm, path in paths.items()}
        for arm, row in rows.items():
            if (not row.get('complete') or row['condition_id'] != condition['condition_id']
                    or row['arm'] != arm or row['init_state_id'] != state):
                raise ValueError('closed-loop trace identity/completion does not match the paired condition')
        traces.extend(map(str, paths.values()))
        for index in _points(min(len(row['records']) for row in rows.values())):
            old, new = (rows[arm]['records'][index] for arm in ('I', 'E+'))
            if (old['noise_seed'] != new['noise_seed'] or old['noise'].shape != (50, 32)
                    or not torch.equal(old['noise'], new['noise'])):
                raise ValueError('paired common-prefix readout lost actual policy-noise pairing')
            point = dict(init_state_id=state, decision_index=index, noise_seed=old['noise_seed'],
                steps={arm: rows[arm]['records'][index]['step'] for arm in ('I', 'E+')},
                trace_paths={arm: str(path) for arm, path in paths.items()},
                noise=old['noise'].detach().cpu(),
                predictions={'I@I': _prediction(old), 'E+@E+': _prediction(new)},
                actual={arm: _actual(rows[arm]['records'][index]) for arm in ('I', 'E+')})
            for key, arm, record in (('E+@I', 'E+', old), ('E0@I', 'E0', old), ('I@E+', 'I', new)):
                jobs.append((point['predictions'], key, arm, _raw(record['raw']), old['noise']))
            points.append(point)
    return points, traces


def _practice(condition, jobs):
    chain = _load(condition['source_experience'])
    event = condition['last_event']
    endpoint, episode = int(event['endpoint']), int(event['episode'])
    if not 0 < endpoint <= len(chain['records']) or endpoint != chain['endpoints'][-1]:
        raise ValueError('last practice endpoint differs from the actual cumulative evidence')
    eligible = [index for index, row in enumerate(chain['records'][:endpoint]) if row['episode'] == episode]
    if not eligible:
        raise ValueError('final actual event has no practice decisions')
    points = []
    for relative in _points(len(eligible)):
        index = eligible[relative]
        row = chain['records'][index]
        if row['behavior_version'] != event['behavior_version']:
            raise ValueError('last practice actor differs from the actual incoming event')
        seed = int(chain['episodes'][episode]['policy_noise_seeds'][int(row['step']) // 5])
        noise = torch.randn((50, 32), generator=torch.Generator().manual_seed(seed))
        point = dict(record_index=index, episode=episode, step=int(row['step']), pre=row['pre'],
            source_experience=condition['source_experience'], noise_seed=seed, noise=noise,
            original_full50_saved=False,
            original={key: row[key] for key in ('actions', 'executed', 'feedback', 'behavior_version')},
            predictions={})
        for arm in ARMS:
            jobs.append((point['predictions'], arm, arm, _raw(chain['observations'][row['pre']]), noise))
        points.append(point)
    return points


def _predict(runtime, states, jobs, language, microbatch):
    for start in range(0, len(jobs), microbatch):
        chunk = jobs[start:start + microbatch]
        inputs = [processed(runtime, raw, language) for _, _, _, raw, _ in chunk]
        batch = {key: torch.cat([item[key] for item in inputs]) for key in inputs[0]}
        noise = torch.stack([item[-1] for item in chunk]).to(runtime.device)
        indices = torch.tensor([ARMS.index(item[2]) for item in chunk], device=runtime.device)
        with runtime.execution.activate(states, batch_indices=indices), autocast(runtime.device):
            owner = NativeVelocity(runtime.policy, batch, capture_phi=False)
            normalized, _ = action_chunk(owner, noise, capture_hidden=False)
        commands = runtime.processor.unnormalize_action(normalized).float().cpu()
        normalized = normalized.float().cpu()
        for (destination, key, _, _, _), norm, command in zip(chunk, normalized, commands, strict=True):
            destination[key] = _prediction(dict(normalized_actions=norm, commands=command))


def _fm_queries(tasks, condition):
    task_id, teacher = int(condition['task_id']), int(condition['teacher_demo'])
    rng = np.random.default_rng(np.random.SeedSequence([20261009, 0xED17, task_id, teacher]))
    demos = rng.choice([demo for demo in range(50) if demo != teacher], 7, replace=False)
    queries = []
    for demo in map(int, demos):
        available = tasks[task_id].episode_lengths[demo] - 1
        if available < 4:
            raise ValueError('FM query episode cannot provide four legal intervals')
        for interval in range(4):
            frame = int(rng.integers(available * interval // 4, available * (interval + 1) // 4))
            queries.append((demo, frame))
    return SimpleNamespace(task_id=task_id, teacher_demo=teacher, queries28=tuple(queries))


def _fm(runtime, asset_root, condition, states, microbatch):
    data = QueryData(Path(asset_root))
    try:
        event = _fm_queries(data.tasks, condition)
        raw = data.raw_query_batch(event)
        batch = runtime.processor.training_batch(raw)
    finally:
        data.close()
    seed = condition_seed(condition['task_id'], condition['teacher_demo'], domain=0xED17)
    sample = flow_sample(runtime.policy, batch, seed=seed, device=runtime.device, random_batch=28, offset=0)
    if sample.action_width != 7 or sample.target.shape != (28, 50, 32):
        raise ValueError('FM lost the canonical full50x32, first-seven consumer')
    owner, predictions = NativeFlowPrediction(runtime.policy), {arm: [] for arm in ARMS}
    for start in range(0, 28, microbatch):
        stop = min(28, start + microbatch)
        chunk = FlowSample(_map_tensors(sample.arguments, lambda value: value[start:stop]),
                           sample.target[start:stop], 7)
        with autocast(runtime.device):
            prepared = owner.prepare(chunk)
        assignment = torch.zeros(stop - start, dtype=torch.long, device=runtime.device)
        for arm, state in zip(ARMS, states, strict=True):
            with runtime.execution.activate([state], batch_indices=assignment), autocast(runtime.device):
                predictions[arm].append(owner(chunk, prepared).float().cpu())
    predictions = {arm: torch.cat(values) for arm, values in predictions.items()}
    cpu_sample = FlowSample((), sample.target.float().cpu(), 7)
    losses = {arm: per_query_loss(value, cpu_sample) for arm, value in predictions.items()}
    errors = {arm: value[..., :7] - cpu_sample.target[..., :7] for arm, value in predictions.items()}
    metrics = {arm: dict(full50_loss=losses[arm].mean().item(), per_query_loss=losses[arm].tolist(),
        per_query=[_stats(value) for value in errors[arm]],
        channel_mse=errors[arm].square().mean((0, 1)).tolist(),
        front5_loss=errors[arm][:, :5].square().mean().item(),
        front5_per_query_loss=errors[arm][:, :5].square().mean((1, 2)).tolist(),
        front5_channel_mse=errors[arm][:, :5].square().mean((0, 1)).tolist()) for arm in ARMS}
    paired = {}
    for label, left, right in (('E+minusI', 'E+', 'I'), ('E0minusI', 'E0', 'I'), ('E+minusE0', 'E+', 'E0')):
        difference = errors[left].square() - errors[right].square()
        paired[label] = {name: dict(mean=delta.mean().item(), per_query=delta.mean((1, 2)).tolist(),
            channel_mse_difference=delta.mean((0, 1)).tolist(),
            per_query_channel_mse_difference=delta.mean(1).tolist())
            for name, delta in (('full50', difference), ('front5', difference[:, :5]))}
    payload = dict(queries28=[list(query) for query in event.queries28], seed=seed, predictions=predictions,
        target=cpu_sample.target, normalized_action_latent=sample.arguments[-3].float().cpu(),
        noise=sample.arguments[-2].float().cpu(), time=sample.arguments[-1].float().cpu(),
        source_actions=raw['action'].cpu(), action_is_pad=raw['action_is_pad'].cpu(),
        action_start_offset=1, loss_padding='all50_first7_including_padded_rows', metrics=metrics, paired=paired)
    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise ValueError('native FM prediction is nonfinite')
    return payload


def _summaries(cross, practice):
    cross_summary, practice_summary = [], []
    pairs = (('E+minusI', 'E+', 'I'), ('E0minusI', 'E0', 'I'), ('E+minusE0', 'E+', 'E0'))
    for point in cross:
        p = point['predictions']
        differences = {'total': ('E+@E+', 'I@I'), 'parameter_on_I': ('E+@I', 'I@I'),
            'state_with_E+': ('E+@E+', 'E+@I'), 'parameter_on_E+': ('E+@E+', 'I@E+'),
            'state_with_I': ('I@E+', 'I@I'), 'experience_on_I': ('E+@I', 'E0@I')}
        point['differences'] = {key: {unit: p[left][unit] - p[right][unit] for unit in p[left]}
                                for key, (left, right) in differences.items()}
        cross_summary.append(dict(init_state_id=point['init_state_id'], decision_index=point['decision_index'],
            steps=point['steps'], noise_seed=point['noise_seed'],
            deltas={key: _difference(p[left], p[right]) for key, (left, right) in differences.items()}))
    for point in practice:
        p, original = point['predictions'], point['original']
        mask = original['executed'].bool()
        if original['actions'].shape != (5, 7) or mask.shape != (5,):
            raise ValueError('saved practice commands must retain their real prefix mask')
        actual_errors = {arm: ((p[arm]['commands'][:5] - original['actions'])[mask].square().mean().item()
                              if mask.any() else None) for arm in ARMS}
        practice_summary.append(dict(record_index=point['record_index'], episode=point['episode'], step=point['step'],
            noise_seed=point['noise_seed'], actual_executed=int(mask.sum()),
            original_full50_saved=False, actual_prefix_command_mse=actual_errors,
            deltas={label: _difference(p[left], p[right]) for label, left, right in pairs}))
    return cross_summary, practice_summary


@torch.no_grad()
def read_condition(runtime, root: Path, condition: dict, *, microbatch: int) -> dict:
    """Consume one frozen condition; complete prediction/summary pairs are idempotent."""
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError('readout microbatch must be a positive physical batch size')
    root = Path(root)
    contract = read_json(root / 'run_contract.json')
    registered = [item for item in contract['conditions'] if item['condition_id'] == condition['condition_id']]
    if registered != [condition] or condition['final_state_ids'] != [32, 33, 34]:
        raise ValueError('readout condition is outside the frozen train panel')
    if hasattr(runtime, 'asset_root') and Path(runtime.asset_root).resolve() != Path(contract['asset_root']).resolve():
        raise ValueError('readout runtime is using different canonical assets')
    identity = _identity(contract, condition)
    destination = root / 'readouts' / condition['condition_id']
    prediction_path, result_path = destination / 'predictions.pt', destination / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result.get('complete'):
            saved = _load(prediction_path)
            if (result.get('identity') != identity or saved.get('identity') != identity
                    or not saved.get('complete') or saved['counts'] != result['counts']
                    or len(saved['cross']) != saved['counts']['cross_points']
                    or len(saved['practice']) != saved['counts']['practice_points']
                    or len(saved['fm']['queries28']) != 28
                    or set(saved['fm']['predictions']) != set(ARMS)
                    or any(value.shape != (28, 50, 32) for value in saved['fm']['predictions'].values())):
                raise ValueError('completed readout does not match the frozen condition')
            return result
    started = time.monotonic()
    _validate_source(condition, root)
    states = [load_file(str(condition['weights'][arm]), device='cpu') for arm in ARMS]
    stamp = time.monotonic()
    timings = {'source_and_weights': stamp - started}
    jobs = []
    cross, trace_paths = _cross(root, condition, jobs)
    now = time.monotonic()
    timings['traces'] = now - stamp
    stamp = now
    practice = _practice(condition, jobs)
    now = time.monotonic()
    timings['practice'] = now - stamp
    stamp = now
    _predict(runtime, states, jobs, condition['language'], microbatch)
    now = time.monotonic()
    timings['native_ten_step'] = now - stamp
    stamp = now
    fm = _fm(runtime, contract['asset_root'], condition, states, microbatch)
    now = time.monotonic()
    timings['fm_including_query_read'] = now - stamp
    stamp = now
    cross_summary, practice_summary = _summaries(cross, practice)
    counts = dict(cross_points=len(cross), practice_points=len(practice),
        extra_ten_step_predictions=len(jobs), fm_queries=28, fm_predictions=84,
        ten_step_physical_batches=(len(jobs) + microbatch - 1) // microbatch,
        fm_prefix_batches=(28 + microbatch - 1) // microbatch,
        fm_suffix_batches=3 * ((28 + microbatch - 1) // microbatch))
    if len(cross) > 18 or len(practice) > 6 or len(jobs) != 3 * (len(cross) + len(practice)):
        raise ValueError('readout expanded the frozen six-point sampling contract')
    payload = dict(identity=identity, complete=True, counts=counts, cross=cross, practice=practice, fm=fm)
    result = dict(identity=identity, condition_id=condition['condition_id'], complete=True, counts=counts,
        physical_microbatch=microbatch, fm_actual_max_batch=min(microbatch, 28),
        predictions=str(prediction_path), trace_paths=trace_paths,
        units=dict(normalized_actions='frozen source normalized action', commands='environment action',
                   fm='native normalized velocity MSE, first7 of50x32', actual='environment action with executed mask'),
        tail_semantics='unexecuted model output; front5 is generated, mask determines actual execution',
        cross=cross_summary, practice=practice_summary, fm=dict(metrics=fm['metrics'], paired=fm['paired'],
        queries28=fm['queries28'], seed=fm['seed'], loss_padding=fm['loss_padding']))
    destination.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name(f'.{prediction_path.name}.{uuid.uuid4().hex}.tmp')
    torch.save(payload, temporary)
    temporary.replace(prediction_path)
    timings['summary_and_prediction_save'] = time.monotonic() - stamp
    result['timings_seconds'] = timings
    result['elapsed_seconds'] = time.monotonic() - started
    write_json_atomic(result_path, result)
    return result
