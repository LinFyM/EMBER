"""Bounded no-gradient fixed-input S/A/B/T consumer; no environment runner."""
from pathlib import Path
import json
import sqlite3
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.practice.execution import NativeVelocity
from ember.writer.runtime import autocast
from .function_data import FunctionInputs
from .learning import memory_peak
from .runtime import finish_phases


def prepare_queue(root):
    root = Path(root)
    manifest = read_json(root/'manifest.json')
    with sqlite3.connect(root/'jobs.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS conditions (identity TEXT PRIMARY KEY, cost INTEGER, status TEXT, worker TEXT, error TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS budget (kind TEXT PRIMARY KEY, used INTEGER, maximum INTEGER)')
        db.executemany('INSERT OR IGNORE INTO budget VALUES (?,0,?)', [('compile',216),('native_query',10368)])
        for condition in manifest['conditions']:
            cost = condition['H']['records'] + condition['sampled_teacher_frames']
            db.execute('INSERT OR IGNORE INTO conditions VALUES (?,?,?,NULL,NULL)',
                       (condition['condition_id'], cost, 'pending'))
        if db.execute('SELECT COUNT(*) FROM conditions').fetchone()[0] != 72:
            raise ValueError('fixed readback queue must retain all72 conditions')


def claim(root, worker):
    with sqlite3.connect(Path(root)/'jobs.sqlite', timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT identity FROM conditions WHERE status='pending' ORDER BY cost DESC,identity LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE conditions SET status='running',worker=? WHERE identity=?", (worker, row[0]))
            return row[0]


def queue_summary(root):
    with sqlite3.connect(Path(root)/'jobs.sqlite') as db:
        return dict(db.execute('SELECT status,COUNT(*) FROM conditions GROUP BY status').fetchall())


def load_points(runtime, manifest):
    points = {}
    if set(manifest['source_checkpoints']) != {'S', 'A', 'B'}:
        raise ValueError('the fixed model panel is exactly S/A/B')
    for name, directory in manifest['source_checkpoints'].items():
        directory = Path(directory)
        proof = read_json(directory/'manifest.json')
        state = torch.load(directory/'state.pt', map_location='cpu', weights_only=False)
        expected = 9 if name == 'S' else 72
        if not proof['complete'] or proof['next_update'] != expected or state['next_update'] != expected:
            raise ValueError('only the registered complete S38349/A72/B72 are allowed')
        if name == 'S' and (state['psi_version'] != 38349 or state['supervision_version'] != 9):
            raise ValueError('S must be the original full G38349/bank9')
        if name != 'S' and (state['arm'] != ('F_only' if name == 'A' else 'full_G')
                           or state['source']['G_cursor'] != 38349):
            raise ValueError('A/B must be the matched sole72 endpoints')
        points[name] = state['psi']
    runtime.freeze_psi()
    runtime.actor.eval().requires_grad_(False)
    runtime.policy.eval().requires_grad_(False)
    return points


def infer(runtime, states, batch, noise, velocity):
    owners = torch.zeros(len(noise), device=runtime.device, dtype=torch.long)
    return runtime.native_actions([states], batch, noise, batch_indices=owners,
                                  checkpointed=False, prepared_velocity=velocity).detach().float().cpu()


def record(root, event):
    kind = 'compile' if 'compile_started' in event else 'native_query'
    amount = event.get('compile_started', event.get('native_queries_started'))
    with sqlite3.connect(Path(root)/'jobs.sqlite', timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        used, maximum = db.execute('SELECT used,maximum FROM budget WHERE kind=?', (kind,)).fetchone()
        if used + amount > maximum:
            raise RuntimeError(f'fixed {kind} boundary would be exceeded: {used}+{amount}>{maximum}')
        db.execute('UPDATE budget SET used=used+? WHERE kind=?', (amount,kind))
    with (Path(root)/'counts.jsonl').open('a') as stream:
        stream.write(json.dumps(event)+'\n')


@torch.no_grad()
def consume_condition(args, runtime, inputs, points, condition):
    tick = time.monotonic()
    destination = args.root/'outputs'/condition['condition_id']
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'result.pt').exists():
        return
    inputs.current_task = condition['task']
    categories = inputs.queries(condition)
    ordered = [(name, q) for name, rows in categories for q in rows]
    batch, noise = inputs.batch([q for _, q in ordered])
    with autocast(runtime.device):
        velocity = NativeVelocity(runtime.policy, batch, capture_phi=False)
    responses, weights = {}, {}
    target = inputs.targets[condition['T_target_id']]
    for point in ('S', 'A', 'B', 'T'):
        response_file = destination/f'{point}_response.pt'
        if response_file.exists():
            responses[point] = torch.load(response_file, weights_only=True)
            continue
        if point == 'T':
            parameters = inputs.state(target['weights'])
        else:
            weights[point] = str(destination/f'{point}.safetensors')
            if Path(weights[point]).exists():
                parameters = load_file(weights[point], device=str(runtime.device))
            else:
                runtime.generator.load_state_dict(points[point])
                runtime.version += 1
                runtime.feature_cache.clear(); runtime.feature_cache_size = 0
                record(args.root, dict(condition=condition['condition_id'], point=point, compile_started=1))
                parameters = inputs.compile(condition)
                save_file({k: v.detach().float().cpu().contiguous() for k, v in parameters.items()}, weights[point])
        record(args.root, dict(condition=condition['condition_id'], point=point, native_queries_started=len(noise)))
        responses[point] = infer(runtime, parameters, batch, noise, velocity)
        if responses[point].shape != (len(noise), 5, 7) or not torch.isfinite(responses[point]).all():
            raise ValueError('fixed native10 readback must retain finite first5x7 responses')
        torch.save(responses[point], response_file)
    for point in ('S', 'A', 'B'):
        weights[point] = str(destination/f'{point}.safetensors')
    finish_phases(runtime)
    payload = dict(condition=condition, target=target, queries=[dict(category=n, weight=q['weight'],
        noise_seed=q['noise_seed'], source=q['source']) for n, q in ordered], responses=responses,
        noise=noise.cpu(), empty_categories=[n for n, rows in categories if not rows], weights=weights,
        source_checkpoints=read_json(args.root/'manifest.json')['source_checkpoints'],
        native_query_count=len(noise)*4, timings=dict(seconds=time.monotonic()-tick,
        peak=memory_peak(runtime.device), frame_chunk=runtime.frame_chunk), no_grad=True)
    torch.save(payload, destination/'result.pending.pt')
    (destination/'result.pending.pt').replace(destination/'result.pt')
    write_json_atomic(destination/'complete.json', dict(complete=True, condition=condition['condition_id'],
        compiler_points=3, native_queries=len(noise)*4, no_environment=True,
        seconds=payload['timings']['seconds'], peak=payload['timings']['peak']))


def readback(args, runtime):
    manifest = read_json(args.root/'manifest.json')
    points = load_points(runtime, manifest)
    inputs = FunctionInputs(runtime, args.root)
    conditions = {c['condition_id']: c for c in manifest['conditions']}
    while (identity := claim(args.root, args.worker_id)) is not None:
        try:
            consume_condition(args, runtime, inputs, points, conditions[identity])
            with sqlite3.connect(args.root/'jobs.sqlite') as db:
                db.execute("UPDATE conditions SET status='complete' WHERE identity=?", (identity,))
        except BaseException as error:
            with sqlite3.connect(args.root/'jobs.sqlite') as db:
                db.execute("UPDATE conditions SET status='failed',error=? WHERE identity=?", (repr(error), identity))
            raise
        if args.max_groups is not None:
            args.max_groups -= 1
            if args.max_groups == 0:
                break
