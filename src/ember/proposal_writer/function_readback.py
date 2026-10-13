"""Frozen endpoint consumer; the consumed S/A/B/T diagnostic is retired."""
from pathlib import Path
import json
import sqlite3
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.practice.execution import NativeVelocity
from ember.writer.runtime import autocast
from .function_data import FunctionInputs, CHECKPOINT_SCHEMA
from .learning import memory_peak, OperatorObjective
from .runtime import finish_phases


def queue_base(root, arm):
    return Path(root)/'arms'/arm/'fixed_inputs'


def prepare_queue(root, arm):
    root = Path(root)
    base = queue_base(root, arm); base.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root/'fixed_input_manifest.json')
    with sqlite3.connect(base/'jobs.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS conditions (identity TEXT PRIMARY KEY, cost INTEGER, status TEXT, worker TEXT, error TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS budget (kind TEXT PRIMARY KEY, used INTEGER, maximum INTEGER)')
        db.executemany('INSERT OR IGNORE INTO budget VALUES (?,0,?)', [('compile',72),('native_query',1344)])
        for condition in manifest['conditions']:
            cost = condition['H']['records'] + condition['sampled_teacher_frames']
            db.execute('INSERT OR IGNORE INTO conditions VALUES (?,?,?,NULL,NULL)',
                       (condition['condition_id'], cost, 'pending'))
        if db.execute('SELECT COUNT(*) FROM conditions').fetchone()[0] != 72:
            raise ValueError('endpoint readback must retain all72 conditions')


def claim(base, worker):
    with sqlite3.connect(base/'jobs.sqlite', timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT identity FROM conditions WHERE status='pending' ORDER BY cost DESC,identity LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE conditions SET status='running',worker=? WHERE identity=?", (worker, row[0]))
            return row[0]


def queue_summary(root, arm):
    with sqlite3.connect(queue_base(root, arm)/'jobs.sqlite') as db:
        return dict(db.execute('SELECT status,COUNT(*) FROM conditions GROUP BY status').fetchall())


def record(base, identity, kind, amount):
    with sqlite3.connect(base/'jobs.sqlite', timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        used, maximum = db.execute('SELECT used,maximum FROM budget WHERE kind=?', (kind,)).fetchone()
        if used+amount > maximum:
            raise RuntimeError(f'endpoint {kind} boundary exceeded: {used}+{amount}>{maximum}')
        db.execute('UPDATE budget SET used=used+? WHERE kind=?', (amount,kind))
    with (base/'counts.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(condition=identity, kind=kind, amount=amount))+'\n')


@torch.no_grad()
def consume_condition(args, runtime, inputs, operator, condition):
    tick = time.monotonic()
    base = queue_base(args.root, args.arm)
    identity = condition['condition_id']
    destination = args.root/'outputs'/args.arm/identity
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'result.pt').exists():
        return
    prior_root = Path(read_json(args.root/'run_contract.json')['prior_fixed_input_root'])
    prior = torch.load(prior_root/'outputs'/identity/'result.pt', map_location='cpu', weights_only=False)
    inputs.current_task = condition['task']
    categories = inputs.queries(condition)
    ordered = [(name, q) for name, rows in categories for q in rows]
    queries = [dict(category=name, weight=q['weight'], noise_seed=q['noise_seed'], source=q['source'])
               for name, q in ordered]
    if prior['condition'] != condition or queries != prior['queries']:
        raise ValueError('endpoint must reuse the actual original fixed condition and query identities')
    batch, _ = inputs.batch([q for _, q in ordered])
    noise = prior['noise'].to(runtime.device)
    weights = destination/'G.safetensors'
    if weights.exists():
        parameters = load_file(weights, device=str(runtime.device))
    else:
        record(base, identity, 'compile', 1)
        parameters = inputs.compile(condition)
        save_file({k:v.detach().float().cpu().contiguous() for k,v in parameters.items()}, weights)
    target = inputs.targets[condition['T_target_id']]
    operator_error, _ = operator(parameters, inputs.state(target['weights']))
    with autocast(runtime.device):
        velocity = NativeVelocity(runtime.policy, batch, capture_phi=False)
    record(base, identity, 'native_query', len(noise))
    owners = torch.zeros(len(noise), device=runtime.device, dtype=torch.long)
    response = runtime.native_actions([parameters], batch, noise, batch_indices=owners,
        checkpointed=False, prepared_velocity=velocity).detach().float().cpu()
    if response.shape != (len(noise), 5, 7) or not torch.isfinite(response).all():
        raise ValueError('endpoint must retain finite full native10 first5x7 responses')
    finish_phases(runtime)
    checkpoint = str(args.root/'arms'/args.arm/'checkpoints/update_00000072')
    payload = dict(condition=condition, target=target, queries=queries, response=response,
        noise=noise.cpu(), empty_categories=prior['empty_categories'], weights=str(weights),
        source_checkpoint=checkpoint, prior_response_source=str(prior_root/'outputs'/identity/'result.pt'),
        native_query_count=len(noise), compiler_calls=1, operator_error=float(operator_error),
        timings=dict(seconds=time.monotonic()-tick, peak=memory_peak(runtime.device),
                     frame_chunk=runtime.frame_chunk), no_grad=True)
    torch.save(payload, destination/'result.pending.pt')
    (destination/'result.pending.pt').replace(destination/'result.pt')
    write_json_atomic(destination/'complete.json', dict(complete=True, condition=identity,
        checkpoint=checkpoint, compiler_calls=1, native_queries=len(noise), no_environment=True))


def readback(args, runtime):
    checkpoint = args.root/'arms'/args.arm/'checkpoints/update_00000072'
    proof = read_json(checkpoint/'manifest.json')
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=False)
    if (not proof['complete'] or proof['schema_version'] != CHECKPOINT_SCHEMA
            or proof['next_update'] != 72 or state['arm'] != args.arm
            or state['source']['objective'] != read_json(args.root/'run_contract.json')['function_objectives'][args.arm]):
        raise ValueError('endpoint readback requires the registered complete same-T arm72')
    runtime.generator.load_state_dict(state['psi']); runtime.version = state['psi_version']
    runtime.freeze_psi()
    runtime.actor.eval().requires_grad_(False); runtime.policy.eval().requires_grad_(False)
    inputs = FunctionInputs(runtime, args.root)
    operator = OperatorObjective(runtime, args.root)
    base = queue_base(args.root, args.arm)
    conditions = {c['condition_id']:c for c in read_json(args.root/'fixed_input_manifest.json')['conditions']}
    while (identity := claim(base, args.worker_id)) is not None:
        try:
            consume_condition(args, runtime, inputs, operator, conditions[identity])
            with sqlite3.connect(base/'jobs.sqlite') as db:
                db.execute("UPDATE conditions SET status='complete' WHERE identity=?", (identity,))
        except BaseException as error:
            with sqlite3.connect(base/'jobs.sqlite') as db:
                db.execute("UPDATE conditions SET status='failed',error=? WHERE identity=?", (repr(error),identity))
            raise
