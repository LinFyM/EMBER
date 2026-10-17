"""Read-only source-practice support and one globally bounded frozen H cache."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import gzip
import io
import os

import numpy as np
import torch

from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_source_checkpoint import read_json
from ember.reference.contract import ASSETS, SOURCE, TASKS
from ember.writer.runtime import autocast

SOURCE_HISTORY_ROOT = Path('/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/feedback_proposals_seed7')
RAW_SCHEMA = 'ember_feedback_raw_H_v1'
CACHE_SCHEMA = 'ember_native_source_history_support_v1'
SOURCE_MODEL = str(ASSETS / SOURCE / 'policy')
CACHE_LIMIT = 4 * 2**30


def _file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _source_rows(task):
    task = int(task)
    conditions = read_json(SOURCE_HISTORY_ROOT / 'conditions.json')
    if set(map(int, conditions)) != set(TASKS) or task not in TASKS:
        raise ValueError('source H is restricted to the original explicit non-held36 allowlist')
    condition = conditions[str(task)]
    states = condition['practice_states']
    if (condition['task'] != task or condition['v'] != 0 or len(states) != 4
            or len(set(states)) != 4 or any(type(v) is not int or not 16 <= v < 50 for v in states)
            or condition['practice_noise_root'] != 202610161
            or condition['query_noise_root'] != 202610151 or condition['query_states'] != [0, 4, 8, 12]
            or condition['original']['task'] != task or condition['original']['kind'] != 'original'
            or condition['teacher_demo'] != condition['original']['demo']):
        raise ValueError('source H condition changed the fixed practice000 panel')
    identity = f'task_{task:04d}_v0_practice_000'
    rows = []
    for state in states:
        path = SOURCE_HISTORY_ROOT / 'episodes' / identity / f'state_{state:03d}.json'
        row = read_json(path)
        _validate_row(row, task, state, identity)
        archive = Path(row['history_path']).resolve(strict=True)
        if archive.parent != path.with_suffix('').resolve(strict=True) or archive.name != 'attempt_001.pt.gz':
            raise ValueError('source H must be the original row-bound practice000 archive')
        rows.append((row, archive, _file_identity(path), _file_identity(archive)))
    return condition, rows


def _validate_row(row, task, state, identity):
    expected = dict(global_task_id=task, init_state_id=state, identity=identity,
        condition_id=f'task_{task:04d}_v0', panel_ordinal=0, role='practice',
        actual_parameter_ref=f'task_{task:04d}_v0_parameter_000', source_frozen=True,
        MT_loaded=False, behavior_actor_uses_teaching=False, full_native_flow_steps=10,
        executed_chunk=5, env_seed=202610161, policy_seed_root=202610161,
        noise_root=202610161, settling_steps=10, restored_control_steps=0, valid_panel_row=True)
    if (any(row.get(k) != v for k, v in expected.items())
            or row.get('producer_parameter_ref') != expected['actual_parameter_ref']
            or len(row.get('behavior_y', [])) != 64 or any(v != 0 for v in row['behavior_y'])
            or type(row.get('success')) is not bool or row.get('engineering_error')
            or row.get('model_failure') or row.get('budget_truncated') or row.get('failure_type')
            or type(row.get('steps')) is not int or row['steps'] <= 0
            or row.get('environment_steps') != row['steps'] + 10):
        raise ValueError('source-practice row is not complete frozen-source zero-code behavior')
    seeds = [policy_noise_seed(202610161, row['suite'], row['task_id'], state, i)
             for i in range((row['steps'] + 4) // 5)]
    if row.get('policy_noise_seeds') != seeds or row.get('first_noise_seed') != seeds[0]:
        raise ValueError('source-practice row changed its actual native policy-noise stream')


def _raw_archive(path, row):
    # Inflate once in memory: torch's zip reader otherwise repeatedly seeks gzip.
    with gzip.open(path, 'rb') as stream:
        payload = torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema') != RAW_SCHEMA or payload.get('complete') is not True:
        raise ValueError('source H needs the original complete raw_H_v1 payload')
    archive_row = payload.get('row', {})
    if not archive_row or any(row.get(k) != v for k, v in archive_row.items()):
        raise ValueError('raw H payload row identity disagrees with the actual source row')
    _validate_row(archive_row, row['global_task_id'], row['init_state_id'], row['identity'])
    history = payload['history']
    expected_key = f'{row["identity"]}_s{row["init_state_id"]:03d}_attempt_001'
    if (payload.get('cache_key') != expected_key or len(history['episodes']) != 1
            or any(row.get(k) != v for k, v in history['episodes'][0].items())
            or history['image_features'] or set(history['states']) != {row['actual_parameter_ref']}):
        raise ValueError('raw H payload lost episode/observation/producer binding')
    if any(key not in history['episodes'][0] for key in ('suite', 'task_id', 'init_state_id',
            'env_seed', 'policy_seed_root', 'policy_noise_seeds', 'steps', 'actual_parameter_ref', 'replans', 'success')):
        raise ValueError('raw H episode payload is missing its original identity fields')
    state = torch.as_tensor(history['states'][row['actual_parameter_ref']])
    if state.shape != (64,) or bool(torch.count_nonzero(state)):
        raise ValueError('source raw H actual stored behavior code is not zero64')
    _validate_records(history, row)
    return history


def _validate_records(history, row):
    records, observations = history['records'], history['observations']
    if not records or len(records) != row['replans'] or len(records) != len(row['policy_noise_seeds']):
        raise ValueError('raw H lost the actual complete native decision stream')
    step = 0
    for ordinal, record in enumerate(records):
        if (record['episode'] != 0 or record['step'] != step
                or record['noise_seed'] != row['policy_noise_seeds'][ordinal]
                or record['parameter_ref'] != row['actual_parameter_ref']
                or record['hidden'].shape != (2, 50, 1024)
                or record['proprio'].shape != (2, 8) or record['actions'].shape != (5, 7)
                or record['executed'].shape != (5,) or record['executed'].dtype != torch.bool
                or record['feedback'].shape != (5,) or record.get('failure_type')
                or record.get('command_kind') != 'model_prediction'):
            raise ValueError('raw H record changed source identity, true facts or complete-flow hidden')
        executed = int(record['executed'].sum())
        if not 1 <= executed <= 5 or record['executed'].tolist() != [i < executed for i in range(5)]:
            raise ValueError('raw H execution mask must retain its actual contiguous executed prefix')
        for key in (record['pre'], record['post']):
            raw = observations[key]
            if raw['images'].dtype != torch.uint8 or raw['images'].shape != (2, 3, 256, 256) or raw['proprio'].shape != (8,):
                raise ValueError('raw H must retain real full-resolution dual RGB and state8')
        if ordinal and record['pre'] != records[ordinal - 1]['post']:
            raise ValueError('raw H decision observations lost their actual chronological chain')
        step += executed
    if step != row['steps']:
        raise ValueError('raw H executed actions do not cover its actual control steps')


def load_source_history(task):
    """CPU-only raw consumer; choose before seeing any success or error value."""
    condition, sources = _source_rows(task)
    total = sum(row['replans'] for row, *_ in sources)
    positions = np.floor(np.linspace(0, total - 1, min(16, total))).astype(int).tolist()
    records, pre, observations, provenance, offset = [], [], {}, [], 0
    for episode, (row, archive, row_id, archive_id) in enumerate(sources):
        history = _raw_archive(archive, row)
        for index in positions:
            local = index - offset
            if not 0 <= local < len(history['records']):
                continue
            record = dict(history['records'][local], episode=episode)
            records.append(record)
            for key in (record['pre'], record['post']):
                observations[key] = history['observations'][key]
            before, after = history['observations'][record['pre']], history['observations'][record['post']]
            if (not torch.equal(record['proprio'][0], before['proprio'])
                    or not torch.equal(record['proprio'][1], after['proprio'])):
                raise ValueError('selected raw H facts disagree with their actual pre/post states')
            pre.append(dict(images=before['images'], proprio=before['proprio']))
            provenance.append(dict(position=index, episode=episode, decision=local,
                init_state_id=row['init_state_id'], step=record['step'], noise_seed=record['noise_seed'],
                pre=record['pre'], post=record['post'], row=row_id, archive=archive_id))
        offset += len(history['records'])
    if len(records) != len(positions) or len(set(positions)) != len(positions):
        raise ValueError('source H support selection lost distinct floor-linspace positions')
    return dict(records=records, pre=tuple(pre), observations=observations,
        positions=positions, provenance=provenance, total_decisions=total,
        practice_states=list(condition['practice_states']),
        historical_control_steps=sum(row['steps'] for row, *_ in sources),
        historical_settling_steps=40, historical_environment_steps=sum(row['environment_steps'] for row, *_ in sources))


def event_facts(processor, record):
    """The existing source quantile units, with unexecuted action slots zero."""
    quantiles = {k: getattr(processor, '_' + k).detach().cpu()
                 for k in ('state_q01', 'state_q99', 'action_q01', 'action_q99')}
    state = processor._quantile_transform(record['proprio'].float(),
        quantiles['state_q01'], quantiles['state_q99'], False)
    actions, mask = torch.zeros_like(record['actions'], dtype=torch.float32), record['executed']
    actions[mask] = processor._quantile_transform(record['actions'][mask].float(),
        quantiles['action_q01'], quantiles['action_q99'], False)
    facts = torch.cat((state.flatten(), actions.flatten(), mask.float(), record['feedback'].float(),
        torch.tensor([record['episode'], record['step']], dtype=torch.float32) / 8192))
    if facts.shape != (63,) or not bool(facts.isfinite().all()):
        raise ValueError('source H facts63 must retain finite actual units')
    return facts.detach()


@torch.no_grad()
def observation_features(runtime, observations, chunk):
    if runtime.execution._active_state is not None or any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError('source-H vision requires frozen source with a closed adapter context')
    result, items = {}, list(observations.items())
    for first in range(0, len(items), chunk):
        part = items[first:first + chunk]
        images = torch.stack([raw['images'] for _, raw in part]).to(runtime.device).float() / 255
        pixels, _ = runtime.policy._preprocess_images({
            'observation.images.base_0_rgb': images[:, 0],
            'observation.images.left_wrist_0_rgb': images[:, 1]})
        with autocast(torch.device(runtime.device)):
            phi = torch.cat([runtime.policy.model.paligemma_with_expert.embed_image(v) for v in pixels[:2]], 1)
        if phi.shape != (len(part), 512, 2048) or not bool(phi.isfinite().all()):
            raise ValueError('actual source-H image patches changed shape or became nonfinite')
        result.update({key: value.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
                       for (key, _), value in zip(part, phi, strict=True)})
    return result


@contextmanager
def _lock(path):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class SourceHistoryStore:
    """All ranks/nodes share root/cache/H; payloads and atomic partials total <=4GiB."""
    def __init__(self, root, runtime, chunk=16):
        if type(chunk) is not int or chunk < 1:
            raise ValueError('actual observation chunk must be positive')
        if str(runtime.source['model_path']) != SOURCE_MODEL:
            raise ValueError('source H features require the canonical frozen source1000 policy')
        self.root, self.runtime, self.chunk = Path(root) / 'cache' / 'H', runtime, chunk
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = dict(model=SOURCE_MODEL, checkpoint=runtime.source['checkpoint'],
            quantiles={key: getattr(runtime.processor, '_' + key).detach().cpu().tolist()
                       for key in ('state_q01', 'state_q99', 'action_q01', 'action_q99')})

    def _identity(self, task):
        condition, sources = _source_rows(task)
        return dict(task=int(task), condition=condition, source=self.source,
            sources=[dict(row=row_id, archive=archive_id, decisions=row['replans'], steps=row['steps'],
                          noise_seeds=row['policy_noise_seeds']) for row, _, row_id, archive_id in sources],
            selection='all-four-practice000 decisions; floor-linspace(0,N-1,min(16,N)); no-success-filter')

    def _build(self, task):
        support = load_source_history(task)
        phi = observation_features(self.runtime, support['observations'], self.chunk)
        records = support.pop('records')
        support.pop('observations')
        images = torch.stack([torch.stack((phi[r['pre']], phi[r['post']])) for r in records])
        hidden = torch.stack([r['hidden'].detach().cpu() for r in records])
        facts = torch.stack([event_facts(self.runtime.processor, r) for r in records])
        seeds = tuple(r['noise_seed'] for r in records)
        noise = torch.stack([torch.randn(50, 32, generator=torch.Generator().manual_seed(seed)) for seed in seeds])
        return dict(**support, images=images, hidden=hidden, facts=facts, noise_seeds=seeds, noise=noise)

    @staticmethod
    def _validate(value, identity):
        if not isinstance(value, dict) or value.get('schema') != CACHE_SCHEMA or value.get('identity') != identity:
            raise ValueError('H cache payload identity/schema differs from the requested actual source histories')
        support = value['support']
        if (support['total_decisions'] != sum(s['decisions'] for s in identity['sources'])
                or support['historical_control_steps'] != sum(s['steps'] for s in identity['sources'])
                or support['historical_settling_steps'] != 40
                or support['historical_environment_steps'] != support['historical_control_steps'] + 40):
            raise ValueError('cached H support disagrees with the original source panel size/cost')
        count = min(16, support['total_decisions'])
        shapes = dict(images=(count, 2, 512, 2048), hidden=(count, 2, 50, 1024),
                      facts=(count, 63), noise=(count, 50, 32))
        if any(support[key].shape != shape or not bool(support[key].isfinite().all())
               or support[key].requires_grad for key, shape in shapes.items()):
            raise ValueError('frozen H cache payload lost finite complete selected-event tensors')
        if (support['images'].dtype != torch.bfloat16 or len(support['pre']) != count
                or len(support['noise_seeds']) != count or len(support['provenance']) != count
                or support['practice_states'] != identity['condition']['practice_states']
                or support['positions'] != np.floor(np.linspace(0, support['total_decisions'] - 1, count)).astype(int).tolist()):
            raise ValueError('frozen H cache payload lost exact original support selection')
        for index, (raw, event) in enumerate(zip(support['pre'], support['provenance'], strict=True)):
            episode = event['episode']
            if (not 0 <= episode < 4 or event['position'] != support['positions'][index]
                    or not 0 <= event['decision'] < identity['sources'][episode]['decisions']
                    or event['position'] != sum(s['decisions'] for s in identity['sources'][:episode]) + event['decision']
                    or event['init_state_id'] != support['practice_states'][episode]
                    or event['row'] != identity['sources'][episode]['row']
                    or event['archive'] != identity['sources'][episode]['archive']
                    or event['noise_seed'] != support['noise_seeds'][index]
                    or event['noise_seed'] != identity['sources'][episode]['noise_seeds'][event['decision']]
                    or raw['images'].dtype != torch.uint8 or raw['images'].shape != (2, 3, 256, 256)
                    or raw['proprio'].shape != (8,)):
                raise ValueError('cached event payload lost actual episode/observation/noise provenance')
        return support

    def get(self, task):
        identity = self._identity(task)
        path = self.root / f'task_{int(task):04d}.pt'
        with _lock(path.with_suffix('.lock')):
            if path.exists():
                return self._validate(torch.load(path, map_location='cpu', weights_only=True), identity)
            value = dict(schema=CACHE_SCHEMA, identity=identity, support=self._build(task))
            support = self._validate(value, identity)
            # A single admission lock counts all workers' ready/control/partial
            # files. Serialization remains under it, so no uncounted reservation
            # can race another writer. A full cache safely streams this support.
            with _lock(self.root / '.budget.lock'):
                used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
                stream = io.BytesIO()
                torch.save(value, stream)
                if used + stream.tell() <= CACHE_LIMIT:
                    temporary = path.with_suffix(f'.pt.partial_{os.getpid()}')
                    try:
                        with temporary.open('wb') as output:
                            output.write(stream.getbuffer())
                            output.flush()
                            os.fsync(output.fileno())
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            return support
