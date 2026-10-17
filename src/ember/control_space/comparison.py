"""Atomic complete first-write LoRAs and label-side finite-panel comparisons."""
from pathlib import Path
from itertools import combinations
import datetime, os, subprocess, time
import torch
from safetensors.torch import load_file, save_file
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.lora import validate_lora_state
from ember.reference.analysis import paired
from ember.proposal_writer.analysis import paired as checked_pair, billing
from ember.reference.accounting import summary, connect
from .contract import (SCHEMA, PHASE, FINAL, TASKS, VIDEO_ROOT, REFERENCE_ROOT, INITIAL_READER,
    ASSETS, SOURCE, NORMALIZATION, OBJECTIVE, PREVIOUS_ROOT, H_ROOT, ARM, IDENTITY_TASKS, checkpoint_path)
from .analysis import grouped


def topology():
    from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'), rank=8)


def generated(root, role, panel):
    if role != 'seen' or panel['task'] not in TASKS:
        raise ValueError('only the registered36 non-held first-write conditions can compile')
    return Path(root)/'generated'/role/f'task_{panel["task"]:04d}'/'v0'


def generation_identity(root, role, panel):
    from .data import report_conditions
    condition = report_conditions()[panel['task']]
    if panel['teacher_demo'] != condition['teacher_demo']:
        raise ValueError('terminal compilation must use the registered report v0 teaching')
    return dict(schema=SCHEMA, phase=PHASE, objective=OBJECTIVE,
        checkpoint=str(checkpoint_path(root, FINAL)), update=FINAL,
        source_checkpoint=str(ASSETS/SOURCE), normalization=str(ASSETS/NORMALIZATION),
        frozen_reader=str(INITIAL_READER), source_history_root=str(H_ROOT),
        practice_states=condition['practice_states'], practice_noise_root=202610161,
        role=role, task=panel['task'], teacher_demo=panel['teacher_demo'],
        rank=8, alpha=8, dropout=0., targets=38, complete=True, held_offline_labels_read=False,
        teacher_input='complete_frozen_ordered_image_memory', query_feedback_read=False,
        interaction_adaptation=True, new_practice=0, MT_loaded=False,
        path=str(generated(root, role, panel)/'lora.safetensors'))


def validate_generated(root, role, panel, *, factors=False):
    folder = generated(root, role, panel); meta = read_json(folder/'metadata.json')
    identity = generation_identity(root, role, panel)
    if any(meta.get(k) != v for k, v in identity.items()) or not (folder/'lora.safetensors').is_file():
        raise ValueError('saved complete controller lost its terminal/report/history identity')
    if factors:
        state = load_file(str(folder/'lora.safetensors'), device='cpu')
        validate_lora_state(state, topology())
        if any(not bool(v.isfinite().all()) for v in state.values()):
            raise ValueError('nonfinite complete generated LoRA')
    return meta


class Materializer:
    """The one compiler seam; four query states share one complete condition LoRA."""
    def __init__(self, root, device, runtime, *, row_chunk=4096, pack=1, frame_chunk=32,
                 loader=None, materialize=None):
        self.root, self.device, self.runtime = Path(root), torch.device(device), runtime
        self.source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        self.row_chunk, self.pack, self.frame_chunk = int(row_chunk), int(pack), int(frame_chunk)
        if min(self.row_chunk, self.pack, self.frame_chunk) < 1:
            raise ValueError('physical generation sizes must be positive')
        self.loader, self.materialize, self.model = loader, materialize, None

    def _ensure(self):
        if self.model is not None:
            return
        if self.loader is None or self.materialize is None:
            from .learning import load_final, materialize
            self.loader = self.loader or load_final
            self.materialize = self.materialize or materialize
        self.model = self.loader(self.root, self.runtime)

    def _adopt(self, role, panel):
        destination = generated(self.root, role, panel); temporary = destination.with_name('v0.partial')
        if destination.exists():
            validate_generated(self.root, role, panel, factors=True)
            return True
        if not temporary.exists():
            return False
        identity = generation_identity(self.root, role, panel)
        if not (temporary/'intent.json').exists():
            if (temporary/'lora.safetensors').exists() or (temporary/'metadata.json').exists():
                raise ValueError('pending controller has no attributable immutable intent')
            return False
        if read_json(temporary/'intent.json') != identity:
            raise ValueError('partial transaction belongs to another condition')
        if not (temporary/'lora.safetensors').exists():
            return False
        state = load_file(str(temporary/'lora.safetensors'), device='cpu')
        validate_lora_state(state, self.runtime.lora)
        if any(not bool(v.isfinite().all()) for v in state.values()):
            raise ValueError('invalid pending complete controller')
        if not (temporary/'metadata.json').exists():
            write_json_atomic(temporary/'metadata.json', dict(identity, recovered_after_parameter_write=True))
        elif any(read_json(temporary/'metadata.json').get(k) != v for k, v in identity.items()):
            raise ValueError('pending metadata lost its immutable compiler identity')
        temporary.rename(destination)
        return True

    def compile(self, role, panels):
        tick = time.monotonic(); unique = {}
        for panel in panels:
            unique.setdefault((panel['task'], panel['teacher_demo']), panel)
        missing = [p for p in unique.values() if not self._adopt(role, p)]
        if missing:
            self._ensure()
        for panel in missing:
            # Native VJP/JVP compilation itself requires autograd even at inference.
            state = self.materialize(self.runtime, self.model, panel['task'], panel['teacher_demo'])
            validate_lora_state(state, self.runtime.lora)
            if any(not bool(v.isfinite().all()) for v in state.values()):
                raise ValueError('nonfinite native-operator compiler output cannot execute')
            destination = generated(self.root, role, panel); temporary = destination.with_name('v0.partial')
            temporary.mkdir(parents=True, exist_ok=True); identity = generation_identity(self.root, role, panel)
            if (temporary/'intent.json').exists() and read_json(temporary/'intent.json') != identity:
                raise ValueError('pending generation identity changed')
            write_json_atomic(temporary/'intent.json', identity)
            payload = temporary/'lora.safetensors.partial'
            save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()}, str(payload))
            with payload.open('rb') as stream:
                os.fsync(stream.fileno())
            payload.replace(temporary/'lora.safetensors')
            write_json_atomic(temporary/'metadata.json', dict(identity,
                materialization_source_commit=self.source_commit,
                generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            temporary.rename(destination)
        receipt = dict(new_controllers=len(missing), adopted_controllers=len(unique)-len(missing),
            query_cases=len(panels), seconds=time.monotonic()-tick, materialize_pack=self.pack,
            row_chunk=self.row_chunk, frame_chunk=self.frame_chunk)
        return [validate_generated(self.root, role, p) for p in panels], receipt

    def close(self):
        if self.model is not None and hasattr(self.model, 'close'):
            self.model.close()
        self.model = None


def case_task(row):
    if 'global_task_id' in row:
        return int(row['global_task_id'])
    return 10*('libero_spatial', 'libero_object', 'libero_goal', 'libero_10').index(row['suite'])+int(row['task_id'])


def _index(rows, panels):
    keys = {(p['task'], p['init_state_id']) for p in panels}; result = {}
    for original in rows:
        row = dict(original); task = case_task(row); row['global_task_id'] = task
        key = task, int(row['init_state_id'])
        if key in result:
            raise ValueError('duplicate historical paired case')
        result[key] = row
    if set(result) != keys:
        raise ValueError('historical reference is not the entire registered144 panel')
    return [result[p['task'], p['init_state_id']] for p in panels]


def historical(root, panels):
    """Label side only; no reference forward, held read, or adaptation dependency."""
    base = REFERENCE_ROOT.parent
    paths = {name: base/name/'analysis/results.json' for name in
        ('feedback_readout_seed7', 'incumbent_compilation_seed7', 'functional_feedback_seed7')}
    feedback = read_json(paths['feedback_readout_seed7']); incumbent = read_json(paths['incumbent_compilation_seed7'])
    functional = read_json(paths['functional_feedback_seed7'])
    values = {a: feedback['raw_rows'][a] for a in ('source', 'reference', 'Reader900', 'first_best', 'full')}
    values['random_search86'] = incumbent['raw_rows']['video']
    values['functional93'] = functional['raw_rows']['functional_feedback']
    expected = dict(source=58, reference=127, Reader900=89, first_best=73, full=79, random_search86=86, functional93=93)
    result = {a: _index(rows, panels['seen']) for a, rows in values.items()}
    if {a: sum(r['success'] for r in rows) for a, rows in result.items()} != expected:
        raise ValueError('actual historical full144 success count changed')
    for rows in result.values():
        checked_pair(result['source'], rows)
    for row, panel in zip(result['Reader900'], panels['seen'], strict=True):
        if row.get('teacher_demo') != panel['teacher_demo']:
            raise ValueError('historical Reader did not consume the same report teaching')
    _check_historical_teaching(base, panels['seen'])
    anchors = _historical_anchors(base, panels['seen'])
    provenance = dict(raw_row_sources={k: str(v) for k, v in paths.items()},
        historical_costs=dict(feedback=feedback.get('costs'), incumbent=incumbent.get('costs'),
            functional=functional.get('costs'), functional_scientific_cost_basis=functional.get('scientific_cost_basis')),
        same_video_verification='Reader raw teacher_demo and frozen v0 condition manifests; old source/reference have no teaching input',
        comparison_limits='Historical methods use different shared theta, searches and/or H. These are capability references, not matched video/history ablations.',
        source_weight_comparison='Only same-task same-video init0 anchors36 are paired. Other108 query states use different old videos and are excluded.')
    return result, anchors, provenance


def _check_historical_teaching(base, panels):
    # Rows from adaptive historical arms do not always repeat teacher_demo.
    # Their frozen producer manifests, rather than invented row fields, bind V.
    report = {p['task']: p['teacher_demo'] for p in panels}
    mappings = read_json(base/'feedback_readout_seed7/manifest.json')['mappings']
    for arm in ('Reader900', 'first_best', 'full'):
        selected = [m for m in mappings if m['arm'] == arm]
        if (len(selected) != 144 or {(m['task'], m['state_id']) for m in selected} !=
                {(p['task'], p['init_state_id']) for p in panels}
                or any(m['teacher_demo'] != report[m['task']] for m in selected)):
            raise ValueError('historical frozen144 manifest changed its report teaching')
    for family, arm in (('incumbent_compilation_seed7', 'video'), ('functional_feedback_seed7', 'functional_feedback')):
        conditions = [c for c in read_json(base/family/'run_contract.json')['conditions'] if c['arm'] == arm]
        if (len(conditions) != 36 or {c['task'] for c in conditions} != set(TASKS)
                or any(c['teacher_demo'] != report[c['task']] for c in conditions)):
            raise ValueError('historical adaptation condition changed its report teaching')


def _historical_anchors(base, panels):
    anchors = {}
    for name, folder in (('source_weight1180', PREVIOUS_ROOT),
            ('factor_credit1180', base/'source_weight_factor_credit_seed7'),
            ('factor_credit1090', base/'source_weight_factor_credit_adjacent1090_seed7')):
        rows = [read_json(folder/'readout/seen'/f'task_{t:04d}'/'state_000.json') for t in TASKS]
        subset = [p for p in panels if p['init_state_id'] == 0]
        anchors[name] = _index(rows, subset)
        for row, panel in zip(anchors[name], subset, strict=True):
            if row.get('teacher_demo') != panel['teacher_demo']:
                raise ValueError('old source-W anchor is not the same task/video condition')
    return anchors


def grouped_evidence(arms, tasks, identity):
    rows = []
    for task in sorted(tasks):
        values = {a: [r for r in cases if case_task(r) == task] for a, cases in arms.items()}
        sets = {a: {r['init_state_id'] for r in v if r['success']} for a, v in values.items()}; comparisons = {}
        for a, b in combinations(values, 2):
            checked = checked_pair(values[a], values[b]); item = paired(values[a], values[b]); union = sets[a] | sets[b]
            item.update(success_set_union=sorted(union), jaccard=len(sets[a] & sets[b])/len(union) if union else 1.,
                first_native_first5_RMSE=checked['first_native_first5_RMSE'])
            comparisons[a+'_to_'+b] = item
        item = dict(task=task, suite=tasks[task]['suite'],
            group='target24' if task < 40 else 'source_seen_support12', identity=task in identity,
            cases=len(values[next(iter(values))]), success_sets={a: sorted(v) for a, v in sets.items()}, comparisons=comparisons)
        if 'reference' in sets:
            gains = sets['reference']-sets['source']
            item.update(reference_new_gains=sorted(gains), reference_gains_preserved={a: sorted(v & gains) for a, v in sets.items()})
        rows.append(item)
    return grouped(rows)


def interaction_evidence(root, rows, panels, limits):
    """Valid cases are unique; every unreturned attempt remains fully charged."""
    with connect(root) as db:
        ledger = {r[0]: r[1:] for r in db.execute('SELECT identity,case_id,status,control,dummy,row_path FROM attempts')}
    expected = set()
    from .readout import row_path
    for row, panel in zip(rows, panels, strict=True):
        case = f'seen_t{panel["task"]:04d}_s{panel["init_state_id"]:03d}'; expected.add(case)
        wanted = (case, 'complete', row['steps'], 10, str(row_path(root, 'seen', panel)))
        if ledger.get(row['attempt_id']) != wanted:
            raise ValueError('final row is not bound to its actual complete physical ledger receipt')
    complete = [r for r in ledger.values() if r[1] == 'complete']
    if len(complete) != 144 or {r[0] for r in complete} != expected or any(r[0] not in expected for r in ledger.values()):
        raise ValueError('only144 unique complete final cases may finish')
    evidence = summary(root); control = sum(v['control'] for v in evidence.values()); dummy = sum(v['dummy'] for v in evidence.values())
    physical = len(ledger); extra = physical-len({r[0] for r in ledger.values()})
    if (physical > limits['physical_submissions'] or control > limits['control_steps'] or dummy > limits['dummy_steps']
            or control+dummy > limits['total_steps'] or extra > limits['engineering_extra_attempts']):
        raise ValueError('actual plus unknown final costs exceed one of the registered simultaneous limits')
    return evidence


def paid_dependencies():
    base = REFERENCE_ROOT.parent
    return dict(shared_source=str(ASSETS/SOURCE), shared_Reader900=str(INITIAL_READER),
        shared_reference=str(REFERENCE_ROOT), source_practice_root=str(H_ROOT),
        source_practice=dict(episodes=144, control_steps=38937, dummy_steps=1440, total_steps=40377,
            successes=52, replans=7808, read_only_reused=True),
        historical_family_billing={name: billing(base/name) for name in
            ('reference_seed7', 'video_initial_seed7', 'feedback_proposals_seed7')},
        costs_are_not_from_zero='Current billing excludes prior source/reference/Reader learning; shared dependencies must not be double counted.')


def analyze(root):
    from .readout import inputs, saved_case
    root = Path(root); run, tasks, panels = inputs(root)
    new = [saved_case(root, 'seen', tasks['seen'][p['task']], p) for p in panels['seen']]
    old, anchors, provenance = historical(root, panels); raw = dict(old, **{ARM: new})
    results = grouped_evidence(raw, tasks['seen'], set(IDENTITY_TASKS))
    anchor_rows = {a: [r for r in rows if r['init_state_id'] == 0] for a, rows in raw.items()}
    anchor_rows.update(anchors)
    anchor_results = grouped_evidence(anchor_rows, tasks['seen'], set(IDENTITY_TASKS))
    interactions = interaction_evidence(root, new, panels['seen'], run['limits'])
    if len(new) != 144 or len({str(generated(root, 'seen', p)) for p in panels['seen']}) != 36:
        raise ValueError('the unique endpoint requires36 full controllers and144 new episodes')
    result = dict(schema=SCHEMA, phase=PHASE, complete=True, checkpoint=str(checkpoint_path(root, FINAL)), update=FINAL,
        panels=dict(seen=results), raw_rows=dict(seen=raw), same_video_init0_anchors=dict(rows=36,
            results=anchor_results, raw_rows=anchor_rows, other108_paired=False), provenance=provenance,
        billing=billing(root), interactions=interactions, limits=run['limits'], new_episodes=144, reused_final_episodes=0,
        zero_interaction=False, new_practice=0, shared_updates=FINAL, held_offline_labels_read=False, held=False, Test=False,
        selection='unique preregistered terminal180;90 is recovery only; finite non-held diagnostic, not strict400 qualification',
        automatic_successor=False, dependencies=paid_dependencies(),
        success_keys={a: sorted([[case_task(r), r['init_state_id']] for r in rows if r['success']]) for a, rows in raw.items()})
    write_json_atomic(root/'analysis/results.json', result)
    write_json_atomic(root/'completion.json', dict(complete=True, rows=144, controllers=36, result=str(root/'analysis/results.json')))
    lines = ['# 原生完整算子首写：唯一180末点', '',
        '非held有限面板144行：'+str(results['total']['successes']),
        '覆盖：'+str(results['total']['breadth']), '',
        '每task同一v0视频复用四新初态；这不是400不同教学条件或held迁移资格。',
        '复用144段source实践/40377步，当前0新实践不能记作zero-interaction。',
        '旧source-W仅init0的36个同视频锚配对；其余108未作同condition比较。',
        '任务、suite、target/support、identity、完整成功键、R/G/L和代价见同目录JSON。']
    (root/'analysis/report.md').write_text('\n'.join(lines)+'\n')
    return result
