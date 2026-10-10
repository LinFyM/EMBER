"""Complete bounded compilation, categorical records and stopped raw histories."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gzip
import math

import torch
from safetensors.torch import save_file

from ember.lora import validate_lora_state, LoRAContractError
from ember.writer.practice import History
from ember.writer.runtime import autocast
from ember.pi05_source_checkpoint import write_json_atomic

from .contract import seed


def cpu_state(state):
    return {k: v.detach().float().cpu().contiguous() for k, v in state.items()}


def history_prefix(history, records, episodes):
    rows = history.records[:records]
    keys = {key for row in rows for key in (row['pre'], row['post'])}
    refs = {'MT300'} | {row['parameter_ref'] for row in rows}
    return History(records=rows, observations={k: history.observations[k] for k in keys},
        episodes=history.episodes[:episodes], image_features={k: history.image_features[k] for k in keys
                                                            if k in history.image_features},
        states={k: v for k, v in history.states.items() if k in refs})


def save_history(path, history):
    """Lossless compression bounds raw RGB storage; labels are not Reader inputs."""
    with gzip.open(path, 'wb', compresslevel=1) as f:
        torch.save(history, f)


def load_history(path):
    with gzip.open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


@dataclass
class Compilation:
    task: int
    demo: int
    identity: str
    path_ordinal: int
    noise_root: int
    history: History
    states: dict
    practiced: list = field(default_factory=lambda: ['MT300'])
    valid: list = field(default_factory=lambda: ['MT300'])
    decisions: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    selected: str | None = None
    environment_steps: int = 0
    video_task: int | None = None

    def manifest(self):
        return dict(task_id=self.task, teacher_demo=self.demo, video_task=self.video_task,
            identity=self.identity, path_ordinal=self.path_ordinal, noise_root=self.noise_root,
            practiced_P=self.practiced, valid_U=self.valid, selected=self.selected,
            decisions=self.decisions, attempts=self.attempts, boundaries=self.boundaries,
            environment_steps=self.environment_steps, final_locked=self.selected is not None)

    def save(self, root, *, retain_U=False):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        save_history(root / 'history.pt.gz', self.history)
        refs = set(self.valid if retain_U else self.practiced) | {'MT300', self.selected}
        refs |= {r['parameter_ref'] for r in self.history.records}
        for ref in refs - {'MT300', None}:
            save_file(self.states[ref], str(root / f'{ref}.safetensors'))
        write_json_atomic(root / 'path.json', {**self.manifest(), 'retained_U': retain_U,
            'untried_reconstruction': 'frozen ψ plus attempt noise seed, exact chronological H and parent',
            'raw_history_compression': 'lossless gzip level1'})


def decision_inputs(runtime, path, decision, features):
    history = history_prefix(path.history, decision['records'], decision['episodes'])
    refs = set(decision['candidates']) - {'STOP'}
    refs |= {row['parameter_ref'] for row in history.records}
    refs.add('MT300')
    states = {k: path.states[k] for k in refs}
    parent = decision.get('parent') if decision['kind'] == 'practice' else None
    responses = runtime.responses({k: states[k] for k in decision['candidates'] if k != 'STOP'},
                                 history, runtime.tasks[path.task].authority.language, parent=parent)
    return history.model_rows(runtime), states, responses


def decision_logits(runtime, path, decision, features):
    rows, states, responses = decision_inputs(runtime, path, decision, features)
    with autocast(runtime.device):
        return runtime.actor(decision['kind'], features, rows, states, decision['candidates'],
                             responses, decision['budget']).float()


def categorical_record(runtime, path, kind, candidates, features, *, budget, rng, uniform,
                       parent=None, baseline=None):
    if not candidates:
        raise ValueError('no legal categorical choice')
    record = dict(kind=kind, candidates=list(candidates), records=len(path.history.records),
        episodes=len(path.history.episodes), budget=list(budget), parent=parent, temperature=1.,
        uniform_behavior=bool(uniform), forced=len(candidates) == 1)
    if len(candidates) == 1:
        probabilities = torch.ones(1)
    elif uniform:
        probabilities = torch.full((len(candidates),), 1 / len(candidates))
    else:
        probabilities = decision_logits(runtime, path, record, features).softmax(-1).detach().cpu()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError('nonfinite categorical probabilities: engineering numerical contract violation')
    chosen = int(torch.multinomial(probabilities, 1, generator=rng))
    record.update(selected=candidates[chosen], selected_index=chosen,
                  probabilities=probabilities.tolist(), log_probability=float(probabilities[chosen].log()))
    if baseline is not None:
        record['baseline_before_batch'] = float(baseline(features, path.history.model_rows(runtime), budget).detach())
    path.decisions.append(record)
    return candidates[chosen]


@torch.no_grad()
def compile_condition(runtime, runner, task, panel, demo, *, identity, path_ordinal, noise_root,
                      uniform=False, snapshots=False, video_task=None, baseline=None):
    if not runtime.psi_frozen:
        raise ValueError('compilation needs a fixed generation/read kernel')
    mt = cpu_state(runtime.mt)
    path = Compilation(task['global_task_id'], int(demo), identity, int(path_ordinal), int(noise_root),
        History(states={'MT300': mt}), {'MT300': mt}, video_task=video_task)
    features = runtime.teaching(path.task, demo, video_task=video_task)
    rng = torch.Generator().manual_seed(seed(30, noise_root, path_ordinal))
    while 1024 - path.environment_steps >= 11:
        budget = [(1024 - path.environment_steps) / 1024, (32 - len(path.attempts)) / 32,
                  len(path.history.episodes) / 100, len(path.practiced) / 33]
        parent_choices = list(path.practiced) + ([] if uniform else ['STOP'])
        parent = categorical_record(runtime, path, 'parent', parent_choices, features,
            budget=budget, rng=rng, uniform=uniform, baseline=baseline)
        if parent == 'STOP':
            break
        proposals = []
        for ordinal in range(min(4, 32 - len(path.attempts))):
            attempt = len(path.attempts)
            ref, xi_seed = f'G_{attempt:03d}', seed(31, noise_root, path_ordinal, attempt)
            refs = {'MT300', parent} | {r['parameter_ref'] for r in path.history.records}
            states = {key: path.states[key] for key in refs}
            with autocast(runtime.device):
                candidate = runtime.generator.generate(path.states[parent], features,
                    path.history.model_rows(runtime), states, noise_seed=xi_seed)
            try:
                validate_lora_state(candidate, runtime.lora)
                valid = all(bool(torch.isfinite(v).all()) for v in candidate.values())
            except LoRAContractError:
                valid = False
            item = dict(reference=ref, parent=parent, noise_seed=xi_seed, valid=valid,
                        records=len(path.history.records), episodes=len(path.history.episodes), euler_steps=16)
            path.attempts.append(item)
            if valid:
                path.states[ref] = cpu_state(candidate)
                path.valid.append(ref); proposals.append(ref)
            del candidate
        practice_budget=[(1024-path.environment_steps)/1024,(32-len(path.attempts))/32,
                         len(path.history.episodes)/100,len(path.practiced)/33]
        chosen = categorical_record(runtime, path, 'practice', [parent, *proposals], features,
            budget=practice_budget, parent=parent, rng=rng, uniform=uniform, baseline=baseline)
        n = len(path.history.episodes)
        state_id = panel['practice_states'][(path_ordinal + n) % len(panel['practice_states'])]
        request = dict(task=task, state=path.states[chosen], parameter_ref=chosen, state_id=state_id,
            noise_root=seed(32, noise_root, n), episode_id=f'{identity}_practice_{n:03d}',
            remaining=1024 - path.environment_steps, snapshots=snapshots, uses_teaching=True)
        result = next(runner.run([request]))
        path.environment_steps += int(result['row']['environment_steps'])
        path.history.append(result['history'])
        if result['row'].get('model_failure'):
            if chosen in path.practiced and chosen != 'MT300':
                path.practiced.remove(chosen)
            if chosen in path.valid and chosen != 'MT300':
                path.valid.remove(chosen)
        elif chosen not in path.practiced:
            path.practiced.append(chosen)
        path.boundaries.append(dict(records=len(path.history.records), episodes=len(path.history.episodes),
            practiced=list(path.practiced), actual_parameter_ref=chosen, row=result['row'],
            environment_steps=path.environment_steps, attempts=len(path.attempts)))
    budget = [(1024 - path.environment_steps) / 1024, (32 - len(path.attempts)) / 32,
              len(path.history.episodes) / 100, len(path.practiced) / 33]
    if 1024-path.environment_steps < 11:
        categorical_record(runtime,path,'parent',['STOP'],features,budget=budget,rng=rng,
                           uniform=uniform,baseline=baseline)
        path.decisions[-1]['forced_reason']='insufficient_environment_budget'
    path.selected = categorical_record(runtime, path, 'final', path.practiced, features,
        budget=budget, rng=rng, uniform=uniform, baseline=baseline)
    return path


def evaluate_fixed(runtime, runner, path, task, state_ids, noise_root, *, references=None):
    if path.selected is None:
        raise ValueError('independent query cannot precede final parameter lock')
    requests = []
    arms = [(ref, ref) for ref in references] if references is not None else [('selected', path.selected), ('MT', 'MT300')]
    for arm, reference in arms:
        for state_id in state_ids:
            requests.append(dict(task=task, state=path.states[reference], parameter_ref=reference,
                state_id=state_id, noise_root=noise_root, snapshots=False, uses_teaching=False,
                episode_id=f'{path.identity}_query_{arm}_{state_id}', remaining=task['horizon'] + 10, arm=arm))
    rows = []
    for result in runner.run(requests):
        row = dict(**result['row'], condition=path.identity, teaching_demo=path.demo,
                   global_task_id=path.task, reference=result['request']['parameter_ref'],
                   arm=result['request']['arm'],
                   selected=path.selected, query_excluded_from_H=True)
        rows.append(row)
    return rows


def restore_compilation(root, runtime):
    from safetensors.torch import load_file
    from ember.pi05_source_checkpoint import read_json
    root = Path(root); meta = read_json(root / 'path.json')
    history = load_history(root / 'history.pt.gz')
    states = dict(history.states)
    states['MT300'] = cpu_state(runtime.mt)
    states.update({file.stem: load_file(str(file)) for file in root.glob('*.safetensors')})
    path = Compilation(meta['task_id'], meta['teacher_demo'], meta['identity'], meta['path_ordinal'],
        meta['noise_root'], history, states, video_task=meta['video_task'])
    for attr, key in [('practiced', 'practiced_P'), ('valid', 'valid_U'), ('selected', 'selected'),
                      ('decisions', 'decisions'), ('attempts', 'attempts'), ('boundaries', 'boundaries'),
                      ('environment_steps', 'environment_steps')]:
        setattr(path, attr, meta[key])
    return path


def score_log_probability(logits, selected, *, forced=False):
    """Forced/single score is zero; complete paths sum these without length division."""
    if forced or len(logits) == 1:
        return logits.sum() * 0
    return logits.log_softmax(-1)[selected]
