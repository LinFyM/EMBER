"""Actual own branches, native score VJP, and one complete accepted task LoRA."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import save_file

from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.proposal_writer.path import save_history, load_history
from ember.proposal_writer.learning import memory_peak
from ember.writer.practice import processed
from .learning import feature_file
from .scoring import VideoTarget, no_video_score

ADAPT_NOISE, FINAL_NOISE = 202610150, 202610151
SCHEMA = 'ember_video_effect_full_parameter_compilation_v1'


def observation_keys(history):
    if not history.records:
        raise ValueError('real scoring episode has no executed decisions')
    return [history.records[0]['pre'], *(r['post'] for r in history.records)]


def physical_trace(history, *, prefix=None, point=None):
    """A restored observation occurs exactly once when joining the real prefix."""
    values = []
    if prefix is not None:
        values.extend(prefix.observations[k]['physical'] for k in observation_keys(prefix)[:point])
    values.extend(history.observations[k]['physical'] for k in observation_keys(history))
    static = values[0]
    if any(not np.array_equal(v['joint_type'], static['joint_type']) for v in values):
        raise ValueError('actual own physical registry changes within one real episode')
    return {**{key: np.stack([v[key] for v in values]) for key in
              ('hand_p', 'hand_R', 'hand_q', 'entity_p', 'entity_R', 'joint_q')},
            'joint_type': static['joint_type'], 'success': bool(history.episodes[-1]['success'])}


def case_request(task, state, parameter_ref, init, identity, *, snapshots=False, snapshot=None, first_actions=None):
    return dict(task=task, state=state, parameter_ref=parameter_ref, state_id=init,
        noise_root=ADAPT_NOISE, episode_id=identity, snapshots=snapshots, snapshot=snapshot,
        capture_history=True, capture_hidden=False, capture_phi=False,
        first_actions=first_actions, uses_teaching=False)


class ActualEpisodes:
    """Save each returned case immediately; canonical events count partial starts."""
    def __init__(self, root, runner):
        self.root, self.runner = Path(root), runner

    def run(self, requests):
        expected = {r['episode_id']: r for r in requests}
        received = {}
        for identity in expected:
            path = self.root / 'episodes' / identity
            if (path / 'row.json').exists():
                row = read_json(path / 'row.json')
                request = expected[identity]
                if (row['actual_parameter_ref'] != request['parameter_ref']
                        or row['init_state_id'] != request['state_id']
                        or row['policy_seed_root'] != request['noise_root']
                        or row.get('actual_override_actions') != (np.asarray(request['first_actions']).tolist()
                            if request.get('first_actions') is not None else None)):
                    raise ValueError('saved own episode has incompatible actual producer or RNG')
                received[identity] = (load_history(path / 'history.pt.gz'), path)
        pending = [r for k, r in expected.items() if k not in received]
        for result in self.runner.run(pending):
            request, row, history = result['request'], result['row'], result['history']
            identity = request['episode_id']; path = self.root / 'episodes' / identity
            row.update(global_task_id=request['task']['global_task_id'], episode_id=identity,
                first_actions_override=request.get('first_actions') is not None,
                actual_override_actions=np.asarray(request['first_actions']).tolist() if request.get('first_actions') is not None else None,
                restored_from_snapshot=request.get('snapshot') is not None)
            if row.get('model_failure') or row.get('budget_truncated'):
                raise RuntimeError('native numerical failure or truncated episode is not a complete scientific case')
            path.mkdir(parents=True, exist_ok=True)
            # Full factors have their own authoritative artifact; avoid H-sized copies.
            history.states = {}
            save_history(path / 'history.pt.gz', history)
            write_json_atomic(path / 'row.json', row)
            received[identity] = (history, path)
        if received.keys() != expected.keys():
            raise ValueError('actual episode consumer returned an incomplete identity set')
        return [received[r['episode_id']] for r in requests]


class FullFactors:
    """All 76 explicit native leaves, in fixed actual PolicyContexts units."""
    def __init__(self, runtime):
        self.center = runtime.mt
        self.scales = runtime.execution.prox
        self.keys = tuple(self.center)
        if len(self.keys) != 76 or len(runtime.lora.targets) != 38:
            raise ValueError('complete 38-target/r8 native factors are required')
        self.u = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(self.center[k])) for k in self.keys])
        self.optimizer = torch.optim.Adam(self.u.parameters(), lr=.1, betas=(.9, .95), eps=1e-8)

    def state(self):
        return {k: self.center[k] + self.scales[k] * value for k, value in zip(self.keys, self.u, strict=True)}

    def snapshot(self):
        return dict(u=[p.detach().clone() for p in self.u], optimizer=deepcopy(self.optimizer.state_dict()))

    @torch.no_grad()
    def restore(self, value):
        for p, old in zip(self.u, value['u'], strict=True): p.copy_(old)
        self.optimizer.load_state_dict(value['optimizer'])


def native_raw(runtime, factors, samples, language, *, gradient=False):
    items = [processed(runtime, raw, language) for raw, _ in samples]
    batch = {k: torch.cat([item[k] for item in items]) for k in items[0]}
    noise = torch.stack([torch.randn(50, 32, generator=torch.Generator().manual_seed(seed))
                         for _, seed in samples]).to(runtime.device)
    with torch.set_grad_enabled(gradient):
        action = runtime.native_actions([factors], batch, noise,
            batch_indices=torch.zeros(len(samples), dtype=torch.long, device=runtime.device), checkpointed=gradient)
        return runtime.processor.unnormalize_action(action)


def joint_branches(raw, rng):
    """Four marginals of ONE Gaussian-motion × Bernoulli-gripper distribution."""
    a = np.asarray(raw, dtype=np.float64)
    u, v = rng.standard_normal((5, 6)), rng.random(5)
    p = 1 / (1 + np.exp(-np.clip(a[:, 6], -80, 80)))
    z = (v < p, 1 - v < p)
    actions = []
    for s in (1, -1):
        for k in (0, 1):
            action = a.copy(); action[:, :6] = np.clip(a[:, :6] + s * .1 * u, -1, 1)
            action[:, 6] = 2 * z[k].astype(float) - 1
            actions.append(action.astype(np.float32))
    return actions, dict(u=u.tolist(), v=v.tolist(), p=p.tolist(), z=[x.tolist() for x in z])


def score_credit(scores, baseline, masks, draw):
    u, p, z = np.asarray(draw['u']), np.asarray(draw['p']), np.asarray(draw['z'], dtype=float)
    g = np.zeros((5, 7), dtype=np.float64)
    for j, (s, k) in enumerate((s, k) for s in (1, -1) for k in (0, 1)):
        m = np.asarray(masks[j], dtype=float)
        advantage = scores[j] - baseline
        g[:, :6] += advantage * s * u * m[:, None] / .4
        g[:, 6] += advantage * (z[k] - p) * m / 4
    if not np.isfinite(g).all(): raise ValueError('actual joint score credit is nonfinite')
    return g.astype(np.float32)


class Compilation:
    """An independent condition; real H and complete Adam/RNG survive acceptance."""
    def __init__(self, root, runtime, runner, context, task, arm, phi):
        self.root, self.runtime, self.context = Path(root), runtime, context
        self.task, self.arm = task, arm
        self.base = self.root / 'conditions' / f'task_{task["global_task_id"]:04d}_{arm}'
        self.base.mkdir(parents=True, exist_ok=True)
        self.actual = ActualEpisodes(root, runner)
        self.factors = FullFactors(runtime)
        self.rng = np.random.default_rng(np.random.SeedSequence([20261015, task['global_task_id']]))
        self.target = None
        if arm != 'no_video':
            item = torch.load(feature_file(root, task['global_task_id'], int(arm[-1])), map_location='cpu', weights_only=False)
            with torch.no_grad(), torch.autocast(runtime.device.type, dtype=torch.bfloat16):
                pred = phi(item['patches'].to(runtime.device), item['language'].to(runtime.device),
                    item['language_mask'].to(runtime.device), item['frame_indices'].to(runtime.device))
            pred = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for k, v in pred.items()}
            self.target = VideoTarget(pred)
            torch.save(pred, self.base / 'predicted_teaching.pt')
        self.parents = []; self.mappings = []
        self.update = self.starts = self.control = self.no_accept = 0
        self.parameter_ref = 'MT300'

    def score(self, history, pos, *, prefix=None, point=None):
        if self.target is None: return no_video_score(bool(history.episodes[-1]['success']))
        return self.target.score(physical_trace(history, prefix=prefix, point=point), self.mappings[pos])

    def initialize(self):
        file = self.base / 'checkpoint.pt'
        if file.exists():
            state = torch.load(file, map_location='cpu', weights_only=False)
            if state['schema'] != SCHEMA or state['arm'] != self.arm or state['task'] != self.task['global_task_id']:
                raise ValueError('own complete checkpoint identity changed')
            self.factors.restore(state['factors']); self.rng.bit_generator.state = state['rng_numpy']
            restore_rng(state['rank_rng'], self.context)
            for key in ('update', 'starts', 'control', 'no_accept', 'parameter_ref', 'mappings'):
                setattr(self, key, state[key])
            self.parents = [(load_history(Path(p) / 'history.pt.gz'), Path(p)) for p in state['parents']]
            return
        self.parents = [(load_history(self.root / 'bootstrap' / f'task_{self.task["global_task_id"]:04d}' /
            f'state_{i:03d}' / 'history.pt.gz'), self.root / 'bootstrap' / f'task_{self.task["global_task_id"]:04d}' /
            f'state_{i:03d}') for i in range(16, 20)]
        self.starts = 4
        self.control = sum(p[0].episodes[0]['steps'] for p in self.parents)
        if self.target is not None:
            self.mappings = [self.target.bind(h.observations[h.records[0]['pre']]['physical']) for h, _ in self.parents]
        self.save()

    def save(self):
        file = self.base / 'checkpoint.pt'; temporary = file.with_suffix('.tmp')
        torch.save(dict(schema=SCHEMA, task=self.task['global_task_id'], arm=self.arm,
            factors=self.factors.snapshot(), rng_numpy=self.rng.bit_generator.state,
            rank_rng=capture_rng(self.context), update=self.update, starts=self.starts, control=self.control,
            no_accept=self.no_accept, parameter_ref=self.parameter_ref, mappings=self.mappings,
            parents=[str(p) for _, p in self.parents], logical_flow='four actual parents; uniform one point each',
            topology=dict(world_size=1, device=str(self.runtime.device)), complete=True), temporary)
        temporary.replace(file)

    def _draw_plan(self):
        file = self.base / f'update_{self.update:03d}_draw.json'
        if file.exists():
            record = read_json(file)
            if record['parent_sources'] != [str(p) for _, p in self.parents]:
                raise ValueError('saved intervention draw has a different current parent')
            self.rng.bit_generator.state = record['rng_numpy_after']
            return record
        points, draws, requests, baselines = [], [], [], []
        for pos, (parent, _path) in enumerate(self.parents):
            point = int(self.rng.integers(len(parent.records)))
            row = parent.records[point]
            if row['snapshot'] is None:
                raise ValueError('uniform parent point lacks live v4 snapshot')
            actions, draw = joint_branches(row['planned_raw_actions'].numpy(), self.rng)
            points.append(point); draws.append(draw); baselines.append(self.score(parent, pos)['R'])
            for j, action in enumerate(actions):
                requests.append(dict(episode_id=f'task_{self.task["global_task_id"]:04d}_{self.arm}_u{self.update:03d}_parent{pos}_branch{j}',
                    first_actions=action.tolist()))
        remaining = [self.task['horizon']-parent.records[point]['snapshot']['episode_control_steps']
            for (parent,_),point in zip(self.parents,points,strict=True)]
        record = dict(points=points,T_e=[len(h.records) for h,_ in self.parents],draws=draws,
            baselines=baselines,parent_sources=[str(p) for _,p in self.parents],
            sampling_probability=[1/len(h.records) for h,_ in self.parents],requests=requests,
            maximum_atomic_control_steps=4*sum(remaining)+4*self.task['horizon'],
            maximum_atomic_starts=20,rng_numpy_after=self.rng.bit_generator.state)
        write_json_atomic(file,record)
        return record

    def _branches(self):
        record=self._draw_plan();requests=[];samples=[]
        state={k:v.detach() for k,v in self.factors.state().items()}
        for pos,((parent,_path),init) in enumerate(zip(self.parents,range(16,20),strict=True)):
            row=parent.records[record['points'][pos]]
            samples.append((parent.observations[row['pre']],row['noise_seed']))
            for data in record['requests'][4*pos:4*pos+4]:
                requests.append(case_request(self.task,state,self.parameter_ref,init,data['episode_id'],
                    snapshot=row['snapshot'],first_actions=np.asarray(data['first_actions'],dtype=np.float32)))
        returned=self.actual.run(requests)
        self.starts+=16;self.control+=sum(h.episodes[-1]['environment_steps'] for h,_ in returned)
        credits,scored=[],[]
        for pos,(parent,_path) in enumerate(self.parents):
            values=returned[4*pos:4*pos+4]
            rows=[self.score(h,pos,prefix=parent,point=record['points'][pos]) for h,_ in values]
            masks=[h.records[0]['executed'].numpy() for h,_ in values]
            credits.append(score_credit([r['R'] for r in rows],record['baselines'][pos],masks,record['draws'][pos]))
            scored.append(dict(scores=rows,masks=[m.tolist() for m in masks],
                traces=[str(p) for _h,p in values],credit=credits[-1].tolist()))
        return samples,credits,record,scored

    def _trust(self, old):
        samples, baseline = [], []
        draw = read_json(self.base / f'update_{self.update:03d}_draw.json')
        for pos, (history, _) in enumerate(self.parents):
            count = len(history.records)
            points = sorted({*np.floor(np.linspace(0, count - 1, 4)).astype(int).tolist(), draw['points'][pos]})
            for point in points:
                row = history.records[point]
                samples.append((history.observations[row['pre']], row['noise_seed']))
                baseline.append(row['planned_raw_actions'])
        baseline = torch.stack(baseline).to(self.runtime.device)
        proposed = [p.detach().clone() for p in self.factors.u]
        tests = []
        for halving in range(5):
            with torch.no_grad():
                for p, initial, end in zip(self.factors.u, old['u'], proposed, strict=True):
                    p.copy_(initial + (end - initial) * (.5 ** halving))
                action = native_raw(self.runtime, self.factors.state(), samples, self.task['language'])
                rms = (action - baseline).square().mean((1, 2)).sqrt()
            tests.append(dict(halvings=halving, maximum_RMS=float(rms.max()), finite=bool(torch.isfinite(action).all())))
            if tests[-1]['finite'] and tests[-1]['maximum_RMS'] <= .1:
                return True, tests
        self.factors.restore(old)
        return False, tests

    def _prepare_proposal(self,samples,credits,draw):
        prepared_file=self.base/f'update_{self.update:03d}_proposal.pt'
        if prepared_file.exists():
            prepared=torch.load(prepared_file,map_location='cpu',weights_only=False)
            if prepared['parent_ref']!=self.parameter_ref:
                raise ValueError('prepared native proposal belongs to another accepted parent')
            old=prepared['old'];credit=prepared['credit'];nonzero=prepared['nonzero']
            trust,tests=prepared['trust'],prepared['tests']
            self.factors.restore(prepared['factors'])
            restore_rng(prepared['rng'],self.context)
        else:
            old=self.factors.snapshot()
            g=torch.tensor(np.stack(credits),device=self.runtime.device)
            self.factors.optimizer.zero_grad(set_to_none=True)
            raw=native_raw(self.runtime,self.factors.state(),samples,self.task['language'],gradient=True)
            lengths=raw.new_tensor(draw['T_e'])
            objective=-(raw*g).sum((1,2)).mul(lengths).mean()
            objective.backward()
            gradient=torch.nn.utils.clip_grad_norm_(self.factors.u.parameters(),1.)
            if not torch.isfinite(gradient):raise RuntimeError('actual native score VJP is nonfinite')
            credit=dict(coordinate_gradient_norm=float(gradient),
                nonzero_factors=sum(p.grad is not None and bool(p.grad.any()) for p in self.factors.u),
                A_nonzero=sum(p.grad is not None and bool(p.grad.any()) for k,p in zip(self.factors.keys,self.factors.u) if '.lora_A.' in k),
                B_nonzero=sum(p.grad is not None and bool(p.grad.any()) for k,p in zip(self.factors.keys,self.factors.u) if '.lora_B.' in k))
            nonzero=bool(g.any());trust,tests=False,[]
            if nonzero:
                self.factors.optimizer.step();trust,tests=self._trust(old)
            temporary=prepared_file.with_suffix('.tmp')
            torch.save(dict(parent_ref=self.parameter_ref,old=old,credit=credit,nonzero=nonzero,
                trust=trust,tests=tests,factors=self.factors.snapshot(),rng=capture_rng(self.context)),temporary)
            temporary.replace(prepared_file)
        return old,credit,nonzero,trust,tests

    def step(self):
        tick = time.monotonic()
        samples, credits, draw, branch_scores = self._branches()
        old,credit,nonzero,trust,tests=self._prepare_proposal(samples,credits,draw)
        accepted,candidate=False,[]
        if trust:
            ref = f'task_{self.task["global_task_id"]:04d}_{self.arm}_proposal_{self.update:03d}'
            state = {k: v.detach() for k, v in self.factors.state().items()}
            validate_lora_state(state, self.runtime.lora)
            weight_file=self.base/f'{ref}.safetensors'
            if not weight_file.exists():
                save_file({k:v.cpu().contiguous() for k,v in state.items()},str(weight_file))
            requests = [case_request(self.task, state, ref, init, f'{ref}_state_{init:03d}', snapshots=True)
                        for init in range(16, 20)]
            candidate = self.actual.run(requests)
            self.starts += 4; self.control += sum(h.episodes[-1]['environment_steps'] for h, _ in candidate)
            old_scores = [self.score(h, pos) for pos, (h, _) in enumerate(self.parents)]
            previous_maps = self.mappings
            new_scores = [self.score(h, pos) for pos, (h, _) in enumerate(candidate)]
            a, b = sum(r['success'] for r in old_scores), sum(r['success'] for r in new_scores)
            accepted = b > a or (b == a and self.target is not None and
                np.mean([r['S_V'] for r in new_scores]) > np.mean([r['S_V'] for r in old_scores]) + 1e-6)
            if accepted: self.parents, self.parameter_ref = candidate, ref
            else: self.mappings = previous_maps
        if not accepted: self.factors.restore(old)
        self.no_accept = 0 if accepted else self.no_accept + 1
        write_json_atomic(self.base / f'update_{self.update:03d}_outcome.json', dict(
            draw=draw, branch_scores=branch_scores, native_credit=credit, nonzero_score_credit=nonzero,
            trust=tests, accepted=accepted, candidate_sources=[str(p) for _, p in candidate],
            parent_scores=old_scores if trust else None,candidate_scores=new_scores if trust else None,
            current_parameter_ref=self.parameter_ref, starts=self.starts, control_steps=self.control,
            seconds=time.monotonic() - tick, peak=memory_peak(self.runtime.device)))
        self.update += 1; self.save()

    def run(self):
        self.initialize()
        reason = 'budget'
        while True:
            if self.target is not None and any(m['status'] == 'no_match' for m in self.mappings):
                reason = 'video_target_formation_failed_MT'; break
            if all(h.episodes[-1]['success'] for h, _ in self.parents): reason = 'all_four_success'; break
            if self.no_accept >= 2: reason = 'two_no_acceptable_proposals'; break
            draw=self._draw_plan()
            if self.starts+draw['maximum_atomic_starts']>96 or self.control+draw['maximum_atomic_control_steps']>64*self.task['horizon']:break
            self.step()
        state = {k: v.detach().cpu().contiguous() for k, v in self.factors.state().items()}
        save_file(state, str(self.base / 'final.safetensors'))
        write_json_atomic(self.base / 'complete.json', dict(complete=True, schema=SCHEMA,
            task=self.task['global_task_id'], arm=self.arm, final_parameter_ref=self.parameter_ref,
            final_LoRA=str(self.base / 'final.safetensors'), factor_targets=38, rank=8,
            updates=self.update, stop_reason=reason, consumed_starts_including_shared_bootstrap=self.starts,
            consumed_control_steps_including_dependencies=self.control, shared_bootstrap_episodes=4,
            final_feedback_read=False, optimizer_exits_before_final=True))
