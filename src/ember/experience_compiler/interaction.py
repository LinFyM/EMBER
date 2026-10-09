"""Independent budget/success-driven slots and batched native control."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time
import uuid

import numpy as np
import torch

from ember.pi05_eval_contract import policy_noise_seed
from ember.writer.runtime import autocast
from .environments import EnvironmentSlots, raw_observation
from .execution import NativeVelocity, action_chunk


def cpu_state(state):
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def processed(runtime, raw, language):
    return runtime.processor({'observation.images.base_0_rgb': raw['images'][0].float().div(255),
        'observation.images.left_wrist_0_rgb': raw['images'][1].float().div(255),
        'observation.state': raw['proprio'], 'task': language})


@dataclass
class Chain:
    states: list = field(default_factory=list)
    records: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    phi: dict = field(default_factory=dict)
    behavior_versions: list = field(default_factory=list)

    def experience(self, endpoint, *, runtime=None):
        rows = self.records[:endpoint]
        if not rows:
            return {}
        if endpoint > len(self.records) or endpoint < 1:
            raise ValueError('experience endpoint exceeds recorded actual decisions')
        ids = {row[phase] for row in rows for phase in ('pre', 'post')}
        missing = {key: self.observations[key] for key in ids if key not in self.phi}
        if missing:
            if runtime is None:
                raise ValueError('recorded observation Phi needs its frozen native owner')
            self.phi.update(runtime.observation_features(missing))
        result = {key: torch.stack([row[key] for row in rows]) for key in
                  ('hidden', 'proprio', 'actions', 'executed', 'feedback')}
        result['images'] = torch.stack([torch.stack((self.phi[row['pre']], self.phi[row['post']])) for row in rows])
        result.update({key: torch.tensor([row[key] for row in rows], dtype=torch.long)
                       for key in ('episode', 'step')})
        return result

    def to_record(self):
        return dict(records=self.records, observations=self.observations,
                    endpoints=self.endpoints, episodes=self.episodes, metrics=self.metrics,
                    behavior_versions=self.behavior_versions)

    def support(self, runtime, language, *, endpoint=None, successful_only=False):
        """Uniform distinct real decisions with their original policy noise."""
        endpoint = len(self.records) if endpoint is None else endpoint
        eligible = [i for i, row in enumerate(self.records[:endpoint])
                    if not successful_only or self.episodes[row['episode']]['success']]
        if not eligible:
            return None
        positions = np.floor(np.linspace(0, len(eligible) - 1, min(16, len(eligible)))).astype(int)
        indices = [eligible[i] for i in positions]
        inputs, noise, seeds = [], [], []
        for index in indices:
            row = self.records[index]
            episode = self.episodes[row['episode']]
            seed = episode['policy_noise_seeds'][row['step'] // 5]
            if row.get('noise_seed', seed) != seed:
                raise ValueError('actual support noise differs from its episode stream')
            inputs.append(processed(runtime, self.observations[row['pre']], language))
            noise.append(torch.randn((50, 32), generator=torch.Generator().manual_seed(seed)))
            seeds.append(seed)
        return dict(indices=torch.tensor(indices, dtype=torch.long, device=runtime.device),
            batch={key: torch.cat([value[key] for value in inputs]) for key in inputs[0]},
            noise=torch.stack(noise).to(runtime.device), noise_seeds=seeds)

    @classmethod
    def from_record(cls, value):
        return cls(**value)


@dataclass
class _Slot:
    request: dict
    current: dict
    chain: Chain
    remaining: int | None
    stream: object = None
    ready: bool = False
    raw: dict | None = None
    observation_id: str = ''
    episode_steps: int = 0
    state_id: int = 0
    noise_root: int = 7
    pending: dict | None = None
    trajectory: list = field(default_factory=list)
    total_replans: int = 0
    reservoir_rng: object = None
    used_states: set = field(default_factory=set)
    state_reuses: int = 0
    native_cost: dict = field(default_factory=lambda: dict(seconds=0., frames=0, cache_hits=0))
    edit_seconds: float = 0.
    started: float = field(default_factory=time.monotonic)
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)


class Runner:
    """Bounded EGL slots; each completed condition releases its own capacity."""
    def __init__(self, runtime, contract, physical_gpu, *, slot_batch=8):
        self.runtime, self.contract = runtime, deepcopy(contract)
        self.slot_batch, self.physical_gpu = slot_batch, int(physical_gpu)
        self.environments = None
        self.started, self.total_environment_steps = time.monotonic(), 0
        self.components = dict(native_prefix_seconds=0., native_flow_seconds=0.,
                               env_operation_seconds=0., env_wait_seconds=0.,
                               edit_seconds=0., plans=0, physical_batch_histogram={})

    def _start(self, index, slot):
        request, condition = slot.request, slot.request.get('condition', {})
        task = request['task']
        if request['kind'] == 'final':
            state_id, root = request['state_id'], request.get('noise_root', 7)
        else:
            state_id = next(slot.stream)
            slot.state_reuses += int(state_id in slot.used_states)
            slot.used_states.add(state_id)
            root = int(np.random.SeedSequence([condition['seed'], len(slot.chain.episodes), 0xADA]).generate_state(1)[0])
        slot.state_id, slot.noise_root, slot.ready, slot.episode_steps = state_id, root, False, 0
        self.environments.submit(index, 'start', task=task,
            contract=request.get('environment_contract', self.contract), state_id=state_id,
            noise_root=request.get('environment_root', root), remaining=slot.remaining)

    def _new_slot(self, request):
        from .contract import state_stream
        kind = request['kind']
        if kind not in {'adapt', 'fixed', 'final'}:
            raise ValueError('unknown condition control consumer')
        current = request.get('state', self.runtime.mt)
        current = {k: v.detach().to(self.runtime.device) for k, v in current.items()}
        chain = Chain(states=[] if kind == 'final' else [cpu_state(current)])
        slot = _Slot(request, current, chain, None if kind == 'final' else request.get('step_budget', 1024))
        if request.get('reservoir'):
            slot.reservoir_rng = np.random.default_rng(request['reservoir_seed'])
        if kind != 'final':
            condition = request['condition']
            slot.stream = state_stream(condition, condition.get('excluded_states', condition.get('final_state_ids', ())))
        return slot

    def _observation(self, slot, raw):
        slot.raw = raw
        if slot.request['kind'] != 'final':
            key = f'{slot.identity}_o{len(slot.chain.observations):05d}'
            slot.chain.observations[key] = raw
            slot.observation_id = key

    def _responses(self, slots, *, block, capture_terminal=True, raise_errors=True):
        tick = time.monotonic()
        replies = self.environments.receive(block=block)
        self.components['env_wait_seconds'] += time.monotonic() - tick
        ended, errors = [], []
        for index, result in replies:
            slot = slots[index]
            if 'error' in result:
                self.total_environment_steps += result['confirmed_environment_steps']
                slot.failure = result
                errors.append(f'actual environment slot{index} failed:\n{result["error"]}')
                continue
            self.components['env_operation_seconds'] += result['operation_seconds']
            if result['kind'] == 'start':
                steps = result['settling_steps']
                self._observation(slot, result['raw'])
            else:
                steps = len(result['executed'])
                self._record(slot, result)
                slot.episode_steps = result['steps']
            self.total_environment_steps += steps
            if slot.remaining is not None:
                slot.remaining -= steps
                if slot.remaining < 0:
                    raise ValueError('actual environment exceeded its registered condition budget')
            if result['episode_ended']:
                ended.append((index, result['row']))
            else:
                slot.ready = True
        # Terminal observations have no subsequent prefix: encode them once.
        if errors and raise_errors:
            raise RuntimeError('\n'.join(errors))
        observations = {slots[i].observation_id: slots[i].raw for i, _ in ended
                        if slots[i].request['kind'] != 'final' and slots[i].chain.records}
        phi = self.runtime.observation_features(observations) if observations and capture_terminal else {}
        for index, _ in ended:
            key = slots[index].observation_id
            if key in phi:
                slots[index].chain.phi[key] = phi[key]
        return ended

    def _record(self, slot, result):
        if slot.request['kind'] == 'final':
            if slot.request.get('capture'):
                pending = slot.pending
                executed = torch.from_numpy(result['executed']).clone()
                actions = torch.zeros(5, 7)
                actions[:len(executed)] = executed
                record = dict(**pending, actions=actions,
                    executed=torch.arange(5) < len(executed),
                    reward=result['reward'], done=result['done'])
                slot.total_replans += 1
                capacity = slot.request.get('reservoir')
                if capacity is None or len(slot.trajectory) < capacity:
                    slot.trajectory.append(record)
                else:
                    replacement = int(slot.reservoir_rng.integers(slot.total_replans))
                    if replacement < capacity:
                        slot.trajectory[replacement] = record
                slot.pending = None
            slot.raw = result['raw']
            return
        pending = slot.pending
        before = slot.raw
        self._observation(slot, result['raw'])
        executed = torch.from_numpy(result['executed'])
        actions = torch.zeros(5, 7)
        actions[:len(executed)] = executed
        truncated = bool(slot.remaining == len(executed) and not result['done'])
        version = self._behavior(slot)
        slot.chain.records.append(dict(pre=pending['pre'], post=slot.observation_id,
            hidden=pending['hidden'], proprio=torch.stack((before['proprio'], slot.raw['proprio'])),
            actions=actions, executed=torch.arange(5) < len(executed),
            feedback=torch.tensor([result['reward'], float(result['done']), float(result['done']), float(truncated)]),
            episode=len(slot.chain.episodes), step=slot.episode_steps, behavior_version=version,
            noise_seed=pending['noise_seed']))
        slot.pending = None

    def _behavior(self, slot):
        if slot.request['kind'] == 'fixed':
            return slot.request.get('fixed_behavior_version', 'MT')
        if len(slot.chain.states) == 1:
            return 'MT'
        return f"{slot.request['behavior_version']}:lambda{len(slot.chain.states) - 1:03d}"

    def _finish_episode(self, index, slot, row):
        kind = slot.request['kind']
        if kind == 'final':
            row['physical_slot_batch'] = self.slot_batch
            row['policy_seed_root'] = slot.noise_root
            result = dict(request=slot.request, row=row)
            if slot.request.get('capture'):
                result.update(trajectory=slot.trajectory, terminal_raw=slot.raw,
                              total_replans=slot.total_replans)
            return result
        chain = slot.chain
        previous_endpoint = chain.endpoints[-1] if chain.endpoints else 0
        chain.episodes.append(row)
        if len(chain.records) > previous_endpoint:
            chain.behavior_versions.append(self._behavior(slot))
            chain.endpoints.append(len(chain.records))
            if kind == 'adapt':
                # Even own success enters editing before stopping/freezing.
                tick = time.monotonic()
                teacher = slot.request.get('teacher')
                if teacher is None:
                    condition = slot.request['condition']
                    teacher = self.runtime.teacher(condition['task_id'], condition['teacher_demo'],
                                                   slot.request.get('role', 'train'))
                    cost = self.runtime.last_teacher_cost
                    slot.native_cost['seconds'] += cost.get('native_encoder_seconds', 0.)
                    slot.native_cost['frames'] += cost.get('native_encoded_frames', 0)
                    slot.native_cost['cache_hits'] += int(cost.get('cache_hit', False))
                slot.teacher_frames = len(teacher['indices'])
                evidence = {} if slot.request.get('masked', False) else chain.experience(len(chain.records))
                support = chain.support(self.runtime, slot.request['task']['language'])
                slot.current = self.runtime.edit(slot.current, teacher, evidence, support=support)
                chain.states.append(cpu_state(slot.current))
                seconds = time.monotonic() - tick
                slot.edit_seconds += seconds
                self.components['edit_seconds'] += seconds
        if row['success'] or slot.remaining == 0:
            self._metrics(slot, row['success'])
            return dict(request=slot.request, chain=chain)
        self._start(index, slot)
        return None

    def _metrics(self, slot, success):
        chain, request = slot.chain, slot.request
        reads = len(chain.states) - 1
        frames = getattr(slot, 'teacher_frames', 0)
        chain.metrics = dict(initial_reads=min(1, reads), rereads=max(0, reads - 1), actual_J=reads,
            full_video_equivalent_reads=float(reads), teacher_frames=frames, read_frames=reads * frames,
            native_teacher_encoded_frames=slot.native_cost['frames'], teacher_feature_cost=slot.native_cost,
            environment_steps=sum(e['environment_steps'] for e in chain.episodes),
            resets=len(chain.episodes), state_reuses=slot.state_reuses,
            stop_reason='own_success' if success else 'step_budget', practice_success=bool(success),
            failure_environment_steps=sum(e['environment_steps'] for e in chain.episodes if not e['success']),
            tail_environment_steps=chain.episodes[-1]['environment_steps'] if not success else 0,
            data_endpoints=len(chain.endpoints), masked_experience=bool(request.get('masked', False)),
            condition_seed=request['condition']['seed'],
            excluded_states=request['condition'].get('excluded_states', request['condition'].get('final_state_ids', [])),
            behavior_actor_uses_teaching=request['kind'] == 'adapt',
            edit_seconds=slot.edit_seconds, wall_seconds=time.monotonic() - slot.started,
            physical_slot_batch=self.slot_batch)

    @torch.no_grad()
    def _infer(self, slots, indices):
        inputs = [processed(self.runtime, slots[i].raw, slots[i].request['task']['language']) for i in indices]
        batch = {key: torch.cat([row[key] for row in inputs]) for key in inputs[0]}
        fact_indices = [p for p, i in enumerate(indices) if slots[i].request['kind'] != 'final']
        collect = bool(fact_indices)
        noise, seeds = [], []
        for index in indices:
            slot, task = slots[index], slots[index].request['task']
            seed = policy_noise_seed(slot.noise_root, task['suite'], int(task['task_id']),
                                     slot.state_id, slot.episode_steps // 5)
            seeds.append(seed)
            noise.append(torch.randn((50, 32), generator=torch.Generator().manual_seed(seed)))
        with self.runtime.execution.activate([slots[i].current for i in indices]), autocast(self.runtime.device):
            tick = time.monotonic()
            velocity = NativeVelocity(self.runtime.policy, batch, capture_phi=collect, capture_indices=fact_indices if collect else None)
            self.components['native_prefix_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            chunks, hidden = action_chunk(velocity, torch.stack(noise).to(self.runtime.device),
                capture_hidden=collect, capture_indices=fact_indices if collect else None)
            sampled, gaussian = chunks.clone(), []
            for position, index in enumerate(indices):
                request = slots[index].request
                if request.get('exploration_sigma') is not None:
                    seed = int(np.random.SeedSequence([request['exploration_root'],
                        slots[index].episode_steps // 5]).generate_state(1)[0])
                    perturbation = torch.randn((5, 7), generator=torch.Generator().manual_seed(seed))
                    value = chunks[position, :5] + request['exploration_sigma'] * perturbation.to(chunks.device)
                    sampled[position, :5] = value
                    gaussian.append(value.detach().float().cpu())
                else:
                    gaussian.append(None)
            actions = self.runtime.processor.unnormalize_action(sampled).cpu().numpy()
            self.components['native_flow_seconds'] += time.monotonic() - tick
        self.components['plans'] += len(indices)
        histogram = self.components['physical_batch_histogram']
        histogram[len(indices)] = histogram.get(len(indices), 0) + 1
        cache_entries = {}
        normalized = chunks.detach().float().cpu() if any(slots[i].request.get('capture') for i in indices) else None
        for position, index in enumerate(indices):
            slot = slots[index]
            if slot.request['kind'] != 'final':
                fact_position = fact_indices.index(position)
                key, phi = slot.observation_id, velocity.phi[fact_position]
                slot.chain.phi[key] = phi
                cache_entries[key] = {'phi': phi}
                # A contiguous slot view still owns the whole native batch's
                # CPU storage; torch.save would retain the other conditions.
                slot.pending = dict(pre=key, hidden=hidden[fact_position].clone(), noise_seed=seeds[position])
            elif slot.request.get('capture'):
                slot.pending = dict(raw=slot.raw, noise=noise[position].clone(), noise_seed=seeds[position],
                    normalized_actions=normalized[position].clone(),
                    commands=torch.from_numpy(actions[position]).clone(), step=slot.episode_steps)
                if gaussian[position] is not None:
                    slot.pending['gaussian_actions'] = gaussian[position].clone()
            slot.ready = False
            capture = {} if not slot.request.get('capture') else dict(normalized_chunk=normalized[position].numpy())
            self.environments.submit(index, 'step', actions=actions[position], noise_seed=seeds[position], **capture)
        if cache_entries:
            from .storage import tensor_bytes
            self.runtime.io.submit(self.runtime.features.put_many, cache_entries, byte_cost=tensor_bytes(cache_entries))

    @torch.no_grad()
    def run(self, requests):
        """Yield each completed slot immediately, then admit its next request."""
        if self.environments is None:
            self.environments = EnvironmentSlots(self.contract, self.physical_gpu, self.slot_batch)
        provider = requests if callable(requests) else None
        source, slots, exhausted = None if provider else iter(requests), {}, False
        self.live_slots = slots
        while slots or not exhausted:
            for index in range(self.slot_batch):
                if index in slots or exhausted:
                    continue
                try:
                    request = provider() if provider else next(source)
                    if request is None:
                        break
                except StopIteration:
                    exhausted = True
                    break
                slots[index] = self._new_slot(request)
                self._start(index, slots[index])
            if not slots:
                break
            ready = [i for i, s in slots.items() if s.ready]
            ended = self._responses(slots, block=not ready)
            for index, row in ended:
                output = self._finish_episode(index, slots[index], row)
                if output is not None:
                    del slots[index]
                    yield output
            # A five-action CPU cohort finishes independently of condition
            # endings; gather its replies before launching the next GPU batch.
            while any(kind == 'step' for kind in self.environments.pending.values()):
                for index, row in self._responses(slots, block=True):
                    output = self._finish_episode(index, slots[index], row)
                    if output is not None:
                        del slots[index]
                        yield output
            ready = [i for i, s in slots.items() if s.ready]
            if ready:
                self._infer(slots, ready)

    def adapt_many(self, conditions, *, role='train', fixed_behavior=False, behavior_version='fresh', step_budget=1024):
        tasks = {t['global_task_id']: t for t in self.contract['tasks']}
        requests = (dict(kind='fixed' if fixed_behavior else 'adapt', condition=c, task=tasks[c['task_id']],
            role=role, behavior_version=behavior_version, step_budget=step_budget) for c in conditions)
        return self.run(requests)

    def final_requests(self, requests):
        return self.run(dict(kind='final', **request) for request in requests)

    def close(self):
        if self.environments is not None:
            # Pending real environment operations remain charged on failure.
            while self.environments.pending and getattr(self, 'live_slots', None):
                self._responses(self.live_slots, block=True, capture_terminal=False, raise_errors=False)
            self.environments.close()
            self.environments = None

    def preserve_partial(self, destination):
        """Keep raw unfinished facts/parameters after failure without GPU work."""
        from pathlib import Path
        from safetensors.torch import save_file
        from ember.pi05_source_checkpoint import write_json_atomic
        for index, slot in getattr(self, 'live_slots', {}).items():
            path = Path(destination) / f'slot{index}_{slot.identity}'
            path.mkdir(parents=True, exist_ok=False)
            torch.save(slot.chain.to_record(), path / 'experience.pt')
            if slot.request.get('capture'):
                torch.save(dict(records=slot.trajectory, terminal_raw=slot.raw,
                    condition=slot.request.get('condition'), arm=slot.request.get('arm'),
                    state_id=slot.state_id, complete=False), path / 'partial_trajectory.pt')
            for ordinal, state in enumerate(slot.chain.states[1:], 1):
                save_file(state, str(path / f'lambda_{ordinal:03d}.safetensors'))
            write_json_atomic(path / 'partial.json', dict(condition=slot.request.get('condition'),
                kind=slot.request['kind'], complete=False, endpoints=slot.chain.endpoints,
                remaining_environment_steps=slot.remaining, failure=getattr(slot, 'failure', None),
                registered_budget=slot.request.get('step_budget', 1024)))
