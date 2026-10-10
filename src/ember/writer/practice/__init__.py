"""One canonical episode engine: explicit fixed task parameters and actual facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time

import numpy as np
import torch

from ember.pi05_eval_contract import policy_noise_seed
from ember.writer.runtime import autocast
from ember.writer.practice.environment import EnvironmentSlots
from ember.writer.practice.execution import NativeVelocity, action_chunk


def processed(runtime, raw, language):
    return runtime.processor({'observation.images.base_0_rgb': raw['images'][0].float().div(255),
        'observation.images.left_wrist_0_rgb': raw['images'][1].float().div(255),
        'observation.state': raw['proprio'], 'task': language})


@dataclass
class History:
    records: list = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    episodes: list = field(default_factory=list)
    image_features: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)

    def append(self, other):
        offset = len(self.episodes)
        self.records.extend([{**row, 'episode': row['episode'] + offset} for row in other.records])
        self.observations.update(other.observations)
        self.episodes.extend(other.episodes)
        self.image_features.update(other.image_features)
        self.states.update(other.states)

    def model_rows(self, runtime):
        missing = {key: raw for key, raw in self.observations.items() if key not in self.image_features}
        self.image_features.update(runtime.observation_features(missing))
        return [dict(**{key: row[key] for key in ('hidden', 'proprio', 'actions', 'executed', 'feedback',
                                                  'episode', 'step', 'parameter_ref')},
                     images=torch.stack((self.image_features[row['pre']], self.image_features[row['post']])))
                for row in self.records]

    def response_support(self, runtime, language):
        if not self.records:
            return None
        positions = np.floor(np.linspace(0, len(self.records) - 1, min(16, len(self.records)))).astype(int)
        selected = [self.records[i] for i in positions]
        items = [processed(runtime, self.observations[row['pre']], language) for row in selected]
        return dict(batch={key: torch.cat([value[key] for value in items]) for key in items[0]},
                    noise=torch.stack([torch.randn(50, 32, generator=torch.Generator().manual_seed(row['noise_seed']))
                                       for row in selected]).to(runtime.device), positions=positions.tolist())


class EpisodeRunner:
    """Persistent cost-balanced slots, one fixed policy per real episode.

    Scheduling/reset/step/batched ODE is lifted from the existing fixed runner.
    No edit operator, learned trajectory fitting, SDE or algorithm fallback.
    """
    def __init__(self, runtime, contract, physical_gpu, *, slots=8):
        self.runtime, self.contract, self.physical_gpu = runtime, deepcopy(contract), int(physical_gpu)
        self.slots, self.environments = slots, None
        self.components = dict(native_prefix_seconds=0., native_flow_seconds=0., env_operation_seconds=0.,
                               env_wait_seconds=0., plans=0, physical_batch_histogram={})
        self.environment_steps = 0

    def _start(self, index, request):
        if 'state' not in request or 'parameter_ref' not in request:
            raise ValueError('episode needs actual complete incoming factors and provenance')
        history = History(states={request['parameter_ref']: {k: v.detach().cpu() for k, v in request['state'].items()}})
        slot = dict(request=request, history=history, raw=None, ready=False, key='', steps=0,
                    pending=None, initial_steps=int(request.get('snapshot', {}).get('episode_control_steps', 0)))
        self.environments.submit(index, 'start', task=request['task'], contract=self.contract,
            state_id=request['state_id'], noise_root=request['noise_root'], remaining=request.get('remaining'),
            snapshot=request.get('snapshot'), capture_snapshot=request.get('snapshots', False))
        return slot

    def _observation(self, slot, raw, snapshot=None):
        history = slot['history']
        key = f"{slot['request']['episode_id']}_o{len(history.observations):05d}"
        history.observations[key] = raw
        slot.update(raw=raw, key=key, snapshot=snapshot)

    def _receive(self, live, *, block):
        started = time.monotonic()
        results = self.environments.receive(block=block)
        self.components['env_wait_seconds'] += time.monotonic() - started
        finished = []
        for index, result in results:
            if 'error' in result:
                raise RuntimeError(result['error'])
            slot = live[index]
            self.components['env_operation_seconds'] += result['operation_seconds']
            if result['kind'] == 'start':
                self.environment_steps += result['settling_steps']
                slot['settling'] = result['settling_steps']
            else:
                before, pending = slot['raw'], slot['pending']
                executed = torch.from_numpy(result['executed'])
                self.environment_steps += len(executed)
                actions = torch.zeros(5, 7)
                actions[:len(executed)] = executed
                self._observation(slot, result['raw'], result.get('snapshot'))
                slot['history'].records.append(dict(pre=pending['pre'], post=slot['key'],
                    hidden=pending['hidden'], snapshot=pending['snapshot'], noise_seed=pending['noise_seed'],
                    failure_type=result.get('failure_type'),
                    normalized_actions=pending['normalized_actions'],
                    parameter_ref=slot['request']['parameter_ref'], episode=0, step=slot['steps'],
                    proprio=torch.stack((before['proprio'], result['raw']['proprio'])), actions=actions,
                    executed=torch.arange(5) < len(executed),
                    feedback=torch.tensor([result['reward'], float(result['done']), float(result['done']),
                                           float(result['episode_ended'] and not result['done']),
                                           float(result.get('failure_type') is not None)])))
                slot['steps'] += len(executed)
            if result['kind'] == 'start':
                self._observation(slot, result['raw'], result.get('snapshot'))
            if result['episode_ended']:
                row = result['row']
                row.update(actual_parameter_ref=slot['request']['parameter_ref'],
                           behavior_actor_uses_teaching=slot['request'].get('uses_teaching', False))
                slot['history'].episodes.append(row)
                finished.append((index, dict(request=slot['request'], row=row, history=slot['history'])))
            else:
                slot['ready'] = True
        return finished

    def _infer(self, live, indices):
        inputs, noises, seeds = [], [], []
        for index in indices:
            slot, request = live[index], live[index]['request']
            task = request['task']
            inputs.append(processed(self.runtime, slot['raw'], task['language']))
            absolute = slot['initial_steps'] + slot['steps']
            value = policy_noise_seed(request['noise_root'], task['suite'], int(task['task_id']),
                                      request['state_id'], absolute // 5)
            seeds.append(value)
            noises.append(torch.randn(50, 32, generator=torch.Generator().manual_seed(value)))
        batch = {key: torch.cat([value[key] for value in inputs]) for key in inputs[0]}
        with self.runtime.execution.activate([live[i]['request']['state'] for i in indices]), autocast(self.runtime.device):
            tick = time.monotonic()
            velocity = NativeVelocity(self.runtime.policy, batch)
            self.components['native_prefix_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            normalized, hidden = action_chunk(velocity, torch.stack(noises).to(self.runtime.device))
            finite = torch.isfinite(normalized).flatten(1).all(1).cpu()
            commands = self.runtime.processor.unnormalize_action(torch.nan_to_num(normalized)).cpu().numpy()
            self.components['native_flow_seconds'] += time.monotonic() - tick
        self.components['plans'] += len(indices)
        histogram = self.components['physical_batch_histogram']
        histogram[len(indices)] = histogram.get(len(indices), 0) + 1
        for pos, index in enumerate(indices):
            slot = live[index]
            slot['history'].image_features[slot['key']] = velocity.phi[pos].clone()
            slot.update(ready=False, pending=dict(pre=slot['key'], hidden=torch.nan_to_num(hidden[pos]).clone(),
                normalized_actions=torch.nan_to_num(normalized[pos, :5].detach().float().cpu()).clone(),
                snapshot=slot['snapshot'], noise_seed=seeds[pos]))
            if bool(finite[pos]):
                self.environments.submit(index, 'step', actions=commands[pos], noise_seed=seeds[pos],
                                         capture_snapshot=slot['request'].get('snapshots', False))
            else:
                self.environments.submit(index, 'numerical_failure')

    @torch.no_grad()
    def run(self, requests):
        if self.environments is None:
            self.environments = EnvironmentSlots(self.contract, self.physical_gpu, self.slots)
        source, live, exhausted = iter(requests), {}, False
        while live or not exhausted:
            for index in range(self.slots):
                if index in live or exhausted:
                    continue
                try:
                    live[index] = self._start(index, next(source))
                except StopIteration:
                    exhausted = True
            if not live:
                break
            ready = [i for i, slot in live.items() if slot['ready']]
            for index, result in self._receive(live, block=not ready):
                del live[index]
                yield result
            ready = [i for i, slot in live.items() if slot['ready']]
            if ready:
                self._infer(live, ready)

    def close(self):
        if self.environments is not None:
            self.environments.close()
            self.environments = None
