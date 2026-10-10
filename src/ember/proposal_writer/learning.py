"""Equal-condition CFM, local final-use learning and one-update on-policy credit."""
from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast

from .contract import TASKS, OPTIMIZER, seed, SCHEMA
from .path import decision_logits, history_prefix, score_log_probability, load_history


class Baseline(nn.Module):
    """Independent ω; frozen teaching/facts only, no actor or generator embedding."""
    def __init__(self):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(2048 + 5 + 4, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, features, rows, budget):
        language = features['language'][features['language_mask']].detach().float().mean(0)
        facts = torch.stack([r['feedback'].to(language.device) for r in rows]).mean(0) if rows else language.new_zeros(5)
        inputs = torch.cat((language, facts, language.new_tensor(budget)))
        return self.value(inputs).squeeze(-1)


def synchronize_gradients(module, context):
    """Sum locally /4-scaled condition gradients; never average path decisions."""
    if context.world_size == 1:
        return
    parameters = list(module.parameters())
    used = torch.tensor([p.grad is not None for p in parameters], device=context.device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, flag in zip(parameters, used.tolist(), strict=True) if flag]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    cursor = 0
    for p in active:
        count = p.numel()
        if p.grad is None:
            p.grad = torch.empty_like(p)
        p.grad.copy_(flat[cursor:cursor + count].reshape_as(p))
        cursor += count


def checkpoint(root, stage, update, runtime, optimizer, context, *, sampler, baseline=None, baseline_optimizer=None):
    root = Path(root) / 'checkpoints' / stage / f'update_{update:08d}'
    rng = capture_rng(context)
    ranks = [None] * context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(rng, ranks, dst=0)
    else:
        ranks = [rng]
    if context.is_main:
        root.mkdir(parents=True, exist_ok=False)
        state = dict(schema_version=SCHEMA, stage=stage, next_update=update,
            psi=runtime.generator.state_dict(), theta=runtime.actor.state_dict(), optimizer=optimizer.state_dict(),
            baseline=baseline.state_dict() if baseline is not None else None,
            baseline_optimizer=baseline_optimizer.state_dict() if baseline_optimizer is not None else None,
            sampler=sampler, rank_rng=ranks, topology=dict(world_size=context.world_size,
                task_order=list(TASKS), logical_condition_batch=4), psi_version=runtime.version,
            coordinates=dict(rank=8,valid=runtime.lora.parameter_count,center=str(runtime.root/'initial.safetensors'),
                fixed_bank_scale=str(runtime.bank_scale_path)),
            fixed_kernel_identity=getattr(runtime,'fixed_kernel_identity',None),
            parameter_names=dict(psi=list(dict(runtime.generator.named_parameters())),
                                 theta=list(dict(runtime.actor.named_parameters()))))
        torch.save(state, root / 'state.pt')
        write_json_atomic(root / 'manifest.json', dict(schema_version=state['schema_version'], stage=stage,
            next_update=update, full_optimizer_rng_sampler=True, complete=True, topology=state['topology']))
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])
    return root


def resume_checkpoint(path, stage, runtime, optimizer, context, *, baseline=None, baseline_optimizer=None):
    state = torch.load(Path(path) / 'state.pt', map_location='cpu', weights_only=False)
    if (state['schema_version'] != SCHEMA or state['stage'] != stage
            or state['topology']['world_size'] != context.world_size):
        raise ValueError('resume stage/topology changed; explicit migration needs a recorded new topology contract')
    runtime.generator.load_state_dict(state['psi']); runtime.actor.load_state_dict(state['theta'])
    optimizer.load_state_dict(state['optimizer']); runtime.version = state['psi_version']
    restore_rng(state['rank_rng'][context.rank], context)
    if baseline is not None:
        baseline.load_state_dict(state['baseline']); baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    return state['next_update']


def metrics(root, stage, record, context):
    destination = Path(root) / 'metrics' / f'{stage}_rank_{context.rank}.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    import json
    with destination.open('a') as f:
        f.write(json.dumps(record) + '\n')


def assigned_tasks(context):
    return [t for i, t in enumerate(TASKS) if i % context.world_size == context.rank]


def train_cfm(runtime, context, root, *, stop, resume=None):
    if stop not in (240, 480):
        raise ValueError('CFM checkpoints are predeclared')
    runtime.psi_frozen = False; runtime.generator.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(runtime.generator.parameters(), **OPTIMIZER)
    initialize_deferred_process_group(context, rendezvous_root=Path(root) / 'rendezvous')
    start = resume_checkpoint(resume, 'G', runtime, optimizer, context) if resume else 0
    if not runtime.bank_scale_path.exists():
        raise ValueError('CFM requires the complete registered teacher bank and fixed S_G')
    if resume is None and context.is_main:
        torch.save(dict(schema_version=SCHEMA,psi=runtime.generator.state_dict(),theta=runtime.actor.state_dict(),
            formal_fresh=True,coordinates=str(runtime.root/'coordinates.json'),bank_scale=str(runtime.bank_scale_path)),
            Path(root)/'initial_models.pt')
    history_cache={}
    for update in range(start, stop):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); records = []
        for task in assigned_tasks(context):
            slots = 4
            rng = np.random.default_rng(seed(40, task, update))
            ordinal = int(rng.integers(slots))
            event_root = Path(root) / 'events' / f'task_{task:04d}_event_{ordinal:02d}'
            event_path = event_root / 'event.json'
            if not event_path.exists():
                raise ValueError('registered multistart event missing before full-bank CFM')
            event = read_json(event_path)
            distribution = read_json(event_root / 'teacher_distribution.json')
            nodes = distribution['legal_nodes']
            if not nodes:
                records.append(dict(task=task, event_ordinal=ordinal, missing_label=True, denominator=4))
                continue
            q = np.asarray([distribution['probabilities'][str(i)] for i in nodes])
            node = int(rng.choice(nodes, p=q / q.sum()))
            from safetensors.torch import load_file
            endpoint = load_file(str(event_root / 'teacher' / 'checkpoints' / f'update_{node:08d}' / 'lora.safetensors'))
            parent = runtime.mt if event['parent_ref'] == 'MT300' else load_file(event['parent'])
            if event['history_path'] not in history_cache:
                history_cache[event['history_path']]=load_history(event['history_path'])
            history=history_cache[event['history_path']]
            features = runtime.teaching(task, event['teacher_demo'])
            packed = runtime.generator.layout.pack(parent).to(runtime.device)
            target = (runtime.generator.layout.pack(endpoint).to(runtime.device) - packed) / runtime.generator.layout.scale
            xi = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed(41, task, update))).to(runtime.device)
            xi = xi * runtime.generator.layout.valid
            t = float(rng.uniform())
            x = (1 - t) * xi + t * target
            with autocast(runtime.device):
                predicted = runtime.generator(x, t, packed, features, history.model_rows(runtime), history.states)
                loss = (predicted.float()[runtime.generator.layout.valid] -
                        (target - xi)[runtime.generator.layout.valid]).square().mean()
            (loss / 4).backward()
            records.append(dict(task=task, event_ordinal=ordinal, node=node, time=t, loss=float(loss.detach()),
                                frames=len(features['z']), denominator=4, missing_label=False))
            del features, predicted, loss, x, xi, target, endpoint
        synchronize_gradients(runtime.generator, context)
        norm = torch.nn.utils.clip_grad_norm_(runtime.generator.parameters(), 1., error_if_nonfinite=True)
        group_norms = {g: sum(float(p.grad.float().square().sum()) for p, name in
                        zip(runtime.generator.meta.values, runtime.generator.meta.groups, strict=True)
                        if name == g and p.grad is not None) ** .5 for g in ('gemma', 'action')}
        optimizer.step(); runtime.mark_psi_update(); torch.cuda.synchronize(context.device)
        metrics(root, 'G', dict(update=update + 1, events=records, gradient_norm=float(norm),
            meta_gradient_norm=group_norms, seconds=time.monotonic() - tick,
            source_has_gradient=any(p.grad is not None for p in runtime.policy.parameters())), context)
        if update + 1 in (240, 480):
            checkpoint(root, 'G', update + 1, runtime, optimizer, context,
                sampler=dict(seed=20261010, next_update=update + 1, events_per_task=slots,
                             task_order=list(TASKS), task_weight=.25, event_weight=.25))


def train_local(runtime, context, root, examples, *, stop=128, resume=None, baseline=None):
    runtime.freeze_psi(); runtime.actor.train()
    optimizer = torch.optim.AdamW(runtime.actor.parameters(), **OPTIMIZER)
    initialize_deferred_process_group(context, rendezvous_root=Path(root) / 'rendezvous')
    start = resume_checkpoint(resume, 'local', runtime, optimizer, context) if resume else 0
    by_task = {t: [x for x in examples if x['task'] == t] for t in TASKS}
    for update in range(start, stop):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); records = []
        for task in assigned_tasks(context):
            rng = np.random.default_rng(seed(42, task, update))
            conditions = by_task[task]
            example = conditions[int(rng.integers(len(conditions)))]
            prefixes = example['prefixes']
            prefix = prefixes[int(rng.integers(len(prefixes)))]
            path = example['path']
            candidates = prefix['practiced']
            decision=prefix.setdefault('decision',dict(kind='final',candidates=candidates,
                records=prefix['records'],episodes=prefix['episodes'],budget=prefix['budget'],parent=None))
            features = runtime.teaching(task, path.demo, video_task=path.video_task)
            logits = decision_logits(runtime, path, decision, features)
            probabilities = logits.softmax(-1)
            rewards = logits.new_tensor([prefix['reward'][name] for name in candidates])
            loss = -(probabilities * rewards).sum() if len(candidates) > 1 else logits.sum() * 0
            (loss / 4).backward()
            records.append(dict(task=task, path=path.identity, records=prefix['records'], candidates=candidates,
                                fixed_query_reward=rewards.detach().cpu().tolist(), loss=float(loss.detach())))
        synchronize_gradients(runtime.actor, context)
        norm = torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize(context.device)
        metrics(root, 'local', dict(update=update + 1, conditions=records, gradient_norm=float(norm),
                                  seconds=time.monotonic() - tick), context)
    checkpoint(root, 'local', stop, runtime, optimizer, context,
        sampler=dict(next_update=stop, examples=[x['path'].identity for x in examples], prefix_uniform=True,
                     condition_weight=.25), baseline=baseline)


def on_policy_update(runtime, context, root, update, paths, query_rewards, control_rewards,
                     optimizer, baseline, baseline_optimizer):
    """Recompute all θ encoding at frozen batch θ, then one update and discard paths."""
    runtime.freeze_psi(); runtime.actor.train(); baseline.eval()
    optimizer.zero_grad(set_to_none=True); baseline_optimizer.zero_grad(set_to_none=True)
    baseline_losses, records = [], []
    for path, reward, control in zip(paths, query_rewards, control_rewards, strict=True):
        features = runtime.teaching(path.task, path.demo, video_task=path.video_task)
        total_logp, actor_loss, decisions = 0., 0., []
        for decision in path.decisions:
            if decision['uniform_behavior']:
                raise ValueError('supervised uniform paths are never on-policy actor samples')
            logits = decision_logits(runtime, path, decision, features)
            logp = score_log_probability(logits, decision['selected_index'], forced=decision['forced'])
            # Stored b is from ω before the batch; current return never fits current b.
            advantage = reward - control - decision['baseline_before_batch']
            (-(advantage * logp) / 4).backward()
            total_logp += float(logp.detach()); actor_loss += -advantage * float(logp.detach())
            history = history_prefix(path.history, decision['records'], decision['episodes'])
            prediction = baseline(features, history.model_rows(runtime), decision['budget'])
            baseline_losses.append((prediction - (reward - control)).square() / 4)
            decisions.append(dict(kind=decision['kind'], forced=decision['forced'], score=float(logp.detach()),
                                  advantage=advantage, behavior_score=decision['log_probability']))
        records.append(dict(condition=path.identity, reward=reward, MT_control=control,
            condition_log_probability_SUM=total_logp, loss=actor_loss, decisions=decisions))
    synchronize_gradients(runtime.actor, context)
    norm = torch.nn.utils.clip_grad_norm_(runtime.actor.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    # The independently parameterized next baseline is fitted only after actor update.
    if baseline_losses:
        torch.stack(baseline_losses).sum().backward()
    synchronize_gradients(baseline, context)
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1., error_if_nonfinite=True)
    baseline_optimizer.step()
    metrics(root, 'RL', dict(update=update, conditions=records, actor_gradient_norm=float(norm),
        score_reduction='SUM all decisions within condition; MEAN four conditions',
        psi_frozen=not any(p.requires_grad for p in runtime.generator.parameters()),
        one_actor_update=True, resample_required=True), context)
