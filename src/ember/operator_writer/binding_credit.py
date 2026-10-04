"""Same-version complete-LoRA FM/own-Q cotangent and canonical target replay."""
from __future__ import annotations

import time
import numpy as np
import torch
from torch.utils.data import default_collate

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample, _add
from ember.writer.runtime import autocast
from .role_binding import GROUPS, RoleObserver, compile_fixed, target


def raw_batch(data, task, queries):
    return default_collate([data.queries[data.rows[task][q['demo']][q['frame']]] for q in queries])


def samples(runtime, batch, queries):
    """Use the original FM noise/time consumer with each registered query seed."""
    rows = []
    for i, q in enumerate(queries):
        one = {k: v[i:i + 1] if isinstance(v, torch.Tensor) and v.ndim and len(v) == len(queries)
               else v for k, v in batch.items()}
        rows.append(flow_sample(runtime.policy, one, seed=q['flow_seed'], device=runtime.device,
                                random_batch=1, offset=0))
    args = tuple([torch.cat([r.arguments[j][k] for r in rows]) for k in range(3)]
                 if j in (0, 1) else torch.cat([r.arguments[j] for r in rows]) for j in range(7))
    return FlowSample(args, torch.cat([r.target for r in rows]), 7)


class RoleFlow(NativeFlowPrediction):
    def forward(self, sample, prepared, q=None):
        if q is None:
            return super().forward(sample, prepared), sample.target.new_zeros(())
        observer = RoleObserver(self.policy, q)
        with observer.capture():
            velocity = super().forward(sample, prepared)
        return velocity, observer.loss()


def paired_credit(runtime, states, batch, queries, micro, q=None):
    owner = RoleFlow(runtime.policy)
    credits = [dict(FM=0., own_Q=0., lora_cotangent={}) for _ in states]
    for start in range(0, 28, micro):
        stop = min(start + micro, 28)
        sliced = {k: v[start:stop] if isinstance(v, torch.Tensor) and v.ndim and len(v) == 28
                  else v for k, v in batch.items()}
        sample = samples(runtime, sliced, queries[start:stop])
        prepared = owner.prepare(sample)
        if not torch.all(sample.arguments[1][0]) or not torch.all(sample.arguments[1][1]) or torch.any(sample.arguments[1][2]):
            raise ValueError('official two visible image blocks / masked third camera changed')
        leaves = {k: torch.stack([s[k] for s in states]).detach().requires_grad_(True) for k in states[0]}
        def forward(values, actual_sample, actual_prepared, role_q):
            return torch.func.functional_call(owner, {'policy.' + k: v for k, v in values.items()},
                                               (actual_sample, actual_prepared, role_q), strict=False)
        velocity, _ = torch.vmap(lambda values: forward(values, sample, prepared, None))(leaves)
        loss = (velocity[..., :7].float() - sample.target[None, ..., :7].float()).square().mean((1, 2, 3))
        role = torch.zeros_like(loss)
        if q is not None:
            # Same real observation/prefix and original epsilon, no target x_tau
            # in the role query and no extra Gaussian/time RNG consumption.
            pad, cache, _, tau = prepared
            endpoint = (pad, cache, sample.arguments[5], torch.ones_like(tau))
            _, role = torch.vmap(lambda values: forward(values, sample, endpoint, q[start:stop]))(leaves)
        gradients = torch.autograd.grad((loss + .1 * role).sum(), tuple(leaves.values()))
        weight = (stop - start) / 28
        for index, credit in enumerate(credits):
            _add(credit['lora_cotangent'], {k: g[index] for k, g in zip(leaves, gradients, strict=True)}, weight / 8)
            credit['FM'] += float(loss[index].detach()) * weight
            credit['own_Q'] += float(role[index].detach()) * weight
    return credits


def labels_for(records, task, queries, device):
    values = []
    for q in queries:
        item = records[task, q['demo']]
        i = int(np.searchsorted(item['frames'], q['frame']))
        if i == len(item['frames']) or item['frames'][i] != q['frame']:
            raise ValueError('query role label is not on the registered observation')
        values.append(item['q'][i])
    return torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)


def step(runtime, cached, data, labels, entry, arm, binding, optimizer, parameters, micro):
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for task in entry['tasks']:
        event = entry['events'][str(task)]
        queries = event['queries']
        batch = runtime.processor.training_batch(raw_batch(data, task, queries))
        q = labels_for(labels, task, queries, runtime.device) if arm != 'F' else None
        states = [compile_fixed(runtime, cached[task, d], binding)[0] for d in (0, 1)]
        runtime.restore_identity()
        with autocast(runtime.device):
            credits = paired_credit(runtime, states, batch, queries, micro, q)
        del states
        for teacher, credit in enumerate(credits):
            fixed = cached[task, teacher]
            with autocast(runtime.device):
                alpha = binding.alpha(fixed).detach().requires_grad_() if binding is not None else None
                for index, name in enumerate(runtime.writer.names):
                    a, b = target(runtime, index, fixed, binding, alpha)
                    torch.autograd.backward((a, b), (credit['lora_cotangent'][name + LORA_A_SUFFIX].to(a),
                                                   credit['lora_cotangent'][name + LORA_B_SUFFIX].to(b)))
                teacher_loss = fixed['H'].new_zeros(())
                if binding is not None:
                    label = labels[task, teacher]
                    expected = fixed['frame_indices'][:-1].cpu().numpy()
                    if not np.array_equal(label['frames'], expected):
                        raise ValueError('teacher alpha mask must be origin t-1, not arrival t')
                    actual = binding.alpha(fixed)
                    teacher_loss = binding.teacher_loss(actual, torch.as_tensor(label['q'][:, 0], device=runtime.device))
                    torch.autograd.backward((actual, teacher_loss), (alpha.grad, torch.full_like(teacher_loss, .1 / 8)))
            rows.append(dict(task=task, teacher=teacher, FM=credit['FM'], own_Q=credit['own_Q'],
                             teacher_selection=float(teacher_loss.detach()), complete_factor_cotangents=76,
                             condition_weight=1 / 8, teacher_loss_applications=int(binding is not None)))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    groups = {g: float(torch.stack([getattr(u, g).weight.grad.float().norm()
                                   for u in runtime.writer.conditional_targets]).norm()) for g in GROUPS}
    if any(p.grad is not None for p in runtime.writer.parameters() if not p.requires_grad):
        raise ValueError('frozen common/interpreter/native acquired gradients')
    binding_gradients = ({n: float(p.grad.float().norm()) if p.grad is not None else 0.
                          for n, p in binding.named_parameters()} if binding is not None else {})
    optimizer.step(); torch.cuda.synchronize()
    return dict(rows=rows, seconds=time.monotonic() - started, microbatch=micro, suffix_conditions_packed=2,
                physical_suffix_queries=2 * micro, frozen_prefix_shared=True,
                unclipped_gradient_norm=float(norm), gradient_groups=groups, binding_gradients=binding_gradients,
                allocated_peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                reserved_peak_GiB=torch.cuda.max_memory_reserved() / 2**30)
