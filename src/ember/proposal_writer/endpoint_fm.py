"""Native function credit for the existing complete endpoint gradient owner."""
from __future__ import annotations

import time

import torch
from torch.func import functional_call

from ember.writer.function_credit import FlowSample, NativeFlowPrediction, mean_velocity_loss
from ember.writer.runtime import autocast

from .teacher import QueryStream, tree_slice
from .runtime import phase


class NativeEndpointCredit:
    """Accumulate all76 factor cotangents, then apply one VJP to the same F."""
    def __init__(self, runtime, specification, microbatch):
        expected = dict(event_ordinal=20261013, query_demos=list(range(46)), logical_queries=28,
                        action_chunk_size=50, action_start_offset=1, action_width=7)
        if any(specification.get(key) != value for key, value in expected.items()):
            raise ValueError('native FM needs the registered28 cross-episode queries and full50x7 target')
        if microbatch < 1 or microbatch > 28:
            raise ValueError('physical native query microbatch must fit logical28')
        self.runtime, self.microbatch, self.streams = runtime, microbatch, {}
        self.prediction = NativeFlowPrediction(runtime.policy)
        self.specification = specification

    def __call__(self, task, demo, visit, generated):
        runtime = self.runtime
        key = task, demo
        # The caller has already generated lambda from legal P/m. Query labels
        # are loaded only here and never returned to the Generator condition.
        if key not in self.streams:
            self.streams[key] = QueryStream(runtime, task, demo, self.specification['event_ordinal'],
                logical_queries=28, query_demo_indices=self.specification['query_demos'])
        with phase(runtime, 'native_query_load'):
            sample, event = self.streams[key].sample(runtime, visit)
        if (sample.action_width != 7 or tuple(sample.target.shape) != (28, 50, 32)
                or any(row['demo'] >= 46 for row in event['queries']) or len(generated) != 76):
            raise ValueError('native FM lost full LoRA, query pool, action padding or logical28')
        keys = tuple(generated)
        leaves = tuple(generated[key].detach().requires_grad_(True) for key in keys)
        # PolicyContexts supplies both the actual merged MT weights and the
        # complete task factors. A factors-only functional call omits that base.
        parameters = runtime.task_parameters(dict(zip(keys, leaves, strict=True)))
        cotangents = [torch.zeros_like(value) for value in leaves]
        torch.cuda.reset_peak_memory_stats(runtime.device)
        native_tick = time.monotonic(); total_loss = 0.
        for first in range(0, 28, self.microbatch):
            last = min(28, first + self.microbatch)
            part = FlowSample(tree_slice(sample.arguments, first, last), sample.target[first:last], 7)
            with phase(runtime, 'native_FM_forward'), autocast(runtime.device):
                prediction = functional_call(self.prediction, parameters, (part,), strict=False)
                loss = mean_velocity_loss(prediction, part.target, 7)
            if not torch.isfinite(loss):
                raise ValueError('non-finite native FM loss')
            total_loss += float(loss.detach()) * (last-first)/28
            with phase(runtime, 'native_factor_cotangents'):
                gradients = torch.autograd.grad(loss * ((last-first)/28/4), leaves)
                for accumulator, gradient in zip(cotangents, gradients, strict=True):
                    accumulator.add_(gradient)
            del part, prediction, loss, gradients
        native_peak = torch.cuda.max_memory_allocated(runtime.device)/2**30
        with phase(runtime, 'endpoint_VJP'):
            torch.autograd.backward(tuple(generated[key] for key in keys), tuple(cotangents))
        return dict(native_FM_loss=total_loss, query_event=event, task_visit=visit,
                    logical_queries=28, denominator=4, microbatch=self.microbatch, factor_leaves=len(leaves),
                    native_and_retained_F_peak_GiB=native_peak, native_seconds=time.monotonic()-native_tick,
                    query_excluded_from_condition_and_H=True, merged_MT_base_supplied=True)

    def close(self):
        for stream in self.streams.values():
            stream.close()


def profile_physical_batches(trainer, runtime, context, root, start, resume, physical_batches):
    """Two discarded real initial batches; complete source/RNG restore after each."""
    from pathlib import Path
    import torch.distributed as dist
    from ember.pi05_source_checkpoint import write_json_atomic
    from .shared_checkpoint import initialize_native
    from .learning import memory_peak
    if resume or tuple(physical_batches)!=(14,28) or start!=38349:
        raise ValueError('only two discarded actual initial batches14/28 are registered')
    results=[]
    for physical in physical_batches:
        initialize_native(runtime,context,root)
        trainer.optimizer.load_state_dict(runtime.generator_optimizer_state)
        trainer.objective.microbatch=physical;torch.cuda.reset_peak_memory_stats(context.device)
        record=trainer.step(start,stage=f'G_profile{physical}',profile=True)
        record['microbatch']=physical;record['process_GPU_peak']=memory_peak(context.device)
        records=[record]
        if context.world_size>1:
            records=[None]*context.world_size;dist.all_gather_object(records,record)
        results.append(dict(microbatch=physical,actual_rank_records=records,
            maximum_rank_seconds=max(r['seconds'] for r in records),
            failed=any(r.get('profile_failed') for r in records)))
    valid=[r for r in results if not r['failed']]
    if not valid:raise ValueError('both registered physical native FM microbatches failed')
    chosen=min(valid,key=lambda r:r['maximum_rank_seconds'])['microbatch']
    initialize_native(runtime,context,root);trainer.optimizer.load_state_dict(runtime.generator_optimizer_state)
    if context.is_main:write_json_atomic(Path(root)/'profile/native_FM_microbatch.json',dict(
        complete=True,discarded=True,results=results,selected_microbatch=chosen,
        source_cursor_restored=runtime.version,rank_RNG_restored=True,new_environment_episodes=0,
        choice_basis='measured maximum-rank complete B4 update wall with actual memory/finite constraints'))
    trainer.objective.close()
