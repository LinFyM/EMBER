"""Cross-episode FM and video teaching share one complete-LoRA Writer replay."""
from __future__ import annotations

import time
import torch

from ember.writer.function_credit import paired_functional_credit
from ember.writer.learning_data import TASKS_PER_UPDATE
from ember.writer.runtime import autocast


class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config, *, frame_parallel_group=None) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device, self.step = context.device, 0
        self.frame_parallel_group = frame_parallel_group
        self.tasks_per_update = int(config["data"].get("tasks_per_update", TASKS_PER_UPDATE))

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def _credit(self, state, batch, trace, offset, *, backward, condition_weight=1., teaching=False):
        endpoint = teaching and bool(configured_endpoint(self.config))
        with autocast(self.device):
            return paired_functional_credit(
                self.runtime.policy, state, self.runtime.lora, batch,
                seed=trace["policy_rng_seed"], device=self.device,
                random_batch=trace.get("policy_random_batch_size", len(trace["action_demos"])), offset=offset,
                microbatch=min(int(self.config["runtime"]["policy_microbatch"]), len(trace["action_demos"])),
                condition_weight=condition_weight, backward=backward,
                noise_endpoint=endpoint, prefix_steps=5 if endpoint else None,
            )

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        hits, misses = self.cache.hits, self.cache.misses
        condition = self.cache.condition(task, demos)
        weight = 1 / self.tasks_per_update
        start = self._time(timings, "input_seconds", start)
        with torch.no_grad():
            state = runtime.compile(condition, frame_parallel_group=self.frame_parallel_group)
        start = self._time(timings, "writer_forward_seconds", start)
        raw, trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
            query_offset=draw["query_offset"], query_count=draw["query_count"],
        )
        batch = runtime.processor.training_batch(raw)
        start = self._time(timings, "query_preparation_seconds", start)
        credit = self._credit(state, batch, trace, draw["query_offset"], backward=True, condition_weight=weight)
        del batch, raw
        start = self._time(timings, "fm_vjp_seconds", start)
        cotangent = credit.pop("lora_cotangent")
        fm_norm = float(torch.stack([value.norm() for value in cotangent.values()]).norm())
        raw, teaching_trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
            query_offset=draw["teaching_offset"], query_count=draw["teaching_count"], teaching=True,
        )
        batch = runtime.processor.training_batch(raw)
        teaching_weight = weight * float(self.config["optimization"]["teaching_weight"])
        teaching = self._credit(state, batch, teaching_trace, draw["teaching_offset"], backward=True,
                                condition_weight=teaching_weight, teaching=True)
        del batch, raw, state
        teaching_cotangent = teaching.pop("lora_cotangent")
        teaching_norm = float(torch.stack([value.norm() for value in teaching_cotangent.values()]).norm())
        for name in cotangent:
            cotangent[name].add_(teaching_cotangent[name])
        del teaching_cotangent
        start = self._time(timings, "teaching_vjp_seconds", start)
        generated = runtime.compile(condition, frame_parallel_group=self.frame_parallel_group)
        torch.autograd.backward(tuple(generated.values()),
                                tuple(cotangent[name].to(value) for name, value in generated.items()))
        self._time(timings, "writer_vjp_seconds", start)
        return {**credit, "teaching_loss": teaching["flow_loss"], "task_weight": weight,
                "condition_weight": weight, "teaching_weight": teaching_weight, "normalizer": 1.,
                "fm_lora_gradient_norm": fm_norm, "queries": len(trace["action_demos"]),
                "teaching_lora_gradient_norm": teaching_norm,
                "teaching_queries": len(teaching_trace["action_demos"]),
                "teaching_compiled_forward_calls": teaching["compiled_forward_calls"],
                **{"teaching_" + name: value for name, value in teaching_trace.items()},
                **trace, **timings, "input_cache_hits": self.cache.hits - hits,
                "input_cache_misses": self.cache.misses - misses, "input_cache_bytes": self.cache.bytes,
                "policy_microbatch": int(self.config["runtime"]["policy_microbatch"])}

    @torch.no_grad()
    def validate(self, task: int, demo: int, *, seed: int, queries: int) -> dict:
        condition = self.cache.condition(task, (demo,))
        state = self.runtime.compile(condition)
        raw, trace = self.data.diagnostic_batch(task, seed=seed, count=queries, teacher_demo=demo)
        batch = self.runtime.processor.training_batch(raw)
        credit = self._credit(state, batch, trace, 0, backward=False)
        credit.pop("lora_cotangent")
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                **credit, "queries": queries, **trace, "gradients": False}


def configured_endpoint(config):
    """Whether the registered second query group uses tau=1 and a five-action prefix."""
    experiment = config.get("experiment")
    if experiment is None:
        return True
    if experiment.get("kind") != "conditional_compilation_diagnostics_20260923":
        raise ValueError("unrecognized conditional teaching objective")
    return experiment.get("extra_endpoint_prefix") is True
