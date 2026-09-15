"""Pure cross-episode FM with complete-LoRA credit and joint Writer replay."""
from __future__ import annotations

import time
import torch

from ember.writer.function_credit import paired_functional_credit
from ember.writer.runtime import autocast


class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device, self.step = context.device, 0

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def _credit(self, state, batch, trace, offset, *, backward):
        with autocast(self.device):
            return paired_functional_credit(
                self.runtime.policy, state, self.runtime.lora, batch,
                seed=trace["policy_rng_seed"], device=self.device,
                random_batch=trace.get("policy_random_batch_size", len(trace["action_demos"])), offset=offset,
                microbatch=min(int(self.config["runtime"]["policy_microbatch"]), len(trace["action_demos"])),
                condition_weight=.25, backward=backward,
            )

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        hits, misses = self.cache.hits, self.cache.misses
        condition = self.cache.condition(task, demos)
        start = self._time(timings, "input_seconds", start)
        with torch.no_grad():
            state = runtime.compile(condition)
        start = self._time(timings, "writer_forward_seconds", start)
        raw, trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
            query_offset=draw["query_offset"], query_count=draw["query_count"],
        )
        batch = runtime.processor.training_batch(raw)
        start = self._time(timings, "query_preparation_seconds", start)
        credit = self._credit(state, batch, trace, draw["query_offset"], backward=True)
        del batch, raw, state
        start = self._time(timings, "fm_vjp_seconds", start)
        cotangent = credit.pop("lora_cotangent")
        fm_norm = float(torch.stack([value.norm() for value in cotangent.values()]).norm())
        generated = runtime.compile(condition)
        torch.autograd.backward(tuple(generated.values()),
                                tuple(cotangent[name].to(value) for name, value in generated.items()))
        self._time(timings, "writer_vjp_seconds", start)
        return {**credit, "task_weight": .25, "condition_weight": .25, "normalizer": 1.,
                "fm_lora_gradient_norm": fm_norm, "queries": len(trace["action_demos"]),
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
