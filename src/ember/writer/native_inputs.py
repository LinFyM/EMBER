"""Read actual native X from the frozen source; never fit or update at deployment."""
from __future__ import annotations

from contextlib import ExitStack

import torch

from ember.ecp.policy_effects import prepare_prefix_kv_cache


class NativeInputReader:
    """One ordinary state-free source forward per chunk, outside both Meta scopes."""

    def __init__(self, policy, contract, probe):
        self.policy, self.contract, self.probe = policy, contract, probe
        self.modules = tuple(policy.get_submodule(target.name).base_layer for target in contract.targets)
        if any(parameter.requires_grad for parameter in policy.parameters()) or probe.shape != (50, 32):
            raise ValueError("native inputs require a frozen source and its complete public probe")

    @torch.no_grad()
    def read_chunk(self, chunk):
        prefix = chunk.on_device(self.probe.device)
        cache = prepare_prefix_kv_cache(self.policy, prefix, native_precision=True)
        captured, copies = {}, {}

        def hook(index):
            def receive(module, arguments):
                if index in captured:
                    raise RuntimeError("native target ran more than once in one source read")
                value = arguments[0]
                expected = (len(prefix.padding), 50, self.contract.targets[index].in_features)
                if value.shape != expected:
                    raise ValueError("actual native input lost a frame, horizon or source coordinate")
                # q/v share their layer input. Retain the tensor reference while
                # hooks are active so its id cannot be recycled before the copy.
                identity = id(value)
                if identity not in copies:
                    copies[identity] = (value, value.detach().float().cpu())
                captured[index] = copies[identity][1]
            return receive

        with ExitStack() as stack:
            for index, module in enumerate(self.modules):
                handle = module.register_forward_pre_hook(hook(index))
                stack.callback(handle.remove)
            noise = self.probe.expand(len(prefix.padding), -1, -1)
            clock = torch.ones(len(prefix.padding), device=self.probe.device)
            with torch.autocast(self.probe.device.type, enabled=False):
                velocity = self.policy.model.denoise_step(prefix.padding, cache, noise, clock)
        if len(captured) != len(self.modules) or not torch.isfinite(velocity).all():
            raise RuntimeError("source native input read is incomplete or nonfinite")
        return tuple(captured[index] for index in range(len(self.modules)))

    @torch.no_grad()
    def read_video(self, chunks):
        captured = [self.read_chunk(chunk) for chunk in chunks]
        if not captured:
            raise ValueError("native input read requires actual video frames")
        joined, shared = [], {}
        for index in range(len(self.modules)):
            key = tuple(id(chunk[index]) for chunk in captured)
            if key not in shared:
                shared[key] = torch.cat([chunk[index] for chunk in captured])
            joined.append(shared[key])
        return tuple(joined)


def native_input_bytes(videos):
    """Count shared q/v storage once in the existing frozen-input CPU budget."""
    tensors = {id(value): value for video in videos for value in video}
    return sum(value.numel() * value.element_size() for value in tensors.values())
