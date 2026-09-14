"""Legal bare source coordinates, projected pullback and chunked exact q credit.

There are no teacher action labels, eta records or learned decoder parameters.
The source is read outside both observer Meta and execution-adapter scopes.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.ecp.policy_effects import prepare_prefix_features_and_cache
from ember.writer.factor import compile_source_lora, native_input_basis
from ember.writer.native import NativeCondition


@dataclass(frozen=True)
class SourceCoordinates:
    """Frozen CPU predictions/bases and the original legal condition; not a Module."""

    reader: NativeCorrectionReader
    condition: NativeCondition
    predictions: torch.Tensor
    bases: dict[str, torch.Tensor]

    def compile(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        return compile_source_lora(self, q)


class NativeCorrectionReader:
    """Read each source graph in its native precision, with only public noise a leaf."""

    def __init__(self, policy, contract, probe):
        self.policy, self.contract, self.probe = policy, contract, probe
        self.modules = tuple(getattr(policy.get_submodule(target.name), "base_layer",
                                     policy.get_submodule(target.name)) for target in contract.targets)
        if (any(parameter.requires_grad for parameter in policy.parameters()) or probe.shape != (50, 32)
                or contract.alpha != contract.rank or not contract.targets
                or not 0 < contract.rank <= min(target.in_features for target in contract.targets)):
            raise ValueError("source pullback requires a frozen policy, public 50x32 probe and alpha=rank")

    def _forward(self, chunk, *, with_grad):
        device = self.probe.device
        prefix = chunk.on_device(device)
        captured = {}

        def hook(target):
            def receive(module, arguments, output):
                if target.name in captured:
                    raise RuntimeError("native target ran more than once in one bare source read")
                if (arguments[0].shape != (len(prefix.padding), 50, target.in_features)
                        or output.shape != (len(prefix.padding), 50, target.out_features)):
                    raise ValueError("source pullback lost a frame, horizon or actual linear coordinate")
                captured[target.name] = (arguments[0], output)
            return receive

        with torch.set_grad_enabled(with_grad), torch.autocast(device.type, enabled=False), ExitStack() as stack:
            _, cache = prepare_prefix_features_and_cache(self.policy, prefix, native_precision=True)
            for target, module in zip(self.contract.targets, self.modules, strict=True):
                stack.callback(module.register_forward_hook(hook(target)).remove)
            noise = self.probe.expand(len(prefix.padding), -1, -1).detach().requires_grad_(with_grad)
            clock = torch.ones(len(prefix.padding), device=device)
            velocity = self.policy.model.denoise_step(prefix.padding, cache, noise, clock)
        if (len(captured) != len(self.modules) or velocity.shape != noise.shape
                or not torch.isfinite(velocity).all()):
            raise RuntimeError("bare source read is incomplete or nonfinite")
        return velocity, captured

    @torch.no_grad()
    def read(self, condition: NativeCondition) -> SourceCoordinates:
        if len(condition.videos) != 1 or not condition.videos[0]:
            raise ValueError("source pullback currently requires one complete real video")
        owners, grams, predictions = {}, {}, []
        for chunk in condition.videos[0]:
            velocity, captured = self._forward(chunk, with_grad=False)
            predictions.append((self.probe[None, :, :7] - velocity[:, :, :7]).float().cpu())
            if not owners:
                shared = {}
                for target in self.contract.targets:
                    identity = id(captured[target.name][0])
                    owners[target.name] = shared.setdefault(identity, target.name)
            for target, owner in owners.items():
                if captured[target][0] is not captured[owner][0]:
                    raise ValueError("shared native input identity changed between frame chunks")
                if owner != target:
                    continue
                value = captured[target][0].detach().float().flatten(0, 1)
                with torch.autocast(value.device.type, enabled=False):
                    if owner not in grams:
                        grams[owner] = value.T @ value
                    else:
                        grams[owner].addmm_(value.T, value)
        predictions = torch.cat(predictions)
        if len(predictions) != len(condition.frame_indices[0]):
            raise ValueError("bare source coordinates omitted a real video frame")
        basis = {owner: native_input_basis(gram, self.contract.rank) for owner, gram in grams.items()}
        return SourceCoordinates(self, condition, predictions, {name: basis[owner] for name, owner in owners.items()})

    @torch.no_grad()
    def pullback(self, coordinates, q):
        """B_l = sum_(frame,horizon) cotangent(y_l)^T (X_l A_l^T) / T."""
        device, total = self.probe.device, len(coordinates.predictions)
        values = [torch.zeros(target.out_features, self.contract.rank, device=device)
                  for target in self.contract.targets]
        cursor = 0
        for chunk in coordinates.condition.videos[0]:
            stop = cursor + len(chunk.padding)
            with torch.enable_grad(), torch.autocast(device.type, enabled=False):
                velocity, captured = self._forward(chunk, with_grad=True)
                outputs = tuple(captured[target.name][1] for target in self.contract.targets)
                cotangent = F.pad(q[cursor:stop].detach().to(device, dtype=torch.float32), (0, 25))
                gradients = torch.autograd.grad(velocity, outputs, grad_outputs=cotangent)
            for target, gradient, value in zip(self.contract.targets, gradients, values, strict=True):
                with torch.autocast(device.type, enabled=False):
                    x = captured[target.name][0].detach().float().flatten(0, 1)
                    address = x @ coordinates.bases[target.name].to(device).T
                    value.addmm_(gradient.float().flatten(0, 1).T, address)
            cursor = stop
            del velocity, captured, outputs, gradients
        for value in values:
            value.div_(total)
            if not torch.isfinite(value).all():
                raise RuntimeError("fixed source pullback produced nonfinite factors")
        return tuple(values)

    @torch.no_grad()
    def adjoint(self, coordinates, gradients):
        """Exact transpose of pullback, using native eager attention's supported AD.

        Differentiate J_y F^T u with respect to dummy u. X A^T and incoming
        factor cotangents are fixed directions; no source-weight derivative
        or finite-difference approximation is constructed.
        """
        device, total = self.probe.device, len(coordinates.predictions)
        incoming = tuple(None if value is None else value.detach().to(device, dtype=torch.float32)
                         for value in gradients)
        result = []
        for chunk in coordinates.condition.videos[0]:
            with torch.enable_grad(), torch.autocast(device.type, enabled=False):
                velocity, captured = self._forward(chunk, with_grad=True)
                outputs = tuple(captured[target.name][1] for target in self.contract.targets)
                dummy = torch.zeros_like(velocity[:, :, :7], requires_grad=True)
                cotangents = torch.autograd.grad(velocity, outputs, F.pad(dummy, (0, 25)), create_graph=True)
                directions = []
                with torch.no_grad():
                    for target, output, gradient in zip(self.contract.targets, outputs, incoming, strict=True):
                        if gradient is None:
                            direction = torch.zeros_like(output)
                        else:
                            x = captured[target.name][0].detach().float()
                            address = x @ coordinates.bases[target.name].to(device).T
                            direction = ((address @ gradient.T) / total).to(output.dtype)
                        directions.append(direction)
                gradient, = torch.autograd.grad(cotangents, dummy, grad_outputs=directions)
            if not torch.isfinite(gradient).all():
                raise RuntimeError("fixed source pullback produced nonfinite q credit")
            result.append(gradient.detach())
            # The first VJP retained its source graph to form the adjoint.
            # Release every graph reference before the next real frame chunk.
            del velocity, captured, outputs, output, dummy, cotangents, directions
        return torch.cat(result)
