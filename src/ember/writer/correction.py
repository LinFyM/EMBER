"""Bare frozen source coordinates and training-only local correction cotangents.

The deployment reader has only legal pre-Gemma inputs and the public probe.
The supervisor separately owns privileged action labels and sealed teacher eta.
"""
from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import torch

from ember.ecp.policy_effects import prepare_prefix_features_and_cache
from ember.pi05_source_checkpoint import read_json
from ember.writer.native import FrozenInputChunk


FIELD_UNIT = 5.823084826577233e-6
FIELD_ETA_MANIFEST = "runs/analysis/native_correction_writer_20260913/correction_labels/registration.json"


def validate_field_config(config, *, model_unit):
    expected = {"positions_per_condition": 4, "future_horizon": 15, "seed": 20260914,
                "coefficient": .1, "unit": FIELD_UNIT, "eta_manifest": FIELD_ETA_MANIFEST}
    if config != expected or model_unit != FIELD_UNIT:
        raise ValueError("registered local correction field supervision changed")


def future_velocity_losses(prediction, noise, actions, counts):
    """Each frame averages only its available future control steps and seven axes."""
    if (actions.ndim != 3 or actions.shape[-1] != 7 or counts.shape != actions.shape[:1]
            or prediction.shape != noise.shape or prediction.shape[0] != len(actions)
            or prediction.shape[1] < actions.shape[1] or prediction.shape[-1] < 7
            or bool((counts < 0).any()) or bool((counts > actions.shape[1]).any())):
        raise ValueError("local future-action target shape or valid-step count changed")
    horizon = actions.shape[1]
    valid = torch.arange(horizon, device=prediction.device)[None] < counts[:, None]
    error = prediction[:, :horizon, :7].float() - (noise[:, :horizon, :7] - actions)
    return (error.square() * valid[..., None]).sum((1, 2)) / (counts.clamp_min(1) * 7)


def local_field_loss(prediction, target, *, unit):
    """One physical-coordinate mean across all frames, horizons and 38 outputs."""
    if not prediction or set(prediction) != set(target) or not math.isfinite(unit) or unit <= 0:
        raise ValueError("local field loss needs matching complete physical fields and a fixed unit")
    numerator, coordinates = 0., 0
    leading = next(iter(prediction.values())).shape[:2]
    for name, value in prediction.items():
        truth = target[name]
        if value.ndim != 3 or value.shape[:2] != leading or value.shape != truth.shape:
            raise ValueError("local field lost a sampled position, horizon or native output coordinate")
        numerator = numerator + (value.float() - truth.to(value.device, dtype=torch.float32)).square().sum()
        coordinates += value.numel()
    if coordinates == 0:
        raise ValueError("local field needs at least one real position")
    return numerator / (coordinates * unit ** 2)


class NativeCorrectionReader:
    """One bare state-free source owner, outside both teacher Meta scopes."""

    def __init__(self, policy, contract, probe):
        self.policy, self.contract, self.probe = policy, contract, probe
        self.modules = tuple(getattr(policy.get_submodule(target.name), "base_layer",
                                     policy.get_submodule(target.name)) for target in contract.targets)
        if any(parameter.requires_grad for parameter in policy.parameters()) or probe.shape != (50, 32):
            raise ValueError("native correction requires a frozen source and its complete public probe")

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
                    raise ValueError("native source lost a frame, horizon or actual linear coordinate")
                captured[target.name] = (arguments[0], output)
            return receive

        with torch.set_grad_enabled(with_grad), torch.autocast(device.type, enabled=False), ExitStack() as stack:
            _, cache = prepare_prefix_features_and_cache(self.policy, prefix, native_precision=True)
            for target, module in zip(self.contract.targets, self.modules, strict=True):
                stack.callback(module.register_forward_hook(hook(target)).remove)
            # Public noise alone connects every Action Expert output to autograd;
            # source weights never become leaves or receive parameter gradients.
            noise = self.probe.expand(len(prefix.padding), -1, -1).detach().requires_grad_(with_grad)
            clock = torch.ones(len(prefix.padding), device=device)
            velocity = self.policy.model.denoise_step(prefix.padding, cache, noise, clock)
        if len(captured) != len(self.modules) or not torch.isfinite(velocity).all():
            raise RuntimeError("bare source read is incomplete or nonfinite")
        return velocity, noise, captured

    @torch.no_grad()
    def read(self, condition):
        if len(condition.videos) != 1:
            raise ValueError("local correction compilation currently requires one complete video")
        chunks = []
        for chunk in condition.videos[0]:
            _, _, captured = self._forward(chunk, with_grad=False)
            copies, values = {}, {}
            for name, (value, _) in captured.items():
                # Shared q/v inputs retain one CPU FP32 copy.
                if id(value) not in copies:
                    copies[id(value)] = value.detach().float().cpu()
                values[name] = copies[id(value)]
            chunks.append(values)
        if not chunks:
            raise ValueError("bare source input read needs actual video frames")
        joined, shared = {}, {}
        for target in self.contract.targets:
            identity = tuple(id(chunk[target.name]) for chunk in chunks)
            if identity not in shared:
                shared[identity] = torch.cat([chunk[target.name] for chunk in chunks])
            joined[target.name] = shared[identity]
        if any(value.shape[0] != len(condition.frame_indices[0]) for value in joined.values()):
            raise ValueError("bare native inputs omitted a teacher frame")
        return (joined,)

    def fields(self, condition, ordinals, actions, counts, *, eta):
        """Training-only output cotangents; labels never become condition inputs."""
        if (len(condition.videos) != 1 or ordinals.ndim != 1 or len(ordinals) != len(actions)
                or not len(ordinals) or not bool((ordinals[1:] > ordinals[:-1]).all())
                or int(ordinals[0]) < 0 or int(ordinals[-1]) >= len(condition.frame_indices[0])
                or not math.isfinite(eta) or eta < 0):
            raise ValueError("local field labels need unique sorted real positions and sealed nonnegative eta")
        collected, cursor = {target.name: [] for target in self.contract.targets}, 0
        for chunk in condition.videos[0]:
            stop = cursor + len(chunk.padding)
            selected = torch.nonzero((ordinals >= cursor) & (ordinals < stop), as_tuple=False).flatten()
            if len(selected):
                local = (ordinals[selected] - cursor).cpu()
                part = FrozenInputChunk(*(value[local] for value in (
                    chunk.embeddings, chunk.padding, chunk.evidence_mask, chunk.task_mask)))
                with torch.enable_grad(), torch.autocast(self.probe.device.type, enabled=False):
                    velocity, noise, captured = self._forward(part, with_grad=True)
                    loss = future_velocity_losses(velocity, noise, actions[selected], counts[selected]).sum()
                    outputs = tuple(captured[target.name][1] for target in self.contract.targets)
                    gradients = torch.autograd.grad(loss, outputs)
                for target, gradient in zip(self.contract.targets, gradients, strict=True):
                    value = gradient.detach().float().mul(-eta)
                    if not torch.isfinite(value).all():
                        raise RuntimeError("bare source produced a nonfinite local correction target")
                    collected[target.name].append(value.cpu())
            cursor = stop
        fields = {name: torch.cat(values) for name, values in collected.items()}
        if any(len(value) != len(ordinals) for value in fields.values()):
            raise ValueError("local correction target omitted a sampled frame")
        return fields


class LocalFieldSupervisor:
    """Training-only authority for sealed eta and online future-action cotangents."""

    def __init__(self, reader, data, processor, config):
        self.reader, self.data, self.processor, self.config = reader, data, processor, config
        path = data.asset_root / config["eta_manifest"]
        registration, seal = read_json(path), read_json(path.parent / "completion.json")
        expected = {"schema": "native_source_correction_labels_v1", "rank": 16, "target_count": 38,
                    "state_contract": "state_free", "probe_seed": 1729, "flow_time": 1.,
                    "source_frame_stride": 5, "source_operator_commit": "f39d594f",
                    "privileged_training_labels_only": True}
        pairs = {(task, demo) for task in data.tasks for demo in data.video_pool}
        if (any(registration.get(key) != value for key, value in expected.items())
                or set(registration["tasks"]) != set(data.tasks)
                or tuple(registration["demos"]) != data.video_pool
                or seal.get("status") != "complete" or seal.get("episodes") != len(pairs)
                or len(registration["entries"]) != len(pairs)):
            raise ValueError("local field eta requires the complete sealed train24 teacher pool")
        self.etas = {}
        for entry in registration["entries"]:
            task, demo = int(entry["task"]), int(entry["demo"])
            metadata = Path(entry["adapter"]).with_suffix(".json")
            record = read_json(metadata)
            eta = float(record["fit"]["eta"])
            if ((task, demo) not in pairs or (task, demo) in self.etas
                    or record.get("task") != task or record.get("teacher_demo") != demo
                    or record.get("suite") != data.tasks[task].suite or record.get("arm") != "state_free"
                    or not math.isfinite(eta) or eta < 0):
                raise ValueError("sealed teacher eta identity or finite-value contract changed")
            # The historical query scores and adapter factors have no consumer.
            self.etas[task, demo] = (eta, str(metadata))

    def targets(self, condition, task, demo, *, query_seed):
        ordinals, raw, counts, trace = self.data.local_field_batch(
            task, demo, condition.frame_indices[0], query_seed=query_seed,
            positions_per_condition=self.config["positions_per_condition"],
            future_horizon=self.config["future_horizon"], seed=self.config["seed"],
        )
        eta, metadata = self.etas[task, demo]
        actions = self.processor.normalize_action(raw)
        fields = self.reader.fields(condition, ordinals, actions, counts.to(actions.device), eta=eta)
        return ordinals, fields, {**trace, "field_eta": eta, "field_eta_source": metadata}
