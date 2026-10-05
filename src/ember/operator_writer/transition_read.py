"""Owned, temporary frozen-T linear/online transition-reading diagnostic."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state

STUDY = "query_conditioned_transition_read_20261005"
ROOT = Path("/data1/user/ymdai/ember_runs/query_conditioned_transition_read_20261005")
PARENT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340")


@dataclass
class FrozenMemory:
    state: dict
    keys: dict
    phi: dict
    frame_indices: torch.Tensor
    raw_frames: int
    sampled_frames: int
    reader: tuple | None = None

    def to(self, device):
        def move(values):
            return {name: value.to(device) for name, value in values.items()}
        encoded = (tuple([value.to(device) for value in group] for group in self.reader)
                   if self.reader is not None else None)
        return FrozenMemory(move(self.state), move(self.keys), move(self.phi),
                            self.frame_indices.to(device), self.raw_frames,
                            self.sampled_frames, encoded)


@torch.no_grad()
def read_frozen_memory(runtime, data, task, demo, *, frame_chunk=8):
    """One original T read; only original action-hidden video supplies content."""
    if any(p.requires_grad for p in runtime.writer.parameters()):
        raise ValueError("transition diagnostic father must be frozen")
    condition, raw, sampled = data.condition(runtime, task, demo)
    state, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
    keys, phi = {}, {}
    with torch.autocast(runtime.device.type, enabled=False):
        h = native["h"].float()
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
        for name, write in zip(runtime.writer.names, runtime.writer.writes, strict=True):
            x = native["x"][name].float()
            key = F.normalize(F.linear(x[:-1], state[name + LORA_A_SUFFIX].float()),
                              dim=-1, eps=1e-6)
            value = F.gelu(write.p(key) + write.c(h[:-1])) * write.d(h[1:] - h[:-1])
            if key.shape[1:] != (50, 128) or value.shape[1:] != (50, 256):
                raise ValueError("original transition slots or widths changed")
            keys[name], phi[name] = key, value
    return FrozenMemory({name: v.detach() for name, v in state.items()}, keys, phi,
                        condition[1].detach(), raw, sampled)


class ReadTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.j = nn.Parameter(torch.zeros(128, 256))
        self.q = nn.Parameter(torch.eye(128))
        self.e = nn.Parameter(torch.eye(128))

    def contents(self, key, phi):
        return F.normalize(F.linear(key.float(), self.e), dim=-1, eps=1e-6), F.linear(phi.float(), self.j)


class TransitionReadout(nn.Module):
    """Checkpoint owner contains only the 2,490,368 newly learned parameters."""
    def __init__(self, lora):
        super().__init__()
        if lora.rank != 128 or len(lora.targets) != 38 or lora.alpha != 128:
            raise ValueError("frozen readout requires complete scale-one rank128")
        self.lora = lora
        self.names = tuple(target.name for target in lora.targets)
        self.targets = nn.ModuleList(ReadTarget() for _ in self.names)
        if sum(p.numel() for p in self.parameters()) != 2490368:
            raise ValueError("new parameter count changed")

    def reader_memory(self, memory):
        e, u = [], []
        with torch.autocast(next(self.parameters()).device.type, enabled=False):
            for name, target in zip(self.names, self.targets, strict=True):
                key, value = target.contents(memory.keys[name], memory.phi[name])
                e.append(key)
                u.append(value)
        return e, u

    def linear(self, memory):
        """FP32 original delta memory, then one actual complete LoRA merge."""
        result = dict(memory.state)
        with torch.autocast(next(self.parameters()).device.type, enabled=False):
            e, u = self.reader_memory(memory)
            for name, target, keys, values in zip(self.names, self.targets, e, u, strict=True):
                r = torch.zeros(128, 128, device=keys.device)
                for key, value in zip(keys, values, strict=True):
                    r = r + (value.T - r @ key.T) @ key / 50
                parent = memory.state[name + LORA_B_SUFFIX].float()
                # B(I+RQ) = B+B(RQ); avoid gratuitously rounding the frozen B through B@I.
                result[name + LORA_B_SUFFIX] = parent + parent @ (r @ target.q)
        validate_lora_state(result, self.lora)
        return result


def _summary_packet(delta, parent, alpha=None, transitions=None):
    # One vector per sample; no per-query or full attention is retained.
    flat = delta.reshape(delta.shape[0], -1).detach()
    base = parent.reshape(parent.shape[0], -1).detach()
    count = torch.full_like(flat[:, 0], flat.shape[1])
    parts = [flat.square().sum(1), base.square().sum(1), count]
    if alpha is not None:
        mass = alpha.detach().mean(1).reshape(len(alpha), transitions, 50)
        frame, slot = mass.sum(2), mass.sum(1)
        frame_axis = torch.arange(transitions, device=mass.device)
        parts += [1 / frame.square().sum(1), 1 / slot.square().sum(1),
                  (frame * frame_axis).sum(1), frame.argmax(1).float(),
                  torch.isfinite(alpha).flatten(1).all(1).float()]
    return torch.stack(parts, dim=1)


def summarize_stats(stats):
    rows = []
    for key, record in sorted(stats.get("records", {}).items()):
        name, flow = key.rsplit("|", 1)
        values = record["sum"]
        d, p, count = values[:3]
        row = {"target": name, "flow": int(flow), "calls": record["calls"],
               "positions": record["positions"], "delta_rms": (d / max(count, 1)) ** .5,
               "parent_lora_rms": (p / max(count, 1)) ** .5,
               "relative_rms": (d / max(p, 1e-30)) ** .5,
               "shape": record["shape"]}
        if len(values) == 8:
            row.update(mean_effective_frames=values[3] / record["calls"],
                       mean_effective_slots=values[4] / record["calls"],
                       mean_frame_index=values[5] / record["calls"],
                       mean_top_frame_index=values[6] / record["calls"],
                       all_finite=values[7] == record["calls"])
        rows.append(row)
    return {"schema_version": "ember_transition_read_effect_v1", "rows": rows,
            "upstream_requires_grad_calls": stats.get("upstream_requires_grad_calls", 0),
            "upstream_gradient_squared": stats.get("upstream_gradient_squared", {})}


class _EffectHooks:
    def __init__(self, policy, lora):
        self.names = tuple(target.name for target in lora.targets)
        self.active = None
        self.handles = [policy.get_submodule(name).register_forward_hook(self._hook(index))
                        for index, name in enumerate(self.names)]

    def _collect(self, index, delta, parent, value, alpha=None, transitions=None):
        stats = self.active["stats"]
        if stats is None:
            return
        flow = self.active["calls"][index]
        self.active["calls"][index] += 1
        packet = _summary_packet(delta, parent, alpha, transitions)
        if len(stats) == 1 and len(packet) != 1:
            packet = torch.cat((packet[:, :3].sum(0), packet[:, 3:].mean(0))).unsqueeze(0)
        self.active["packets"].append((index, flow, packet, tuple(value.shape[1:]), value.requires_grad))
        if value.requires_grad:
            destination = self.active["grad_packets"]
            def observe_credit(gradient):
                destination.append((index, gradient.detach().float().square().sum()))
            value.register_hook(observe_credit)

    def _flush(self):
        packets = self.active["packets"]
        if not packets:
            return
        # One device-to-host synchronization per activation rather than per hook.
        stacked = torch.stack([p[2] for p in packets]).cpu().tolist()
        for (index, flow, _, shape, credit), values in zip(packets, stacked, strict=True):
            for stats, vector in zip(self.active["stats"], values, strict=True):
                key = self.names[index] + "|" + str(flow)
                record = stats.setdefault("records", {}).setdefault(
                    key, {"sum": [0.] * len(vector), "calls": 0, "positions": 0, "shape": list(shape)})
                record["sum"] = [a + b for a, b in zip(record["sum"], vector, strict=True)]
                record["calls"] += 1
                record["positions"] += shape[0] if shape else 1
                stats["upstream_requires_grad_calls"] = stats.get("upstream_requires_grad_calls", 0) + int(credit)
        gradients = self.active["grad_packets"]
        if gradients:
            # Training has one shared condition. The hook observes, never replaces, its gradient.
            values = torch.stack([value for _, value in gradients]).cpu().tolist()
            for (index, _), value in zip(gradients, values, strict=True):
                for stats in self.active["stats"]:
                    terms = stats.setdefault("upstream_gradient_squared", {})
                    name = self.names[index]
                    terms[name] = terms.get(name, 0.) + value

    def close(self):
        if self.active is not None:
            raise ValueError("cannot close active transition hooks")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ReaderHooks(_EffectHooks):
    """Actual own-h credit; physical parent can be installed or identity only."""
    def __init__(self, policy, lora, readout):
        self.readout = readout
        super().__init__(policy, lora)

    def _hook(self, index):
        name, target = self.names[index], self.readout.targets[index]

        def read(_module, inputs, output):
            if self.active is None:
                return output
            value = inputs[0]
            batch = value.shape[0]
            with torch.autocast(value.device.type, enabled=False):
                h = value.float().reshape(batch, -1, value.shape[-1])
                a, b, key, u, valid, transitions = self.active["targets"][index]
                q = torch.matmul(h, a.transpose(1, 2))
                s = F.linear(q, target.q)
                score = torch.matmul(s, key.transpose(1, 2)).masked_fill(~valid[:, None], -torch.inf)
                alpha = score.softmax(dim=-1)
                r = torch.matmul(alpha, u)
                parent = torch.matmul(q, b.transpose(1, 2))
                delta = torch.matmul(r, b.transpose(1, 2))
                added = delta if self.active["parent_installed"] else parent + delta
            self._collect(index, delta, parent, value, alpha, transitions)
            return output + added.reshape_as(output).to(output.dtype)
        return read

    @contextmanager
    def activate(self, memories, *, parent_installed=False, stats=None):
        if self.active is not None or not memories:
            raise ValueError("transition reader activation is not reentrant/empty")
        encoded = [m.reader if m.reader is not None else self.readout.reader_memory(m) for m in memories]
        targets = []
        for index, name in enumerate(self.names):
            lengths = [e[index].shape[0] for e, _ in encoded]
            longest = max(lengths)
            keys, values, masks = [], [], []
            for (e, u), length in zip(encoded, lengths, strict=True):
                keys.append(F.pad(e[index], (0, 0, 0, 0, 0, longest - length)).flatten(0, 1))
                values.append(F.pad(u[index], (0, 0, 0, 0, 0, longest - length)).flatten(0, 1))
                masks.append(torch.arange(longest * 50, device=e[index].device) < length * 50)
            targets.append((torch.stack([m.state[name + LORA_A_SUFFIX].float() for m in memories]),
                            torch.stack([m.state[name + LORA_B_SUFFIX].float() for m in memories]),
                            torch.stack(keys), torch.stack(values), torch.stack(masks), longest))
        self.active = {"targets": targets, "parent_installed": parent_installed,
                       "stats": stats, "calls": [0] * 38, "packets": [], "grad_packets": []}
        try:
            yield
        finally:
            self._flush()
            self.active = None


class LinearEffectHooks(_EffectHooks):
    """Passive comparison of installed final L and its original parent factors."""
    def _hook(self, index):
        name = self.names[index]
        def observe(_module, inputs, output):
            if self.active is None:
                return output
            value = inputs[0]
            with torch.autocast(value.device.type, enabled=False):
                a, b, change = self.active["targets"][index]
                q = torch.matmul(value.float().reshape(len(value), -1, value.shape[-1]), a.transpose(1, 2))
                self._collect(index, torch.matmul(q, change.transpose(1, 2)),
                              torch.matmul(q, b.transpose(1, 2)), value)
            return output
        return observe

    @contextmanager
    def activate(self, states, parents, *, stats):
        if self.active is not None:
            raise ValueError("linear passive hooks are not reentrant")
        targets = []
        for name in self.names:
            a = torch.stack([s[name + LORA_A_SUFFIX].float() for s in states])
            b = torch.stack([s[name + LORA_B_SUFFIX].float() for s in parents])
            change = torch.stack([s[name + LORA_B_SUFFIX].float() - p[name + LORA_B_SUFFIX].float()
                                  for s, p in zip(states, parents, strict=True)])
            targets.append((a, b, change))
        self.active = {"targets": targets, "stats": stats, "calls": [0] * 38, "packets": [], "grad_packets": []}
        try:
            yield
        finally:
            self._flush()
            self.active = None
