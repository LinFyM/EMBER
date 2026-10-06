"""Joint-Gaussian score credit through full execution LoRA and native Writer.

Saved collection means form a local velocity cotangent. No sampled path is
differentiated; every nonzero credit uses the real denoise suffix and a complete
same-version video compilation replay, including both public factors.
"""
from __future__ import annotations

import time

import torch

from ember.writer.runtime import autocast
from .credit import native_credit
from .denoising_collection import restore_raw
from .denoising_policy import NativeVelocity


def score_cotangent(transition: dict, advantage: float, q: int, m: int) -> torch.Tensor:
    if m != min(16, q) or q <= 0 or not 0 <= transition["step"] < 10:
        raise ValueError("path reservoir/transition sampling identity changed")
    tau = 1 - transition["step"] / 10
    if abs(tau - transition["tau"]) > 1e-7:
        raise ValueError("saved transition time changed")
    return (transition["z_next"] - transition["m"]) * (advantage / 16 * q / m * 5
                                                                   * (1 + .5 * (1 - tau)) / tau)


def score_group(runtime, condition: tuple, state: dict, group: dict,
                *, microbatch: int, frame_chunk: int, force_zero: bool = False) -> dict:
    started = time.perf_counter()
    entries = []
    for row, advantage in zip(group["rows"], group["advantages"], strict=True):
        saved = torch.load(row["score_records"]["path"], map_location="cpu", weights_only=False)
        if len(saved["decisions"]) != row["saved_decisions"]:
            raise ValueError("score reservoir count mismatch")
        for decision in saved["decisions"]:
            for transition in decision["transitions"]:
                entries.append((decision["observation"], transition,
                                score_cotangent(transition, advantage, row["actual_replans"], row["saved_decisions"])))
    if not entries or len(state) != 76:
        raise ValueError("score must cover the full 38 A/B targets and real replans")
    if not group["nonzero_LOO"] and not force_zero:
        return {"scored_transitions": len(entries), "nonzero_LOO": False,
                "score_seconds": 0., "replay_seconds": 0., "lora_cotangent_norm": 0.,
                "native_cotangent_norm": {"h": 0., "x": 0., "passes": []}}
    cotangent = {name: torch.zeros_like(value, dtype=torch.float32) for name, value in state.items()}
    for start in range(0, len(entries), microbatch):
        block = entries[start:start + microbatch]
        processed = [runtime.processor(restore_raw(item[0])) for item in block]
        batch = {key: torch.cat([item[key] for item in processed]) for key in processed[0]}
        z = torch.stack([item[1]["z"] for item in block]).to(runtime.device)
        tau = torch.tensor([item[1]["tau"] for item in block], device=runtime.device)
        credit = torch.stack([item[2] for item in block]).to(runtime.device)
        leaves = {name: value.detach().requires_grad_(True) for name, value in state.items()}
        with torch.enable_grad(), autocast(runtime.device):
            owner = NativeVelocity(runtime.policy, batch)
            velocity = torch.func.functional_call(owner, {"policy." + name: value for name, value in leaves.items()},
                                                  (z, tau), strict=False)
            gradients = torch.autograd.grad(velocity, tuple(leaves.values()), grad_outputs=credit.to(velocity))
        for name, value in zip(leaves, gradients, strict=True):
            cotangent[name].add_(value.detach().float())
    torch.cuda.synchronize(runtime.device)
    scoring = time.perf_counter() - started
    if any(not torch.isfinite(value).all() for value in cotangent.values()):
        raise ValueError("nonfinite complete-LoRA score credit")
    with torch.enable_grad():
        replay, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
        torch.autograd.backward(tuple(replay.values()), tuple(cotangent[name].to(value) for name, value in replay.items()))
    torch.cuda.synchronize(runtime.device)
    return {"scored_transitions": len(entries), "nonzero_LOO": bool(group["nonzero_LOO"]),
            "score_seconds": scoring, "replay_seconds": time.perf_counter() - started - scoring,
            "lora_cotangent_norm": float(torch.stack([value.norm() for value in cotangent.values()]).norm()),
            "native_cotangent_norm": native_credit(native)}
