"""One operator condition's full FM and optional direct public FM credit."""
from __future__ import annotations

import time

import torch

from ember.writer.function_credit import dual_functional_credit, paired_functional_credit
from ember.writer.runtime import autocast


def gradient_groups(writer) -> dict[str, float]:
    def norm(parameters):
        values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return float(torch.stack(values).norm()) if values else 0.0

    context = ({"value_context": norm(writer.value_context.parameters()),
                "u": norm(write.u.weight for write in writer.writes)}
               if writer.value_context is not None else {})
    conditional = ({"interpreter": norm(writer.interpreter.parameters()),
                    **{group: norm(parameter for unit in writer.conditional_targets
                                   for parameter in getattr(unit, group).parameters())
                       for group in ("a_x", "a_z", "a_context", "a_dynamic", "a_out",
                                     "b_key", "b_delta", "b_context", "b_dynamic", "b_out")}}
                   if writer.interpreter is not None else {})
    return {**context, **conditional, "public_A": norm(writer.common.values[i] for i, name in enumerate(writer.common.names)
                             if name.endswith(".lora_A.default.weight")),
            "public_B0": norm(writer.common.values[i] for i, name in enumerate(writer.common.names)
                              if name.endswith(".lora_B.default.weight")),
            "independent_S": norm(writer.separate_keys or ()),
            **{name: norm(parameter for unit in writer.writes
                          for parameter in getattr(unit, name).parameters())
               for name in ("p", "c", "d", "o")}}


def apply_public_cotangent(writer, cotangent: dict[str, torch.Tensor]) -> None:
    """Add beta execution risk only to the 76 public factor parameters."""
    public = writer.public_state()
    if (len(public) != 76 or set(public) != set(cotangent)
            or any(not torch.isfinite(value).all() for value in cotangent.values())):
        raise ValueError("public FM cotangent lost 76 complete finite A/B0 factors")
    torch.autograd.backward(tuple(public.values()),
                            tuple(cotangent[name].to(public[name]) for name in public))


def native_credit(native: dict) -> dict:
    def norm(values):
        terms = [value.grad.float().norm() for value in values if value.grad is not None]
        return float(torch.stack(terms).norm()) if terms else 0.0
    passes = [{"h": norm((item["h"],)), "x": norm(item["x"].values()),
               "B": norm(value for name, value in item["state"].items()
                         if name.endswith(".lora_B.default.weight"))}
              for item in native["passes"]]
    return {"h": passes[-1]["h"], "x": passes[-1]["x"], "passes": passes}


def one_job(runtime, data, event: dict, microbatch: int,
            frame_chunk: int, loss_variant: str, *, target_executor=None) -> dict:
    if loss_variant not in ("full", "full_plus_public_beta"):
        raise ValueError("operator pilot loss identity changed")
    started = time.perf_counter()
    condition, raw, sampled = data.condition(runtime, event["task"], event["teacher_demo"])
    with torch.no_grad():
        state, _ = runtime.compile(condition, frame_chunk=frame_chunk,
                                   **({"target_executor": target_executor} if target_executor else {}))
    torch.cuda.synchronize(runtime.device)
    compilation = time.perf_counter() - started
    batch = runtime.processor.training_batch(data.batch(event))
    with autocast(runtime.device):
        arguments = dict(seed=event["flow_seed"], device=runtime.device, random_batch=28,
                         offset=0, microbatch=microbatch, condition_weight=0.25)
        if loss_variant == "full_plus_public_beta":
            credit, beta_credit = dual_functional_credit(
                runtime.policy, state, runtime.writer.public_state(), runtime.lora, batch, **arguments)
        else:
            credit = paired_functional_credit(runtime.policy, state, runtime.lora, batch, **arguments)
            beta_credit = None
    torch.cuda.synchronize(runtime.device)
    fm = time.perf_counter() - started - compilation
    cotangent = credit["lora_cotangent"]
    with torch.enable_grad():
        if beta_credit is not None:
            apply_public_cotangent(runtime.writer, beta_credit["lora_cotangent"])
        replay, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True,
                                        **({"target_executor": target_executor} if target_executor else {}))
        if set(replay) != set(cotangent) or any(not torch.isfinite(v).all() for v in cotangent.values()):
            raise ValueError("full-rank FM cotangent incomplete or nonfinite")
        torch.autograd.backward(tuple(replay.values()),
                                tuple(cotangent[name].to(replay[name]) for name in replay))
    torch.cuda.synchronize(runtime.device)
    native_norm = native_credit(native)
    return {"task": event["task"], "teacher_demo": event["teacher_demo"],
            "queries": len(event["queries"]), "query_demos": [row["demo"] for row in event["queries"]],
            "query_frames": [row["frame"] for row in event["queries"]], "flow_seed": event["flow_seed"],
            "raw_frames": raw, "sampled_frames": sampled, "flow_loss": credit["flow_loss"],
            "public_flow_loss": beta_credit["flow_loss"] if beta_credit else None,
            "loss_variant": loss_variant, "public_query_reuse": beta_credit is not None,
            **({"target_execution": target_executor.record()} if target_executor else {}),
            "fm_cotangent_norm": float(torch.stack([v.norm() for v in cotangent.values()]).norm()),
            "public_cotangent_norm": (float(torch.stack([v.norm() for v in
                                      beta_credit["lora_cotangent"].values()]).norm())
                                      if beta_credit else None),
            "native_cotangent_norm": native_norm,
            "compile_seconds": compilation, "fm_seconds": fm,
            "replay_seconds": time.perf_counter() - started - compilation - fm,
            "total_seconds": time.perf_counter() - started}
