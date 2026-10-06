"""CPU-only readback statistics for frozen G450/C450; never steps an optimizer.

credit_report inputs:
  grads_by_term: {"FM": {name: CPU tensor/None}, "KD": ..., "rel": ...}.
    Caller already applied four-task .25, query means and registered loss weights.
  named_parameter_metadata: original named_parameters order, list of dicts with
    name, shape, numel, dtype and requires_grad (the ORIGINAL optimizer eligibility).
  optimizer_state: original AdamW state_dict for G or C, without cloning it.
Missing gradients/None are zero. Caller owns scientific autograd and detach.
Moment sampling is tensor-stratified; no dtype repair or precision experiment.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
import re

import torch

MATRICES = ("a_x", "a_z", "a_context", "a_dynamic", "a_out", "b_key",
            "b_delta", "b_context", "b_dynamic", "b_out")
SAMPLE_N = 2048


def group_of(name, model):
    """Leaf group; model is 'G' or 'C', names use original writer registration."""
    if model not in {"G", "C"}:
        raise ValueError("only frozen G450/C450 are in scope")
    name = name.removeprefix(model + ".")
    common = re.fullmatch(r"common\.values\.(\d+)", name)
    if common:
        # DirectLoRAParameters sorts the 38 complete A/B template pairs.
        return "public_A" if int(common[1]) % 2 == 0 else "public_B0"
    if name.startswith("common."):
        if "lora_A" in name:
            return "public_A"
        if "lora_B" in name:
            return "public_B0"
    head = name.split(".", 1)[0]
    if model == "G" and head.lower() in {"phi", "omega", "read"}:
        return {"phi": "Phi", "omega": "Omega", "read": "read"}[head.lower()]
    if model == "C" and head == "interpreter":
        return "interpreter"
    parts = name.split(".")
    if head == "conditional_targets" and len(parts) == 4 and parts[2] in MATRICES:
        return "Compiler." + parts[2]
    raise ValueError(f"unregistered {model} parameter: {name}")


def _major(group):
    if group in {"public_A", "public_B0"}:
        return "public"
    return "Compiler" if group.startswith("Compiler.") else group


def _cpu(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("statistics consume CPU tensors only")
    if value.requires_grad or value.is_sparse or not value.is_floating_point():
        raise ValueError("statistics require detached dense floating tensors")
    return value


def accumulate_(target, grads):
    """Add already weighted CPU gradients, preserving the original tensor dtype."""
    for name, value in grads.items():
        if value is None:
            target.setdefault(name, None)
            continue
        value = _cpu(value)
        previous = target.get(name)
        if previous is None:
            target[name] = value.clone()
        else:
            _cpu(previous)
            if previous.shape != value.shape or previous.dtype != value.dtype:
                raise ValueError(f"gradient shape/dtype changed for {name}")
            previous.add_(value)
    return target


def _sum(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    result = present[0].clone()
    for value in present[1:]:
        if result.dtype != value.dtype or result.shape != value.shape:
            raise ValueError("term gradients do not share original parameter dtype/shape")
        result.add_(value)
    return result


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _blank(labels, pairs):
    return {"tensors": 0, "numel": 0, "nonfinite_gradient_elements": 0,
            "sq": dict.fromkeys(labels, 0.), "dot": dict.fromkeys(pairs, 0.),
            "sqrtv": [], "attenuation": [], "dtype": Counter(), "eps": Counter(),
            "sample_n": 0, "moment_numel": 0, "missing_state": 0,
            "weighted": {label: [0., 0., 0.] for label in labels},
            "preconditioned_FM_dot": dict.fromkeys(labels, 0.),
            "saved_moment_descent_dot": dict.fromkeys(labels, 0.)}


def _finish(stats, labels, pairs):
    norms = {label: math.sqrt(value) for label, value in stats["sq"].items()}
    angles = {}
    for a, b in pairs:
        dot = stats["dot"][(a, b)]
        denominator = norms[a] * norms[b]
        angles[a + "__" + b] = {"dot": _number(dot),
            "cosine": _number(dot / denominator) if denominator else None}
    result = {key: stats[key] for key in ("tensors", "numel", "nonfinite_gradient_elements")}
    result.update(norms={label: _number(value) for label, value in norms.items()},
                  inner_products=angles)
    if not stats["sqrtv"]:
        result["adam_saved450"] = {"sample_n": 0, "missing_state": stats["missing_state"]}
        return result
    sqrtv, attenuation = torch.cat(stats["sqrtv"]), torch.cat(stats["attenuation"])
    eps_counts = {str(key): value for key, value in stats["eps"].items()}
    # All authorized optimizers use the same eps; do not manufacture a mixed threshold.
    if len(eps_counts) != 1:
        raise ValueError("mixed Adam eps outside fixed optimizer contract")
    eps = next(iter(stats["eps"]))
    weighted = {}
    for label, (g2, atten_g2, preconditioned_g2) in stats["weighted"].items():
        weighted[label] = {
            "sample_expanded_gradient_sq": _number(g2),
            "gradient_sq_weighted_epsfactor": _number(atten_g2 / g2) if g2 else None,
            "sample_expanded_g_D_g": _number(preconditioned_g2),
            "sample_expanded_FM_D_g": _number(stats["preconditioned_FM_dot"][label]),
            "sample_expanded_g_mhat_over_denominator":
                _number(stats["saved_moment_descent_dot"][label])}
    result["adam_saved450"] = {
        "sample_n": stats["sample_n"], "moment_numel": stats["moment_numel"],
        "missing_state": stats["missing_state"],
        "moment_dtype_tensor_counts": dict(stats["dtype"]), "eps": eps,
        "nonfinite_samples": int((~torch.isfinite(sqrtv)).sum()),
        "sqrtv_q10_q50_q90": [_number(v) for v in torch.quantile(
            sqrtv, torch.tensor([.1, .5, .9], dtype=torch.float64))],
        "fraction_sqrtv_lt_eps": float((sqrtv < eps).double().mean()),
        "median_epsfactor": _number(torch.quantile(attenuation, .5)),
        "gradient_weighted": weighted}
    return result


def credit_report(grads_by_term, named_parameter_metadata, optimizer_state, model):
    """Return JSON-compatible raw/major/global credit with fixed saved450 moments.

    D=diag(1/(sqrt(vhat450)+eps)). g_D_g/FM_D_g are sampled frozen-denominator
    gradient geometry, not an Adam update. mhat is the saved first moment, without
    incorporating these queries; its descent dot is <g,mhat/(sqrt(vhat)+eps)>.
    No LR, weight decay, step, loss reweighting or training counterfactual is applied.
    """
    terms = list(grads_by_term)
    if "FM" not in terms or set(terms) - {"FM", "KD", "rel"}:
        raise ValueError("expected FM and optional already weighted KD/rel")
    metadata = [row for row in named_parameter_metadata if row.get("requires_grad", True)]
    names = [row["name"] for row in metadata]
    if len(set(names)) != len(names):
        raise ValueError("duplicate original parameter names")
    if any(set(grads) - set(names) for grads in grads_by_term.values()):
        raise ValueError("gradient parameter absent from original metadata")
    saved_groups = optimizer_state["param_groups"]
    parameter_ids = [(pid, group) for group in saved_groups for pid in group["params"]]
    if len(parameter_ids) != len(metadata):
        raise ValueError("original namedparameter/optimizer order or eligibility changed")
    auxiliary = [term for term in terms if term != "FM"]
    labels = terms + (["AUX"] if auxiliary else []) + ["TOTAL"]
    pairs = list(combinations(terms, 2)) + [("FM", label) for label in ("AUX", "TOTAL")
                                          if label in labels]
    groups, native_norms, order = {}, {label: [] for label in labels}, []
    for row, (pid, adam_group) in zip(metadata, parameter_ids, strict=True):
        name = row["name"]
        leaf = group_of(name, model)
        memberships = list(dict.fromkeys(["ALL", _major(leaf), leaf]))
        states = [groups.setdefault(key, _blank(labels, pairs)) for key in memberships]
        order.append({"name": name, "saved_optimizer_parameter_id": pid, "group": leaf})
        base = {term: grads_by_term[term].get(name) for term in terms}
        for value in base.values():
            if value is not None:
                _cpu(value)
                if tuple(value.shape) != tuple(row["shape"]) or value.numel() != row["numel"]:
                    raise ValueError(f"gradient shape differs from original metadata: {name}")
                if "dtype" in row and str(value.dtype) != str(row["dtype"]):
                    raise ValueError(f"gradient dtype differs from original parameter: {name}")
        vectors = dict(base)
        if auxiliary:
            vectors["AUX"] = _sum(base[term] for term in auxiliary)
        vectors["TOTAL"] = _sum(base.values())
        precise = {label: value.double() for label, value in vectors.items() if value is not None}
        nonfinite = sum(int((~torch.isfinite(value)).sum()) for value in base.values()
                        if value is not None)
        for stats in states:
            stats["tensors"] += 1
            stats["numel"] += row["numel"]
            stats["nonfinite_gradient_elements"] += nonfinite
        for label, value in vectors.items():
            if value is not None:
                native_norms[label].append(torch.linalg.vector_norm(value, 2))
                sq = float(precise[label].square().sum())
                for stats in states:
                    stats["sq"][label] += sq
        for a, b in pairs:
            dot = float((precise[a] * precise[b]).sum()) if a in precise and b in precise else 0.
            for stats in states:
                stats["dot"][(a, b)] += dot
        saved = optimizer_state["state"].get(pid)
        if saved is None:
            for stats in states:
                stats["missing_state"] += 1
            continue
        m, v = _cpu(saved["exp_avg"]), _cpu(saved["exp_avg_sq"])
        step = float(saved["step"])
        if step != 450 or m.shape != v.shape or tuple(v.shape) != tuple(row["shape"]):
            raise ValueError(f"saved450 moment identity/shape changed: {name}")
        beta1, beta2 = adam_group["betas"]
        eps = float(adam_group["eps"])
        if tuple(adam_group["betas"]) != (.9, .95) or eps != 1e-8:
            raise ValueError("optimizer differs from original fixed AdamW contract")
        count = min(SAMPLE_N, v.numel())
        index = torch.linspace(0, v.numel() - 1, count, dtype=torch.float64).round().long()
        # PyTorch Adam's sqrt then bias-correction division, in saved v's dtype.
        sqrtv = (v.reshape(-1)[index].sqrt() / math.sqrt(1 - beta2 ** step)).double()
        mhat = (m.reshape(-1)[index] / (1 - beta1 ** step)).double()
        denominator = sqrtv + eps
        attenuation = sqrtv / denominator
        sampled = {label: value.reshape(-1)[index].double()
                   for label, value in vectors.items() if value is not None}
        expansion = v.numel() / count
        fm = sampled.get("FM")
        for stats in states:
            stats["sqrtv"].append(sqrtv)
            stats["attenuation"].append(attenuation)
            stats["sample_n"] += count
            stats["moment_numel"] += v.numel()
            stats["dtype"][f"m={m.dtype};v={v.dtype}"] += 1
            stats["eps"][eps] += count
        for label, gradient in sampled.items():
            g2 = gradient.square()
            values = (float(g2.sum()) * expansion,
                      float((g2 * attenuation).sum()) * expansion,
                      float((g2 / denominator).sum()) * expansion)
            fm_dot = float((fm * gradient / denominator).sum()) * expansion if fm is not None else 0.
            moment_dot = float((gradient * mhat / denominator).sum()) * expansion
            for stats in states:
                stats["weighted"][label] = [a + b for a, b in zip(stats["weighted"][label], values)]
                stats["preconditioned_FM_dot"][label] += fm_dot
                stats["saved_moment_descent_dot"][label] += moment_dot
    clip = {}
    for label, values in native_norms.items():
        norm = torch.linalg.vector_norm(torch.stack(values), 2) if values else torch.tensor(0.)
        factor = torch.clamp(1. / (norm + 1e-6), max=1.)
        clip[label] = {"pytorch_native_total_norm": _number(norm),
                       "clip1_factor": _number(factor), "finite": bool(torch.isfinite(norm))}
    return {"model": model, "already_weighted_terms": terms, "clip1": clip,
        "groups": {key: _finish(value, labels, pairs) for key, value in groups.items()},
        "parameter_order": order,
        "definitions": {
            "full_gradient_statistics": "Full elements; FP64 reductions only. Source tensors unchanged.",
            "TOTAL": "Native-dtype sum in supplied term order; None/missing means zero.",
            "clip1": "Native per-tensor torch linalg L2, stacked L2; clamp(1/(norm+1e-6),max=1). No clipping mutation.",
            "sampling": "Each tensor: min(2048,numel) unique rounded linspace endpoints. Median/fraction use tensor-stratified pooled samples.",
            "weighted_sampling": "Actual sampled gradients; weight=numel/sample_n for estimated population sums. Not full-element Adam statistics.",
            "sqrtv": "Original saved v dtype sqrt/div bias-correction at step450; source dtype retained. Ratios/reductions FP64, no state mutation.",
            "epsfactor": "sqrtv/(sqrtv+saved_eps), algebra at fixed saved moments; no eps=0 training.",
            "credit": "Positive FM_D_g or g_mhat_over_denominator indicates local descent dot only; excludes new moments, LR, decay and behavior.",
            "limits": "Frozen query credit; no optimizer step, dtype intervention, performance or closed-loop causal claim."}}
