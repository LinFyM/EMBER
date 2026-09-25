#!/usr/bin/env python3
"""One frozen RAW/Rao-Blackwell score comparison on sealed return-credit inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from return_credit_gradient import (_add, _collection, _collection_lora,
                                    _decision_payload, _gradient_inputs, _parameters,
                                    _relative_lora_difference, _save, _snapshot)
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed
from ember.writer.data import RawTeacherVideoStore
from ember.writer.flow import flow_mean_lora_gradient
from ember.writer.learning_data import load_learning_tasks
from ember.writer.return_credit import dot, norm, score_cotangent
from ember.writer.runtime import build_runtime
from ember.writer.score_conditioning import correlated_sign_score


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / "EMBER"
SPEC = REPO / "configs/return_score_conditioning_v1/experiment_spec.json"
THRESHOLD = -1. + 2. / (2. + 1e-6)


def authority(formal=True):
    spec = read_json(SPEC)
    source = spec["source"]
    old = read_json(REPO / source["spec"])
    state = git_state(REPO)
    if (formal and (state["branch"] or state["dirty_paths"]
                    or not git_state_is_clean_pushed_or_frozen_authority(state))):
        raise ValueError("score conditioning requires a clean pushed detached commit")
    old_root, config, _ = _gradient_inputs(old, formal=False)
    root = Path(spec["resources"]["study_root"])
    if (str(old_root) != source["study_root"]
            or old["implementation"]["collection_commit_exception"] != source["collection_commit"]
            or old["parent"]["training_commit"] != source["parent_training_commit"]
            or old["parent"]["macro"] != source["parent_macro"]
            or sorted(source["task_ids"]) != sorted(row["task"] for row in old["collection"]["conditions"])
            or old["collection"]["episodes"] != source["existing_episodes"]):
        raise ValueError("registered original collection identity changed")
    if root.exists():
        registration = read_json(root / "registration.json")
        if (registration["implementation_commit"] != state["commit"]
                or registration["source_study"] != str(old_root)
                or registration["study_id"] != spec["study_id"]):
            raise ValueError("score-conditioning root already belongs to another run")
    return spec, old, old_root, root, config, state


def _decision_row(spec, task, state, episode, payload, index, item, advantage):
    old_mean = item["mu_old_full"][:5, :7].reshape(35).float()
    latent = item["latent_u"].reshape(35).float()
    if (item["applied_normalized"].shape != (5, 7)
            or not torch.equal(item["applied_normalized"].reshape(35), latent)
            or not torch.isfinite(old_mean).all() or not torch.isfinite(latent).all()):
        raise ValueError("saved exploration and applied action differ")
    means = old_mean.reshape(5, 7)[:, 6].numpy().astype(np.float64)
    sampled = latent.reshape(5, 7)[:, 6].numpy().astype(np.float64)
    signs = np.sign(sampled - THRESHOLD)
    prob, conditional, tail, bound = correlated_sign_score(
        means, signs, seed=spec["numerics"]["cdf_seed"])
    repeat = 0.
    if np.min(np.abs(means - THRESHOLD)) < .4:
        p2, c2, _, _ = correlated_sign_score(
            means, signs, seed=spec["numerics"]["independent_cpu_seed"])
        repeat = max(abs(prob-p2), float(np.max(np.abs(conditional-c2))))
    raw = score_cotangent(latent, old_mean, float(advantage),
                          Q=payload["Q"], M=payload["M"])
    rb = raw.clone()
    weight = float(advantage) * payload["Q"] / payload["M"] / 128.
    rb[6::7] = torch.as_tensor(weight * conditional, dtype=torch.float32)
    if (not torch.equal(raw.reshape(5, 7)[:, :6], rb.reshape(5, 7)[:, :6])
            or not torch.isfinite(rb).all()
            or np.any(signs != np.sign(means - THRESHOLD))):
        raise ValueError("registered sign or untouched score coordinates changed")
    row = {"task": task, "state": state, "replica": episode["replica"],
           "replan": item["replan_index"], "policy_noise_seed": item["policy_noise_seed"],
           "exploration_seed": item["exploration_seed"],
           "payload": episode["return_credit_collection"], "decision_index": index,
           "bank_record": episode["bank_record"], "teacher": episode["teacher_demo"],
           "success": episode["success"], "advantage": float(advantage),
           "Q": payload["Q"], "M": payload["M"], "weight": weight,
           "mean_gripper": means.tolist(), "latent_gripper": sampled.tolist(),
           "executed_sign": signs.astype(int).tolist(), "sign_probability": prob,
           "flip_union_upper_bound": tail,
           "conditional_score_coordinate_upper_bound": bound,
           "conditional_gripper_score": conditional.tolist(),
           "RAW_output_cotangent": raw.tolist(), "RB_output_cotangent": rb.tolist(),
           "executed_mask": item["executed_mask"]}
    return row, repeat


def _reference_rows_match(old_root, rows):
    original_audit = old_root / "coordination" / "main_gripper_score_rows.jsonl"
    independent = [json.loads(line) for line in original_audit.read_text().splitlines()]
    if (len(independent) != len(rows) or any(
            (row["task"], row["state"], row["replica"], row["replan"], row["executed_sign"])
            != (reference["task"], reference["state"], reference["replica"],
                reference["replan"], reference["sign"])
            or abs(row["advantage"] - reference["advantage"]) > 1e-6
            or not np.allclose(row["mean_gripper"], reference["mean_gripper"], atol=1e-7)
            for row, reference in zip(rows, independent, strict=True))):
        raise ValueError("independent sealed CPU source audit disagrees on 512 decision identities")


def _decisions(spec, old, old_root):
    groups, nonzero_groups = _collection(old, old_root)
    seen, rows = set(), []
    independent_max = 0.
    for task, state, episodes, advantages in groups:
        for episode, advantage in zip(episodes, advantages, strict=True):
            payload = _decision_payload(episode)
            for index, item in enumerate(payload["decisions"]):
                key = (task, state, episode["replica"], item["replan_index"])
                if key in seen:
                    raise ValueError("duplicate sealed decision")
                seen.add(key)
                row, repeat = _decision_row(spec, task, state, episode, payload,
                                            index, item, advantage)
                rows.append(row)
                independent_max = max(independent_max, repeat)
    active = [row for row in rows if row["advantage"] != 0.]
    if (len(rows) != spec["source"]["unique_saved_decisions"]
            or len(active) != spec["compute"]["gradient_unique_inputs"]
            or len(groups) != spec["source"]["groups"]
            or nonzero_groups != 6
            or sorted({r["task"] for r in active}) != spec["source"]["nonzero_task_ids"]
            or independent_max > 1e-3):
        raise ValueError("saved decision count, active identities or CDF repeat changed")
    _reference_rows_match(old_root, rows)
    summary = {"rows": len(rows), "active": len(active), "groups": len(groups),
                  "nonzero_groups": nonzero_groups, "cdf_independent_seed_maxabs": independent_max,
                  "active_min_margin_sigma": min(abs(m-THRESHOLD)/.1 for r in active
                                                 for m in r["mean_gripper"]),
                  "active_max_flip_union_bound": max(r["flip_union_upper_bound"] for r in active),
                  "active_max_conditional_score_bound": max(
                      r["conditional_score_coordinate_upper_bound"] for r in active)}
    if (summary["active_min_margin_sigma"] <
            spec["numerics"]["nonzero_min_boundary_distance_sigma"] - 1e-5
            or summary["active_max_flip_union_bound"] >
            spec["numerics"]["nonzero_joint_flip_probability_upper_bound"] * 1.001
            or summary["active_max_conditional_score_bound"] >
            spec["numerics"]["nonzero_conditional_score_maxabs_upper_bound"] * 1.001):
        raise ValueError("registered gripper sign margin or analytic bound changed")
    return rows, summary


def audit():
    spec, old, old_root, root, _, state = authority()
    output = root / "decision_rows.jsonl"
    if output.exists():
        raise ValueError("decision audit already exists; do not recalculate the frozen batch")
    rows, summary = _decisions(spec, old, old_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(output)
    write_json_atomic(root / "analysis" / "cpu_audit.json", summary)
    print(json.dumps({"stage": "audit", "commit": state["commit"], **summary}), flush=True)


def _active_rows(spec, root):
    path = root / "decision_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    active = [row for row in rows if row["advantage"] != 0.]
    if (len(rows) != spec["artifacts"]["decision_rows"]
            or len(active) != spec["compute"]["gradient_unique_inputs"]
            or len({(r["task"],r["state"],r["replica"],r["replan"]) for r in rows}) != len(rows)):
        raise ValueError("CPU audit row identity or active set incomplete")
    grouped = defaultdict(list)
    for row in active:
        grouped[(row["task"], row["state"] % 2)].append(row)
    return grouped


def _prepared(runtime, tasks, store, task, teacher):
    video = store.load(task, teacher)
    return runtime.prepare((torch.from_numpy(np.array(video.frames)),),
                           (torch.from_numpy(np.array(video.frame_indices)),),
                           tasks[task].authority.language)


def _policy_cotangents(runtime, lora, task, parity, teacher, rows, *, pilot):
    cotangents = {arm: {name: torch.zeros_like(value, dtype=torch.float32)
                        for name, value in lora.items()} for arm in ("RAW", "RB")}
    squared = maxima = 0.
    count = 0
    cached_path = None
    payload = None
    for row in rows:
        if row["teacher"] != teacher or row["task"] != task or row["state"] % 2 != parity:
            raise ValueError("active task/parity group identity changed")
        path = row["payload"]["path"]
        if path != cached_path:
            if Path(path).stat().st_size != row["payload"]["bytes"]:
                raise ValueError("saved decision payload size changed after CPU audit")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            cached_path = path
        item = payload["decisions"][row["decision_index"]]
        if (item["replan_index"] != row["replan"]
                or item["policy_noise_seed"] != row["policy_noise_seed"]
                or item["exploration_seed"] != row["exploration_seed"]):
            raise ValueError("saved active decision RNG or replan identity changed")
        batch = {key: value.to(runtime.device) for key, value in item["processed"].items()}
        noise = item["flow_noise"].unsqueeze(0).to(runtime.device)
        old_mean = item["mu_old_full"][:5, :7].float()
        for arm in ("RAW", "RB"):
            score = torch.tensor(row[f"{arm}_output_cotangent"],
                                 dtype=torch.float32, device=runtime.device).unsqueeze(0)
            gradient, predicted = flow_mean_lora_gradient(
                runtime.policy, lora, runtime.lora, batch, noise, score)
            _add(cotangents[arm], gradient)
            difference = predicted[0].detach().float().cpu() - old_mean
            if not torch.isfinite(predicted).all() or not all(
                    torch.isfinite(value).all() for value in gradient.values()):
                raise ValueError("nonfinite full ten-flow policy VJP")
            squared += float(difference.square().sum())
            maxima = max(maxima, float(difference.abs().max()))
            if pilot and count == 0 and arm == "RB" and math.sqrt(squared / 70) > .01:
                raise ValueError("first registered decision replay pilot failed")
        count += 1
    rms = math.sqrt(squared / (count * 2 * 35))
    if rms > .01:
        raise ValueError(f"actual-bank flow replay RMS exceeded .01: {task=} {parity=} {rms=}")
    return cotangents, count, rms, maxima


def _group_gradient(runtime, parameters, old, old_root, task, parity, rows,
                    tasks, store, *, pilot):
    teacher = rows[0]["teacher"]
    prepared = _prepared(runtime, tasks, store, task, teacher)
    lora, adapter = _collection_lora(old, old_root, task, teacher,
                                     runtime, tasks[task], prepared)
    cotangents, count, rms, maxima = _policy_cotangents(
        runtime, lora, task, parity, teacher, rows, pilot=pilot)
    generated = runtime.compile(prepared)
    recompile = _relative_lora_difference(generated, lora)
    vectors = {}
    for arm in ("RAW", "RB"):
        runtime.state.zero_grad(set_to_none=True)
        torch.autograd.backward(tuple(generated.values()),
            tuple(cotangents[arm][name].to(value) for name, value in generated.items()),
            retain_graph=(arm == "RAW"))
        vectors[arm] = _snapshot(parameters)
        if not all(torch.isfinite(value).all() for value in vectors[arm].values()):
            raise ValueError("nonfinite full Writer VJP")
    activity = {arm: {part: sum(float(value.square().sum()) for name, value in vectors[arm].items()
                            if part in name) for part in
                      ("text_meta_lora", "vl_meta_lora", "action_meta_lora")}
                for arm in vectors}
    if any(value <= 0 for value in activity["RAW"].values()):
        raise ValueError("full Writer VJP missed a registered Meta component")
    return vectors, {"task": task, "parity": parity, "decisions": count,
                     "forward_calls": 2*count, "replay_rms": rms, "replay_maxabs": maxima,
                     "writer_recompile": recompile, "bank": adapter,
                     "teacher": teacher, "activity": activity,
                     "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "source_frozen": not any(p.requires_grad for p in runtime.policy.parameters())}


def _formal_group(runtime, parameters, old, old_root, task, parity, rows, tasks, store,
                  *, pilot):
    if not pilot:
        return _group_gradient(runtime, parameters, old, old_root, task, parity,
                               rows, tasks, store, pilot=False)
    # Complete the first registered decision, including its Writer VJP, before
    # consuming the other decisions. Both parts sum to the unchanged group VJP.
    first, pilot_report = _group_gradient(runtime, parameters, old, old_root,
                                          task, parity, rows[:1], tasks, store, pilot=True)
    rest, report = _group_gradient(runtime, parameters, old, old_root,
                                   task, parity, rows[1:], tasks, store, pilot=False)
    if pilot_report["bank"] != report["bank"]:
        raise ValueError("pilot and remaining decisions used different collection banks")
    vectors = {arm: {name: first[arm][name] + rest[arm][name] for name in first[arm]}
               for arm in ("RAW", "RB")}
    count = pilot_report["decisions"] + report["decisions"]
    report.update({
        "decisions": count, "forward_calls": 2*count,
        "replay_rms": math.sqrt((pilot_report["replay_rms"]**2 * pilot_report["decisions"]
                                 + report["replay_rms"]**2 * report["decisions"]) / count),
        "replay_maxabs": max(pilot_report["replay_maxabs"], report["replay_maxabs"]),
        "writer_recompile": {"pilot": pilot_report["writer_recompile"],
                             "remainder": report["writer_recompile"]},
        "activity": {arm: {part: sum(float(value.square().sum()) for name, value in
                                      vectors[arm].items() if part in name)
                           for part in ("text_meta_lora", "vl_meta_lora", "action_meta_lora")}
                     for arm in ("RAW", "RB")},
        "pilot": {"decision": (rows[0]["task"], rows[0]["state"], rows[0]["replica"],
                               rows[0]["replan"]), "full_writer_vjp": True}})
    return vectors, report


def _group_plan(grouped):
    first = next(iter(grouped))  # registered first nonzero decision
    groups = [first] + sorted((group for group in grouped if group != first),
                              key=lambda group: (-len(grouped[group]), group))
    if sum(map(len, grouped.values())) != 96 or len(groups) != 5:
        raise ValueError("nonzero task/parity scope changed")
    loads = [0, 0]
    owners = {}
    for group in groups:
        rank = min(range(2), key=lambda value: (loads[value], value))
        owners[group] = rank
        loads[rank] += len(grouped[group])
    if owners[first] != 0:
        raise ValueError("first registered nonzero group must own the formal pilot")
    return first, groups, owners, loads


def _owned_group(group, grouped, first, runtime, parameters, old, old_root,
                 root, tasks, store):
    task, parity = group
    suffix = f"task_{task:03d}_{'even' if parity == 0 else 'odd'}"
    paths = {arm: root / "gradient" / "groups" / f"{arm}_{suffix}.safetensors"
             for arm in ("RAW", "RB")}
    report_path = root / "gradient" / "groups" / f"{suffix}.json"
    existing = [path.exists() for path in (*paths.values(), report_path)]
    if any(existing) and not all(existing):
        raise ValueError("preserved partial gradient group needs a scoped failure audit")
    if all(existing):
        report = read_json(report_path)
        if (report["task"], report["parity"], report["decisions"]) != (
                task, parity, len(grouped[group])):
            raise ValueError("completed gradient group does not match frozen scope")
        return report
    vectors, report = _formal_group(runtime, parameters, old, old_root,
                                    task, parity, grouped[group], tasks, store,
                                    pilot=(group == first))
    for arm in ("RAW", "RB"):
        _save(paths[arm], vectors[arm])
    write_json_atomic(report_path, report)
    return report


def _save_totals(root, parameters, groups, arm):
    totals = {name: {key: torch.zeros_like(parameter, dtype=torch.float32,
                                           device="cpu") for key, parameter in parameters}
              for name in ("total", "even", "odd")}
    for task, parity in groups:
        path = (root / "gradient" / "groups" /
                f"{arm}_task_{task:03d}_{'even' if parity == 0 else 'odd'}.safetensors")
        values = load_file(str(path), device="cpu")
        for name in ("total", "even" if parity == 0 else "odd"):
            for key in totals[name]:
                totals[name][key].add_(values[key])
    for name, values in totals.items():
        _save(root / "gradient" / f"{arm}_{name}.safetensors", values)


def _finish_gradient(spec, root, state, parameters, groups, loads, reports):
    complete = [report for rank in reports for report in rank]
    if (len(complete) != 5 or sum(report["decisions"] for report in complete) != 96
            or sum(report["forward_calls"] for report in complete) != 192
            or any(not report["allow_tf32"] or not report["source_frozen"] for report in complete)):
        raise ValueError("full frozen gradient group contract incomplete")
    for arm in ("RAW", "RB"):
        _save_totals(root, parameters, groups, arm)
    write_json_atomic(root / "gradient" / "completion.json", {
        "implementation_commit": state["commit"], "source_gradient_commit": spec["source"]["gradient_commit"],
        "collection_commit": spec["source"]["collection_commit"], "world_size": 2,
        "rank_allow_tf32": [True, True], "rank_decisions": loads,
        "groups": sorted(complete, key=lambda item: (item["task"], item["parity"])),
        "zero_groups": [{"task": task, "parity": parity} for task in spec["source"]["task_ids"]
                        for parity in (0, 1) if (task, parity) not in groups],
        "unique_inputs": 96, "ten_flow_forward_calls": 192,
        "all_task_replay_rms_max": max(item["replay_rms"] for item in complete)})


def gradient():
    torch.backends.cuda.matmul.allow_tf32 = True
    spec, old, old_root, root, config, state = authority()
    grouped = _active_rows(spec, root)
    first, groups, owners, loads = _group_plan(grouped)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2:
        raise ValueError("registered gradient requires two same-node ranks")
    torch.set_num_threads(4)
    runtime = build_runtime(ASSETS, config, context.device)
    runtime.policy.requires_grad_(False).eval()
    parent = load_file(str(Path(old["parent"]["checkpoint"]) / "ecp.safetensors"), device="cpu")
    runtime.state.load_state_dict(parent, strict=True)
    runtime.state.train()
    parameters = _parameters(runtime)
    if any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError("Source policy unexpectedly trainable")
    initialize_deferred_process_group(context, rendezvous_root=root / "rendezvous")
    tasks = load_learning_tasks(ASSETS, spec["source"]["nonzero_task_ids"],
                                protocol_path=config["data"]["protocol"])
    store = RawTeacherVideoStore([task.authority for task in tasks.values()],
                                 frame_stride=5, camera_view=config["observer"]["camera_view"])
    local = []
    if context.rank != 0:
        barrier(context)
    for group in groups:
        if owners[group] != context.rank:
            continue
        local.append(_owned_group(group, grouped, first, runtime, parameters,
                                  old, old_root, root, tasks, store))
        if group == first:
            barrier(context)
    store.close()
    barrier(context)
    reports = [None] * 2
    dist.all_gather_object(reports, local)
    if context.rank == 0:
        _finish_gradient(spec, root, state, parameters, groups, loads, reports)
    barrier(context)
    dist.destroy_process_group()


def _geometry(left, right):
    names = sorted(left)
    a, b = [left[name] for name in names], [right[name] for name in names]
    n1, n2 = norm(a), norm(b)
    return {"left_norm": n1, "right_norm": n2,
            "dot": dot(a, b), "cosine": dot(a, b) / (n1*n2) if n1*n2 else None}


def _module_norms(names, raw, rb, change):
    modules = {}
    meta = ("text_meta_lora", "vl_meta_lora", "action_meta_lora")
    for module in (*meta, "other_writer"):
        selected = [name for name in names if (module in name if module != "other_writer"
                                               else not any(tag in name for tag in meta))]
        modules[module] = {"parameter_tensors": len(selected),
                           "RAW_norm": norm([raw[name] for name in selected]),
                           "RB_norm": norm([rb[name] for name in selected]),
                           "removed_norm": norm([change[name] for name in selected])}
    return modules


def _task37(root, names, gradients):
    answer = {}
    for arm in ("RAW", "RB"):
        pieces = [load_file(str(root / "gradient" / "groups" /
                   f"{arm}_task_037_{parity}.safetensors")) for parity in ("even", "odd")
                  if (root / "gradient" / "groups" /
                      f"{arm}_task_037_{parity}.safetensors").exists()]
        source = {name: sum(part[name] for part in pieces) for name in names}
        total = gradients[arm]["total"]
        answer[arm] = {"norm": norm(list(source.values())),
                       "dot_fraction_of_total_square": dot(list(source.values()),
                                                           [total[name] for name in names]) /
                       (norm(list(total.values()))**2)}
    return answer


def _group_directions(root, names, entries):
    answer = []
    for entry in entries:
        suffix = f"task_{entry['task']:03d}_{'even' if entry['parity'] == 0 else 'odd'}.safetensors"
        pair = {arm: load_file(str(root / "gradient" / "groups" / f"{arm}_{suffix}"))
                for arm in ("RAW", "RB")}
        relative = (norm([pair["RAW"][name]-pair["RB"][name] for name in names]) /
                    norm([pair["RAW"][name] for name in names]))
        answer.append({"task": entry["task"], "parity": entry["parity"],
                       "decisions": entry["decisions"],
                       "RAW_vs_RB": _geometry(pair["RAW"], pair["RB"]),
                       "relative_removed_norm": relative})
    return answer


def analyze():
    spec, _, _, root, _, state = authority()
    completion = read_json(root / "gradient" / "completion.json")
    if (completion["implementation_commit"] != state["commit"]
            or completion["unique_inputs"] != 96 or completion["ten_flow_forward_calls"] != 192):
        raise ValueError("formal gradient incomplete")
    gradients = {arm: {part: load_file(str(root / "gradient" / f"{arm}_{part}.safetensors"))
                       for part in ("total", "even", "odd")} for arm in ("RAW", "RB")}
    raw, rb = gradients["RAW"]["total"], gradients["RB"]["total"]
    names = sorted(raw)
    change = {name: raw[name] - rb[name] for name in names}
    old_root = Path(spec["source"]["study_root"])
    if read_json(old_root / "gradient" / "completion.json")["implementation_commit"] != spec["source"]["gradient_commit"]:
        raise ValueError("saved E2 comparison gradient implementation changed")
    old = load_file(str(old_root / "gradient" / "return.safetensors"))
    if set(old) != set(raw):
        raise ValueError("saved E2 return direction parameter identity changed")
    modules = _module_norms(names, raw, rb, change)
    task37 = _task37(root, names, gradients)
    group_directions = _group_directions(root, names, completion["groups"])
    geometry = _geometry(raw, rb)
    old_geometry = _geometry(raw, old)
    relative_removed = norm(list(change.values())) / geometry["left_norm"]
    relative_old = norm([raw[name]-old[name] for name in names]) / norm([old[name] for name in names])
    report = {"implementation_commit": state["commit"], "decision_rows": 512,
              "active_unique_inputs": 96, "actual_forward_calls": 192,
              "RAW_vs_RB": geometry,
              "relative_removed_norm": relative_removed,
              "removed_dot_fraction": dot([change[name] for name in names],
                                           [raw[name] for name in names]) / geometry["left_norm"]**2,
              "recomputed_RAW_vs_saved_E2": old_geometry,
              "recomputed_RAW_relative_E2_difference": relative_old,
              "E2_difference_over_intervention": (relative_old / relative_removed
                                                  if relative_removed else None),
              "even_odd": {arm: _geometry(parts["even"], parts["odd"])
                           for arm, parts in gradients.items()},
              "task_parity": completion["groups"],
              "task_parity_directions": group_directions,
              "modules": modules, "task37": task37,
              "task_replay_rms_max": completion["all_task_replay_rms_max"],
              "writer_recompile": [{"task": x["task"], "parity": x["parity"],
                                    **x["writer_recompile"]} for x in completion["groups"]],
              "interpretation_limits": ["single frozen batch, no variance estimate or closed-loop claim",
                                        "actual-bank policy VJP with recompiled full-Writer Jacobian",
                                        "native BF16/TF32 reduction and anchored-Jacobian residual",
                                        "old E2 RAW numerical difference must be compared with intervention"]}
    write_json_atomic(root / "analysis" / "direction_summary.json", report)
    write_json_atomic(root / "analysis" / "completion.json", {
        "status": "complete", "study_id": spec["study_id"], "implementation_commit": state["commit"],
        "decision_rows": 512, "active_inputs": 96, "ten_flow_forward_calls": 192,
        "group_vectors_per_arm": 5, "zero_groups": completion["zero_groups"],
        "analysis": str(root / "analysis" / "direction_summary.json")})
    print(json.dumps({"stage": "analysis", "commit": state["commit"],
                      "relative_removed_norm": report["relative_removed_norm"],
                      "cosine": geometry["cosine"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "gradient", "analyze"))
    arguments = parser.parse_args()
    {"audit": audit, "gradient": gradient, "analyze": analyze}[arguments.stage]()
