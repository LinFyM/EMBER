#!/usr/bin/env python3
"""Full ten-flow score and original FM Writer VJPs from the fixed parent."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import traceback

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.lora import validate_lora_state
from ember.pi05_eval.return_credit import authority
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed
from ember.writer.flow import flow_actions, flow_mean_lora_gradient
from ember.writer.learning_data import WriterTrainingData
from ember.writer.materialization import file_record
from ember.writer.return_credit import (candidate_step, dot, inspect_parent_events,
                                        loo_advantages, norm, score_cotangent)
from ember.writer.runtime import VideoConditionCache, build_runtime
from ember.writer.supervised import SupervisedEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT.parent / "EMBER"


def _gradient_inputs(spec, *, formal: bool):
    state = git_state(REPO_ROOT)
    if formal and (state["branch"] or state["dirty_paths"]
                   or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("return-credit gradient requires one clean pushed detached commit")
    root = Path(spec["resources"]["study_root"])
    config = read_json(REPO_ROOT / spec["parent"]["config"])
    run = read_json(Path(spec["parent"]["checkpoint"]).parent.parent / "run_contract.json")
    if run["git"]["commit"] != spec["parent"]["training_commit"] or run["config"] != config:
        raise ValueError("return-credit parent training identity changed")
    return root, config, state


def _collection(spec, root):
    groups, episodes, nonzero = [], 0, 0
    for row in spec["collection"]["conditions"]:
        for state_id in row["init_state_ids"]:
            path = root / "collection" / "groups" / f"task_{row['task']:03d}_state_{state_id:02d}" / "completion.json"
            if not path.is_file():
                raise ValueError(f"return-credit collection group is incomplete: {path}")
            group = read_json(path)
            rows = group["rows"]
            if ((group["task"], group["state"], group["teacher_demo"]) !=
                    (row["task"], state_id, row["teacher_demo"])
                    or len(rows) != 4 or [r["replica"] for r in rows] != [0, 1, 2, 3]
                    or any(r["global_task_id"] != row["task"] or
                           r["teacher_demo"] != row["teacher_demo"] or
                           r["init_state_id"] != state_id for r in rows)):
                raise ValueError("return-credit collection identity or four replicas changed")
            rewards = [int(r["success"]) for r in rows]
            advantages = loo_advantages(rewards)
            nonzero += int(bool(torch.count_nonzero(advantages)))
            episodes += len(rows)
            groups.append((row["task"], state_id, rows, advantages))
    if len(groups) != 32 or episodes != 128:
        raise ValueError("return-credit collection lost a registered group or episode")
    return groups, nonzero


def _parameters(runtime):
    return tuple((name, p) for name, p in runtime.state.named_parameters() if p.requires_grad)


def _save(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"return-credit gradient output already exists: {path}")
    save_file({name: value.detach().float().cpu().contiguous() for name, value in values.items()}, str(path))


def _zero(parameters):
    return {name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in parameters}


def _add(destination, source):
    for name in destination:
        destination[name].add_(source[name].float())


def _snapshot(parameters):
    return {name: (parameter.grad.detach().float().clone() if parameter.grad is not None
                   else torch.zeros_like(parameter, dtype=torch.float32))
            for name, parameter in parameters}


def _decision_payload(row):
    record = row["return_credit_collection"]
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError("return-credit saved decision evidence changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["schema_version"] != "ember_return_credit_decisions_v1"
            or payload["Q"] != record["Q"] or payload["M"] != record["M"]
            or payload["replica"] != row["replica"]
            or len(payload["decisions"]) != min(4, payload["Q"])):
        raise ValueError("return-credit Q/M or selected decision identity changed")
    return payload


def _collection_lora(spec, root, task_id, teacher, runtime):
    record = read_json(root / "banks" / "collection" / "P" /
                       f"task_{task_id:03d}_teacher_{teacher:02d}" / "bank_record.json")
    condition = record["condition"]
    adapter = condition["adapter"]
    if (record["implementation_commit"] != spec["implementation"]["collection_commit_exception"]
            or record["arm"] != "P" or record["phase"] != "collection"
            or (record["task"], record["teacher"]) != (task_id, teacher)
            or record["writer"]["path"] != str(Path(spec["parent"]["checkpoint"]) / "ecp.safetensors")
            or condition["teacher_demo_indices"] != [teacher]
            or condition["global_task_id"] != task_id
            or adapter != file_record(Path(adapter["path"]))):
        raise ValueError("return-credit gradient bank differs from actual collection condition")
    lora = load_file(adapter["path"], device=str(runtime.device))
    validate_lora_state(lora, runtime.lora)
    return lora, adapter


def _relative_lora_difference(generated, reference):
    square_difference = sum(float((generated[name].detach().float() -
                                   reference[name].float()).square().sum())
                            for name in reference)
    square_reference = sum(float(value.float().square().sum())
                           for value in reference.values())
    return math.sqrt(square_difference / square_reference)


def _score_task(runtime, cache, data, parameters, spec, root, task_id, groups, *, pilot=False):
    runtime.state.train()
    teacher = next(row["teacher_demo"] for row in spec["collection"]["conditions"]
                   if row["task"] == task_id)
    condition = cache.condition(task_id, (teacher,))
    lora, adapter = _collection_lora(spec, root, task_id, teacher, runtime)
    parity_gradients = []
    audit = {"task": task_id, "teacher_demo": teacher, "decisions": 0,
             "nonzero_advantage_decisions": 0, "squared_replay_error": 0.,
             "replay_coordinates": 0, "replay_maxabs": 0., "collection_adapter": adapter,
             "writer_recompile_relative_l2": [], "episodes": []}
    for parity in (0, 1):
        cotangent = {name: torch.zeros_like(value, dtype=torch.float32)
                     for name, value in lora.items()}
        for group_task, state_id, rows, advantages in groups:
            if group_task != task_id or state_id % 2 != parity:
                continue
            for row, advantage in zip(rows, advantages, strict=True):
                payload = _decision_payload(row)
                Q, M = payload["Q"], payload["M"]
                audit["episodes"].append({"state": state_id, "replica": row["replica"],
                    "success": row["success"], "advantage": float(advantage), "Q": Q, "M": M,
                    "selected_replans": [item["replan_index"] for item in payload["decisions"]]})
                for item in payload["decisions"]:
                    batch = {key: value.to(runtime.device) for key, value in item["processed"].items()}
                    noise = item["flow_noise"].unsqueeze(0).to(runtime.device)
                    old = item["mu_old_full"][:5, :7].reshape(35)
                    score = score_cotangent(item["latent_u"], old, float(advantage), Q=Q, M=M)
                    with torch.set_grad_enabled(bool(advantage)):
                        if bool(advantage):
                            gradient, predicted = flow_mean_lora_gradient(
                                runtime.policy, lora, runtime.lora, batch, noise,
                                score.unsqueeze(0).to(runtime.device))
                            _add(cotangent, gradient)
                            audit["nonzero_advantage_decisions"] += 1
                        else:
                            predicted = flow_actions(runtime.policy, lora, runtime.lora,
                                                     batch, noise, checkpoint_steps=False)[:, :5, :7]
                    diff = predicted[0].detach().float().cpu() - old.reshape(5, 7)
                    audit["squared_replay_error"] += float(diff.square().sum())
                    audit["replay_coordinates"] += 35
                    audit["replay_maxabs"] = max(audit["replay_maxabs"], float(diff.abs().max()))
                    audit["decisions"] += 1
                    if pilot:
                        break
                if pilot and audit["decisions"]:
                    break
            if pilot and audit["decisions"]:
                break
        runtime.state.zero_grad(set_to_none=True)
        if any(torch.count_nonzero(value).item() for value in cotangent.values()):
            runtime.state.train()
            generated = runtime.compile(condition)
            audit["writer_recompile_relative_l2"].append(
                _relative_lora_difference(generated, lora))
            torch.autograd.backward(tuple(generated.values()),
                tuple(cotangent[name].to(value) for name, value in generated.items()))
        gradient = _snapshot(parameters)
        parity_gradients.append(gradient)
        if pilot:
            break
        _save(root / "gradient" / "tasks" / f"task_{task_id:03d}_{'even' if parity == 0 else 'odd'}.safetensors",
              gradient)
    audit["replay_rms"] = math.sqrt(audit["squared_replay_error"] / max(1, audit["replay_coordinates"]))
    if audit["replay_rms"] > spec["implementation"]["flow_parity_first5_normalized_rms_max"]:
        raise ValueError(f"return-credit real ten-flow replay RMS exceeded .01: {audit['replay_rms']}")
    return parity_gradients, audit


def _fm_task(engine, data, runtime, parameters, event, task_id):
    draw = {"job_id": 0, "condition_index": 0, "task": task_id,
            "occurrence": event["occurrence"], "video_demos": (event["teacher_demo"],),
            "query_seed": event["query_seed"], "query_offset": 0,
            "query_count": 21, "teaching_offset": 0, "teaching_count": 7,
            "frames": event["frames"]}
    runtime.state.zero_grad(set_to_none=True)
    row = engine.backward(draw)
    result = _snapshot(parameters)
    if (row["queries"] != 21 or row["teaching_queries"] != 7
            or not math.isfinite(row["flow_loss"]) or not math.isfinite(row["teaching_loss"])):
        raise ValueError("return-credit original 21+7 FM gradient became invalid")
    return result, {"task": task_id, "main_loss": row["flow_loss"],
                    "extra_loss": row["teaching_loss"],
                    "weighted_loss": row["flow_loss"] / 8 + row["teaching_loss"] / 24,
                    "gradient_norm": norm(tuple(result.values()))}


def _all_reduce(values, parameters):
    names = tuple(name for name, _ in parameters)
    flat = torch.cat(tuple(values[name].reshape(-1) for name in names))
    if dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    result, start = {}, 0
    for name, parameter in parameters:
        count = parameter.numel()
        result[name] = flat[start:start + count].reshape_as(parameter).clone()
        start += count
    return result


def _candidate(runtime, parameters, parent, direction, root, arm, sign, radius, commit,
               parent_checkpoint):
    output = root / "candidates" / arm
    if output.exists():
        raise ValueError("return-credit candidate already exists or has a partial attempt")
    output.mkdir(parents=True)
    info, optimizer = candidate_step(parameters, parent, direction, sign=sign, radius=radius)
    save_file({name: value.detach().cpu().contiguous()
               for name, value in runtime.state.state_dict().items()},
              str(output / "weights.safetensors"))
    torch.save(optimizer, output / "fresh_sgd_optimizer.pt")
    write_json_atomic(output / "completion.json", {
        "schema_version": "ember_return_credit_candidate_v1", "arm": arm,
        "implementation_commit": commit, "parent_checkpoint": str(parent_checkpoint),
        "weights": {"path": str(output / "weights.safetensors"),
                    "bytes": (output / "weights.safetensors").stat().st_size},
        "optimizer_state": {"path": str(output / "fresh_sgd_optimizer.pt"),
                            "bytes": (output / "fresh_sgd_optimizer.pt").stat().st_size},
        "schedule": {"kind": "fresh_constant", "steps": 1},
        "sampler_continuation": False, "parent_adam_inherited": False, **info})


def _rank_task_gradients(spec, root, runtime, cache, data, engine, parameters,
                         context, groups, events, nonzero):
    local = {key: _zero(parameters) for key in ("return", "even", "odd", "fm_loss")}
    audits, fm_rows, error = [], [], None
    try:
        for position, task_id in enumerate(spec["data"]["gradient_tasks"]):
            if position % 2 != context.rank:
                continue
            if nonzero:
                halves, audit = _score_task(runtime, cache, data, parameters, spec, root,
                                            task_id, groups)
                audits.append(audit)
                for parity, gradient in enumerate(halves):
                    _add(local["even" if parity == 0 else "odd"], gradient)
                    _add(local["return"], gradient)
            runtime.state.train()
            fm_gradient, row = _fm_task(engine, data, runtime, parameters,
                                        events[position], task_id)
            _add(local["fm_loss"], fm_gradient)
            _save(root / "gradient" / "tasks" / f"task_{task_id:03d}_fm_loss.safetensors",
                  fm_gradient)
            fm_rows.append(row)
    except Exception:
        error = traceback.format_exc()
    failures = [None] * context.world_size
    dist.all_gather_object(failures, error)
    if any(failures):
        raise RuntimeError(f"return-credit rank task VJP failed: {failures}")
    return local, audits, fm_rows


def _finish_gradient(spec, root, state, context, runtime, parameters, parent,
                     totals, all_audits, nonzero):
    all_scores = [row for rank in all_audits for row in rank["audits"]]
    all_fm = [row for rank in all_audits for row in rank["fm"]]
    rms = math.sqrt(sum(row["squared_replay_error"] for row in all_scores) /
                    max(1, sum(row["replay_coordinates"] for row in all_scores))) if all_scores else None
    if rms is not None and rms > spec["implementation"]["flow_parity_first5_normalized_rms_max"]:
        raise ValueError("return-credit aggregate native flow replay exceeded .01")
    geometry = {"return_norm": norm(tuple(totals["return"].values())),
                "fm_direction_norm": norm(tuple(totals["fm_loss"].values())),
                "even_norm": norm(tuple(totals["even"].values())),
                "odd_norm": norm(tuple(totals["odd"].values())),
                "return_fm_loss_dot": dot(tuple(totals["return"].values()),
                                           tuple(totals["fm_loss"].values())),
                "even_odd_dot": dot(tuple(totals["even"].values()),
                                     tuple(totals["odd"].values()))}
    if nonzero:
        if geometry["return_norm"] == 0 or geometry["fm_direction_norm"] == 0:
            raise ValueError("nonzero advantages produced an unidentifiable zero Writer direction")
        for arm, direction, sign in (("R", totals["return"], 1),
                                     ("NEG", totals["return"], -1),
                                     ("FM", {name: -value for name, value in totals["fm_loss"].items()}, 1)):
            _candidate(runtime, parameters, parent, direction, root, arm, sign,
                       spec["intervention"]["step_norm"], state["commit"],
                       spec["parent"]["checkpoint"])
    write_json_atomic(root / "gradient" / "completion.json", {
        "schema_version": "ember_return_credit_gradient_v1",
        "implementation_commit": state["commit"], "groups": 32,
        "collection_episodes": 128, "nonzero_advantage_groups": nonzero,
        "all_advantages_zero": nonzero == 0,
        "fm_event_queries": 224, "return_episode_divisor": 128,
        "geometry": geometry, "replay_rms": rms,
        "task_replay_audits": sorted(all_scores, key=lambda row: row["task"]),
        "task_fm_audits": sorted(all_fm, key=lambda row: row["task"]),
        "candidate_states": [] if nonzero == 0 else ["R", "NEG", "FM"],
        "source_frozen": True, "parent_optimizer_inherited": False,
        "world_size": context.world_size})


def _gradient(spec, root, config, state):
    groups, nonzero = _collection(spec, root)
    events = inspect_parent_events(spec)
    if (root / "gradient" / "completion.json").exists() or (root / "gradient").exists():
        raise ValueError("return-credit gradient completed or has a preserved partial attempt")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2:
        raise ValueError("return-credit gradient uses two same-node ranks")
    torch.set_num_threads(4)
    runtime = build_runtime(ASSET_ROOT, config, context.device)
    runtime.policy.requires_grad_(False).eval()
    if any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError("return-credit Source is not frozen")
    weights = load_file(str(Path(spec["parent"]["checkpoint"]) / "ecp.safetensors"), device="cpu")
    runtime.state.load_state_dict(weights, strict=True)
    runtime.state.train()
    parameters = _parameters(runtime)
    parent = {name: parameter.detach().float().clone() for name, parameter in parameters}
    initialize_deferred_process_group(context, rendezvous_root=root / "rendezvous")
    if context.rank == 0:
        (root / "gradient" / "tasks").mkdir(parents=True)
    barrier(context)
    data = WriterTrainingData(ASSET_ROOT, config["data"],
                              camera_view=config["observer"]["camera_view"])
    if data.event_plan() != read_json(Path(spec["parent"]["event_plan"])):
        raise ValueError("return-credit FM event plan changed from the parent")
    cache = VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"]))
    engine_config = deepcopy(config)
    engine_config["data"]["tasks_per_update"] = 8
    engine = SupervisedEngine(runtime, data, cache, context, engine_config)
    local, audits, fm_rows = _rank_task_gradients(
        spec, root, runtime, cache, data, engine, parameters, context,
        groups, events, nonzero)
    totals = {key: _all_reduce(value, parameters) for key, value in local.items()}
    if context.rank == 0:
        for key, values in totals.items():
            _save(root / "gradient" / f"{key}.safetensors", values)
    all_audits = [None] * context.world_size
    dist.all_gather_object(all_audits, {"audits": audits, "fm": fm_rows})
    if context.rank == 0:
        _finish_gradient(spec, root, state, context, runtime, parameters, parent,
                         totals, all_audits, nonzero)
    barrier(context)
    data.close()
    dist.destroy_process_group()


def _pilot_vjp(spec, root, config, state):
    path = root / "collection" / "groups" / "task_002_state_00" / "completion.json"
    group = read_json(path)
    runtime = build_runtime(ASSET_ROOT, config, torch.device("cuda:0"))
    runtime.policy.requires_grad_(False).eval()
    runtime.state.load_state_dict(load_file(str(Path(spec["parent"]["checkpoint"]) /
                                           "ecp.safetensors"), device="cpu"), strict=True)
    runtime.state.train()
    data = WriterTrainingData(ASSET_ROOT, config["data"],
                              camera_view=config["observer"]["camera_view"])
    cache = VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"]))
    teacher = spec["collection"]["conditions"][0]["teacher_demo"]
    condition = cache.condition(2, (teacher,))
    with torch.no_grad():
        lora = {key: value.detach() for key, value in runtime.compile(condition).items()}
    payload = _decision_payload(group["rows"][0])
    if not payload["decisions"]:
        raise ValueError("formal collection pilot has no actual policy decision")
    item = payload["decisions"][0]
    old = item["mu_old_full"][:5, :7].reshape(35)
    cotangent = score_cotangent(item["latent_u"], old, 1., Q=payload["Q"], M=payload["M"])
    gradient, predicted = flow_mean_lora_gradient(runtime.policy, lora, runtime.lora,
        {key: value.to(runtime.device) for key, value in item["processed"].items()},
        item["flow_noise"].unsqueeze(0).to(runtime.device),
        cotangent.unsqueeze(0).to(runtime.device))
    runtime.state.zero_grad(set_to_none=True)
    runtime.state.train()
    generated = runtime.compile(condition)
    torch.autograd.backward(tuple(generated.values()),
        tuple(gradient[name].to(value) for name, value in generated.items()))
    parameters = _parameters(runtime)
    rms = float((predicted[0].float().cpu().reshape(35) - old).square().mean().sqrt())
    active = sum(int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)))
                 for _, parameter in parameters)
    if rms > .01 or active == 0 or any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError("return-credit pilot native flow or Writer VJP failed")
    write_json_atomic(root / "audit" / "pilot_vjp.json", {
        "schema_version": "ember_return_credit_pilot_vjp_v1",
        "implementation_commit": state["commit"], "task": 2, "state": 0,
        "replay_rms": rms, "active_writer_parameters": active,
        "source_frozen": True, "cotangent_norm": float(cotangent.norm()),
        "lora_gradients": len(gradient), "native_flow_steps": 10,
        "technical_probe_advantage": 1., "formal_advantage_not_overridden": True})
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "pilot-vjp", "gradient"))
    args = parser.parse_args()
    spec = authority()
    root, config, state = _gradient_inputs(spec, formal=args.stage != "inspect")
    if args.stage == "inspect":
        print({"tasks": [row["task"] for row in inspect_parent_events(spec)]})
    elif args.stage == "pilot-vjp":
        _pilot_vjp(spec, root, config, state)
    else:
        _gradient(spec, root, config, state)


if __name__ == "__main__":
    main()
