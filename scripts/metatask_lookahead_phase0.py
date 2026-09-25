#!/usr/bin/env python3
"""Run one independent registered phase0 case and its frozen functional reads."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.checkpoint import load_ecp_checkpoint
from ember.lora import validate_lora_state
from ember.pi05_eval.crossed_video_predict import scaled_osc_actions
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import WriterTrainingData
from ember.writer.metatask_lookahead import (
    REPO_ROOT, SPEC_PATH, authority, inspect_cases, preconditioner,
    query_weights, tensor_norm, virtual_alpha, weight_audit,
)
from ember.writer.relational_contract import validate_config
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import VideoConditionCache, autocast, build_runtime
from ember.writer.supervised import SupervisedEngine
from ember.writer.task_execution import condition_assignment
from ember.writer.training import RUN_SCHEMA, STAGE, _optimization, _training_state


def _parameters(state):
    return tuple((name, param) for name, param in state.named_parameters() if param.requires_grad)


def _save_weights(path: Path, parameters) -> None:
    save_file({name: param.detach().float().cpu().contiguous()
               for name, param in parameters}, str(path))


def _copy_weights(parameters, weights) -> None:
    if set(weights) != {name for name, _ in parameters}:
        raise ValueError("weights-only state lost an active Writer parameter")
    with torch.no_grad():
        for name, param in parameters:
            param.copy_(weights[name].to(device=param.device, dtype=param.dtype))


def _collect_error(context, error):
    if context.world_size == 1:
        failures = [error]
    else:
        failures = [None] * context.world_size
        dist.all_gather_object(failures, error)
    if any(failures):
        raise RuntimeError(f"lookahead worker failed before collective: {failures}")


def _rank_norm_agreement(context, values) -> float:
    norm = tensor_norm(values)
    observed = torch.tensor([norm], dtype=torch.float64, device=context.device)
    high, low = observed.clone(), observed.clone()
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    if float(high-low) > 1e-6 * max(1., float(high)):
        raise ValueError("two ranks have different Writer parameter or gradient state")
    return norm


def _cosine(left, right) -> float:
    nleft, nright = tensor_norm(left), tensor_norm(right)
    if nleft == 0 or nright == 0:
        return 0.0
    dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right, strict=True))
    return dot / (nleft * nright)


def _run_group(engine, data, context, optimizer, parameters, draws, group, side,
               assignment, *, gradients):
    optimizer.zero_grad(set_to_none=True)
    local_loss, local_calls, error = 0.0, 0, None
    try:
        for position in assignment[context.rank]:
            draw = draws[position]
            weights = ((1.0,) * 28 if group == "ORIGINAL" else
                       query_weights(group, side, position))
            if gradients:
                row = engine.backward(draw, query_weights=weights)
                local_loss += (row["flow_loss"] / 4 + row["teaching_loss"] / 12)
                local_calls += row["compiled_forward_calls"] + row["teaching_compiled_forward_calls"]
            else:
                row = engine.event_loss(draw, query_weights=weights)
                local_loss += row["main"] + row["extra"]
                local_calls += row["compiled_forward_calls"]
    except Exception:
        error = traceback.format_exc()
    _collect_error(context, error)
    totals = torch.tensor([local_loss, float(local_calls)], device=context.device, dtype=torch.float64)
    if context.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    result = {"loss": float(totals[0]), "physical_compiled_forwards": int(totals[1]),
              "gradient_vjp": gradients}
    if gradients:
        sum_writer_gradients(tuple(param for _, param in parameters), world_size=context.world_size)
        result["gradient"] = tuple(param.grad.detach().clone() for _, param in parameters)
        result["gradient_norm"] = _rank_norm_agreement(context, result["gradient"])
    return result


def _restore_parent(spec, config, data, runtime, optimizer, scheduler, context):
    checkpoint = Path(spec["parent"]["checkpoint"])
    restored = {}
    macro, _ = load_ecp_checkpoint(
        checkpoint=checkpoint, stage=STAGE, context=context, model=runtime.state,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA,
        restored_state=restored)
    if (macro != 1155 or restored["training_state"] != _training_state(config, 1155)
            or restored["sampler_state"]["next_step"] != 1155
            or scheduler.last_epoch != 1155):
        raise ValueError("lookahead case requires the complete original parent state")
    data.restore_sampler(restored["sampler_state"])
    return restored["sampler_state"]


def _candidate(name, gradient, spec, config, data, runtime, optimizer, scheduler,
               context, parameters, parent, output):
    _restore_parent(spec, config, data, runtime, optimizer, scheduler, context)
    optimizer.zero_grad(set_to_none=True)
    for (_, param), value in zip(parameters, gradient, strict=True):
        param.grad = value.detach().clone()
    unclipped = float(torch.nn.utils.clip_grad_norm_(
        [param for _, param in parameters], float(config["optimization"]["grad_clip"]),
        error_if_nonfinite=True))
    lr = float(optimizer.param_groups[0]["lr"])
    decay = float(optimizer.param_groups[0]["weight_decay"])
    optimizer.step()
    scheduler.step()
    if scheduler.last_epoch != 1156 or any(int(optimizer.state[p]["step"]) != 1156
                                          for _, p in parameters):
        raise ValueError("candidate did not apply exactly one parent AdamW update")
    delta = tuple(p.detach().float() - base for (_, p), base in zip(parameters, parent, strict=True))
    _rank_norm_agreement(context, tuple(param.detach() for _, param in parameters))
    gradient_delta = tuple(p.detach().float() - base * (1 - lr * decay)
                           for (_, p), base in zip(parameters, parent, strict=True))
    if context.rank == 0:
        _save_weights(output / "weights" / f"{name}.safetensors", parameters)
    return {"candidate": name, "lr": lr, "grad_norm_before_clip": unclipped,
            "clip_factor": min(1.0, float(config["optimization"]["grad_clip"]) / (unclipped + 1e-6)),
            "total_step_norm": tensor_norm(delta),
            "gradient_displacement_norm": tensor_norm(gradient_delta),
            "optimizer_step": 1156, "scheduler_last_epoch": scheduler.last_epoch}, gradient_delta


def _virtual(parameters, parent, p_values, gradient, alpha, path, *, rank):
    with torch.no_grad():
        for (_, param), base, pre, grad in zip(parameters, parent, p_values, gradient, strict=True):
            param.copy_(base - alpha * pre * grad)
    if rank == 0:
        _save_weights(path, parameters)


def _predict_flow_query(runtime, infer, lora, sliced, trace, task, index, osc_audit,
                        output, final_output, case_id, name):
    seed = task_logical_batch_policy_rng_seed(
        optimization_seed=20260925, task_id=task, task_visit=0,
        demo_indices=[int(trace["action_demos"][index])],
        frame_indices=[int(trace["action_frames"][index])])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, 50, 32), generator=generator,
                        dtype=torch.float32, device="cpu").to(runtime.device)
    flow_input = {key: value for key, value in sliced.items() if key != "action"}
    runtime.policy.reset()
    with infer.activate([lora]):
        predicted = runtime.policy.predict_action_chunk(flow_input, noise=noise, num_steps=10)
    env = runtime.processor.unnormalize_action(predicted)
    if (predicted.shape != (1, 50, 7) or env.shape != (1, 50, 7)
            or not torch.isfinite(predicted).all() or not torch.isfinite(env).all()):
        raise ValueError("real 10-flow output changed shape or became nonfinite")
    normalized = predicted.float().cpu().numpy()
    environment = env.float().cpu().numpy()
    osc = scaled_osc_actions(environment, osc_audit)
    expert = sliced["action"].float().cpu().numpy()
    path = output / "flow" / f"task_{task:03d}" / f"query_{index:02d}_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, normalized=normalized[0], environment=environment[0],
                            osc_first5=osc[0], expert_normalized=expert[0],
                            policy_noise=noise[0].float().cpu().numpy())
    return {"case": case_id, "state": name, "task": task,
            "query_index": index, "query_demo": int(trace["action_demos"][index]),
            "query_frame": int(trace["action_frames"][index]), "noise_seed": seed,
            "path": str(final_output / path.relative_to(output)), "bytes": path.stat().st_size,
            "expert_action_used_as_input": False, "environment_steps": 0}


def _readout_case(spec, config, case_id, data, runtime, context, parameters, parent, output):
    states = spec["readouts"]["parameter_states"]
    tasks = spec["cases"][case_id - 1]["tasks"]
    ordered = [(position, int(task)) for position, task in enumerate(tasks)
               if position % context.world_size == context.rank]
    audit_path = Path("/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924/query_index.jsonl")
    with audit_path.open(encoding="utf-8") as handle:
        osc_audit = json.loads(handle.readline())["osc_channel_audit"]
    runtime.state.eval()
    runtime.policy.requires_grad_(False).eval()
    if any(param.requires_grad for param in runtime.policy.parameters()):
        raise ValueError("Source policy became trainable during the readout")
    infer = BatchedLoRAInference(runtime.policy, runtime.lora)
    final_output = output.parent / f"case_{case_id:02d}"
    fm_rows, flow_rows = [], []
    try:
        with torch.inference_mode(), autocast(context.device):
            for position, task in ordered:
                raw, trace = data.diagnostic_batch(task, seed=20260925, count=16, teacher_demo=46)
                if (not set(trace["action_demos"]) <= {47, 48, 49}
                        or len(trace["action_demos"]) != 16):
                    raise ValueError("independent readout touched training or teacher actions")
                batch = runtime.processor.training_batch(raw)
                condition = runtime.prepare(*data.load_videos(task, (46,)),
                                            data.tasks[task].authority.language)
                for name in states:
                    if name != "P":
                        saved = load_file(str(output / "weights" / f"{name}.safetensors"),
                                          device="cpu")
                        _copy_weights(parameters, saved)
                    else:
                        with torch.no_grad():
                            for (_, param), value in zip(parameters, parent, strict=True):
                                param.copy_(value)
                    lora = runtime.compile(condition)
                    validate_lora_state(lora, runtime.lora)
                    bank = output / "banks" / name / f"task_{task:03d}.safetensors"
                    bank.parent.mkdir(parents=True, exist_ok=True)
                    save_file({key: value.detach().float().cpu().contiguous()
                               for key, value in lora.items()}, str(bank))
                    for index in range(16):
                        sliced = {key: value[index:index + 1] if isinstance(value, torch.Tensor)
                                  and value.ndim and len(value) == 16 else value
                                  for key, value in batch.items()}
                        credit = paired_functional_credit(
                            runtime.policy, lora, runtime.lora, sliced,
                            seed=trace["policy_rng_seed"], device=context.device,
                            random_batch=16, offset=index, microbatch=1,
                            condition_weight=1.0, backward=False)
                        if credit["lora_cotangent"] or not math.isfinite(credit["flow_loss"]):
                            raise ValueError("nonfinite or differentiating frozen FM readout")
                        fm_rows.append({"case": case_id, "state": name, "task": task,
                                        "query_index": index, "teacher_demo": 46,
                                        "query_demo": int(trace["action_demos"][index]),
                                        "query_frame": int(trace["action_frames"][index]),
                                        "logical_policy_rng_seed": int(trace["policy_rng_seed"]),
                                        "fm_loss": credit["flow_loss"],
                                        "bank": str(final_output / bank.relative_to(output)),
                                        "no_grad": True})
                        if name not in ("P", "BASE", "TASK", "MIX") or index >= 4:
                            continue
                        flow_rows.append(_predict_flow_query(
                            runtime, infer, lora, sliced, trace, task, index, osc_audit,
                            output, final_output, case_id, name))
    finally:
        infer.close()
    for label, rows in (("fm", fm_rows), ("flow", flow_rows)):
        path = output / f"{label}_rank_{context.rank}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(fm_rows), len(flow_rows)


def _compute_updates(spec, config, data, runtime, optimizer, scheduler, context,
                     engine, parameters, draws, assignment, partial):
    sampler = _restore_parent(spec, config, data, runtime, optimizer, scheduler, context)
    parent = tuple(param.detach().float().clone() for _, param in parameters)
    parent_norm = _rank_norm_agreement(context, parent)
    p_values = preconditioner(tuple(param for _, param in parameters), optimizer, step=1155)
    original = _run_group(engine, data, context, optimizer, parameters, draws,
                          "ORIGINAL", "A", assignment, gradients=True)
    groups = {group: {side: _run_group(engine, data, context, optimizer, parameters,
                      draws, group, side, assignment, gradients=True)
                      for side in ("A", "B")} for group in ("TASK", "MIX")}
    g0 = original["gradient"]
    identities = {}
    for group in groups:
        avg = tuple((a+b)*0.5 for a,b in zip(
            groups[group]["A"]["gradient"], groups[group]["B"]["gradient"], strict=True))
        identities[group] = {
            "gradient_sum_residual_norm": tensor_norm(tuple(x-y for x,y in zip(avg,g0,strict=True))),
            "loss_sum_residual": (groups[group]["A"]["loss"]+
                                  groups[group]["B"]["loss"])/2-original["loss"]}
    base, d0 = _candidate("BASE", g0, spec, config, data, runtime, optimizer,
                          scheduler, context, parameters, parent, partial)
    candidate_rows = [base]
    candidate_deltas = {"BASE": d0}
    virtual_rows = []
    forward_counts = original["physical_compiled_forwards"] + sum(
        row["physical_compiled_forwards"] for group in groups.values() for row in group.values())
    for group in ("TASK", "MIX"):
        alpha, denominator, zero = virtual_alpha(
            d0, p_values, groups[group]["A"]["gradient"], groups[group]["B"]["gradient"])
        opposing = []
        for side, opposite in (("A", "B"), ("B", "A")):
            _restore_parent(spec, config, data, runtime, optimizer, scheduler, context)
            path = partial / "weights" / f"{group}_{side}.safetensors"
            _virtual(parameters, parent, p_values, groups[group][side]["gradient"],
                     alpha, path, rank=context.rank)
            virtual_delta = tuple(param.detach().float()-base for (_, param), base in
                                  zip(parameters, parent, strict=True))
            virtual_step_norm = _rank_norm_agreement(context, virtual_delta)
            for reading_side in ("A", "B"):
                reading = _run_group(engine, data, context, optimizer, parameters, draws,
                    group, reading_side, assignment, gradients=reading_side == opposite)
                forward_counts += reading["physical_compiled_forwards"]
                virtual_rows.append({"state": f"{group}_{side}", "read_group": reading_side,
                                     "loss": reading["loss"], "virtual_step_norm": virtual_step_norm,
                                     "gradient_norm": reading.get("gradient_norm"),
                                     "gradient_vjp": reading_side == opposite})
                if reading_side == opposite:
                    opposing.append(reading["gradient"])
        look = tuple((a+b)*0.5 for a,b in zip(opposing[0], opposing[1], strict=True))
        row, delta = _candidate(group, look, spec, config, data, runtime, optimizer,
                                scheduler, context, parameters, parent, partial)
        row.update(alpha=alpha, alpha_denominator=denominator,
                   virtual_zero_rule=zero, lookahead_grad_norm=tensor_norm(look))
        candidate_rows.append(row)
        candidate_deltas[group] = delta
    return {"sampler": sampler, "parent": parent, "parent_norm": parent_norm,
            "original": original, "groups": groups, "identities": identities,
            "candidate_rows": candidate_rows, "candidate_deltas": candidate_deltas,
            "virtual_rows": virtual_rows, "forward_counts": forward_counts}


def _publish_case(spec, case_id, state, context, assignment, partial, output, started, result):
    fm_rows, flow_rows = [], []
    for rank in range(2):
        for label, target in (("fm", fm_rows), ("flow", flow_rows)):
            target.extend(json.loads(line) for line in
                (partial / f"{label}_rank_{rank}.jsonl").read_text().splitlines())
    if (len(fm_rows) != 512 or len(flow_rows) != 64
            or len({(r["state"], r["task"], r["query_index"]) for r in fm_rows}) != 512
            or len({(r["state"], r["task"], r["query_index"]) for r in flow_rows}) != 64):
        raise ValueError("one case lost registered FM or true-flow rows")
    groups = result["groups"]
    deltas = result["candidate_deltas"]
    write_json_atomic(partial / "case_contract.json", {
        "schema_version": "ember_metatask_lookahead_case_v1", "case": case_id,
        "study_spec": str(REPO_ROOT / SPEC_PATH), "implementation_commit": state["commit"],
        "parent_checkpoint": spec["parent"]["checkpoint"], "parent_macro": 1155,
        "sampler_next_step_at_parent": result["sampler"]["next_step"],
        "source_macro": spec["cases"][case_id-1]["source_macro"],
        "event_indices": spec["cases"][case_id-1]["event_indices"],
        "event_assignment": assignment,
        "original_loss": result["original"]["loss"],
        "original_grad_norm": result["original"]["gradient_norm"],
        "parent_parameter_norm": result["parent_norm"],
        "parent_group_losses": {g: {s: groups[g][s]["loss"] for s in ("A","B")}
                                for g in ("TASK","MIX")},
        "parent_group_gradient_norms": {
            g: {s: groups[g][s]["gradient_norm"] for s in ("A", "B")}
            for g in ("TASK", "MIX")},
        "candidate_gradient_displacement_cosines": {
            "TASK_BASE": _cosine(deltas["TASK"], deltas["BASE"]),
            "MIX_BASE": _cosine(deltas["MIX"], deltas["BASE"]),
            "TASK_MIX": _cosine(deltas["TASK"], deltas["MIX"]),
        },
        "identities": result["identities"], "virtual_rows": result["virtual_rows"],
        "candidate_rows": result["candidate_rows"],
        "gradient_forward_calls": result["forward_counts"],
        "fm_rows": 512, "true_flow_rows": 64, "no_environment_steps": True,
        "weights_are_diagnostic_not_resume": True,
        "elapsed_seconds": time.monotonic()-started,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(context.device)/2**30,
    })
    os.replace(partial, output)


def run(case_id: int) -> None:
    spec = authority()
    config = validate_config(read_json(REPO_ROOT / spec["parent"]["config"]))
    cases = inspect_cases(spec, config)
    weight_audit()
    if case_id not in range(1, 8):
        raise ValueError("case is outside seven registered independent source macros")
    state = git_state(REPO_ROOT)
    if (state["branch"] or state["dirty_paths"]
            or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal lookahead requires a clean pushed detached implementation")
    root = Path(spec["resources"]["study_root"])
    output = root / f"case_{case_id:02d}"
    partial = root / f".case_{case_id:02d}.partial"
    if output.exists() or partial.exists():
        raise ValueError("case already completed or has a preserved failed attempt")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 2:
        raise ValueError("parent and lookahead case require two ranks on one node")
    torch.set_num_threads(4)
    seed_everything(int(config["optimization"]["seed"]) - context.rank, context)
    data = WriterTrainingData(REPO_ROOT.parent / "EMBER", config["data"],
                              camera_view=config["observer"]["camera_view"])
    if data.event_plan() != read_json(Path(spec["parent"]["event_plan"])):
        raise ValueError("regenerated original task/query/noise events differ from parent")
    runtime = build_runtime(REPO_ROOT.parent / "EMBER", config, context.device)
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise ValueError("lookahead source policy must remain frozen")
    runtime.state.train()
    optimizer, scheduler = _optimization(runtime.state, config)
    initialize_deferred_process_group(context, rendezvous_root=root / "rendezvous")
    if context.rank == 0:
        partial.mkdir(parents=True)
        (partial / "weights").mkdir()
    barrier(context)
    engine = SupervisedEngine(runtime, data,
        VideoConditionCache(runtime, data, int(config["runtime"]["raw_video_cache_bytes"])),
        context, config)
    parameters = _parameters(runtime.state)
    draws = cases[case_id - 1]
    assignment = condition_assignment(range(4),
        {position: int(draw["frames"]) for position, draw in enumerate(draws)}, world_size=2)
    started = time.monotonic()
    try:
        result = _compute_updates(spec, config, data, runtime, optimizer, scheduler,
                                  context, engine, parameters, draws, assignment, partial)
        _restore_parent(spec, config, data, runtime, optimizer, scheduler, context)
        barrier(context)
        _readout_case(spec, config, case_id, data, runtime, context, parameters,
                      result["parent"], partial)
        barrier(context)
        if context.rank == 0:
            _publish_case(spec, case_id, state, context, assignment, partial, output,
                          started, result)
        barrier(context)
    finally:
        data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    spec = authority()
    config = validate_config(read_json(REPO_ROOT / spec["parent"]["config"]))
    cases = inspect_cases(spec, config)
    if args.inspect:
        print(json.dumps({"cases": len(cases), "tasks": [draw["task"] for group in cases
                           for draw in group], "weights": weight_audit()}))
        return
    if args.case is None:
        parser.error("--case is required unless --inspect is used")
    run(args.case)


if __name__ == "__main__":
    main()
