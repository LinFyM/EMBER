"""Four independent train episodes and reward-independent path reservoirs.

Only own execution observations enter the score records. Teacher conditioning
is handled by FormalData/Runtime; this module never reads a teacher HDF5.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ember.lora import copy_task_lora_state_
from ember.pi05_eval.episode import start_fixed_episode, finish_episode_row, update_stage_predicates
from ember.pi05_eval.trajectory_capture import record_replan, record_passive_step
from ember.pi05_processing import libero_policy_input
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.runtime import autocast

from .denoising_policy import sample_action_chunk


PURPOSE = {name: index for index, name in enumerate(
    ("init", "env", "initial_noise", "SDE", "reservoir", "steps"), 1)}


def episode_seed(macro: int, task: int, replica: int, purpose: str, replan: int = 0) -> int:
    """Stateless independent streams; physical rank/worker is deliberately absent."""
    values = [20261006, macro, task, replica, PURPOSE[purpose], replan]
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def raw_record(raw: dict) -> dict:
    return {key: (value.mul(255).round().to(torch.uint8) if key.startswith("observation.images.")
                  else value.detach().cpu().clone()) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()}


def restore_raw(record: dict) -> dict:
    return {key: value.float().div(255) if key.startswith("observation.images.") else value
            for key, value in record.items()}


def _start_episode(env, init_states, task, environment, root, macro, global_task, replica):
    output = root / f"replica_{replica}"
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(environment, output_dir=str(output))
    contract["rng"] = {"inference_seed": episode_seed(macro, global_task, replica, "env")}
    contract["diagnostic_stage_predicates"] = {"full_conditions_only": False}
    contract["diagnostic_occupancy_capture"] = {
        "mode": "compact", "full_conditions": [], "trajectory_root": str(output / "trajectory"),
        "passive_trace": {"trace_root": str(output / "continuous")}}
    initial = int(np.random.default_rng(episode_seed(macro, global_task, replica, "init")).integers(32))
    try:
        slot = start_fixed_episode(env=env, init_state_id=initial, init_states=init_states,
            task=task, contract=contract, root_seed=contract["rng"]["inference_seed"],
            dummy=np.asarray(environment["environment"]["dummy_action"], dtype=np.float32),
            task_adapter=None, capture_level="compact")
    except Exception:
        write_json_atomic(output / "initialization_failure.json", {"init_state_id": initial,
            "env_seed": contract["rng"]["inference_seed"], "error": traceback.format_exc()})
        raise
    slot.update(replica=replica, episode_done=False, finished=False, denoising_noise_seeds=[])
    return slot, contract, np.random.default_rng(episode_seed(macro, global_task, replica, "reservoir"))


def _save_rows(slots, contracts, reservoirs, rngs, task, event, macro, started):
    errors = []
    for slot, contract, reservoir, rng in zip(slots, contracts, reservoirs, rngs, strict=True):
        row = {"macro": macro, "global_task": event["task"], "teacher_demo": event["teacher_demo"],
               "replica": slot["replica"], "completed": slot["finished"], "success": slot["episode_done"],
               "steps": slot["steps"], "init_state_id": slot["init_state_id"],
               "actual_replans": slot["replan_index"], "saved_decisions": len(reservoir),
               "SDE_seeds": slot["denoising_noise_seeds"], "parameter_version": macro - 1,
               "init_seed": episode_seed(macro, event["task"], slot["replica"], "init")}
        record_path = Path(contract["output_dir"]) / "score_records.pt"
        # Preserve the score path before a potentially failing trace serializer.
        torch.save({"row": row, "decisions": sorted(reservoir, key=lambda item: item["replan"]),
                    "reservoir_RNG": rng.bit_generator.state}, record_path)
        row["score_records"] = {"path": str(record_path), "bytes": record_path.stat().st_size}
        try:
            row.update(finish_episode_row(slot=slot, task=task, contract=contract,
                                         task_adapter=None, worker_started=started))
        except Exception:
            row["capture_error"] = traceback.format_exc()
            errors.append(row["capture_error"])
        write_json_atomic(Path(contract["output_dir"]) / "row.json", row)
    return errors


def collect_group(runtime, pool, task: dict, environment: dict, event: dict,
                  state: dict, root: Path, macro: int) -> dict:
    """One current compiled function, four iid init0..31 episodes, no extra case."""
    started = time.monotonic()
    task = dict(task, init_state_ids=list(range(32)))
    envs, init_states = pool.switch(task)
    if len(envs) != 4:
        raise ValueError("a training group requires its four independent environments")
    copy_task_lora_state_(runtime.policy, state, runtime.lora)
    slots, contracts, reservoirs, rngs = [], [], [], []
    try:
        for replica, env in enumerate(envs):
            slot, contract, rng = _start_episode(env, init_states, task, environment, root, macro,
                                                event["task"], replica)
            slots.append(slot)
            contracts.append(contract)
            reservoirs.append([])
            rngs.append(rng)
        while any(not slot["finished"] for slot in slots):
            planning = [slot for slot in slots if not slot["finished"] and not slot["action_plan"]]
            if planning:
                raw = [libero_policy_input(slot["obs"], task["language"]) for slot in planning]
                processed = [runtime.processor(value) for value in raw]
                batch = {key: torch.cat([item[key] for item in processed]) for key in processed[0]}
                initial_seeds = [episode_seed(macro, event["task"], slot["replica"],
                                             "initial_noise", slot["replan_index"]) for slot in planning]
                sde_seeds = [episode_seed(macro, event["task"], slot["replica"],
                                         "SDE", slot["replan_index"]) for slot in planning]
                noise = torch.stack([torch.randn(50, 32, generator=torch.Generator().manual_seed(seed))
                                     for seed in initial_seeds]).to(runtime.device)
                with torch.no_grad(), autocast(runtime.device):
                    result = sample_action_chunk(runtime.policy, batch, noise=noise,
                                                 seeds=sde_seeds, return_path=True)
                    physical = runtime.processor.unnormalize_action(result.chunks[:, :5]).float().cpu().numpy()
                if not torch.isfinite(result.chunks).all() or not np.isfinite(physical).all():
                    raise ValueError("nonfinite actual SDE command")
                for index, slot in enumerate(planning):
                    replica, decision = slot["replica"], slot["replan_index"]
                    chosen = sorted(np.random.default_rng(episode_seed(macro, event["task"], replica,
                                                                      "steps", decision)).choice(10, 2, replace=False).tolist())
                    record = {"replan": decision, "control_step": slot["steps"],
                              "observation": raw_record(raw[index]), "initial_noise_seed": initial_seeds[index],
                              "SDE_seed": sde_seeds[index], "selected_steps": chosen,
                              "transitions": [{"step": k, "tau": result.transitions[k].tau,
                                  **{name: getattr(result.transitions[k], name)[index].detach().float().cpu().clone()
                                     for name in ("z", "m", "z_next")}} for k in chosen]}
                    q = decision + 1
                    if q <= 16:
                        reservoirs[replica].append(record)
                    else:
                        replacement = int(rngs[replica].integers(q))
                        if replacement < 16:
                            reservoirs[replica][replacement] = record
                    slot["policy_noise_seeds"].append(initial_seeds[index])
                    slot["denoising_noise_seeds"].append(sde_seeds[index])
                    record_replan(slot, raw[index], processed[index], result.chunks[index], physical[index])
                    slot["action_plan"].extend(physical[index])
                    slot["replan_index"] += 1
            for env, slot, contract in zip(envs, slots, contracts, strict=True):
                if slot["finished"]:
                    continue
                action = np.asarray(slot["action_plan"].popleft(), dtype=np.float32)
                slot["obs"], _, done, _ = env.step(action)
                slot["steps"] += 1
                update_stage_predicates(env, slot)
                record_passive_step(env, slot, action, contract["diagnostic_occupancy_capture"])
                slot["episode_done"] = bool(done)
                slot["finished"] = bool(done) or slot["steps"] >= int(task["horizon"])
    finally:
        original_error = sys.exc_info()[1]
        errors = _save_rows(slots, contracts, reservoirs, rngs, task, event, macro, started)
        if errors and original_error is None:
            raise RuntimeError("own trace serialization failed: " + "\n".join(errors))
    rows = [json.loads((Path(contract["output_dir"]) / "row.json").read_text())
            for contract in contracts]
    returns = [int(row["success"]) for row in rows]
    advantages = [reward - (sum(returns) - reward) / 3 for reward in returns]
    return {"event": {key: event[key] for key in ("task", "teacher_demo", "visit")},
            "parameter_version": macro - 1, "rows": rows, "returns": returns, "advantages": advantages,
            "nonzero_LOO": any(advantages), "collection_seconds": time.monotonic() - started}
