"""Condition-scoped adaptive compilation and strict persistent-worker readout.

The existing PI05 SQLite queue owns claims. One job adapts one teaching
condition, then freezes its LoRA across all registered final states. Initial
and null arms are separate raw sidecars, never additional adaptation chains.
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from ember.pi05_eval_queue import (EvaluationShard, claim_next, complete_job, completed_jobs,
                                   fail_job, initialize_queue, publish_json_exclusive,
                                   queue_summary, read_json_with_size)
from ember.pi05_eval_results import paired_success_comparison
from .contract import (ASSET_ROOT, RUN_ROOT, SCHEMA, MT_RESULTS, T_RESULTS, _LEARNING_CONTRACT, formal_environment,
                       learning_environment, formal400_mapping, panel_contract)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f"immutable evaluation record changed: {path}")
    else:
        publish_json_exclusive(path, value)


def prepare(output: Path, checkpoint: Path, stage: str = "meta27", asset_root: Path = ASSET_ROOT,
            *, recover_claims: bool = False, retry_failed: bool = False) -> Path:
    """Freeze the panel before scores; queue recovery requires exited workers."""
    if stage not in {"meta27", "meta54", "formal"}:
        raise ValueError("only the preregistered meta27/meta54/formal panels are allowed")
    output, checkpoint, asset_root = Path(output).resolve(), Path(checkpoint).resolve(), Path(asset_root).resolve()
    if not checkpoint.is_dir():
        raise ValueError("evaluation requires an existing checkpoint directory")
    formal = stage == "formal"
    conditions = list(formal400_mapping(asset_root=asset_root) if formal
                      else panel_contract(asset_root=asset_root)["conditions"])
    environment = (formal_environment if formal else learning_environment)(asset_root=asset_root)
    environment["parallel"] = {"envs_per_replica": 1 if formal else 3}
    adaptation_environment = deepcopy(environment)
    adaptation_environment.pop("operator_read_write_scene", None)
    if not formal:
        environment["operator_read_write_scene"] = read_json(_LEARNING_CONTRACT)["operator_read_write_scene"]
    arms = ["end"] if formal else ["initial", "end"] + (["null"] if stage == "meta54" else [])
    horizon = environment["environment"]["horizons"]
    contract = {"schema_version": SCHEMA, "stage": stage, "checkpoint": str(checkpoint),
                "asset_root": str(asset_root), "environment_contract": environment,
                "adaptation_environment_contract": adaptation_environment,
                "arms": arms, "conditions": conditions, "adaptation_step_budget": 1024,
                "policy_seed_root": 7, "expected_rows_per_arm": 400 if formal else 48,
                "references": {"MT": str(MT_RESULTS), "T2340": str(T_RESULTS)} if formal else {}}
    _validate_panel(contract)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_contract.json"
    immutable(path, contract)
    shards = tuple(EvaluationShard(
        job_id=f"{stage}-{condition['condition_id']}", ordinal=index,
        suite=condition["suite"], task_id=int(condition["suite_task_id"]),
        horizon=int(horizon[condition["suite"]]),
        init_state_ids=tuple(condition["final_state_ids"]),
        estimated_cost=1024 + len(arms) * len(condition["final_state_ids"]) * int(horizon[condition["suite"]]))
        for index, condition in enumerate(conditions))
    initialize_queue(output / "queue.sqlite3", shards, contract_reference=str(path),
                     recover_claims=recover_claims, retry_failed=retry_failed)
    return path


def _validate_panel(contract: dict) -> None:
    conditions, formal = contract["conditions"], contract["stage"] == "formal"
    expected_count = 400 if formal else 16
    if len(conditions) != expected_count or len({c["condition_id"] for c in conditions}) != expected_count:
        raise ValueError("condition panel has missing or duplicate teaching conditions")
    tasks = defaultdict(list)
    for c in conditions:
        if not c["condition_id"] or "/" in c["condition_id"] or ".." in c["condition_id"]:
            raise ValueError("condition identity is not a safe queue/artifact component")
        states = c["final_state_ids"]
        if len(states) != (1 if formal else 3) or len(set(states)) != len(states) or not all(0 <= s < 50 for s in states):
            raise ValueError("condition final states differ from the fixed panel")
        if not 0 <= c["teacher_demo"] < 50:
            raise ValueError("teacher ordinal is outside the legal 50-video pool")
        tasks[c["task_id"]].append(c)
    if len(tasks) != 8:
        raise ValueError("readout must cover the registered eight tasks")
    for rows in tasks.values():
        if formal:
            if ({c["teacher_demo"] for c in rows} != set(range(50))
                    or {c["final_state_ids"][0] for c in rows} != set(range(50))
                    or len(rows) != 50):
                raise ValueError("formal task/video/state conditions must be without replacement")
        elif len(rows) != 2 or len({c["teacher_demo"] for c in rows}) != 2:
            raise ValueError("train readout requires two distinct videos per task")


def _validate_rows(rows: list[dict], condition: dict, arm: str, task: dict) -> None:
    if len(rows) != len(condition["final_state_ids"]):
        raise ValueError("raw arm rows differ from the condition shard state count")
    if {row["init_state_id"] for row in rows} != set(condition["final_state_ids"]):
        raise ValueError("raw arm changed final state coverage")
    for row in rows:
        expected = {"suite": condition["suite"], "task_id": condition["suite_task_id"],
                    "condition_id": condition["condition_id"], "teacher_demo": condition["teacher_demo"],
                    "arm": arm, "language": task["language"], "split_role": task["split_role"],
                    "env_seed": 7, "policy_seed_root": 7}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("raw row changed condition, language, final RNG or task authority")
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("policy_noise_seeds"), list):
            raise ValueError("raw final row must preserve success and actual policy-noise evidence")
        if "video_ordinal" in condition and row.get("video_ordinal") != condition["video_ordinal"]:
            raise ValueError("raw formal row changed the frozen teacher schedule")


def _task(contract: dict, condition: dict) -> dict:
    tasks = contract["environment_contract"]["tasks"]
    matches = [task for task in tasks if task["suite"] == condition["suite"]
               and int(task["task_id"]) == int(condition["suite_task_id"])]
    if len(matches) != 1:
        raise ValueError("condition does not identify exactly one registered environment task")
    return matches[0]


def execute_condition(runtime, runner, contract: dict, condition: dict, destination: Path) -> dict:
    """Adapt once; initial/end/null evaluate the same excluded final states."""
    import torch
    from safetensors.torch import save_file

    started = time.time()
    task = _task(contract, condition)
    role = "validation" if contract["stage"] == "formal" else "train"
    teacher = runtime.teacher(condition["task_id"], condition["teacher_demo"], role=role)
    runner.contract = contract["adaptation_environment_contract"]
    chain = runner.adapt(task, teacher, condition_seed=condition["seed"],
                         excluded_states=tuple(condition["final_state_ids"]))
    if len(chain.states) != len(chain.endpoints) + 1:
        raise ValueError("chain state endpoints do not match actual shared revisions")
    destination.mkdir(parents=True, exist_ok=False)
    with gzip.open(destination / "chain.pt.gz", "xb", compresslevel=1) as stream:
        torch.save(chain.to_record(), stream)
    states = {"initial": chain.states[0], "end": chain.states[-1]}
    if "null" in contract["arms"]:
        states["null"] = runtime.replay(teacher, chain, masked=True)[-1]
    runner.contract = contract["environment_contract"]
    rows, weights, saved = {}, {}, {}
    for arm in contract["arms"]:
        lora = states[arm]
        if id(lora) not in saved:
            name = arm + ".safetensors"
            save_file({key: value.detach().cpu().contiguous() for key, value in lora.items()}, str(destination / name))
            saved[id(lora)] = name
        weights[arm] = saved[id(lora)]
        state_ids = condition["final_state_ids"]
        raw = ([runner.final(task, state_ids[0], lora, noise_root=7)] if contract["stage"] == "formal"
               else runner.final_many(task, state_ids, lora, noise_root=7))
        rows[arm] = [{**row, "condition_id": condition["condition_id"],
                      "teacher_demo": condition["teacher_demo"], "arm": arm,
                      **({"video_ordinal": condition["video_ordinal"]} if "video_ordinal" in condition else {})}
                     for row in raw]
        _validate_rows(rows[arm], condition, arm, task)
    return {"rows": rows, "metrics": chain.metrics, "actual_J": len(chain.endpoints),
            "weights": weights, "chain": "chain.pt.gz", "wall_seconds": time.time() - started}


def worker(args) -> dict:
    """One visible GPU, no NCCL, persistent policy and real environment pool."""
    output = Path(args.output).resolve()
    contract = read_json(output / "evaluation_contract.json")
    if Path(args.checkpoint).resolve() != Path(contract["checkpoint"]):
        raise ValueError("worker checkpoint differs from the frozen panel")
    gpu = int(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("evaluation worker must expose exactly its assigned physical GPU")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(gpu))
    import torch
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .runtime import Runtime
    from .interaction import Runner
    from .run import check_budget

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("evaluation worker requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if affinity is None:
        raise ValueError("evaluation requires verified GPU-local NUMA affinity")
    torch.set_grad_enabled(False)
    worker_id = f"{socket.gethostname()}-{gpu}-{os.getpid()}"
    started, claimed, completed, runner = time.time(), None, [], None
    status, error = "failed", None
    try:
        runtime = Runtime(Path(contract["asset_root"]), getattr(args, "device", "cuda:0"),
                          frame_chunk=args.frame_chunk, decoder_chunk=args.decoder_chunk,
                          experience_chunk=getattr(args, "experience_chunk", 16))
        runtime.load_checkpoint(Path(contract["checkpoint"]))
        runtime.compiler.eval()
        runner = Runner(runtime, contract["environment_contract"], gpu)
        by_id = {f"{contract['stage']}-{c['condition_id']}": c for c in contract["conditions"]}
        while True:
            check_budget(RUN_ROOT)
            claimed = claim_next(output / "queue.sqlite3", worker_id=worker_id, physical_gpu=gpu)
            if claimed is None:
                break
            condition = by_id[claimed.shard.job_id]
            relative = Path("conditions") / claimed.shard.job_id / claimed.claim_token
            result = execute_condition(runtime, runner, contract, condition, output / relative)
            arm_paths = {}
            for arm, rows in result.pop("rows").items():
                path = Path("shards") / f"{claimed.shard.job_id}-{claimed.claim_token}-{arm}.json"
                publish_json_exclusive(output / path, {"condition": condition, "arm": arm, "rows": rows})
                arm_paths[arm] = str(path)
            primary = read_json(output / arm_paths["end"])
            primary.update({"arm_paths": arm_paths, "artifact_root": str(relative), **result})
            # A distinct primary record keeps each arm's row count honest.
            path = Path("shards") / f"{claimed.shard.job_id}-{claimed.claim_token}.json"
            size = publish_json_exclusive(output / path, primary)
            complete_job(output / "queue.sqlite3", job_id=claimed.shard.job_id, worker_id=worker_id,
                         claim_token=claimed.claim_token, rows_path=str(path), rows_bytes=size,
                         row_count=len(primary["rows"]), successes=sum(r["success"] for r in primary["rows"]))
            completed.append(claimed.shard.job_id)
            claimed = None
        status = "complete"
    except BaseException:
        error = traceback.format_exc()
        if claimed is not None:
            fail_job(output / "queue.sqlite3", job_id=claimed.shard.job_id, worker_id=worker_id,
                     claim_token=claimed.claim_token, error=error)
        raise
    finally:
        if runner is not None:
            runner.close()
        receipt = {"worker_id": worker_id, "host": socket.gethostname(), "physical_gpu": gpu,
                   "started_unix": started, "finished_unix": time.time(), "status": status,
                   "cpu_affinity": list(affinity), "completed_jobs": completed, "error": error}
        immutable(output / "workers" / f"{worker_id}.json", receipt)
    return receipt


def _indexed(rows: list[dict]) -> dict:
    result = {(r["suite"], r["task_id"], r["condition_id"], r["init_state_id"]): r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("paired train rows are empty or duplicated")
    return result


def compare_train(reference: list[dict], candidate: list[dict]) -> dict:
    left, right = _indexed(reference), _indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("train comparison changed condition/state coverage")
    for key, before in left.items():
        after = right[key]
        fields = ("language", "env_seed", "policy_seed_root", "split_role", "teacher_demo")
        common = min(len(before["policy_noise_seeds"]), len(after["policy_noise_seeds"]))
        if any(before.get(f) != after.get(f) for f in fields) or common < 1 or before["policy_noise_seeds"][:common] != after["policy_noise_seeds"][:common]:
            raise ValueError("train comparison changed language/video/final RNG pairing")
    def counts(keys):
        before, after = {k for k in keys if left[k]["success"]}, {k for k in keys if right[k]["success"]}
        return {"rows": len(keys), "reference_successes": len(before), "candidate_successes": len(after),
                "retained": len(before & after), "gained": len(after - before), "lost": len(before - after),
                "churn_count": len(before ^ after), "churn_fraction": len(before ^ after) / len(keys),
                "success_set_jaccard": len(before & after) / len(before | after) if before | after else None,
                "retained_keys": sorted(before & after), "gained_keys": sorted(after - before), "lost_keys": sorted(before - after)}
    tasks = sorted({k[:2] for k in left})
    per_task = [{"suite": suite, "task_id": task, **counts([k for k in left if k[:2] == (suite, task)])}
                for suite, task in tasks]
    return {**counts(list(left)), "per_task": per_task,
            "per_suite": [{"suite": suite, **counts([k for k in left if k[0] == suite])}
                          for suite in sorted({k[0] for k in left})],
            "reference_breadth": sum(p["reference_successes"] > 0 for p in per_task),
            "candidate_breadth": sum(p["candidate_successes"] > 0 for p in per_task)}


def _summary(rows: list[dict]) -> dict:
    tasks, suites = defaultdict(list), defaultdict(list)
    for row in rows:
        tasks[(row["suite"], row["task_id"])].append(row)
        suites[row["suite"]].append(row)
    def counts(values):
        return {"row_count": len(values), "successes": sum(r["success"] for r in values)}
    return {**counts(rows), "breadth": sum(any(r["success"] for r in values) for values in tasks.values()),
            "per_task": [{"suite": k[0], "task_id": k[1], **counts(v)} for k, v in sorted(tasks.items())],
            "per_suite": [{"suite": k, **counts(v)} for k, v in sorted(suites.items())]}


def compare_formal(reference: dict, candidate: dict) -> dict:
    result = paired_success_comparison(reference, candidate)
    key = lambda row: (row["suite"], row["task_id"], row["init_state_id"])
    previous = {key(row): row for row in reference["rows"]}
    for row in candidate["rows"]:
        before = previous[key(row)]
        if row.get("scene_reference") != before.get("scene_reference"):
            raise ValueError("formal comparison changed the actual canonical scene reference")
        teacher = before.get("operator_read_write_lora")
        if teacher and any(row.get(field) != teacher[field] for field in ("condition_id", "teacher_demo", "video_ordinal")):
            raise ValueError("formal comparison changed the original T teaching condition")
    return result


def _cost(output: Path, conditions: list[dict]) -> dict:
    intervals, receipts = defaultdict(list), []
    for path in sorted((output / "workers").glob("*.json")):
        row = read_json(path)
        receipts.append(row)
        intervals[(row["host"], row["physical_gpu"])].append((row["started_unix"], row["finished_unix"]))
    allocated = 0.0
    for periods in intervals.values():
        start, stop = None, None
        for a, b in sorted(periods):
            if start is None:
                start, stop = a, b
            elif a <= stop:
                stop = max(stop, b)
            else:
                allocated += stop - start
                start, stop = a, b
        if start is not None:
            allocated += stop - start
    metrics = defaultdict(list)
    for row in conditions:
        for key, value in {**row["metrics"], "actual_J": row["actual_J"], "condition_wall_seconds": row["wall_seconds"]}.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key].append(value)
    distributions = {}
    for key, values in metrics.items():
        ordered = sorted(values)
        distributions[key] = {"count": len(values), "sum": sum(values), "mean": sum(values) / len(values),
                              "p50": ordered[int(.5 * (len(values) - 1))], "p90": ordered[int(.9 * (len(values) - 1))], "max": ordered[-1]}
    return {"allocated_physical_GPUh": allocated / 3600, "worker_receipts": receipts,
            "condition_cost_distributions": distributions}


def aggregate(output: Path, references: dict | None = None) -> dict:
    output = Path(output).resolve()
    contract = read_json(output / "evaluation_contract.json")
    _validate_panel(contract)
    queue = queue_summary(output / "queue.sqlite3")
    if queue["status_counts"] != {"complete": len(contract["conditions"])}:
        raise ValueError("aggregation requires the whole condition queue to complete")
    panels, conditions = {arm: [] for arm in contract["arms"]}, []
    expected = {c["condition_id"]: c for c in contract["conditions"]}
    seen = set()
    for job in completed_jobs(output / "queue.sqlite3"):
        primary, size = read_json_with_size(output / job["rows_path"])
        c = primary["condition"]
        if c != expected.get(c["condition_id"]) or c["condition_id"] in seen or size != job["rows_bytes"]:
            raise ValueError("completed artifact changed the frozen condition or queue identity")
        seen.add(c["condition_id"])
        for arm in contract["arms"]:
            raw = read_json(output / primary["arm_paths"][arm])
            if raw["condition"] != c or raw["arm"] != arm:
                raise ValueError("raw sidecar belongs to another condition/arm")
            _validate_rows(raw["rows"], c, arm, _task(contract, c))
            panels[arm].extend(raw["rows"])
        if primary["rows"] != read_json(output / primary["arm_paths"]["end"])["rows"] or len(primary["rows"]) != job["row_count"]:
            raise ValueError("primary queue rows disagree with the actual end-arm raw rows")
        conditions.append({key: primary[key] for key in ("condition", "metrics", "actual_J", "weights", "chain", "artifact_root", "wall_seconds")})
    if seen != expected.keys() or any(len(rows) != contract["expected_rows_per_arm"] for rows in panels.values()):
        raise ValueError("aggregate arm coverage does not match the preregistered panel")
    comparison = {}
    if "initial" in panels:
        comparison["initial_to_end"] = compare_train(panels["initial"], panels["end"])
    if "null" in panels:
        comparison["null_to_end"] = compare_train(panels["null"], panels["end"])
    if contract["stage"] == "meta54":
        previous = output.parent / "meta27" / "results.json"
        if previous.exists():
            prior = read_json(previous)
            comparison["meta27_to_meta54"] = compare_train(prior["arms"]["end"]["rows"], panels["end"])
    for name, path in (contract["references"] if references is None else references).items():
        comparison[name] = compare_formal(read_json(Path(path)), {"rows": panels["end"]})
    result = {"schema_version": SCHEMA, "stage": contract["stage"], "checkpoint": contract["checkpoint"],
              "queue": queue, "arms": {arm: {**_summary(rows), "rows": rows} for arm, rows in panels.items()},
              "conditions": conditions, "comparisons": comparison, "cost": _cost(output, conditions)}
    immutable(output / "results.json", result)
    return result
