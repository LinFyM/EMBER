#!/usr/bin/env python3
"""Mechanical phase0 aggregation; seven source macros are bootstrap clusters."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.metatask_lookahead import authority


STATES = ("P", "TASK_A", "TASK_B", "MIX_A", "MIX_B", "BASE", "TASK", "MIX")
FLOW_STATES = ("P", "BASE", "TASK", "MIX")
COMPARISONS = (("TASK", "BASE"), ("MIX", "BASE"), ("TASK", "MIX"))


def _read_rows(case: Path, label: str) -> list[dict]:
    return [json.loads(line) for rank in range(2)
            for line in (case / f"{label}_rank_{rank}.jsonl").read_text().splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _bootstrap(case_values: list[float], draws: np.ndarray) -> dict:
    values = np.asarray(case_values, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("seven original source macros are required for bootstrap")
    if draws.shape != (20000, 7):
        raise ValueError("registered joint case bootstrap draws changed")
    estimates = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in
            np.quantile(estimates, [.025, .975])], "case_values": values.tolist(),
            "clusters": 7, "replicates": 20000, "seed": 20260925,
            "interval_type": "descriptive_state_fixed_case_bootstrap"}


def _validate_rows(fm, flows) -> None:
    if (len({(r["case"],r["state"],r["task"],r["query_index"]) for r in fm}) != 3584
            or len({(r["case"],r["state"],r["task"],r["query_index"]) for r in flows}) != 448
            or set(r["state"] for r in fm) != set(STATES)
            or set(r["state"] for r in flows) != set(FLOW_STATES)
            or not all(r["no_grad"] for r in fm)
            or not all(not r["expert_action_used_as_input"] and r["environment_steps"] == 0
                       for r in flows)):
        raise ValueError("phase0 row pairing or readout wall changed")


def _load_matrix(root: Path, spec):
    cases, fm, flows, commits = [], [], [], set()
    for case_id in range(1, 8):
        case = root / f"case_{case_id:02d}"
        contract = json.loads((case / "case_contract.json").read_text())
        if (contract["case"] != case_id or contract["parent_macro"] != 1155
                or contract["source_macro"] != 1155+case_id
                or contract["event_indices"] != spec["cases"][case_id-1]["event_indices"]
                or contract["fm_rows"] != 512 or contract["true_flow_rows"] != 64
                or not contract["no_environment_steps"]):
            raise ValueError("lookahead case identity, readout or information wall changed")
        commits.add(contract["implementation_commit"])
        cases.append(contract)
        fm.extend(_read_rows(case, "fm"))
        flows.extend(_read_rows(case, "flow"))
        if len(list((case / "banks").glob("*/*.safetensors"))) != 32:
            raise ValueError("one source case lost an eight-state by four-task LoRA bank")
        if len(list((case / "weights").glob("*.safetensors"))) != 7:
            raise ValueError("one source case lost diagnostic virtual/candidate weights")
    if len(commits) != 1 or len(fm) != 3584 or len(flows) != 448:
        raise ValueError("phase0 retained rows or single implementation identity incomplete")
    _validate_rows(fm, flows)
    return cases, fm, flows, commits


def _flow_query(case_id, task, query_index, flow_by_key):
    records = {state:flow_by_key[(case_id,task,query_index,state)] for state in FLOW_STATES}
    loaded = {}
    for state, record in records.items():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError("stored full50 true-flow output changed")
        with np.load(path, allow_pickle=False) as saved:
            loaded[state] = {key:saved[key].copy() for key in saved.files}
        if (loaded[state]["normalized"].shape != (50,7)
                or loaded[state]["environment"].shape != (50,7)
                or loaded[state]["osc_first5"].shape != (5,6)
                or loaded[state]["policy_noise"].shape != (50,32)):
            raise ValueError("true-flow saved H50/OSC/noise dimensions changed")
    parent_prediction = loaded["P"]["normalized"]
    expert = loaded["P"]["expert_normalized"]
    if any(not np.array_equal(loaded[s]["expert_normalized"],expert)
           or not np.array_equal(loaded[s]["policy_noise"],loaded["P"]["policy_noise"])
           for s in FLOW_STATES):
        raise ValueError("same query/noise/expert identity changed across parameter states")
    metrics = {}
    for state in FLOW_STATES:
        prediction = loaded[state]["normalized"]
        delta = prediction-parent_prediction
        for horizon, sl in (("first5",slice(0,5)),("full50",slice(None))):
            error = float(np.mean((prediction[sl]-expert[sl])**2))
            energy = float(np.mean(delta[sl]**2))
            alignment = float(-2*np.mean(delta[sl]*(expert[sl]-parent_prediction[sl])))
            residual = (error-float(np.mean((parent_prediction[sl]-expert[sl])**2))
                        -energy-alignment)
            if abs(residual) > 1e-4:
                raise ValueError("finite true-flow error identity failed")
            metrics[f"{state}_{horizon}"] = {"mse":error,"displacement_energy":energy,
                "direction_term":alignment,"identity_residual":residual}
    return {"case":case_id,"task":task,"query_index":query_index,
        "query_demo":records["P"]["query_demo"],"query_frame":records["P"]["query_frame"],
        "noise_seed":records["P"]["noise_seed"],
        "first5_osc_xy":{s:loaded[s]["osc_first5"][:,:2].tolist() for s in FLOW_STATES},
        "metrics":metrics}


def _group_transfer(case):
    records = case["virtual_rows"]
    return {group: {side: {read: next(row["loss"] for row in records
             if row["state"] == f"{group}_{side}" and row["read_group"] == read)
             - case["parent_group_losses"][group][read] for read in ("A", "B")}
             for side in ("A", "B")} for group in ("TASK", "MIX")}


def _aggregate(spec, cases, fm, flows):
    fm_by_key = {(r["case"],r["task"],r["query_index"],r["state"]): r for r in fm}
    flow_by_key = {(r["case"],r["task"],r["query_index"],r["state"]): r for r in flows}
    task_rows, flow_detail, case_effects = [], [], []
    boot_values = defaultdict(lambda: [[] for _ in range(7)])
    for case in cases:
        case_id = case["case"]
        case_effects.append({"case":case_id,"source_macro":case["source_macro"],
                             "finite_group_loss_transfer":_group_transfer(case),
                             "identities":case["identities"],
                             "candidate_rows":case["candidate_rows"],
                             "candidate_gradient_displacement_cosines":
                                 case["candidate_gradient_displacement_cosines"]})
        for task in spec["cases"][case_id-1]["tasks"]:
            fm_mean = {state: float(np.mean([fm_by_key[(case_id,task,i,state)]["fm_loss"]
                                              for i in range(16)])) for state in STATES}
            if not np.isfinite(list(fm_mean.values())).all():
                raise ValueError("phase0 independent-episode FM is nonfinite")
            row = {"case":case_id,"task":task,"fm_mean_by_state":fm_mean,
                   "fm_contrasts":{f"{a}-{b}":fm_mean[a]-fm_mean[b] for a,b in COMPARISONS},
                   "TASK_group_member": "A" if spec["cases"][case_id-1]["tasks"].index(task)%2 == 0 else "B",
                   "MIX_contains_all_four_tasks":True}
            task_rows.append(row)
            for a,b in COMPARISONS:
                boot_values[("fm",a,b)][case_id-1].append(row["fm_contrasts"][f"{a}-{b}"])
            for query_index in range(4):
                row_flow = _flow_query(case_id, task, query_index, flow_by_key)
                flow_detail.append(row_flow)
                for a,b in COMPARISONS:
                    boot_values[("flow_first5_mse",a,b)][case_id-1].append(
                        row_flow["metrics"][f"{a}_first5"]["mse"]-
                        row_flow["metrics"][f"{b}_first5"]["mse"])
    return task_rows, flow_detail, case_effects, boot_values


def main() -> None:
    spec = authority()
    root = Path(spec["resources"]["study_root"])
    output = root / "analysis"
    if output.exists():
        raise ValueError("registered phase0 analysis already exists")
    cases, fm, flows, commits = _load_matrix(root, spec)
    task_rows, flow_detail, case_effects, boot_values = _aggregate(spec,cases,fm,flows)
    output.mkdir(parents=True)
    _write_jsonl(output/"fm_rows.jsonl",fm)
    _write_jsonl(output/"flow_rows.jsonl",flows)
    _write_jsonl(output/"per_task.jsonl",task_rows)
    _write_jsonl(output/"flow_decomposition.jsonl",flow_detail)
    _write_jsonl(output/"case_effects.jsonl",case_effects)
    draws = np.random.default_rng(20260925).integers(0,7,size=(20000,7))
    bootstrap = {"/".join(key):_bootstrap([float(np.mean(x)) for x in groups],draws)
                 for key,groups in sorted(boot_values.items())}
    write_json_atomic(output/"comparisons.json",bootstrap)
    conservative_gpu_hours = sum(2*row["elapsed_seconds"]/3600 for row in cases)
    if conservative_gpu_hours > spec["resources"]["formal_gpu_hours_max"]:
        raise ValueError("formal computation exceeded its GPU-hour cap")
    write_json_atomic(output/"completion.json",{
        "schema_version":"ember_metatask_lookahead_phase0_completion_v1",
        "study_id":spec["study_id"],"implementation_commit":next(iter(commits)),
        "cases":7,"source_macros":list(range(1156,1163)),
        "original_training_queries":784,"diagnostic_banks":224,
        "fm_rows":len(fm),"true_flow_rows":len(flows),"flow_query_rows":len(flow_detail),
        "virtual_weights_only":28,"candidate_weights_only":21,
        "new_environment_steps":0,"new_held_inputs":0,
        "conservative_gpu_hours":conservative_gpu_hours,
        "gpu_hour_cap":spec["resources"]["formal_gpu_hours_max"],
        "per_task_rows":len(task_rows),"case_bootstrap_replicates":20000,
        "analysis_files":[str(path) for path in sorted(output.iterdir())],
        "performance_or_video_benefit_claim":False,
    })


if __name__ == "__main__":
    main()
