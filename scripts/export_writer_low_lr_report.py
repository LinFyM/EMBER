#!/usr/bin/env python3
"""Export the completed Writer low-LR phase into reviewable report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ember.pi05_eval_results import AGGREGATE_SCHEMA
from ember.pi05_source_checkpoint import read_json, write_json_atomic


SELECTION_SCHEMA = "ember_writer_low_lr_selection_v1"
PHASE_SCHEMA = "ember_writer_low_lr_phase_history_v1"
REFERENCE_STEP = 1000
PARENT_STEP = 1800


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy_portable_json(source: Path, destination: Path, replacements: Mapping[str, str]) -> None:
    """Copy JSON evidence while replacing host-local roots with stable provenance labels."""
    def portable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            for prefix, label in replacements.items():
                if value.startswith(prefix):
                    return label + value[len(prefix):]
        return value

    write_json_atomic(destination, portable(read_json(source)))


def load_panel(directory: Path, *, expected_arm: str, expected_step: int) -> dict[str, Any]:
    required = [directory / "results.json", directory / "run_contract.json",
                directory / "launcher_completion.json"]
    if not all(path.is_file() for path in required):
        raise ValueError(f"missing formal evaluation completion in {directory}")
    results, contract, completion = map(read_json, required)
    if (results.get("schema_version") != AGGREGATE_SCHEMA
            or contract.get("role") != "validation" or contract.get("mode") != "formal"
            or len({results.get("contract_reference"), contract.get("contract_reference"),
                    completion.get("contract_reference")}) != 1
            or not completion.get("return_codes")
            or any(code != 0 for code in completion["return_codes"].values())):
        raise ValueError(f"invalid formal evaluation completion in {directory}")
    rows = results.get("rows", ())
    keys = {(row["suite"], int(row["task_id"]), int(row["init_state_id"])) for row in rows}
    successes = sum(bool(row["success"]) for row in rows)
    adapters = [row.get("horizon_writer_lora", {}) for row in rows]
    if (len(rows) != 400 or len(keys) != 400
            or results.get("overall", {}).get("episodes") != 400
            or results.get("overall", {}).get("successes") != successes
            or completion.get("queue", {}).get("completed_rows") != 400
            or completion.get("queue", {}).get("successes") != successes):
        raise ValueError(f"panel is not a complete unique Validation400 in {directory}")
    if (any(adapter.get("arm") != expected_arm for adapter in adapters)
            or any(adapter.get("writer_checkpoint", {}).get("macro") != expected_step for adapter in adapters)):
        raise ValueError(f"panel arm or Writer checkpoint changed in {directory}")
    return results


def success_map(panel: Mapping[str, Any]) -> dict[tuple[str, int, int], bool]:
    return {(row["suite"], int(row["task_id"]), int(row["init_state_id"])): bool(row["success"])
            for row in panel["rows"]}


def paired_counts(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, int]:
    left, right = success_map(reference), success_map(candidate)
    if left.keys() != right.keys():
        raise ValueError("paired panels do not share identical task/state rows")
    retained = sum(left[key] and right[key] for key in left)
    gained = sum(not left[key] and right[key] for key in left)
    lost = sum(left[key] and not right[key] for key in left)
    return {"retained": retained, "gained": gained, "lost": lost, "churn": gained + lost}


def task_rows(label: str, step: int, panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for row in panel["rows"]:
        grouped[(row["suite"], int(row["task_id"]))].append(bool(row["success"]))
    return [{"panel": label, "global_step": step, "suite": suite, "task_id": task,
             "successes": sum(values), "episodes": len(values)}
            for (suite, task), values in sorted(grouped.items())]


def suite_rows(label: str, step: int, panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in panel["rows"]:
        grouped[row["suite"]].append(bool(row["success"]))
    return [{"panel": label, "global_step": step, "suite": suite,
             "successes": sum(values), "episodes": len(values)}
            for suite, values in sorted(grouped.items())]


def _training_cost(study: Path, *, expected_end_step: int) -> dict[str, Any]:
    metrics_path = study / "training" / "writer" / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    phase = [row for row in metrics if int(row.get("step", row.get("optimizer_updates", 0))) > PARENT_STEP]
    if not phase:
        raise ValueError("low-LR phase metrics are missing")
    steps = [int(row.get("step", row["optimizer_updates"])) for row in phase]
    if steps != list(range(PARENT_STEP + 1, steps[-1] + 1)):
        raise ValueError("low-LR phase metrics are not a contiguous update sequence")
    if steps[-1] != expected_end_step or len(steps) != expected_end_step - PARENT_STEP:
        raise ValueError("low-LR phase metrics do not reach the registered stopping node")
    seconds = [float(row["seconds"]) for row in phase]
    applied = {float(row["lr_applied"]) for row in phase}
    if len(applied) != 1:
        raise ValueError("low-LR phase did not retain one applied learning rate")
    return {
        "schema_version": "ember_writer_low_lr_cost_v1",
        "global_start_step": steps[0], "global_end_step": steps[-1],
        "phase_updates": len(phase), "applied_lr": next(iter(applied)),
        "update_seconds_sum": sum(seconds), "update_seconds_mean": sum(seconds) / len(seconds),
        "update_seconds_p95": sorted(seconds)[math.ceil(.95 * len(seconds)) - 1],
        "peak_reserved_gib": max(float(row["peak_reserved_gib"]) for row in phase),
        "main_queries": 84 * len(phase), "teaching_queries": 28 * len(phase),
        "total_queries": 112 * len(phase),
    }


def _plot_curve(output: Path, nodes: list[dict[str, Any]], selected_step: int, selected_score: int,
                stop_step: int, *, owner_stopped: bool) -> None:
    old = [row for row in nodes if row["phase"] == "coverage"]
    phase = [row for row in nodes if row["phase"] == "low_lr"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    fig.subplots_adjust(left=.085, right=.97, bottom=.2, top=.78)
    fig.text(.085, .94, "EMBER Writer: coverage training and fixed low-LR continuation",
             fontsize=16, weight="bold")
    fig.text(.085, .875, "Complete correct Validation400 nodes · intervention after global step 1800",
             fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.7)
    ax.plot([row["global_step"] for row in old], [row["successes"] for row in old],
            color="#777777", marker="o", linewidth=1.4, label="Original coverage Writer")
    ax.plot([PARENT_STEP, *[row["global_step"] for row in phase]],
            [old[-1]["successes"], *[row["successes"] for row in phase]],
            color="#0072B2", marker="o", linewidth=1.8, label="Fixed low-LR phase")
    ax.axvline(PARENT_STEP, color="#D55E00", linestyle=":", linewidth=1.5,
               label="LR intervention")
    ax.axhline(117, color="#E69F00", linestyle="--", linewidth=1.2,
               label="Original N1000 reference: 117")
    ax.scatter([selected_step], [selected_score], marker="*", s=210, color="#009E73", zorder=5,
               label=f"Selected: {selected_step} ({selected_score}/400)")
    stop_score = next(row["successes"] for row in phase if row["global_step"] == stop_step)
    stop_label = f"Last complete {stop_score}" if owner_stopped else f"Phase stop {stop_score}"
    ax.annotate(stop_label, (stop_step, stop_score), xytext=(-8, -21),
                textcoords="offset points", ha="right")
    ax.set_xlabel("Global Writer optimizer update")
    ax.set_ylabel("Successful rollouts / 400")
    ax.set_xticks([row["global_step"] for row in nodes])
    ax.tick_params(axis="x", rotation=45)
    values = [row["successes"] for row in nodes] + [117]
    ax.set_ylim(max(0, min(values) - 12), min(400, max(values) + 18))
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"writer_low_lr_validation_curve.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    svg = output / "writer_low_lr_validation_curve.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def export(study: Path, original: Path, output: Path) -> dict[str, Any]:
    phase = read_json(study / "writer_phase_validation_history.json")
    selection = read_json(study / "writer_low_lr_selection.json")
    owner_stopped = selection.get("termination") == "owner_requested_stop"
    controller_exit = study / "launch" / "writer_training_controller.exit"
    if not controller_exit.is_file():
        raise ValueError("low-LR training controller completion record is missing")
    if not owner_stopped and controller_exit.read_text().strip() != "0":
        raise ValueError("low-LR training controller has not completed successfully")
    if (phase.get("schema_version") != PHASE_SCHEMA
            or (not phase.get("decision", {}).get("stop") and not owner_stopped)
            or selection.get("schema_version") != SELECTION_SCHEMA or selection.get("status") != "selected"):
        raise ValueError("low-LR phase is not stopped and finalized")
    owner_stop = None
    if owner_stopped:
        owner_stop_path = Path(selection["owner_stop"])
        owner_stop = read_json(owner_stop_path)
        if (owner_stop.get("schema_version") != "ember_writer_low_lr_owner_stop_v1"
                or owner_stop.get("formal_evidence_cutoff_step") != phase["history"][-1]["step"]
                or owner_stop.get("training_completed_through_step") != selection.get("phase_training_end_step")):
            raise ValueError("owner stop record changed after selection")
    old_history = read_json(original / "writer_validation_history.json")
    old_nodes = old_history["history"]
    if [(row["step"], row["successes"]) for row in old_nodes] != [
        (200, 110), (400, 92), (600, 115), (800, 87), (1000, 117),
        (1200, 80), (1400, 97), (1600, 111), (1800, 92),
    ]:
        raise ValueError("original coverage Writer reference changed")
    nodes = [{"phase": "coverage", "global_step": row["step"], "phase_step": "",
              "successes": row["successes"], "episodes": 400} for row in old_nodes]
    phase_steps = [int(row["step"]) for row in phase["history"]]
    if phase_steps != list(range(1900, phase_steps[-1] + 1, 100)):
        raise ValueError("low-LR phase history skipped or reordered a registered Validation400 node")
    if int(phase["decision"].get("last_step", -1)) != phase_steps[-1]:
        raise ValueError("low-LR phase stopping cursor disagrees with its complete history")
    nodes += [{"phase": "low_lr", "global_step": row["step"], "phase_step": row["phase_step"],
               "successes": row["successes"], "episodes": 400} for row in phase["history"]]

    panels: dict[str, tuple[int, dict[str, Any]]] = {
        "original_n1000_correct": (REFERENCE_STEP, load_panel(
            original / "evaluation" / "writer_00001000", expected_arm="correct", expected_step=REFERENCE_STEP)),
        "parent_n1800_correct": (PARENT_STEP, load_panel(
            original / "evaluation" / "writer_00001800", expected_arm="correct", expected_step=PARENT_STEP)),
    }
    for row in phase["history"]:
        step = int(row["step"])
        panel = load_panel(study / "evaluation" / f"writer_{step:08d}",
                           expected_arm="correct", expected_step=step)
        if panel["overall"]["successes"] != int(row["successes"]):
            raise ValueError(f"phase history score changed at step {step}")
        panels[f"phase_{step}_correct"] = (step, panel)
    selected_step = int(selection["selected_macro"])
    if selection["selected_study"] == "original":
        selected_label = "original_n1000_correct"
    else:
        selected_label = f"phase_{selected_step}_correct"
        for arm in ("other", "cross_suite_wrong"):
            panels[f"selected_{arm}"] = (selected_step, load_panel(
                study / "evaluation" / f"writer_{selected_step:08d}_{arm}",
                expected_arm="same_task_other" if arm == "other" else arm, expected_step=selected_step))
    selected_panel = panels[selected_label][1]
    selected_score = int(selected_panel["overall"]["successes"])
    if selected_score != int(selection["correct_successes"]):
        raise ValueError("selection score changed after finalization")

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "writer_low_lr_validation_nodes.csv", nodes,
               ["phase", "global_step", "phase_step", "successes", "episodes"])
    per_task = [row for label, (step, panel) in panels.items() for row in task_rows(label, step, panel)]
    per_suite = [row for label, (step, panel) in panels.items() for row in suite_rows(label, step, panel)]
    _write_csv(output / "writer_low_lr_per_task.csv", per_task,
               ["panel", "global_step", "suite", "task_id", "successes", "episodes"])
    _write_csv(output / "writer_low_lr_per_suite.csv", per_suite,
               ["panel", "global_step", "suite", "successes", "episodes"])

    ordered = [("parent_n1800_correct", PARENT_STEP, panels["parent_n1800_correct"][1])]
    ordered += [(f"phase_{row['step']}_correct", int(row["step"]), panels[f"phase_{row['step']}_correct"][1])
                for row in phase["history"]]
    adjacent = []
    for (left_label, left_step, left), (right_label, right_step, right) in zip(ordered, ordered[1:]):
        adjacent.append({"reference": left_label, "candidate": right_label,
                         "reference_step": left_step, "candidate_step": right_step,
                         **paired_counts(left, right)})
    _write_csv(output / "writer_low_lr_adjacent_pairs.csv", adjacent,
               ["reference", "candidate", "reference_step", "candidate_step",
                "retained", "gained", "lost", "churn"])

    raw = []
    for label, (step, panel) in panels.items():
        if not label.startswith("phase_") and not label.startswith("selected_") and label != selected_label:
            continue
        for row in panel["rows"]:
            lora = row.get("horizon_writer_lora", {})
            raw.append({"panel": label, "global_step": step, "suite": row["suite"],
                        "task_id": row["task_id"], "init_state_id": row["init_state_id"],
                        "success": bool(row["success"]), "video_ordinal": lora.get("video_ordinal"),
                        "teacher_demo_indices": json.dumps(lora.get("teacher_demo_indices", []))})
    _write_csv(output / "writer_low_lr_validation_rows.csv", raw,
               ["panel", "global_step", "suite", "task_id", "init_state_id", "success",
                "video_ordinal", "teacher_demo_indices"])
    failures = [row for row in raw if row["panel"] == selected_label and not row["success"]]
    _write_csv(output / "selected_correct_failures.csv", failures,
               ["panel", "global_step", "suite", "task_id", "init_state_id", "success",
                "video_ordinal", "teacher_demo_indices"])

    comparisons = {"selected_vs_original_n1000": paired_counts(panels["original_n1000_correct"][1], selected_panel)}
    if selection["selected_study"] == "phase":
        comparisons["correct_vs_other"] = paired_counts(selected_panel, panels["selected_other"][1])
        comparisons["correct_vs_cross_suite_wrong"] = paired_counts(
            selected_panel, panels["selected_cross_suite_wrong"][1])
    write_json_atomic(output / "writer_low_lr_paired_statistics.json", comparisons)
    training_end_step = int(selection.get("phase_training_end_step", phase_steps[-1]))
    cost = _training_cost(study, expected_end_step=training_end_step)
    write_json_atomic(output / "writer_low_lr_cost.json", cost)
    portable_roots = {str(study): "writer-low-lr-study:", str(original): "coverage-study:"}
    _copy_portable_json(study / "writer_phase_validation_history.json",
                        output / "writer_phase_validation_history.json", portable_roots)
    _copy_portable_json(study / "writer_low_lr_selection.json",
                        output / "writer_low_lr_selection.json", portable_roots)
    if owner_stopped:
        _copy_portable_json(owner_stop_path, output / owner_stop_path.name, portable_roots)
    _plot_curve(output, nodes, selected_step, selected_score, phase_steps[-1], owner_stopped=owner_stopped)

    phase_scores = ", ".join(f"{row['step']}:{row['successes']}" for row in phase["history"])
    controls = "未执行；phase未严格超过117，按合同保留原N1000。"
    if selection["selected_study"] == "phase":
        other = panels["selected_other"][1]["overall"]["successes"]
        wrong = panels["selected_cross_suite_wrong"][1]["overall"]["successes"]
        controls = f"选中phase节点的same-task-other为 **{other}/400**，cross-suite-wrong为 **{wrong}/400**。"
    termination = (f"Owner在global{training_end_step}训练完成、对应Validation尚未形成正式400行时明确要求停止；"
                   f"因此正式曲线截止global{phase_steps[-1]}，未宣称触发预登记早停。"
                   if owner_stopped else
                   f"预登记早停在global{phase['decision']['last_step']}触发，原因为`{phase['decision']['reason']}`。")
    status = ("Owner在五个完整phase节点后明确停止；正式证据已封存并按资格规则完成选点。"
              if owner_stopped else
              "唯一低学习率phase已按登记早停规则结束并完成选点。")
    report = f"""# Writer N1800 恒定低学习率修复续训报告

> 状态：{status}本报告不包含Test、FT、RL、外部比较或其它补救分支。

## 合同与执行

从覆盖重训正式N1800完整状态恢复Writer、三个Meta、AdamW、采样器和四rank RNG；global1801起实际学习率固定为
`{cost['applied_lr']:.12g}`。训练任务、监督、四卡拓扑、视频seed与评测协议不变。每100更新只用完整correct
Validation400执行phase早停。首次启动在任何更新前因可选diagnostics历史文件被误设为必需而失败；修复后从原N1800
重新启动，没有恢复不完整输出，也没有改变科学变量。

## 完整曲线、停止与选点

新phase完整节点为 **{phase_scores}**（均为成功数/400）。{termination} phase最高为
**{selection['phase_best_successes']}/400**。最终裁决为`{selection['outcome']}`，选中global
**{selected_step}**，correct为 **{selected_score}/400**。phase分数必须严格超过原N1000的117才可替换；并列correct只用
same-task-other破同分，wrong从不参与选点。

![完整训练与修复续训曲线](writer_low_lr_validation_curve.png)

## 视频对照

{controls}

## 配对结果、失败案例与成本

选中模型相对原N1000的配对保留/获得/丢失为
**{comparisons['selected_vs_original_n1000']['retained']}/{comparisons['selected_vs_original_n1000']['gained']}/{comparisons['selected_vs_original_n1000']['lost']}**。
逐节点逐任务、逐suite、相邻节点R/G/L和选中correct失败行分别见随附CSV；phase及实际执行controls的400行精简原始记录见
`writer_low_lr_validation_rows.csv`，正式完整JSON仍保留在study。

phase共执行 **{cost['phase_updates']}** 次更新（其中正式Validation截止{phase_steps[-1]}），训练循环合计 **{cost['update_seconds_sum']:.2f}秒**，平均
**{cost['update_seconds_mean']:.3f}秒/更新**，记录峰值reserved **{cost['peak_reserved_gib']:.3f} GiB/卡**；累计
**{cost['total_queries']}** queries。训练耗时不含物化、闭环评测和阶段切换。

## 结论边界与材料

本实验只检验N1800后固定低学习率能否恢复并稳定超过原N1000。科学负结果不会解释成工程待修复，也不触发其它
起点、学习率、seed、任务权重、loss或架构补救。原始早停历史、选点记录、成本JSON、全曲线PNG/SVG及绘图导出源码
均随本目录或仓库脚本保留；旧Test的78/74/82及其+40未通过结论不受本实验改写。
"""
    (output / "report.md").write_text(report)
    return {"selected_macro": selected_step, "selected_score": selected_score,
            "phase_nodes": len(phase["history"]), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--original-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.study.resolve(), args.original_study.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
