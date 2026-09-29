"""Read the registered A28 train panel from the fresh change-clock 270 Writer."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.change_clock import MODE, ROOT, TASK
from ember.operator_writer.data import FormalData
from ember.operator_writer.native import read_native_video
from ember.operator_writer.run import CHANGE_CLOCK_SPEC_PATH, build_runtime, specification
from ember.writer.function_credit import flow_sample, paired_functional_credit
from ember.writer.runtime import autocast


ASSET = Path("/data1/user/ymdai/projects/EMBER")
OLD_PANEL = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
                 "/continuation900/analysis/teacher_to_query_270/rows.jsonl")
TASKS = (0, 12, 20, 32)
PREFIX = "model.paligemma_with_expert.gemma_expert.model.layers.8.self_attn."
TARGETS = (PREFIX + "q_proj", PREFIX + "v_proj", "model.action_out_proj")


@contextmanager
def capture_query(policy):
    inputs = {name: [] for name in TARGETS}
    outputs, handles = [], []
    try:
        for name in TARGETS:
            def before(_module, arguments, selected=name):
                inputs[selected].append(arguments[0].detach().float().cpu())
            handles.append(policy.get_submodule(name).register_forward_pre_hook(before))
        def after(_module, _arguments, output):
            outputs.append(output.detach().float().cpu()[..., :7])
        handles.append(policy.get_submodule("model.action_out_proj").register_forward_hook(after))
        yield inputs, outputs
    finally:
        for handle in handles:
            handle.remove()


def evaluate(runtime, state, batch, seed, target, microbatch, *, backward):
    with capture_query(runtime.policy) as (inputs, outputs), autocast(runtime.device):
        credit = paired_functional_credit(
            runtime.policy, state, runtime.lora, batch, seed=seed, device=runtime.device,
            random_batch=28, offset=0, microbatch=microbatch, condition_weight=1.,
            backward=backward)
    prediction = torch.cat(outputs)
    if prediction.shape != target.shape or prediction.shape != (28, 50, 7):
        raise ValueError("fixed A28 FM prediction/target shape changed")
    readback = float((prediction - target).square().mean())
    if abs(readback - credit["flow_loss"]) > 2e-4:
        raise ValueError("captured action_out differs from real FM loss")
    return prediction, credit, ({name: torch.cat(parts) for name, parts in inputs.items()}
                                if backward else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, choices=(7, 14, 28), default=14)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    expected = ROOT / MODE / "train/attempts/fresh/checkpoints/macro_00000270/ecp.safetensors"
    if checkpoint != expected or not checkpoint.is_file():
        raise ValueError("diagnostic requires only this completed fresh 270 candidate")
    output = ROOT / "analysis/teacher_to_query_270"
    if output.exists():
        raise ValueError("diagnostic output already exists")
    old = [json.loads(line) for line in OLD_PANEL.read_text().splitlines()]
    old = {(row["task"], row["teacher_demo"]): row for row in old if row["mode"] == "T"}
    if len(old) != 8:
        raise ValueError("old T270 fixed panel is incomplete")
    torch.set_num_threads(4)
    spec = specification(CHANGE_CLOCK_SPEC_PATH)
    data = FormalData(ASSET, spec, task_ids=TASKS)
    output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    (output / "run_contract.json").write_text(json.dumps({
        "task": TASK, "mode": MODE,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "checkpoint": str(checkpoint), "old_panel": str(OLD_PANEL),
        "tasks": list(TASKS), "targets": list(TARGETS), "query_count": 28,
        "optimization_updates": 0, "environment_episodes": 0,
    }, indent=2) + "\n")
    try:
        runtime = build_runtime(ASSET, spec, torch.device(args.device), MODE)
        runtime.writer.load_state_dict(load_file(str(checkpoint), device=args.device), strict=True)
        runtime.writer.eval()
        runtime.policy.eval()
        for task in TASKS:
            teachers = np.random.default_rng(np.random.SeedSequence([20260928, 1, task])).permutation(50)[:2]
            reference = old[(task, int(teachers[0]))]
            queries, seed = reference["queries"], reference["flow_seed"]
            if (old[(task, int(teachers[1]))]["queries"] != queries
                    or old[(task, int(teachers[1]))]["flow_seed"] != seed
                    or len(queries) != 28 or set(q["demo"] for q in queries) & set(teachers.tolist())):
                raise ValueError("old panel query/teacher/noise pairing changed")
            event = {"task": task, "teacher_demo": int(teachers[0]), "queries": queries}
            batch = runtime.processor.training_batch(data.batch(event))
            runtime.restore_identity()
            with torch.no_grad(), autocast(runtime.device):
                sample = flow_sample(runtime.policy, batch, seed=seed, device=runtime.device,
                                     random_batch=28, offset=0)
            target = sample.target[..., :7].detach().float().cpu()
            flow_time = sample.arguments[-1].detach().float().cpu()
            del sample
            common = runtime.writer.public_state()
            runtime.restore_identity()
            public_prediction, public_credit, _ = evaluate(runtime, common, batch, seed, target,
                                                             args.microbatch, backward=False)
            public_path = output / f"public_task{task:03d}.pt"
            torch.save({"prediction": public_prediction, "target": target, "flow_time": flow_time},
                       public_path)
            rows.append({"kind": "public_beta", "task": task, "queries": queries,
                         "flow_seed": seed, "fm": public_credit["flow_loss"], "raw": str(public_path)})
            for teacher in teachers:
                teacher = int(teacher)
                condition, raw_frames, sampled_frames = data.condition(runtime, task, teacher)
                runtime.restore_identity()
                with torch.no_grad(), autocast(runtime.device):
                    native_x, h = read_native_video(runtime.policy, common, runtime.writer.probe,
                                                    condition, runtime.writer.names, frame_chunk=8)
                    state = runtime.writer(native_x, h)
                validate_lora_state(state, runtime.lora)
                runtime.restore_identity()
                prediction, credit, query_x = evaluate(runtime, state, batch, seed, target,
                                                        args.microbatch, backward=True)
                cotangent = credit["lora_cotangent"]
                if len(cotangent) != 76 or any(name + LORA_B_SUFFIX not in cotangent for name in TARGETS):
                    raise ValueError("full LoRA functional cotangent changed")
                path = output / f"full_task{task:03d}_teacher{teacher:02d}.pt"
                torch.save({
                    "prediction": prediction, "target": target, "flow_time": flow_time,
                    "public_prediction": public_prediction,
                    "native_h": h.detach().float().cpu(),
                    "native_x": {name: native_x[name].detach().float().cpu() for name in TARGETS},
                    "query_x": query_x,
                    "memory_M": {name: (state[name + LORA_B_SUFFIX] - common[name + LORA_B_SUFFIX]).detach().float().cpu()
                                 for name in TARGETS},
                    "G_B": {name: cotangent[name + LORA_B_SUFFIX].detach().float().cpu()
                            for name in TARGETS},
                    "video_frame_indices": condition[1].detach().cpu(),
                }, path)
                rows.append({"kind": "full", "task": task, "teacher_demo": teacher,
                             "queries": queries, "flow_seed": seed, "fm": credit["flow_loss"],
                             "public_fm": public_credit["flow_loss"], "raw_frames": raw_frames,
                             "sampled_frames": sampled_frames, "raw": str(path)})
        if len(rows) != 12 or sum(row["kind"] == "full" for row in rows) != 8:
            raise ValueError("fixed A28 candidate panel incomplete")
        (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (output / "completion.json").write_text(json.dumps({"status": "complete", "rows": len(rows),
            "full": 8, "public_beta": 4, "updates": 0, "environment_episodes": 0,
            "checkpoint": str(checkpoint), "old_panel": str(OLD_PANEL),
            "seconds": time.monotonic() - started}, indent=2) + "\n")
    except Exception:
        (output / "failure.json").write_text(json.dumps({"completed_rows": len(rows),
            "seconds": time.monotonic() - started, "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    finally:
        data.close()


if __name__ == "__main__":
    main()
