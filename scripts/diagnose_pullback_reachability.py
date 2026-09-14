"""Train-only matched terminal-Writer/free-q/free-LoRA reachability diagnostic.

This is offline support fitting, never a deployable Writer or initialization.
The canonical source compiler, official full50 FM and native full10 inference
own all model computation. Heldout readouts never enter optimization/selection.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import default_collate

from ember.lora import validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.function_credit import flow_sample, paired_functional_credit
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    scoped_policy_randomness,
)
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import frozen_authority, inspect_writer_checkpoint, source_matches
from ember.writer.native import autocast
from ember.writer.runtime import build_runtime

SCHEMA = "train_pullback_reachability_task_v1"
SEED = 20260915
TEACHER = 16
ARMS = ("writer900", "free_q", "free_lora")
SUPPORT = tuple(range(17, 42))
HELDOUT = tuple(range(42, 46))


def read(path):
    return json.loads(path.read_text())


def write(path, value, *, exclusive=False):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.link(handle.name, path) if exclusive else os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def tensor_file(path, tensors, *, metadata=None):
    save_file({name: value.detach().float().cpu().contiguous() for name, value in tensors.items()},
              str(path), metadata=metadata)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def optimizer_options(smoke):
    return {"lr": 1., "max_iter": 2 if smoke else 32, "max_eval": 48, "history_size": 10,
            "tolerance_grad": 1e-7, "tolerance_change": 1e-9, "line_search_fn": "strong_wolfe"}


def sample_panel(dataset, task, pool, count, seed):
    """Same episode-uniform/frame-uniform replacement sampling as WriterTrainingData."""
    rng, rows, selected = random.Random(seed), dataset.task_episode_rows[task], []
    for _ in range(count):
        demo = rng.choice(pool)
        frame = rng.randrange(len(rows[demo]))
        selected.append((demo, frame, rows[demo][frame]))
    raw = default_collate([dataset[index] for _, _, index in selected])
    return raw, {"sampling_seed": seed, "pool": list(pool), "count": count,
        "sampling": "uniform_episode_then_uniform_frame_with_replacement",
        "unique_positions": len({(demo, frame) for demo, frame, _ in selected}),
        "action_demos": [demo for demo, _, _ in selected], "action_frames": [frame for _, frame, _ in selected],
        "action_start_indices": [frame + 1 for _, frame, _ in selected], "action_chunk_size": 50,
        "action_start_offset": 1, "padding": "repeat_last_action; official FM includes all50"}


def sliced(batch, start, stop, total):
    return {name: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim and len(value) == total else value
            for name, value in batch.items()}


@dataclass
class Panel:
    batch: dict
    seeds: tuple[int, ...]
    trace: dict
    full10_noise: torch.Tensor


@torch.no_grad()
def prepare_panel(runtime, dataset, task, pool, count, realizations, rng, chunk, path):
    raw, trace = sample_panel(dataset, task, pool, count, rng.getrandbits(63))
    batch = runtime.processor.training_batch(raw)
    seeds = tuple(rng.getrandbits(63) for _ in range(realizations))
    noises, times = [], []
    for seed in seeds:
        samples = [flow_sample(runtime.policy, sliced(batch, start, min(start + chunk, count), count),
                    seed=seed, device=runtime.observer.device, random_batch=count, offset=start)
                   for start in range(0, count, chunk)]
        noises.append(torch.cat([sample.arguments[-2].cpu() for sample in samples]))
        times.append(torch.cat([sample.arguments[-1].cpu() for sample in samples]))
    full10_seed = rng.getrandbits(63)
    with scoped_policy_randomness(full10_seed, runtime.observer.device):
        noise = runtime.policy.model.sample_noise((count, 50, 32), runtime.observer.device).cpu()
    np.savez(path.with_suffix(".npz"), flow_noise=torch.stack(noises).numpy(), flow_time=torch.stack(times).numpy(),
             full10_noise=noise.numpy(), normalized_actions=batch["action"].cpu().numpy(),
             action_is_pad=raw["action_is_pad"].numpy())
    trace.update({"policy_rng_seeds": list(seeds), "policy_random_batch_size": count,
        "policy_batch_offset": 0, "microbatch_offsets": list(range(0, count, chunk)),
        "flow_time_sampling_scheme": INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        "flow_noise_sampling_scheme": INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        "flow_realizations_per_position": realizations, "prediction_count": count * realizations,
        "full10_noise_seed": full10_seed, "full10_noise_independent_of_fm": True,
        "random_tensors_and_labels": str(path.with_suffix(".npz").resolve()), "gradients": pool == SUPPORT})
    write(path.with_suffix(".json"), trace)
    return Panel(batch, seeds, trace, noise)


def fm(runtime, state, panel, chunk, *, backward):
    loss, gradients = 0., {}
    for seed in panel.seeds:
        with autocast(runtime.observer.device):
            credit = paired_functional_credit(runtime.policy, state, runtime.lora, panel.batch,
                seed=seed, device=runtime.observer.device, random_batch=panel.trace["count"], offset=0,
                microbatch=min(chunk, panel.trace["count"]), condition_weight=1. / len(panel.seeds), backward=backward)
        loss += credit["flow_loss"] / len(panel.seeds)
        for name, value in credit["lora_cotangent"].items():
            if name in gradients:
                gradients[name].add_(value)
            else:
                gradients[name] = value
    return loss, gradients


class ActionPrediction(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, noise):
        return self.policy.predict_action_chunk(batch, noise=noise, num_steps=10)


@torch.no_grad()
def full10(runtime, state, panel, chunk):
    device, owner = runtime.observer.device, ActionPrediction(runtime.policy)
    replacements = {"policy." + name: value.to(device=device, dtype=runtime.policy.get_parameter(name).dtype)
                    for name, value in state.items()}
    predictions, count = [], panel.trace["count"]
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        with torch.autocast(device.type, enabled=False):
            value = torch.func.functional_call(owner, replacements,
                (sliced(panel.batch, start, stop, count), panel.full10_noise[start:stop].to(device)), strict=False)
        predictions.append(value.float().cpu())
    result = torch.cat(predictions)
    if result.shape != (count, 50, 7):
        raise ValueError("full10 changed actual horizon/action dimensions")
    return result


@torch.no_grad()
def readouts(runtime, state, support, heldout, chunk, path):
    result, arrays = {}, {}
    for name, panel in (("support", support), ("heldout", heldout)):
        loss, _ = fm(runtime, state, panel, chunk, backward=False)
        predictions = full10(runtime, state, panel, chunk)
        labels = panel.batch["action"].cpu()
        per_query = (predictions.double() - labels.double()).square().mean((1, 2))
        finite = math.isfinite(loss) and bool(torch.isfinite(per_query).all())
        result[name] = {"fm": loss if math.isfinite(loss) else None,
                       "full10_mse": float(per_query.mean()) if finite else None, "finite": finite,
                       "positions": panel.trace["count"], "fm_predictions": panel.trace["prediction_count"]}
        arrays[f"{name}_full10_predictions"] = predictions.numpy()
        arrays[f"{name}_full10_mse"] = per_query.numpy()
    np.savez(path, **arrays)
    return result


def optimize(runtime, coordinates, initial_q, initial_state, support, chunk, arm, options, path):
    variables = {"q": initial_q.detach().clone().requires_grad_(True)} if arm == "free_q" else {
        name: value.detach().clone().requires_grad_(True) for name, value in initial_state.items()}
    optimizer = torch.optim.LBFGS(list(variables.values()), **options)
    history, calls, stage, started = [], 0, "not_started", time.monotonic()

    def closure():
        nonlocal calls, stage
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        stage = "compile"
        with torch.no_grad():
            state = coordinates.compile(variables["q"]) if arm == "free_q" else variables
        stage = "support_fm"
        loss, credit = fm(runtime, state, support, chunk, backward=True)
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite support FM in LBFGS closure")
        if arm == "free_q":
            stage = "compiler_adjoint"
            variables["q"].grad = coordinates.adjoint(credit).to(variables["q"])
        else:
            for name, value in variables.items():
                value.grad = credit[name].to(value)
        if any(value.grad is None or not torch.isfinite(value.grad).all() for value in variables.values()):
            raise FloatingPointError("nonfinite or missing optimization gradient")
        stage = "gradient_check"
        row = {"closure": calls, "support_fm": loss,
               "lbfgs_n_iter": int(optimizer.state[next(iter(variables.values()))].get("n_iter", 0)),
               "gradient_max": max(float(value.grad.abs().max()) for value in variables.values()),
               "wall_seconds": time.monotonic() - started}
        history.append(row)
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        return torch.tensor(loss, device=runtime.observer.device)

    error, final, step_returned = None, None, False
    try:
        optimizer.step(closure)
        step_returned = True
        with torch.no_grad():
            final = coordinates.compile(variables["q"]) if arm == "free_q" else variables
        validate_lora_state(final, runtime.lora)
        if any(not torch.isfinite(value).all() for value in final.values()):
            raise FloatingPointError("nonfinite final factors")
        final = {name: value.detach() for name, value in final.items()}
    except (RuntimeError, ValueError, FloatingPointError) as failure:
        error = {"type": type(failure).__name__, "message": str(failure),
                 "classification": "resource_failure" if isinstance(failure, torch.cuda.OutOfMemoryError)
                 else "numerical_or_contract_failure", "capacity_impossible": False}
    native = optimizer.state[next(iter(variables.values()))]
    step = native.get("t")
    report = {"algorithm": "torch.optim.LBFGS", "options": options, "returned_normally": error is None,
        "optimizer_returned_normally": step_returned,
        "error": error, "closure_calls": calls, "successful_closures": len(history),
        "last_closure_stage": stage, "func_evals": int(native.get("func_evals", 0)),
        "n_iter": int(native.get("n_iter", 0)), "optimizer_step_calls": 1,
        "optimized_tensor_count": len(variables), "optimized_parameter_count": sum(v.numel() for v in variables.values()),
        "last_step_length": float(step) if step is not None and math.isfinite(float(step)) else None,
        "soft_max_eval_exceeded": calls > options["max_eval"],
        "iteration_limit_reached": int(native.get("n_iter", 0)) >= options["max_iter"],
        "last_closure_gradient_max": history[-1]["gradient_max"] if history else None,
        "line_search_status": "not exposed by native LBFGS; normal return does not certify Wolfe/convergence",
        "convergence": "not certified", "heldout_used_by_optimizer": False,
        "closure_trace": str(path.resolve()), "wall_seconds": time.monotonic() - started}
    return final if error is None else None, variables.get("q"), report


@torch.no_grad()
def initial_condition(runtime, videos, task):
    video = videos.load(task.authority.task_id, TEACHER)
    if video.raw_frame_count != task.episode_lengths[TEACHER]:
        raise ValueError("teacher frame count differs from fixed task authority")
    native = runtime.observer.prepare((torch.from_numpy(video.frames),), (torch.from_numpy(video.frame_indices),),
                                      task.authority.language)
    with autocast(runtime.observer.device):
        responses, inputs = runtime.observer.read(native)
        coordinates = runtime.correction.read(native)
        encoded = runtime.state.writer.encode(responses, *inputs)
        q = runtime.state.writer.action_cotangents(encoded, coordinates.predictions)
        state = coordinates.compile(q)
    return q.detach(), {name: value.detach() for name, value in state.items()}, coordinates, {
        "demo_index": TEACHER, "raw_frame_count": video.raw_frame_count,
        "sampled_frame_count": len(video.frame_indices), "frame_indices": video.frame_indices.tolist(),
        "hdf5": str(task.authority.path), "frame_stride": 5, "include_last_frame": True, "camera_view": "dual"}


def arm_record(path, state, runtime, name, metrics, optimization):
    validate_lora_state(state, runtime.lora)
    if len(state) != 76 or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("diagnostic adapter must contain finite complete 38-target rank16 factors")
    adapter = tensor_file(path, state, metadata={"schema": SCHEMA, "arm": name, "offline_diagnostic": "true"})
    return {"arm": name, "adapter": adapter, "single_complete_rank16": True,
        "rank": 16, "alpha": 16, "target_count": 38, "factor_count": 76,
        "factor_shapes": {key: list(value.shape) for key, value in state.items()},
        "generation": "frozen_terminal_Writer900" if name == "writer900" else "offline_support_fitted_" + name,
        "only_q_optimized": name == "free_q", "only_lora_factors_optimized": name == "free_lora",
        "source_and_writer_frozen": True, "support_only_gradients": True,
        "optimizer_updates": optimization.get("n_iter", 0),
        "optimizer_updates_semantics": "native LBFGS attempted inner iterations; zero for Writer900",
        "optimization": optimization, "final_readouts": metrics,
        "readout_file": str(path.with_name(path.stem + "_readouts.npz").resolve())}


def run_task(runtime, dataset, videos, task, output, args, checkpoint, source):
    task_id, started = task.authority.task_id, time.monotonic()
    torch.cuda.reset_peak_memory_stats(runtime.observer.device)
    base = output / f"task_{task_id:02d}"
    base.mkdir(exist_ok=True)
    rng = random.Random(SEED + task_id)
    support = prepare_panel(runtime, dataset, task_id, SUPPORT, 64, 2, rng, args.query_chunk, base / "support")
    heldout = prepare_panel(runtime, dataset, task_id, HELDOUT, 128, 1, rng, args.query_chunk, base / "heldout")
    q, state, coordinates, video = initial_condition(runtime, videos, task)
    initial = readouts(runtime, state, support, heldout, args.query_chunk, base / "writer900_readouts.npz")
    records = {"writer900": arm_record(base / "writer900.safetensors", state, runtime, "writer900", initial, {})}
    records["writer900"]["q"] = tensor_file(base / "writer900_q.safetensors", {"q": q})
    for name in ARMS[1:]:
        trace = base / f"{name}_closures.jsonl"
        trace.write_text("")  # Incomplete tasks restart from the same frozen condition, never a partial fit.
        final, final_q, report = optimize(runtime, coordinates, q, state, support, args.query_chunk,
                                           name, optimizer_options(args.smoke), trace)
        write(base / f"{name}_optimizer.json", report)
        if final is None:
            records[name] = {"arm": name, "usable": False, "optimization": report}
            continue
        metrics = readouts(runtime, final, support, heldout, args.query_chunk, base / f"{name}_readouts.npz")
        records[name] = arm_record(base / f"{name}.safetensors", final, runtime, name, metrics, report)
        if final_q is not None:
            records[name]["q"] = tensor_file(base / "free_q_values.safetensors", {"q": final_q})
        del final, final_q
    if any(p.requires_grad or p.grad is not None for p in (*runtime.policy.parameters(), *runtime.state.parameters())):
        raise RuntimeError("source and complete Writer/Meta must stay frozen")
    usable = all("adapter" in record and all(row["finite"] for row in record["final_readouts"].values())
                 for record in records.values())
    manifest = {"schema": SCHEMA, "global_task_id": task_id, "suite": task.suite, "task_id": task.suite_task_id,
        "language": task.authority.language, "teacher_demo_indices": [TEACHER], "teacher_videos": [video],
        "source": source, "writer_checkpoint": checkpoint, "smoke": args.smoke, "diagnostic_git": args.diagnostic_git,
        "run_contract": str((output / "run_contract.json").resolve()),
        "support": support.trace, "heldout": heldout.trace, "initial_readouts": initial, "arms": records,
        "offline_diagnostic_only": True, "future_initialization_allowed": False, "checkpoint_selection": False,
        "source_trainable": 0, "writer_meta_trainable": 0, "heldout_gradients": False,
        "complete": True, "all_arms_usable": usable, "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.observer.device) / 2**30}
    write(base / "manifest.json", manifest)
    print(json.dumps({"task": task_id, "all_arms_usable": usable, "wall_seconds": manifest["wall_seconds"]}), flush=True)
    return usable


def prepare_runtime(root, run, checkpoint_path, args):
    from ember.writer.topology import bind_current_process_to_cuda_numa
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    if not bind_current_process_to_cuda_numa(0):
        raise RuntimeError("GPU-local NUMA placement is required")
    config = copy.deepcopy(run["config"])
    config["observer"]["frame_chunk"] = args.frame_chunk
    runtime = build_runtime(root, config, torch.device("cuda:0"))
    runtime.state.load_state_dict(load_file(str(checkpoint_path / "ecp.safetensors"), device="cuda:0"), strict=True)
    runtime.state.requires_grad_(False).eval()
    runtime.policy.eval()
    expected = torch.randn(50, 32, generator=torch.Generator().manual_seed(int(config["observer"]["probe_seed"])))
    if (not source_matches(runtime.source, run["source"]) or runtime.lora.rank != 16 or len(runtime.lora.targets) != 38
            or not torch.equal(runtime.state.probe.cpu(), expected)):
        raise ValueError("runtime source, public probe or full38 rank16 contract differs from authority")
    return runtime


def run_pending(root, run, checkpoint_path, args, tasks, pending, output, checkpoint):
    runtime = prepare_runtime(root, run, checkpoint_path, args)
    authorities = tuple(tasks[task].authority for task in pending)
    dataset = FunctionalQueryDataset(authorities, demo_indices=(*SUPPORT, *HELDOUT), action_chunk_size=50, action_start_offset=1)
    videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
    failures = []
    try:
        for task in pending:
            try:
                if not run_task(runtime, dataset, videos, tasks[task], output, args, checkpoint, run["source"]):
                    failures.append(task)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                write(output / f"task_{task:02d}" / "failure.json", {"complete": False,
                    "error_type": type(error).__name__, "message": str(error), "capacity_impossible": False})
                raise
    finally:
        dataset.close()
        videos.close()
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--frame-chunk", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only first train task/demo16, max_iter2; separate result root.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root, checkpoint_path = args.asset_root.resolve(), args.checkpoint.resolve()
    run, checkpoint = inspect_writer_checkpoint(checkpoint_path)
    args.diagnostic_git = git_state(Path(__file__).resolve().parents[1])
    tasks = load_learning_tasks(root, run["config"]["data"]["task_ids"])
    selected = args.tasks if args.tasks is not None else ([next(iter(tasks))] if args.smoke else list(tasks))
    if (checkpoint["macro"] != 900 or len(tasks) != 24 or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(tasks) or min(args.frame_chunk, args.query_chunk) < 1
            or (args.smoke and selected != [next(iter(tasks))])):
        raise ValueError("requires terminal900, fixed train24 subsets, positive chunks; smoke only first train task")
    contract = {"schema": SCHEMA, "writer_checkpoint": checkpoint, "source": run["source"],
        "tasks": list(tasks), "teacher_demo": TEACHER, "support_pool": list(SUPPORT), "heldout_pool": list(HELDOUT),
        "support_positions": 64, "support_fm_realizations": 2, "heldout_positions": 128, "heldout_fm_realizations": 1,
        "seed": SEED, "smoke": args.smoke, "optimizer": optimizer_options(args.smoke),
        "frame_chunk": args.frame_chunk, "query_chunk": args.query_chunk,
        "fm": "official full50x7 paired_functional_credit; fixed independent beta time/Gaussian noise",
        "full10": "native predict_action_chunk; independent saved fixed noise; support and heldout initial/end only",
        "selection": "none; native LBFGS returned iterate, heldout never used in optimizer",
        "initialization": "identical full q900 and A/B900; source and entire Writer/Meta frozen",
        "torch_version": str(torch.__version__), "diagnostic_git": args.diagnostic_git,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip()}
    if args.validate_only:
        print(json.dumps({**contract, "selected_tasks": selected, "GPU_used": False}))
        return
    if args.output is None:
        parser.error("--output is required for GPU diagnostics")
    if not frozen_authority(args.diagnostic_git):
        raise ValueError("diagnostic execution requires a clean pushed detached frozen checkout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        write(output / "run_contract.json", contract, exclusive=True)
    except FileExistsError:
        if read(output / "run_contract.json") != contract:
            raise ValueError("diagnostic output cannot mix contracts, smoke, implementation or checkpoints")
    pending, failures = [], []
    for task in selected:
        base = output / f"task_{task:02d}"
        if base.exists() and not args.resume:
            raise FileExistsError(f"existing task requires --resume: {base}")
        if (base / "manifest.json").is_file():
            record = read(base / "manifest.json")
            if record.get("complete") is not True or record.get("schema") != SCHEMA or record["global_task_id"] != task:
                raise ValueError("invalid task completion manifest")
            if not record["all_arms_usable"]:
                failures.append(task)
        else:
            pending.append(task)
    if pending:
        failures.extend(run_pending(root, run, checkpoint_path, args, tasks, pending, output, checkpoint))
    write(output / ("worker_" + "_".join(map(str, selected)) + ".json"),
          {"schema": SCHEMA, "tasks": selected, "computed_tasks": pending, "failed_tasks": failures,
           "complete": True, "smoke": args.smoke})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
