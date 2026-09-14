"""Bounded train24 oracle-q functional check of the canonical source pullback.

Teacher labels are privileged diagnostic inputs, never deployment Writer inputs.
Reuse the fixed 96 conditions, eta and query predictions from f39d594f; the two
readouts are t1 endpoint and ten Euler steps, both with each query's own state.
Only the full-video PCA16 outlet is under test. No fitting or dense G is saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

SCHEMA = "process_pullback_function_v1"
ORIGINAL = Path("runs/analysis/native_corrective_transfer_20260913/formal")
DEMOS = (16, 17, 18, 19)


def read(path):
    return json.loads(path.read_text())


def write(path, value, *, replace=False):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
    try:
        if replace:
            os.replace(handle.name, path)
        else:
            os.link(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def source_assets(root):
    from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract

    reuse = read(root / "configs/pi05_writer_data_v1.json")["authorities"]
    authorities = load_evaluation_authorities(root / "configs/pi05_target_evaluation_v1.json", root)
    checkpoint = root / reuse["source_checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / reuse["lora_contract"]), rank=16)
    if len(contract.targets) != 38 or contract.rank != contract.alpha or contract.rank != 16:
        raise ValueError("pullback check needs the canonical 38-target rank16 scale-one contract")
    return reuse, source, contract


def manifest(root, source):
    from ember.writer.learning_data import load_learning_tasks

    ids = read(root / "configs/pi05_writer_data_v1.json")["authority_id_order"]["target_global_task_ids"]
    tasks = load_learning_tasks(root, ids)
    workers, queries, construction = {}, {}, {}
    for path in sorted((root / ORIGINAL).glob("worker_*.json")):
        worker = read(path)
        if (worker.get("complete") is not True or worker.get("smoke") is not False
                or worker.get("physical_source_trainable") != 0 or worker.get("optimizer_updates") != 0
                or worker.get("query_gradients") is not False or worker.get("source") != source):
            raise ValueError(f"original source reference identity changed: {path}")
        for task in worker["tasks"]:
            if task in workers:
                raise ValueError("original worker task coverage overlaps")
            workers[task] = str(path.resolve())
    if len(tasks) != 24 or set(workers) != set(tasks):
        raise ValueError("original reference must cover the complete fixed train24")
    for task, entry in tasks.items():
        base = root / ORIGINAL / f"task_{task:02d}"
        queries[task] = read(base / "queries.json")
        if len(queries[task]) != 16:
            raise ValueError("original diagnostic query panel must have 16 positions per task")
        for clip, row in enumerate(queries[task]):
            expected = {"clip": clip, "demo": 42 + clip // 4, "frame": row["frame"],
                        "action_start": row["frame"] + 1, "action_stop": row["frame"] + 16,
                        "noise_seed": 20260924 + 100 * task + clip}
            if row != expected or not 0 <= row["frame"] < entry.episode_lengths[row["demo"]] - 15:
                raise ValueError(f"original query/noise mapping changed: task {task} clip {clip}")
        with np.load(base / "reference.npz", allow_pickle=False) as reference:
            if (set(reference.files) != {"source", "labels"} or reference["source"].shape != (2, 16, 15, 7)
                    or reference["labels"].shape != (16, 15, 7)
                    or not all(np.isfinite(reference[key]).all() for key in reference.files)):
                raise ValueError("unsupported original source prediction/truth reference")
        for demo in DEMOS:
            path = base / f"demo_{demo}_state_free.json"
            record = read(path)
            valid = list(range(0, entry.episode_lengths[demo] - 15, 5))
            positions = [valid[len(valid) * k // 5] for k in range(1, 5)]
            expected = {"task": task, "suite": entry.suite, "teacher_demo": demo, "arm": "state_free",
                        "teacher_positions": positions, "teacher_observed_endpoints": [p + 15 for p in positions],
                        "teacher_actions": [[p + 1, p + 16] for p in positions]}
            eta = record["fit"]["eta"]
            if (len(set(positions)) != 4 or any(record.get(key) != value for key, value in expected.items())
                    or not np.isfinite(eta) or eta < 0
                    or not all(path.with_suffix(suffix).is_file() for suffix in (".npz", ".safetensors"))):
                raise ValueError(f"original teacher/eta/G authority changed: {path}")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as original:
                shapes = {"teacher_actions": (4, 15, 7), "teacher_source_velocity": (4, 15, 7),
                          "predictions": (2, 16, 15, 7)}
                if any(original[key].shape != shape or not np.isfinite(original[key]).all()
                       for key, shape in shapes.items()):
                    raise ValueError(f"original q/G prediction authority is invalid: {path}")
            construction[task, demo] = {"positions": positions, "eta": eta, "metadata": str(path.resolve())}
    return tasks, queries, construction, workers


class NativeVelocity(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, noise, flow_time):
        return self.policy.model.denoise_step(padding, cache, noise, flow_time)


@torch.no_grad()
def predict(runtime, samples, rows, state, query_chunk):
    """Original f39d594f stateful readout, batched without query derivatives."""
    from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device = runtime.state.probe.device
    replacements = {"policy." + name: value.to(device=device, dtype=runtime.policy.get_parameter(name).dtype)
                    for name, value in state.items()}
    owner, outputs = NativeVelocity(runtime.policy), []
    for start in range(0, len(samples), query_chunk):
        selected = samples[start:start + query_chunk]
        batch = {}
        for camera, target in (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb")):
            pixels = torch.stack([torch.as_tensor(q[f"observation.images.{camera}"]) for q in selected]).to(device)
            batch[f"observation.images.{target}"] = pixels.float().div(255) if pixels.dtype == torch.uint8 else pixels.float()
        states = torch.stack([torch.as_tensor(q["observation.state"]) for q in selected]).to(device)
        tokens, mask = runtime.processor._tokenize_prompts(states, [q["task"] for q in selected])
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens, mask
        prefix = prepare_execution_policy_prefix(runtime.policy, batch, native_precision=True)
        cache = prepare_prefix_kv_cache(runtime.policy, prefix, native_precision=True)
        noise = torch.stack([torch.randn((50, 32), generator=torch.Generator().manual_seed(row["noise_seed"]))
                             for row in rows[start:start + query_chunk]]).to(device)

        def velocity(value, clock):
            with torch.autocast(device.type, enabled=False):
                return torch.func.functional_call(owner, replacements, (prefix.padding, cache, value,
                    torch.full((len(value),), clock, device=device)), strict=False)

        endpoint = noise - velocity(noise, 1.)
        value = noise
        for step in range(10):
            value = value - .1 * velocity(value, 1. - .1 * step)
        outputs.append(torch.stack((endpoint[:, :15, :7], value[:, :15, :7])).float().cpu().numpy())
    return np.concatenate(outputs, axis=1)


def oracle_q(predictions, frame_indices, positions, truth, original_velocity, probe, eta):
    """T/4 cancels the canonical compiler's 1/T against the original 4-frame loss."""
    if truth.shape != (4, 15, 7) or original_velocity.shape != truth.shape:
        raise ValueError("oracle support must contain the original four 15x7 labels and velocities")
    indices = [int(value) for value in frame_indices]
    if len(set(indices)) != len(indices) or not set(positions) <= set(indices):
        raise ValueError("original support positions are absent from the complete sampled video")
    if predictions.shape != (len(indices), 50, 7):
        raise ValueError("source coordinates must cover every frame and all 50 action positions")
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be the original finite nonnegative value")
    residual = truth - (probe.detach().cpu()[None, :15, :7] - original_velocity)
    q = torch.zeros_like(predictions, device="cpu", dtype=torch.float32)
    q[[indices.index(position) for position in positions], :15] = -2. * eta * residual / (15 * 7) * (len(indices) / 4.)
    if not torch.isfinite(q).all():
        raise ValueError("oracle q is nonfinite")
    return q


def mse(predictions, labels):
    with np.errstate(invalid="ignore", over="ignore"):
        return ((predictions.astype(np.float64) - labels[None]) ** 2).mean(axis=(-2, -1))


def finite_list(values):
    return [[float(value) if np.isfinite(value) else None for value in row] for row in values]


@torch.no_grad()
def run_task(runtime, tasks, queries, construction, workers, dataset, videos, task, output, args):
    base = output / f"task_{task:02d}"
    base.mkdir(exist_ok=True)
    original = Path(workers[task]).parent / f"task_{task:02d}"
    rows, index = queries[task], dataset.task_episode_rows
    samples = [dataset[index[task][row["demo"]][row["frame"]]] for row in rows]
    with np.load(original / "reference.npz", allow_pickle=False) as reference:
        source, labels = reference["source"], reference["labels"]
    actual = runtime.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in samples])).cpu().numpy()
    if any(q["action_is_pad"].any() for q in samples) or not np.allclose(actual, labels, rtol=1e-6, atol=1e-7):
        raise ValueError("query labels differ from original authority")
    for demo in DEMOS:
        started = time.monotonic()
        condition = construction[task, demo]
        metadata = Path(condition["metadata"])
        with np.load(metadata.with_suffix(".npz"), allow_pickle=False) as old:
            truth = torch.from_numpy(old["teacher_actions"].copy())
            old_velocity = torch.from_numpy(old["teacher_source_velocity"].copy())
            old_predictions = old["predictions"].copy()
        if old_predictions.shape != source.shape or not np.isfinite(old_predictions).all():
            raise ValueError("original G prediction authority is invalid")
        teachers = [dataset[index[task][demo][position]] for position in condition["positions"]]
        actual_truth = runtime.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in teachers])).cpu()
        if any(q["action_is_pad"].any() for q in teachers) or not torch.allclose(actual_truth, truth, rtol=1e-6, atol=1e-7):
            raise ValueError("teacher labels differ from original eta authority")
        video = videos.load(task, demo)
        native = runtime.observer.prepare((torch.as_tensor(video.frames),), (torch.as_tensor(video.frame_indices),),
                                           tasks[task].authority.language)
        coordinates = runtime.correction.read(native)
        q = oracle_q(coordinates.predictions, video.frame_indices, condition["positions"], truth, old_velocity,
                     runtime.state.probe, condition["eta"])
        state = coordinates.compile(q.to(runtime.state.probe.device))
        predictions = predict(runtime, samples, rows, state, args.query_chunk)
        values = {"pullback": mse(predictions, labels), "original_G": mse(old_predictions, labels), "source": mse(source, labels)}
        tag = base / f"demo_{demo}"
        np.savez(tag.with_suffix(".npz"), predictions=predictions, original_G=old_predictions, source=source, labels=labels)
        if args.save_lora:
            from safetensors.torch import save_file
            save_file({name: value.detach().float().cpu().contiguous() for name, value in state.items()},
                      str(tag.with_suffix(".safetensors")))
        record = {"schema": SCHEMA, "task": task, "suite": tasks[task].suite, "teacher_demo": demo,
            "teacher_positions": condition["positions"], "eta": condition["eta"], "eta_refitted": False,
            "frames": len(video.frames), "raw_frame_count": video.raw_frame_count, "rank": 16, "target_count": 38,
            "teacher_state": "state_free", "query_state": "own_execution_state", "readouts": ["t1_endpoint", "full10"],
            "original_condition": str(metadata), "original_source_worker": workers[task],
            "reference": str(original / "reference.npz"), "queries": str(original / "queries.json"),
            "q": "original residual; four supports x first15 x 7; -2*eta*residual/(15*7)*T/4; zero elsewhere",
            "prediction_finite": bool(np.isfinite(predictions).all()),
            "mse": {key: finite_list(value) for key, value in values.items()},
            "privileged_oracle_only": True, "optimizer_updates": 0, "query_gradients": False,
            "complete": True, "wall_seconds": time.monotonic() - started}
        write(tag.with_suffix(".json"), record, replace=True)
        print(json.dumps({"task": task, "demo": demo, "frames": len(video.frames),
                          "prediction_finite": record["prediction_finite"], "wall_seconds": record["wall_seconds"]}), flush=True)
        del native, coordinates, q, state, video
    if any(p.requires_grad or p.grad is not None for p in runtime.policy.parameters()):
        raise RuntimeError("physical source must remain frozen with no accumulated gradients")
    write(base / "completion.json", {"schema": SCHEMA, "task": task, "conditions": 4, "complete": True}, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_process_pullback_writer.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--frame-chunk", type=int)
    parser.add_argument("--query-chunk", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed tasks; recompute incomplete tasks in place.")
    parser.add_argument("--save-lora", action="store_true", help="Optional complete low-rank factors; never dense G.")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    config_path = args.config if args.config.is_absolute() else Path(__file__).resolve().parents[1] / args.config
    config = read(config_path)
    if args.frame_chunk is not None:
        config["observer"]["frame_chunk"] = args.frame_chunk
    if (args.query_chunk < 1 or config["observer"]["frame_chunk"] < 1
            or config["observer"]["probe_seed"] != 1729 or config["observer"]["camera_view"] != "dual"
            or config["data"]["frame_stride"] != 5 or not config["data"]["include_last_frame"]):
        raise ValueError("positive chunks and original public probe/full dual-camera stride5 video are required")
    _, source, _ = source_assets(root)
    tasks, queries, construction, workers = manifest(root, source)
    selected = args.tasks if args.tasks is not None else list(tasks)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(tasks):
        raise ValueError("tasks must be a unique subset of the fixed train24")
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "tasks": selected, "conditions": len(selected) * 4,
                          "query_conditions": len(selected) * 64, "readouts": ["t1_endpoint", "full10"],
                          "original_reference_root": str(root / ORIGINAL), "GPU_used": False}))
        return
    if args.output is None:
        parser.error("--output is required unless --validate-only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = {"schema": SCHEMA, "source": source, "original_reference_root": str((root / ORIGINAL).resolve()),
        "original_implementation": "f39d594f", "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "observer": config["observer"], "model": config["model"], "query_chunk": args.query_chunk,
        "save_lora": args.save_lora, "demos": list(DEMOS), "tasks": list(tasks),
        "q_authority": "original teacher_source_velocity/teacher_actions/eta; -2eta*residual/(15*7)*T/4",
        "projection": "canonical NativeCorrectionReader: full stride5+last video x full50 native X PCA16",
        "readouts": ["own_execution_state_t1_endpoint", "own_execution_state_full10"],
        "rank": 16, "alpha": 16, "optimizer_updates": 0, "query_gradients": False,
        "privileged_oracle_only": True, "source_and_G_predictions_recomputed": False}
    try:
        write(output / "run_contract.json", registration)
    except FileExistsError:
        if read(output / "run_contract.json") != registration:
            raise ValueError("output root cannot mix implementations, source, config or numerical batch settings")
    pending = []
    for task in selected:
        base = output / f"task_{task:02d}"
        completion = base / "completion.json"
        if base.exists() and not args.resume:
            raise FileExistsError(f"existing task output requires --resume: {base}")
        if completion.exists():
            expected = {"schema": SCHEMA, "task": task, "conditions": 4, "complete": True}
            if read(completion) != expected or any(not (base / f"demo_{demo}{suffix}").is_file()
                    for demo in DEMOS for suffix in ((".json", ".npz", ".safetensors") if args.save_lora else (".json", ".npz"))):
                raise ValueError(f"incomplete or incompatible task completion: {completion}")
        else:
            pending.append(task)
    if not pending:
        print(json.dumps({"complete": True, "skipped_tasks": selected, "GPU_used": False}))
        return
    from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
    from ember.writer.runtime import build_runtime
    from ember.writer.topology import bind_current_process_to_cuda_numa

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if not affinity:
        raise RuntimeError("GPU-local NUMA placement is required")
    started = time.monotonic()
    runtime = build_runtime(root, config, torch.device("cuda:0"))
    runtime.state.requires_grad_(False)
    if runtime.source != source:
        raise ValueError("runtime source differs from original source identity")
    authorities = tuple(tasks[task].authority for task in pending)
    dataset = FunctionalQueryDataset(authorities, demo_indices=(*DEMOS, 42, 43, 44, 45),
                                     action_chunk_size=15, action_start_offset=1)
    videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
    try:
        for task in pending:
            run_task(runtime, tasks, queries, construction, workers, dataset, videos, task, output, args)
        torch.cuda.synchronize()
    finally:
        dataset.close()
        videos.close()
    write(output / ("worker_" + "_".join(map(str, selected)) + ".json"),
          {"schema": SCHEMA, "tasks": selected, "computed_tasks": pending, "complete": True,
           "source": source, "physical_source_trainable": 0, "optimizer_updates": 0, "query_gradients": False,
           "privileged_oracle_only": True, "numa_affinity": sorted(affinity),
           "wall_seconds": time.monotonic() - started,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}, replace=True)


if __name__ == "__main__":
    main()
