"""Disposable train24 local-cotangent operator audit; never a deployment Writer.

Source loading and prediction follow f39d594f's native corrective transfer audit.
Eta, source predictions and query identities are read from its original records.
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
from safetensors.torch import load_file, save_file

SCHEMA = "local_correction_field_operator_v1"
ORIGINAL = Path("runs/analysis/native_corrective_transfer_20260913/formal")
DEMOS = (16, 17, 18, 19)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
    try:
        os.link(handle.name, path)  # Publish a complete record atomically without overwriting.
    finally:
        Path(handle.name).unlink()


def source_assets(root):
    from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract

    reuse = read(root / "configs/pi05_writer_data_v1.json")["authorities"]
    authorities = load_evaluation_authorities(root / "configs/pi05_target_evaluation_v1.json", root)
    checkpoint = root / reuse["source_checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / reuse["lora_contract"]), rank=16)
    if len(contract.targets) != 38 or contract.rank != contract.alpha or contract.rank != 16:
        raise ValueError("local field needs the canonical 38-target rank16 scale-one contract")
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
            construction[task, demo] = {"positions": positions, "eta": eta, "metadata": str(path.resolve())}
    return tasks, queries, construction, workers


class NativeVelocity(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, noise, flow_time):
        return self.policy.model.denoise_step(padding, cache, noise, flow_time)


class SourceOperator:
    def __init__(self, root, device, reuse, source, contract):
        from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
        from ember.pi05_source_setup import load_policy
        from ember.writer.functional import prepare_frozen_writer_policy

        self.source, self.contract, self.device = source, contract, device
        self.policy = load_policy(Path(source["model_path"]), read(root / reuse["source_base_config"]), device)
        prepare_frozen_writer_policy(self.policy, contract)
        self.policy.model.gradient_checkpointing_disable()
        self.owner = NativeVelocity(self.policy)
        tokenizer = root / reuse["tokenizer"]
        self.processor = Pi05LiberoProcessor(read(root / reuse["source_normalization"])["stats"], tokenizer, 200, str(device))
        self.tokenizer = Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device))
        self.weight_names = tuple(t.name + ".base_layer.weight" for t in contract.targets)
        self.weights = tuple(self.policy.get_parameter(name).detach() for name in self.weight_names)
        self.probe = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(1729)).to(device)

    @torch.no_grad()
    def prefix(self, observations, *, own_state=False):
        from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = {}
        for camera, target in (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb")):
            pixels = torch.stack([torch.as_tensor(q[f"observation.images.{camera}"]) for q in observations]).to(self.device)
            batch[f"observation.images.{target}"] = pixels.float().div(255) if pixels.dtype == torch.uint8 else pixels.float()
        language = [q["task"] for q in observations]
        if own_state:
            states = torch.stack([torch.as_tensor(q["observation.state"]) for q in observations]).to(self.device)
            tokens, mask = self.processor._tokenize_prompts(states, language)
        else:
            if any("observation.state" in q for q in observations):
                raise ValueError("teacher prefix must be state-free")
            tokens, mask, _ = self.tokenizer(language)
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens, mask
        prefix = prepare_execution_policy_prefix(self.policy, batch, native_precision=True)
        return prefix.padding, prepare_prefix_kv_cache(self.policy, prefix, native_precision=True)

    def velocity(self, prefix, noise, clock=1., *, state=None, weights=None):
        if state is not None and weights is not None:
            raise ValueError("source weight leaves and one functional LoRA are exclusive")
        replacements = {}
        if state is not None:
            replacements = {"policy." + name: value.to(self.policy.get_parameter(name).dtype) for name, value in state.items()}
        if weights is not None:
            replacements = {"policy." + name: value for name, value in zip(self.weight_names, weights, strict=True)}
        with torch.autocast(self.device.type, enabled=False):
            return torch.func.functional_call(self.owner, replacements,
                (*prefix, noise, torch.full((len(noise),), clock, device=self.device)), strict=False)

    def construct(self, prefix, truth, eta, *, check_identity=False):
        from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state

        if truth.shape != (4, 15, 7) or not np.isfinite(eta) or eta < 0:
            raise ValueError("construction requires four original 15x7 labels and a fixed nonnegative eta")
        captured = [[] for _ in self.contract.targets]
        handles = [self.policy.get_submodule(spec.name + ".base_layer").register_forward_hook(
            lambda module, args, output, i=i: captured[i].append((args[0].detach(), output)))
            for i, spec in enumerate(self.contract.targets)]
        leaves = tuple(weight.detach().requires_grad_(True) for weight in self.weights)
        try:
            noise = self.probe.expand(4, -1, -1)
            prediction = self.velocity(prefix, noise, weights=leaves)[:, :15, :7]
            loss = (prediction.float() - (noise[:, :15, :7] - truth)).square().mean()
            if not torch.isfinite(loss) or any(len(values) != 1 for values in captured):
                raise ValueError("source forward must yield one finite, complete local-field graph")
            outputs = tuple(values[0][1] for values in captured)
            gradients = torch.autograd.grad(loss, (*outputs, *leaves) if check_identity else outputs)
        finally:
            for handle in handles:
                handle.remove()
        before = float(loss.detach())
        factors, fields, energies = {}, {}, []
        identity_error = identity_energy = 0.
        with torch.no_grad(), torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []):
            torch.manual_seed(20260923)
            for i, (spec, values) in enumerate(zip(self.contract.targets, captured, strict=True)):
                x, y = values[0]
                if x.shape != (4, 50, spec.in_features) or y.shape != (4, 50, spec.out_features):
                    raise ValueError(f"local field lost the actual four-position/full50 linear axes: {spec.name}")
                x, c = x.flatten(0, 1).float(), (-4. * eta * gradients[i].float()).flatten(0, 1)
                if not torch.isfinite(x).all() or not torch.isfinite(c).all():
                    raise ValueError("nonfinite bare-source input or local correction cotangent")
                fields[spec.name] = c.reshape(4, 50, -1).cpu().contiguous()
                u, singular, v = torch.svd_lowrank(c, q=min(24, *c.shape), niter=2)
                rank = min(self.contract.rank, len(singular))
                a, b = x.new_zeros(self.contract.rank, spec.in_features), c.new_zeros(spec.out_features, self.contract.rank)
                a[:rank] = (u[:, :rank] * singular[:rank]).T @ x / 4.
                b[:, :rank] = v[:, :rank]
                factors[spec.name + LORA_A_SUFFIX], factors[spec.name + LORA_B_SUFFIX] = a, b
                energy = float(c.square().sum())
                energies.append({"target": spec.name, "C_shape": [4, 50, spec.out_features], "C_energy": energy,
                    "rank16_C_energy_fraction": float(singular[:rank].square().sum()) / energy if energy > 0 else 0.})
                if check_identity:
                    expected = -eta * gradients[len(outputs) + i].float()
                    identity_error += float((c.T @ x / 4. - expected).double().square().sum())
                    identity_energy += float(expected.double().square().sum())
        validate_lora_state(factors, self.contract)
        if (not all(torch.isfinite(value).all() for value in factors.values())
                or any(p.requires_grad or p.grad is not None for p in self.policy.parameters())):
            raise ValueError("local factors must be finite and the physical source must stay frozen")
        fit = {"teacher_mse_before": before, "eta": eta, "eta_refitted": False, "local_field_energy": energies}
        if check_identity:
            fit["cotangent_weight_gradient_relative_error"] = (identity_error / max(identity_energy, 1e-30)) ** .5
        return factors, fields, fit

    @torch.no_grad()
    def predict(self, prefix, noise, state):
        endpoint = noise - self.velocity(prefix, noise, state=state)
        x = noise
        for step in range(10):
            x = x - .1 * self.velocity(prefix, x, 1. - .1 * step, state=state)
        return torch.stack((endpoint[:, :15, :7], x[:, :15, :7])).float().cpu().numpy()


def update_difference(state, old, contract):
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state

    validate_lora_state(old, contract)
    error = energy = 0.
    for spec in contract.targets:
        a, b, oa, ob = [group[spec.name + suffix].double() for group in (state, old)
                       for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)]
        joined_a, joined_b = torch.cat((a, oa)), torch.cat((b, -ob), dim=1)
        error += float(((joined_a @ joined_a.T) * (joined_b.T @ joined_b)).sum())
        energy += float(((oa @ oa.T) * (ob.T @ ob)).sum())
    if not np.isfinite(error) or not np.isfinite(energy):
        raise ValueError("original G or new total update is nonfinite")
    ratio = max(0., error) / max(energy, 1e-30)
    return {"total_update_relative_squared_error_old_G": ratio,
            "total_update_relative_frobenius_error_old_G": ratio ** .5}


def run_task(operator, tasks, queries, construction, workers, dataset, task, output, smoke):
    base = output / f"task_{task:02d}"
    base.mkdir()
    original = Path(workers[task]).parent / f"task_{task:02d}"
    index, rows = dataset.task_episode_rows, queries[task]
    samples = [dataset[index[task][row["demo"]][row["frame"]]] for row in rows]
    with np.load(original / "reference.npz", allow_pickle=False) as reference:
        source, labels = reference["source"], reference["labels"]
    actual_labels = operator.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in samples])).cpu().numpy()
    if any(q["action_is_pad"].any() for q in samples) or not np.allclose(actual_labels, labels, rtol=1e-6, atol=1e-7):
        raise ValueError("current own-state query labels differ from the original reference; no recomputation permitted")
    query_prefix = operator.prefix(samples, own_state=True)
    query_noise = torch.stack([torch.randn((50, 32), generator=torch.Generator().manual_seed(row["noise_seed"]))
                               for row in rows]).to(operator.device)
    source_mse = ((source.astype(np.float64) - labels[None]) ** 2).mean(axis=(-2, -1))
    demos = (16,) if smoke else DEMOS
    for demo in demos:
        started = time.monotonic()
        condition = construction[task, demo]
        teachers = [dataset[index[task][demo][p]] for p in condition["positions"]]
        truth = operator.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in teachers])).to(operator.device)
        metadata = Path(condition["metadata"])
        with np.load(metadata.with_suffix(".npz"), allow_pickle=False) as old:
            if (any(q["action_is_pad"].any() for q in teachers)
                    or not np.allclose(truth.cpu().numpy(), old["teacher_actions"], rtol=1e-6, atol=1e-7)):
                raise ValueError("construction truth differs from the original eta authority")
        observations = [{key: q[key] for key in ("observation.images.camera1", "observation.images.camera2", "task")} for q in teachers]
        prefix = operator.prefix(observations)
        state, fields, fit = operator.construct(prefix, truth, condition["eta"], check_identity=smoke)
        saved_state = {name: value.detach().float().cpu().contiguous() for name, value in state.items()}
        fit.update(update_difference(saved_state, load_file(str(metadata.with_suffix(".safetensors"))), operator.contract))
        predictions = operator.predict(query_prefix, query_noise, state)
        tag = base / f"demo_{demo}"
        save_file(saved_state, str(tag.with_suffix(".safetensors")))
        save_file(fields, str(base / f"demo_{demo}_field.safetensors"))
        np.savez(tag.with_suffix(".npz"), predictions=predictions)
        with np.errstate(invalid="ignore", over="ignore"):
            mse = ((predictions.astype(np.float64) - labels[None]) ** 2).mean(axis=(-2, -1))
        finite_mse = [[float(v) if np.isfinite(v) else None for v in row] for row in mse]
        record = {"schema": SCHEMA, "task": task, "suite": tasks[task].suite, "teacher_demo": demo,
                  "teacher_positions": condition["positions"], "teacher_actions": [[p + 1, p + 16] for p in condition["positions"]],
                  "teacher_observed_endpoints": [p + 15 for p in condition["positions"]], "eta": condition["eta"],
                  "fit": fit, "original_condition": condition["metadata"], "original_source_worker": workers[task],
                  "source_checkpoint": operator.source["checkpoint"], "source_training_commit": operator.source["source_training_commit"],
                  "rank": 16, "target_count": 38, "field_file": str(base / f"demo_{demo}_field.safetensors"),
                  "reference": str(original / "reference.npz"), "queries": str(original / "queries.json"),
                  "prediction_finite": bool(np.isfinite(predictions).all()), "mse_t1": finite_mse[0], "mse_full10": finite_mse[1],
                  "source_mse_t1": source_mse[0].tolist(), "source_mse_full10": source_mse[1].tolist(),
                  "physical_source_trainable": 0, "optimizer_updates": 0, "query_gradients": False,
                  "privileged_oracle_only": True, "complete": True, "smoke": smoke, "wall_seconds": time.monotonic() - started}
        write(tag.with_suffix(".json"), record)
        print(json.dumps({"task": task, "demo": demo, "eta": fit["eta"], "prediction_finite": record["prediction_finite"],
                          "wall_seconds": record["wall_seconds"]}), flush=True)
        del prefix, state, fields, saved_state
    write(base / "completion.json", {"schema": SCHEMA, "task": task, "conditions": len(demos), "complete": True, "smoke": smoke})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    reuse, source, contract = source_assets(root)
    tasks, queries, construction, workers = manifest(root, source)
    selected = args.tasks if args.tasks is not None else ([0] if args.smoke else list(tasks))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(tasks) or (args.smoke and selected != [0]):
        raise ValueError("tasks must be a unique train24 subset; smoke is fixed to task0/demo16")
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "tasks": list(tasks), "selected_tasks": selected, "demos": list(DEMOS),
            "construction_conditions": len(selected) * 4, "construction_positions": len(selected) * 16,
            "query_positions": len(selected) * 16, "predicted_query_conditions": len(selected) * 64,
            "raw_C_float32_bytes": len(selected) * 4 * 4 * 50 * 4 * sum(t.out_features for t in contract.targets),
            "original_reference_root": str(root / ORIGINAL), "source_checkpoint": source["checkpoint"], "GPU_used": False}))
        return
    if args.output is None:
        parser.error("--output is required for the operator diagnostic or smoke")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_path = output / ("worker_" + "_".join(map(str, selected)) + ".json")
    if worker_path.exists() or any((output / f"task_{task:02d}").exists() for task in selected):
        raise FileExistsError("refusing to overwrite an existing worker or condition directory")
    registration = {"schema": SCHEMA, "smoke": args.smoke, "source": source, "original_reference_root": str(root / ORIGINAL),
        "source_operator_commit": "f39d594f", "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "C": "-4*eta*d(mean_four_15x7_t1_loss)/dy", "LoRA": "B=V16; A=(U16*s16).T@X/4",
        "horizon": 50, "rank": 16, "alpha": 16, "svd_q_max": 24, "svd_niter": 2, "svd_seed": 20260923,
        "teacher_probe_seed": 1729, "eta_refitted": False, "source_reference_recomputed": False}
    try:
        write(output / "run_contract.json", registration)
    except FileExistsError:
        if read(output / "run_contract.json") != registration:
            raise ValueError("diagnostic roots cannot mix smoke/formal, source or implementation identities")
    from ember.writer.data import FunctionalQueryDataset
    from ember.writer.topology import bind_current_process_to_cuda_numa

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if not affinity:
        raise RuntimeError("GPU-local NUMA placement is required")
    started = time.monotonic()
    operator = SourceOperator(root, torch.device("cuda:0"), reuse, source, contract)
    dataset = FunctionalQueryDataset(tuple(tasks[t].authority for t in selected),
        demo_indices=(*(DEMOS if not args.smoke else (16,)), 42, 43, 44, 45), action_chunk_size=15, action_start_offset=1)
    try:
        for task in selected:
            run_task(operator, tasks, queries, construction, workers, dataset, task, output, args.smoke)
        torch.cuda.synchronize()
    finally:
        dataset.close()
    write(worker_path, {"schema": SCHEMA, "tasks": selected, "source": source, "complete": True, "smoke": args.smoke,
        "physical_source_trainable": sum(p.numel() for p in operator.policy.parameters() if p.requires_grad),
        "optimizer_updates": 0, "query_gradients": False, "privileged_oracle_only": True, "target_count": 38, "rank": 16,
        "numa_affinity": sorted(affinity), "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})


if __name__ == "__main__":
    main()
