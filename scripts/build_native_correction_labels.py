"""One-time privileged training labels from the frozen source correction oracle.

The SourceOperator is transplanted from f39d594f's transfer audit, without its
query/evaluation path. Retire this entrypoint once the 624 labels are sealed.
No loss, autograd, action labels or task-local update belongs to deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from safetensors.torch import load_file, save_file


SCHEMA = "native_source_correction_labels_v1"
DEMOS = tuple(range(16, 42))
NEW_DEMOS = tuple(range(20, 42))
REUSED_ROOT = Path("runs/analysis/native_corrective_transfer_20260913/formal")
SOURCE_COMMIT = "f39d594f"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(root):
    from ember.writer.learning_data import load_learning_tasks

    ids = read(root / "configs/pi05_writer_data_v1.json")["authority_id_order"]["target_global_task_ids"]
    tasks = load_learning_tasks(root, ids)
    if len(tasks) != 24 or 0 not in tasks:
        raise ValueError("fixed train24 authority changed")
    construction = {}
    for task, entry in tasks.items():
        for demo in DEMOS:
            valid = list(range(0, entry.episode_lengths[demo] - 15, 5))
            positions = [valid[len(valid) * k // 5] for k in range(1, 5)]
            if len(set(positions)) != 4:
                raise ValueError(f"task {task} demo {demo} lacks four legal construction positions")
            construction[task, demo] = positions
    return tasks, construction


class NativeVelocity(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, noise, flow_time):
        return self.policy.model.denoise_step(padding, cache, noise, flow_time)


class SourceOperator:
    def __init__(self, root, device):
        from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
        from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
        from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
        from ember.pi05_source_setup import load_policy
        from ember.writer.functional import prepare_frozen_writer_policy

        reuse = read(root / "configs/pi05_writer_data_v1.json")["authorities"]
        authorities = load_evaluation_authorities(root / "configs/pi05_target_evaluation_v1.json", root)
        checkpoint = root / reuse["source_checkpoint"]
        self.source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
        self.policy = load_policy(Path(self.source["model_path"]), read(root / reuse["source_base_config"]), device)
        self.contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / reuse["lora_contract"]), rank=16)
        prepare_frozen_writer_policy(self.policy, self.contract)
        self.policy.model.gradient_checkpointing_disable()
        self.owner = NativeVelocity(self.policy)
        tokenizer = root / reuse["tokenizer"]
        stats = read(root / reuse["source_normalization"])["stats"]
        self.processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(device))
        self.tokenizer = Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device))
        self.device = device
        self.weight_names = tuple(t.name + ".base_layer.weight" for t in self.contract.targets)
        self.weights = tuple(self.policy.get_parameter(name).detach() for name in self.weight_names)
        self.probe = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(1729)).to(device)
        assert len(self.weights) == 38 and not any(p.requires_grad for p in self.policy.parameters())

    @torch.no_grad()
    def prefix(self, observations):
        from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = {}
        for source, target in (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb")):
            pixels = torch.stack([torch.as_tensor(q[f"observation.images.{source}"]) for q in observations]).to(self.device)
            batch[f"observation.images.{target}"] = pixels.float().div(255) if pixels.dtype == torch.uint8 else pixels.float()
        language = [q["task"] for q in observations]
        assert all("observation.state" not in q for q in observations)
        tokens, mask, _ = self.tokenizer(language)
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens, mask
        prefix = prepare_execution_policy_prefix(self.policy, batch, native_precision=True)
        cache = prepare_prefix_kv_cache(self.policy, prefix, native_precision=True)
        return prefix.padding, cache

    def velocity(self, prefix, noise, clock=1., *, state=None, weights=None):
        assert state is None or weights is None
        arguments = (*prefix, noise, torch.full((len(noise),), clock, device=self.device))
        replacements = {}
        if state is not None:
            replacements = {"policy." + name: value.to(self.policy.get_parameter(name).dtype)
                            for name, value in state.items()}
        if weights is not None:
            replacements = {"policy." + name: value for name, value in zip(self.weight_names, weights, strict=True)}
        with torch.autocast(self.device.type, enabled=False):
            return torch.func.functional_call(self.owner, replacements, arguments, strict=False)

    def construct(self, prefix, truth):
        from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, identity_lora_state, validate_lora_state

        noise = self.probe.expand(len(truth), -1, -1)
        target = noise[:, :15, :7] - truth
        # Same-valued temporary arguments carry derivatives; source registration stays frozen.
        leaves = tuple(weight.detach().requires_grad_(True) for weight in self.weights)
        prediction = self.velocity(prefix, noise, weights=leaves)[:, :15, :7]
        residual = prediction.float() - target
        loss = residual.square().mean()
        assert torch.isfinite(loss)
        gradients = torch.autograd.grad(loss, leaves)
        factors, directions, energies = {}, [], []
        with torch.no_grad(), torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.manual_seed(20260923)
            for spec, gradient, weight in zip(self.contract.targets, gradients, self.weights, strict=True):
                gradient = gradient.float()
                assert torch.isfinite(gradient).all()
                rank = self.contract.rank
                u, singular, v = torch.svd_lowrank(gradient, q=min(24, *gradient.shape), niter=2)
                a = singular[:rank].sqrt()[:, None] * v[:, :rank].T
                b = -u[:, :rank] * singular[:rank].sqrt()[None, :]
                factors[spec.name + LORA_A_SUFFIX] = a
                factors[spec.name + LORA_B_SUFFIX] = b
                directions.append((b @ a).to(weight.dtype))
                energies.append(float(singular[:rank].square().sum() / gradient.square().sum().clamp_min(1e-30)))
        residual = residual.detach()
        before = float(loss.detach())
        del prediction, loss, gradients, leaves

        def response(*parameters):
            return self.velocity(prefix, noise, weights=parameters)[:, :15, :7].float()

        # Reverse-over-reverse JVP evaluates one fixed direction, without parameter updates.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            _, derivative = torch.autograd.functional.jvp(response, self.weights, tuple(directions), create_graph=False)
        numerator = -(residual * derivative).sum()
        denominator = derivative.square().sum()
        if not torch.isfinite(derivative).all():
            raise RuntimeError("non-finite native directional derivative")
        eta = 0.5 * max(0., float(numerator / denominator)) if float(denominator) > 0 else 0.
        assert np.isfinite(eta)
        del directions
        if eta == 0:
            factors = identity_lora_state(self.contract, device=self.device)
        else:
            for name in factors:
                if name.endswith(LORA_B_SUFFIX):
                    factors[name] = factors[name] * eta
        validate_lora_state(factors, self.contract)
        assert all(torch.isfinite(value).all() for value in factors.values())
        with torch.no_grad():
            actual = self.velocity(prefix, noise, state=factors)[:, :15, :7].float()
            after = float((actual.double() - target.double()).square().mean())
            linear = float((residual + eta * derivative).square().mean())
        assert not any(p.requires_grad for p in self.policy.parameters())
        arrays = {"teacher_source_velocity": (target + residual).cpu().numpy(),
                  "teacher_directional_velocity": derivative.cpu().numpy(),
                  "teacher_actual_velocity": actual.detach().cpu().numpy()}
        return factors, {"teacher_mse_before": before, "teacher_mse_after": after if np.isfinite(after) else None,
                         "teacher_prediction_finite": bool(torch.isfinite(actual).all()),
                         "teacher_linear_mse": linear, "eta": eta,
                         "direction_numerator": float(numerator), "direction_denominator": float(denominator),
                         "rank16_gradient_energy": energies}, arrays


def condition_paths(root, output, task, demo):
    if demo < 20:
        base = root / REUSED_ROOT / f"task_{task:02d}" / f"demo_{demo}_state_free"
    else:
        base = output / f"task{task}_demo{demo}"
    return tuple(base.with_suffix(suffix) for suffix in (".safetensors", ".json", ".npz"))


def run_task(operator, tasks, construction, dataset, task, output, smoke):
    index = dataset.task_episode_rows
    for demo in ((20,) if smoke else NEW_DEMOS):
        paths = condition_paths(Path(), output, task, demo)
        if any(path.exists() for path in paths):
            raise FileExistsError(f"refusing to overwrite condition {task}/{demo}")
        positions = construction[task, demo]
        teachers = [dataset[index[task][demo][position]] for position in positions]
        if any(q["action_is_pad"].any() for q in teachers):
            raise ValueError("construction actions must all be observed and unpadded")
        truth = operator.processor.normalize_action(
            torch.stack([torch.as_tensor(q["action"]) for q in teachers])).to(operator.device)
        observations = [{key: q[key] for key in (
            "observation.images.camera1", "observation.images.camera2", "task")}
            for q in teachers]
        started = time.monotonic()
        prefix = operator.prefix(observations)
        state, fit, arrays = operator.construct(prefix, truth)
        save_file({name: value.detach().float().cpu().contiguous() for name, value in state.items()}, str(paths[0]))
        np.savez(paths[2], **arrays, teacher_actions=truth.cpu().numpy())
        record = {
            "schema": SCHEMA, "task": task, "suite": tasks[task].suite, "teacher_demo": demo,
            "arm": "state_free", "teacher_positions": positions,
            "teacher_observed_endpoints": [p + 15 for p in positions],
            "teacher_actions": [[p + 1, p + 16] for p in positions], "fit": fit,
            "source_trainable": 0, "optimizer_updates": 0, "query_gradients": False,
            "privileged_training_labels_only": True, "source_operator_commit": SOURCE_COMMIT,
            "complete": True, "smoke": smoke, "wall_seconds": time.monotonic() - started,
        }
        write(paths[1], record)
        print(json.dumps({key: record[key] for key in ("task", "teacher_demo", "fit", "wall_seconds")}), flush=True)
        del prefix, state


def check_condition(root, output, task, demo, positions, contract):
    """Audit only teacher construction arrays, including for reused oracle files."""
    from ember.lora import validate_lora_state

    adapter, metadata, raw = condition_paths(root, output, task, demo)
    record = read(metadata)
    expected = {
        "task": task, "teacher_demo": demo, "arm": "state_free", "teacher_positions": positions,
        "teacher_observed_endpoints": [p + 15 for p in positions],
        "teacher_actions": [[p + 1, p + 16] for p in positions],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"condition authority differs: {metadata}")
    if demo >= 20 and (record.get("schema") != SCHEMA or record.get("complete") is not True
                       or record.get("smoke") is not False or record.get("source_trainable") != 0
                       or record.get("optimizer_updates") != 0 or record.get("query_gradients") is not False):
        raise ValueError(f"new condition is incomplete or diagnostic-only: {metadata}")
    with np.load(raw, allow_pickle=False) as stored:
        # Reused NPZs also contain query predictions. They are deliberately never accessed.
        arrays = {name: stored[name].astype(np.float64) for name in (
            "teacher_actions", "teacher_source_velocity", "teacher_directional_velocity", "teacher_actual_velocity")}
    if any(value.shape != (4, 15, 7) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"invalid teacher truth/source/directional/actual arrays: {raw}")
    probe = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(1729))[:, :15, :7].numpy()
    target = probe - arrays["teacher_actions"]
    residual = arrays["teacher_source_velocity"] - target
    derivative = arrays["teacher_directional_velocity"]
    numerator = float(-(residual * derivative).sum())
    denominator = float(np.square(derivative).sum())
    eta = .5 * max(0., numerator / denominator) if denominator > 0 else 0.
    actual = arrays["teacher_actual_velocity"]
    expected_fit = {
        "direction_numerator": numerator, "direction_denominator": denominator, "eta": eta,
        "teacher_mse_before": float(np.square(residual).mean()),
        "teacher_mse_after": float(np.square(actual - target).mean()),
        "teacher_linear_mse": float(np.square(residual + eta * derivative).mean()),
    }
    fit = record["fit"]
    for name, value in expected_fit.items():
        if fit.get(name) is None or not np.isclose(fit[name], value, rtol=1e-5, atol=1e-7):
            raise ValueError(f"raw teacher amplitude/MSE mismatch for {name}: {metadata}")
    energy = np.asarray(fit["rank16_gradient_energy"], dtype=np.float64)
    if fit.get("teacher_prediction_finite") is not True or energy.shape != (38,) or not np.isfinite(energy).all():
        raise ValueError(f"nonfinite teacher construction: {metadata}")
    state = load_file(str(adapter), device="cpu")
    validate_lora_state(state, contract)
    if len(state) != 76 or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"invalid complete rank16 adapter: {adapter}")
    return {"task": task, "demo": demo, "adapter": str(adapter.resolve())}


def seal(root, output, tasks, construction):
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract

    if (output / "registration.json").exists() or (output / "completion.json").exists():
        raise FileExistsError("label seal already exists; no overwrite")
    reuse = read(root / "configs/pi05_writer_data_v1.json")["authorities"]
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / reuse["lora_contract"]), rank=16)
    covered = set()
    for worker_path in output.glob("worker_*.json"):
        worker = read(worker_path)
        if worker.get("smoke") is True:
            raise ValueError("smoke output must not be mixed with the full label build")
        if (worker.get("complete") is not True or worker.get("source_trainable") != 0
                or worker.get("optimizer_updates") != 0 or worker.get("query_gradients") is not False):
            raise ValueError(f"worker is incomplete: {worker_path}")
        worker_tasks = set(worker["tasks"])
        if covered & worker_tasks or not worker_tasks <= set(tasks):
            raise ValueError("worker task coverage overlaps or escapes train24")
        covered.update(worker_tasks)
    if covered != set(tasks):
        raise ValueError("all 24 task workers must finish before sealing")
    started = time.monotonic()
    entries = [check_condition(root, output, task, demo, construction[task, demo], contract)
               for task in tasks for demo in DEMOS]
    if len(entries) != 624 or sum(entry["demo"] >= 20 for entry in entries) != 528:
        raise ValueError("label coverage changed")
    write(output / "registration.json", {
        "schema": SCHEMA, "tasks": list(tasks), "demos": list(DEMOS), "rank": 16, "target_count": 38,
        "state_contract": "state_free", "entries": entries,
        "privileged_training_labels_only": True, "source_operator_commit": SOURCE_COMMIT,
        "probe_seed": 1729, "flow_time": 1., "rank_projection_seed": 20260923,
        "source_frame_stride": 5, "query_reads": 0,
    })
    completion = {
        "schema": SCHEMA, "status": "complete", "episodes": 624, "new_episodes": 528, "reused_episodes": 96,
        "source_trainable": 0, "optimizer_updates": 0, "query_gradients": False,
        "privileged_training_labels_only": True, "raw_teacher_audit": "passed",
        "seal_wall_seconds": time.monotonic() - started,
    }
    write(output / "completion.json", completion)
    print(json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    tasks, construction = manifest(root)
    selected = args.tasks if args.tasks is not None else ([0] if args.smoke else list(tasks))
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(tasks):
        raise ValueError("--tasks must be a unique subset of fixed train24")
    if args.smoke and selected != [0]:
        raise ValueError("smoke is fixed to task0/demo20")
    if args.seal and set(selected) != set(tasks):
        raise ValueError("seal requires complete train24 coverage")
    if args.validate_only:
        print(json.dumps({
            "schema": SCHEMA, "tasks": list(tasks), "selected_tasks": selected, "demos": list(DEMOS),
            "episodes": len(construction), "new_episodes": len(tasks) * len(NEW_DEMOS), "reused_episodes": 96,
            "construction_positions": 4 * len(construction), "query_reads": 0,
            "state_contract": "state_free", "privileged_training_labels_only": True,
        }))
        return
    if args.output is None:
        parser.error("--output is required for build, smoke and seal")
    output = args.output.resolve()
    if args.seal:
        seal(root, output, tasks, construction)
        return
    output.mkdir(parents=True, exist_ok=True)
    worker_path = output / ("worker_" + "_".join(map(str, selected)) + ".json")
    if worker_path.exists():
        raise FileExistsError(f"worker record already exists: {worker_path}")
    # Check every destination before loading a GPU model; partial data are never overwritten.
    for task in selected:
        for demo in ((20,) if args.smoke else NEW_DEMOS):
            if any(path.exists() for path in condition_paths(root, output, task, demo)):
                raise FileExistsError(f"condition already exists: {task}/{demo}")
    from ember.writer.data import FunctionalQueryDataset
    from ember.writer.topology import bind_current_process_to_cuda_numa

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    started = time.monotonic()
    operator = SourceOperator(root, torch.device("cuda:0"))
    demos = (20,) if args.smoke else NEW_DEMOS
    dataset = FunctionalQueryDataset(tuple(tasks[t].authority for t in selected), demo_indices=demos,
                                     action_chunk_size=15, action_start_offset=1)
    try:
        for task in selected:
            run_task(operator, tasks, construction, dataset, task, output, args.smoke)
        torch.cuda.synchronize()
    finally:
        dataset.close()
    write(worker_path, {
        "schema": SCHEMA, "tasks": selected, "demos": list(demos), "source": operator.source,
        "episodes": len(selected) * len(demos), "complete": True, "smoke": args.smoke,
        "source_trainable": sum(p.numel() for p in operator.policy.parameters() if p.requires_grad),
        "optimizer_updates": 0, "query_gradients": False, "privileged_training_labels_only": True,
        "target_count": 38, "rank": 16, "numa_affinity": sorted(affinity),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    })


if __name__ == "__main__":
    main()
