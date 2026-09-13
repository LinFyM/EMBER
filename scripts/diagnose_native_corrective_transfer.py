"""Bounded privileged native-gradient transfer audit; never a deployment Writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from safetensors.torch import save_file


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(root):
    from ember.writer.learning_data import load_learning_tasks

    task_ids = read(root / "configs/pi05_writer_data_v1.json")["authority_id_order"]["target_global_task_ids"]
    tasks = load_learning_tasks(root, task_ids)
    original = root / "runs/outputs/local_action_grounded_20260912/ordered/local_diagnostics.jsonl"
    queries = [json.loads(line) for line in original.read_text().splitlines()]
    queries = sorted((row for row in queries if row["step"] == 200), key=lambda r: (r["task"], r["clip"]))
    assert len(tasks) == 24 and len(queries) == 384
    assert {(r["task"], r["clip"]) for r in queries} == {(t, c) for t in tasks for c in range(16)}
    for row in queries:
        assert row["local_action_demo"] == 42 + row["clip"] // 4
        assert row["local_action_start"] == row["local_start_frame"] + 1
        assert row["local_action_stop"] == row["local_start_frame"] + 16
        assert row["gradients"] is False
    construction = {}
    for task, entry in tasks.items():
        for demo in range(16, 20):
            valid = list(range(0, entry.episode_lengths[demo] - 15, 5))
            positions = [valid[len(valid) * k // 5] for k in range(1, 5)]
            assert len(set(positions)) == 4
            construction[task, demo] = positions
    return tasks, queries, construction


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
    def prefix(self, observations, arm):
        from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = {}
        for source, target in (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb")):
            pixels = torch.stack([torch.as_tensor(q[f"observation.images.{source}"]) for q in observations]).to(self.device)
            batch[f"observation.images.{target}"] = pixels.float().div(255) if pixels.dtype == torch.uint8 else pixels.float()
        language = [q["task"] for q in observations]
        if arm == "state_free":
            assert all("observation.state" not in q for q in observations)
            tokens, mask, _ = self.tokenizer(language)
        else:
            assert arm == "state_true_oracle"
            states = torch.stack([torch.as_tensor(q["observation.state"]) for q in observations]).to(self.device)
            tokens, mask = self.processor._tokenize_prompts(states, language)
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

    @torch.no_grad()
    def predict(self, prefix, noise, state=None):
        endpoint = noise - self.velocity(prefix, noise, state=state)
        x = noise
        for step in range(10):
            x = x - .1 * self.velocity(prefix, x, 1. - .1 * step, state=state)
        result = torch.stack((endpoint[:, :15, :7], x[:, :15, :7])).float()
        if state is None and not torch.isfinite(result).all():
            raise RuntimeError("non-finite frozen source reference")
        return result.cpu().numpy()


def observation(query, arm):
    keys = ["observation.images.camera1", "observation.images.camera2", "task"]
    if arm == "state_true_oracle":
        keys.append("observation.state")
    return {key: query[key] for key in keys}


def run_task(operator, tasks, query_rows, construction, dataset, task, output, smoke):
    index = dataset.task_episode_rows
    rows = [row for row in query_rows if row["task"] == task]
    queries = [dataset[index[task][row["local_action_demo"]][row["local_start_frame"]]] for row in rows]
    assert len(queries) == 16 and all(not query["action_is_pad"].any() for query in queries)
    device = operator.device
    labels = operator.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in queries])).cpu().numpy()
    query_prefix = operator.prefix([observation(q, "state_true_oracle") for q in queries], "state_true_oracle")
    query_noise = torch.stack([torch.randn((50, 32), generator=torch.Generator().manual_seed(
        20260924 + 100 * task + row["clip"])) for row in rows]).to(device)
    source = operator.predict(query_prefix, query_noise)
    query_metadata = [{"clip": r["clip"], "demo": r["local_action_demo"], "frame": r["local_start_frame"],
                       "action_start": r["local_action_start"], "action_stop": r["local_action_stop"],
                       "noise_seed": 20260924 + 100 * task + r["clip"]} for r in rows]
    destination = output / f"task_{task:02d}"
    destination.mkdir()
    np.savez(destination / "reference.npz", source=source, labels=labels)
    write(destination / "queries.json", query_metadata)
    records = []
    for demo in ([16] if smoke else range(16, 20)):
        positions = construction[task, demo]
        teachers = [dataset[index[task][demo][position]] for position in positions]
        assert all(not q["action_is_pad"].any() for q in teachers)
        truth = operator.processor.normalize_action(torch.stack([torch.as_tensor(q["action"]) for q in teachers])).to(device)
        for arm in ("state_free", "state_true_oracle"):
            started = time.monotonic()
            prefix = operator.prefix([observation(q, arm) for q in teachers], arm)
            state, fit, teacher_arrays = operator.construct(prefix, truth)
            predictions = operator.predict(query_prefix, query_noise, state)
            tag = f"demo_{demo}_{arm}"
            save_file({name: value.detach().float().cpu().contiguous() for name, value in state.items()},
                      str(destination / f"{tag}.safetensors"))
            np.savez(destination / f"{tag}.npz", predictions=predictions, **teacher_arrays,
                     teacher_actions=truth.cpu().numpy())
            with np.errstate(invalid="ignore", over="ignore"):
                mse = ((predictions.astype(np.float64) - labels[None]) ** 2).mean(axis=(-2, -1))
            prediction_finite = bool(np.isfinite(predictions).all())
            source_mse = ((source.astype(np.float64) - labels[None]) ** 2).mean(axis=(-2, -1))
            mse_rows = [[float(value) if np.isfinite(value) else None for value in row] for row in mse]
            record = {"task": task, "suite": tasks[task].suite, "teacher_demo": demo, "arm": arm,
                      "teacher_positions": positions, "teacher_observed_endpoints": [p + 15 for p in positions],
                      "teacher_actions": [[p + 1, p + 16] for p in positions], "fit": fit,
                      "prediction_finite": prediction_finite, "mse_t1": mse_rows[0], "mse_full10": mse_rows[1],
                      "source_mse_t1": source_mse[0].tolist(), "source_mse_full10": source_mse[1].tolist(),
                      "wall_seconds": time.monotonic() - started}
            write(destination / f"{tag}.json", record)
            records.append(record)
            print(json.dumps({"task": task, "demo": demo, "arm": arm, "fit": {k: v for k, v in fit.items()
                              if k != "rank16_gradient_energy"},
                              "mse": mse.mean(1).tolist() if prediction_finite else None,
                              "source_mse": source_mse.mean(1).tolist(), "wall_seconds": record["wall_seconds"]}), flush=True)
            del prefix, state
    write(destination / "completion.json", {"task": task, "conditions": len(records), "complete": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    tasks, rows, construction = manifest(root)
    if args.validate_only:
        print(json.dumps({"tasks": list(tasks), "construction_conditions_per_arm": len(construction),
                          "construction_positions_per_arm": 4 * len(construction), "query_positions": len(rows),
                          "construction_demos": [16, 17, 18, 19], "query_demos": [42, 43, 44, 45]}))
        return
    assert args.output and args.tasks and len(set(args.tasks)) == len(args.tasks) and set(args.tasks) <= set(tasks)
    assert not args.smoke or args.tasks == [min(tasks)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from ember.writer.data import FunctionalQueryDataset
    from ember.writer.topology import bind_current_process_to_cuda_numa

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    started = time.monotonic()
    operator = SourceOperator(root, torch.device("cuda:0"))
    dataset = FunctionalQueryDataset(tuple(tasks[t].authority for t in args.tasks),
                                     demo_indices=(16, 17, 18, 19, 42, 43, 44, 45),
                                     action_chunk_size=15, action_start_offset=1)
    for task in args.tasks:
        run_task(operator, tasks, rows, construction, dataset, task, output, args.smoke)
    torch.cuda.synchronize()
    dataset.close()
    write(output / ("worker_" + "_".join(map(str, args.tasks)) + ".json"), {
        "tasks": args.tasks, "source": operator.source, "complete": True, "smoke": args.smoke,
        "physical_source_trainable": 0, "optimizer_updates": 0, "query_gradients": False,
        "privileged_oracle_only": True, "target_count": 38, "rank": 16, "numa_affinity": sorted(affinity),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})


if __name__ == "__main__":
    main()
