"""Fixed E0--E3 diagnostics for the old and coverage Writer trajectories.

This module owns measurement mechanics only. It never selects checkpoints or
changes the registered training runs. Experiment orchestration and immutable
asset identities live in the companion script and design document.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data._utils.collate import default_collate

from ember.batched_lora import BatchedLoRAInference
from ember.lora import copy_task_lora_state_, task_lora_state_dict
from ember.pi05_eval_contract import load_evaluation_authorities
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_policy
from ember.source_sft.contract import authority_path, load_source_sft_config
from ember.writer.data import FunctionalQueryDataset
from ember.writer.function_credit import (
    NativeFlowPrediction,
    flow_sample,
    mean_velocity_loss,
    paired_functional_credit,
)
from ember.writer.learning_data import WriterTrainingData
from ember.writer.runtime import VideoConditionCache, autocast, build_runtime
from ember.writer.training import STAGE, _optimization


DIAGNOSTIC_SCHEMA = "ember_writer_stability_diagnostics_v1"
HIGH_LR = 1.6279650115e-4
LOW_LR = 2.959936e-5
TEACHING_WEIGHT = 1.0 / 3.0
PROBE_TEACHER_QUERY = ((46, 47), (48, 49))
PROBE_FRACTIONS = (0.25, 0.75)
PROBE_NOISE_SEEDS = (2026092101, 2026092102)


@dataclass
class LoadedWriter:
    name: str
    checkpoint: Path
    run: dict[str, Any]
    trainer: dict[str, Any]
    runtime: Any
    optimizer: torch.optim.Optimizer
    parameter_names: tuple[str, ...]


@dataclass
class FrozenPolicy:
    name: str
    policy: torch.nn.Module
    processor: Pi05LiberoProcessor
    device: torch.device
    generated_state: Mapping[str, torch.Tensor] | None = None


def parameter_group(name: str) -> str:
    """Return the mutually exclusive scientific owner of a Writer parameter."""
    groups = {
        "text_meta": ".text_meta_lora.",
        "vl_meta": ".vl_meta_lora.",
        "action_meta": ".action_meta_lora.",
    }
    matches = [group for group, marker in groups.items() if marker in name]
    if len(matches) > 1:
        raise ValueError(f"Writer parameter belongs to multiple groups: {name}")
    return matches[0] if matches else "writer_main"


def grouped_gradient_metrics(
    names: Sequence[str], gradients: Sequence[torch.Tensor | None]
) -> dict[str, float]:
    accum = {group: 0.0 for group in ("writer_main", "text_meta", "vl_meta", "action_meta")}
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is not None:
            accum[parameter_group(name)] += float(gradient.detach().float().square().sum())
    return {group + "_grad_norm": math.sqrt(value) for group, value in accum.items()}


def gradient_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    if len(left) != len(right):
        raise ValueError("gradient vectors have different parameter counts")
    return sum(float(a.float().flatten().dot(b.float().flatten())) for a, b in zip(left, right, strict=True))


def _finite_tensor_mapping(values: Mapping[str, torch.Tensor], *, label: str) -> None:
    if not values or any(not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError(f"{label} is empty or nonfinite")


def _checkpoint_files(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if (
        manifest.get("schema_version") != "ember_ecp_checkpoint_v1"
        or manifest.get("stage") != STAGE
        or trainer.get("schema_version") != "ember_ecp_checkpoint_v1"
        or trainer.get("stage") != STAGE
        or int(manifest.get("next_macro", -1)) != int(trainer.get("next_macro", -2))
    ):
        raise ValueError("Writer diagnostic parent checkpoint contract changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Writer diagnostic checkpoint file changed: {path}")
    return manifest, trainer


def load_writer(
    *, name: str, checkpoint: Path, asset_root: Path, device: torch.device, fixed_lr: float | None = None
) -> LoadedWriter:
    checkpoint = checkpoint.resolve()
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    manifest, trainer = _checkpoint_files(checkpoint)
    if int(run["topology"]["world_size"]) != int(manifest["world_size"]):
        raise ValueError("Writer run and checkpoint topology differ")
    runtime = build_runtime(asset_root, run["config"], device)
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.train()
    optimizer, _scheduler = _optimization(runtime.state, run["config"])
    optimizer.load_state_dict(trainer["optimizer"])
    if fixed_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(fixed_lr)
            group["initial_lr"] = float(fixed_lr)
    names = tuple(name for name, _ in runtime.state.named_parameters())
    validate_optimizer_state(runtime.state, optimizer, names)
    return LoadedWriter(name, checkpoint, run, trainer, runtime, optimizer, names)


def validate_optimizer_state(
    state: torch.nn.Module, optimizer: torch.optim.Optimizer, names: Sequence[str]
) -> dict[str, Any]:
    parameters = tuple(state.parameters())
    grouped = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    if (tuple(names) != tuple(name for name, _ in state.named_parameters())
            or len(grouped) != len(parameters)
            or any(left is not right for left, right in zip(grouped, parameters, strict=True))):
        raise ValueError("Writer optimizer parameter order differs from named model order")
    if len(optimizer.state) != len(parameters):
        raise ValueError("Writer optimizer does not own every Writer parameter")
    steps = set()
    for name, parameter in zip(names, parameters, strict=True):
        cell = optimizer.state.get(parameter)
        if set(cell or {}) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Writer AdamW state fields changed: {name}")
        if cell["exp_avg"].shape != parameter.shape or cell["exp_avg_sq"].shape != parameter.shape:
            raise ValueError(f"Writer AdamW state shape changed: {name}")
        if not torch.isfinite(cell["exp_avg"]).all() or not torch.isfinite(cell["exp_avg_sq"]).all():
            raise FloatingPointError(f"Writer AdamW state is nonfinite: {name}")
        steps.add(float(cell["step"].item()))
    if len(steps) != 1:
        raise ValueError("Writer AdamW parameters have different step cursors")
    return {"parameters": len(parameters), "optimizer_states": len(optimizer.state), "adam_step": steps.pop()}


def current_training_data(
    asset_root: Path, current_run: Mapping[str, Any], *, planned_updates: int
) -> WriterTrainingData:
    return WriterTrainingData(
        asset_root,
        current_run["config"]["data"],
        camera_view=current_run["config"]["observer"]["camera_view"],
        planned_updates=planned_updates,
    )


def event_window(data: WriterTrainingData, *, first_step: int, last_step: int) -> list[tuple[dict[str, Any], ...]]:
    if not 1 <= first_step <= last_step:
        raise ValueError("invalid one-based diagnostic event window")
    rows = []
    for step in range(1, last_step + 1):
        draws = data.next_iteration()
        if step >= first_step:
            rows.append(draws)
    return rows


def _condition_cotangent(
    loaded: LoadedWriter,
    data: WriterTrainingData,
    cache: VideoConditionCache,
    draw: Mapping[str, Any],
    *,
    teaching: bool,
    condition_weight: float,
) -> tuple[tuple, dict[str, torch.Tensor], dict[str, Any]]:
    condition = cache.condition(int(draw["task"]), draw["video_demos"])
    with torch.no_grad():
        generated = loaded.runtime.compile(condition)
    raw, trace = data.action_batch(
        int(draw["task"]), int(draw["occurrence"]), draw["video_demos"],
        query_seed=int(draw["query_seed"]), query_offset=0,
        query_count=7 if teaching else 21, teaching=teaching,
    )
    batch = loaded.runtime.processor.training_batch(raw)
    with autocast(loaded.runtime.device):
        credit = paired_functional_credit(
            loaded.runtime.policy,
            generated,
            loaded.runtime.lora,
            batch,
            seed=int(trace["policy_rng_seed"]),
            device=loaded.runtime.device,
            random_batch=int(trace["policy_random_batch_size"]),
            offset=0,
            microbatch=min(int(loaded.run["config"]["runtime"]["policy_microbatch"]), len(trace["action_demos"])),
            condition_weight=float(condition_weight),
            backward=True,
            noise_endpoint=teaching,
            prefix_steps=5 if teaching else None,
        )
    cotangent = credit.pop("lora_cotangent")
    _finite_tensor_mapping(cotangent, label="LoRA cotangent")
    return condition, cotangent, {**credit, **trace}


def writer_condition_gradient(
    loaded: LoadedWriter,
    data: WriterTrainingData,
    cache: VideoConditionCache,
    draw: Mapping[str, Any],
    *,
    component: str,
    task_weight: float = 1.0,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Measure one full Writer gradient without mutating parameters or AdamW."""
    if component not in {"main", "teaching", "joint"}:
        raise ValueError("unknown Writer gradient component")
    loaded.optimizer.zero_grad(set_to_none=True)
    condition = None
    credits: dict[str, Any] = {}
    total: dict[str, torch.Tensor] = {}
    for teaching, label in ((False, "main"), (True, "teaching")):
        if component != "joint" and component != label:
            continue
        scale = task_weight * (TEACHING_WEIGHT if teaching and component == "joint" else 1.0)
        observed_condition, cotangent, credit = _condition_cotangent(
            loaded, data, cache, draw, teaching=teaching, condition_weight=scale
        )
        condition = observed_condition if condition is None else condition
        for name, value in cotangent.items():
            total[name] = value if name not in total else total[name] + value
        credits[label + "_loss"] = float(credit["flow_loss"])
        credits[label + "_lora_grad_norm"] = float(torch.stack([v.float().norm() for v in cotangent.values()]).norm())
    generated = loaded.runtime.compile(condition)
    torch.autograd.backward(tuple(generated.values()), tuple(total[name].to(value) for name, value in generated.items()))
    gradients = tuple(
        torch.zeros_like(parameter, device="cpu") if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
        for parameter in loaded.runtime.state.parameters()
    )
    if any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("Writer parameter gradient is nonfinite")
    return gradients, {**credits, **grouped_gradient_metrics(loaded.parameter_names, gradients)}


def combine_gradients(
    destination: Sequence[torch.nn.Parameter], gradients: Sequence[Sequence[torch.Tensor]], weights: Sequence[float]
) -> None:
    if not gradients or len(gradients) != len(weights):
        raise ValueError("invalid gradient combination")
    for index, parameter in enumerate(destination):
        parameter.grad = sum(
            gradient[index].to(device=parameter.device, dtype=parameter.dtype).mul(float(weight))
            for gradient, weight in zip(gradients, weights, strict=True)
        )


def optimizer_step(
    loaded: LoadedWriter, *, gradients: Sequence[Sequence[torch.Tensor]] | None, weights: Sequence[float] | None,
    lr: float, grad_clip: float = 1.0,
) -> dict[str, float]:
    loaded.optimizer.zero_grad(set_to_none=True)
    if gradients is not None:
        combine_gradients(tuple(loaded.runtime.state.parameters()), gradients, weights or ())
    for group in loaded.optimizer.param_groups:
        group["lr"] = float(lr)
    parameters = tuple(loaded.runtime.state.parameters())
    before = tuple(value.detach().float().cpu().clone() for value in parameters)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)) if gradients is not None else 0.0
    loaded.optimizer.step()
    displacement = math.sqrt(sum(
        float((parameter.detach().float().cpu() - prior).square().sum())
        for parameter, prior in zip(parameters, before, strict=True)
    ))
    return {"preclip_grad_norm": norm, "parameter_displacement_norm": displacement, "lr": float(lr)}


def snapshot_parent(loaded: LoadedWriter) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return (
        {name: value.detach().cpu().clone() for name, value in loaded.runtime.state.state_dict().items()},
        copy.deepcopy(loaded.optimizer.state_dict()),
    )


def restore_parent(loaded: LoadedWriter, snapshot: tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]) -> None:
    model, optimizer = snapshot
    loaded.runtime.state.load_state_dict(model, strict=True)
    loaded.optimizer.load_state_dict(optimizer)


def save_diagnostic_checkpoint(loaded: LoadedWriter, output: Path, *, local_step: int, parent_step: int) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu().contiguous() for name, value in loaded.runtime.state.state_dict().items()},
              str(output / "ecp.safetensors"))
    torch.save({"optimizer": loaded.optimizer.state_dict(), "local_step": local_step, "parent_step": parent_step},
               output / "trainer_state.pt")
    return output


def probe_dataset(asset_root: Path, data: WriterTrainingData) -> FunctionalQueryDataset:
    return FunctionalQueryDataset(
        tuple(task.authority for task in data.tasks.values()),
        demo_indices=(47, 49), action_chunk_size=50, action_start_offset=1,
    )


def probe_items(dataset: FunctionalQueryDataset, task: int) -> Iterable[tuple[int, int, int, dict[str, Any]]]:
    for pair_index, (_teacher, query_demo) in enumerate(PROBE_TEACHER_QUERY):
        rows = dataset.task_episode_rows[task][query_demo]
        for fraction_index, fraction in enumerate(PROBE_FRACTIONS):
            position = min(len(rows) - 1, int(round((len(rows) - 1) * fraction)))
            yield pair_index, fraction_index, position, dataset[rows[position]]


def _prediction_metrics(prediction: torch.Tensor, target: torch.Tensor, width: int) -> dict[str, Any]:
    error = (prediction[..., :width].float() - target[..., :width].float()).square()
    return {
        "full_mse": float(error.mean()),
        "first5_mse": float(error[:, :5].mean()),
        "first5_time_mse": error[:, :5].mean(dim=(0, 2)).detach().cpu().tolist(),
        "first5_dimension_mse": error[:, :5].mean(dim=(0, 1)).detach().cpu().tolist(),
    }


@torch.no_grad()
def probe_one(
    frozen: FrozenPolicy,
    raw: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    from lerobot.utils.constants import ACTION

    batch = frozen.processor.training_batch(default_collate([raw]))
    state = frozen.generated_state
    sample = flow_sample(frozen.policy, batch, seed=seed, device=frozen.device, random_batch=1, offset=0)
    endpoint = flow_sample(frozen.policy, batch, seed=seed, device=frozen.device, random_batch=1, offset=0,
                           noise_endpoint=True)
    owner = NativeFlowPrediction(frozen.policy)
    if state is None:
        prediction = owner(sample)
        endpoint_prediction = owner(endpoint)
    else:
        mapping = {"policy." + name: value for name, value in state.items()}
        prediction = torch.func.functional_call(owner, mapping, (sample,), strict=False)
        endpoint_prediction = torch.func.functional_call(owner, mapping, (endpoint,), strict=False)
    width = sample.action_width
    result = _prediction_metrics(prediction, sample.target, width)
    endpoint_error = (endpoint_prediction[..., :width].float() - endpoint.target[..., :width].float()).square()
    result.update(
        tau1_first5_mse=float(endpoint_error[:, :5].mean()),
        tau1_first5_time_mse=endpoint_error[:, :5].mean(dim=(0, 2)).cpu().tolist(),
        tau1_first5_dimension_mse=endpoint_error[:, :5].mean(dim=(0, 1)).cpu().tolist(),
    )
    noise = sample.arguments[-2]
    if state is None:
        predicted_actions = frozen.policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    else:
        physical = task_lora_state_dict(frozen.policy, clone=True)
        copy_task_lora_state_(frozen.policy, state, load_pi05_lora_contract_from_policy(frozen.policy, state))
        try:
            predicted_actions = frozen.policy.predict_action_chunk(batch, noise=noise, num_steps=10)
        finally:
            copy_task_lora_state_(frozen.policy, physical, load_pi05_lora_contract_from_policy(frozen.policy, state))
    target_actions = frozen.policy.prepare_action(batch)[..., :width]
    action_error = (predicted_actions[..., :width].float() - target_actions.float()).square()
    result.update(
        inference_first5_mse=float(action_error[:, :5].mean()),
        inference_first5_actions=predicted_actions[:, :5, :width].float().cpu().tolist(),
        target_first5_actions=target_actions[:, :5].float().cpu().tolist(),
        inference_first5_time_mse=action_error[:, :5].mean(dim=(0, 2)).cpu().tolist(),
        inference_first5_dimension_mse=action_error[:, :5].mean(dim=(0, 1)).cpu().tolist(),
    )
    return result


def load_pi05_lora_contract_from_policy(policy: torch.nn.Module, state: Mapping[str, torch.Tensor]):
    """Recover the already injected rank16 contract without a second injection."""
    del state
    contract = getattr(policy, "_ember_diagnostic_lora_contract", None)
    if contract is None:
        raise ValueError("diagnostic policy lacks its injected LoRA contract")
    return contract


def writer_frozen_policy(loaded: LoadedWriter, generated: Mapping[str, torch.Tensor]) -> FrozenPolicy:
    loaded.runtime.policy._ember_diagnostic_lora_contract = loaded.runtime.lora
    return FrozenPolicy(loaded.name, loaded.runtime.policy, loaded.runtime.processor, loaded.runtime.device, generated)


def source_policy(
    *, name: str, asset_root: Path, current_run: Mapping[str, Any], device: torch.device,
    mtbc_config: Path | None = None, mtbc_checkpoint: Path | None = None,
) -> FrozenPolicy:
    source_config = current_run["config"]["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    source = current_run["source"]
    policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    tokenizer = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(device))
    if mtbc_config is not None:
        config = load_source_sft_config(mtbc_config)
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        from ember.lora import inject_task_lora

        inject_task_lora(policy, lora)
        state = load_file(str(mtbc_checkpoint / "lora.safetensors"), device=str(device))
        copy_task_lora_state_(policy, state, lora)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return FrozenPolicy(name, policy, processor, device)


def writer_probe_state(loaded: LoadedWriter, cache: VideoConditionCache, *, task: int, teacher_demo: int) -> dict[str, torch.Tensor]:
    with torch.no_grad(), autocast(loaded.runtime.device):
        state = loaded.runtime.compile(cache.condition(task, (teacher_demo,)))
    _finite_tensor_mapping(state, label="Writer probe LoRA")
    return {name: value.detach() for name, value in state.items()}


def displacement_from_parent(loaded: LoadedWriter, parent: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((value.detach().float().cpu() - parent[name].float()).square().sum())
                         for name, value in loaded.runtime.state.state_dict().items()))
