"""Native cached-prefix SDE sampling for the bounded denoising-return study.

No weights are loaded here. ``NativeVelocity`` wraps the caller's injected policy;
functional_call keys are ``policy.`` plus its actual LoRA parameter names. Its
cached prefix is detached; each forward is the native single denoise_step.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


STUDY = "denoising_return_writer_20261006"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
SCHEMA = "ember_denoising_return_sampler_v1"
STEPS, H, SHAPE = 10, 0.1, (50, 32)
FORMULA = "m=z-h*((1+0.5*(1-tau))*v+0.5*z); z_next=m+sqrt(h*tau)*xi"


def sampler_contract(node: str) -> dict:
    """The only registered SDE evaluation arms; validation retains native ODE."""
    if node not in {"parent", "72"}:
        raise ValueError("SDE evaluation is registered only for parent/72 seen144")
    return {
        "schema_version": SCHEMA, "study": STUDY, "node": node,
        "runtime_mode": "SDE", "formula": FORMULA, "h": H,
        "num_steps": STEPS, "tau": [1.0 - step * H for step in range(STEPS)],
        "noise_shape": list(SHAPE), "output_action_dim": 7,
        "sde_seed_root": 20261006, "seed_purpose": f"{STUDY}:eval_sde",
        "seed_schedule": "policy_noise_seed(root,purpose:suite,task,init,replan)",
        "generator": "independent_cpu_torch_generator_per_slot_replan",
        "initial_policy_noise_preserved": True, "intermediate_clipping": False,
    }


def validate_sampler_contract(contract: Mapping[str, Any]) -> None:
    """Check the fixed scientific panel on preparation and recovery."""
    spec = contract.get("denoising_return_sampler")
    if spec is None:
        return
    from ember.operator_writer.scope import STATES, registration, task_keys_from_ids
    from ember.pi05_assets import Pi05EvaluationError

    node = spec.get("node") if isinstance(spec, Mapping) else None
    if node not in {"parent", "72"} or dict(spec) != sampler_contract(node):
        raise Pi05EvaluationError("denoising-return sampler formula/seed registration changed")
    tasks = contract.get("tasks", ())
    keys = [(row["suite"], int(row["task_id"])) for row in tasks]
    expected = task_keys_from_ids(registration()["global_task_ids"])
    adapter = contract.get("adapter") or {}
    marker = adapter.get("denoising_return") or {}
    policy = contract.get("policy") or {}
    if (contract.get("role") != "operator_seen_training36"
            or contract.get("mode") != "formal"
            or len(keys) != 36 or set(keys) != set(expected)
            or any(tuple(row.get("init_state_ids", ())) != STATES for row in tasks)
            or adapter.get("kind") != "operator_read_write_lora_bank"
            or marker.get("node") != node or marker.get("panel") != "seen"
            or marker.get("runtime_mode") != "T"
            or Path(str(contract.get("output_dir", ""))).resolve()
            != ROOT / "readouts" / node / "seen" / "SDE" / "evaluation"
            or policy.get("num_inference_steps") != STEPS
            or policy.get("replan_steps") != 5
            or (contract.get("diagnostic_exploration") or {}).get("enabled", False)):
        raise Pi05EvaluationError("denoising-return SDE requires its complete seen144 bank/panel")


def _native_prefix(policy: nn.Module, batch: Mapping[str, torch.Tensor]):
    """Exactly the native sample_actions prefix path, executed once per decision."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    model = policy.model
    if model._rtc_enabled():
        raise ValueError("denoising-return sampling does not register RTC")
    images, image_masks = policy._preprocess_images(batch)
    embeddings, padding, attention = model.embed_prefix(
        images, image_masks, batch["observation.language.tokens"],
        batch["observation.language.attention_mask"])
    masks = model._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=masks, position_ids=torch.cumsum(padding, dim=1) - 1,
        past_key_values=None, inputs_embeds=[embeddings, None], use_cache=True)
    return padding, cache


class NativeVelocity(nn.Module):
    """One actual suffix evaluation, with a shared loaded policy and frozen prefix."""

    def __init__(self, policy: nn.Module, batch: Mapping[str, torch.Tensor]):
        super().__init__()
        if (policy.config.chunk_size, policy.config.max_action_dim) != SHAPE:
            raise ValueError("denoising-return requires the native 50x32 latent")
        if policy.config.output_features["action"].shape[0] != 7:
            raise ValueError("denoising-return requires the canonical seven action outputs")
        self.policy = policy
        policy.eval()
        with torch.no_grad():
            padding, self.past_key_values = _native_prefix(policy, batch)
        self.register_buffer("prefix_pad_masks", padding, persistent=False)

    def forward(self, z: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or tuple(z.shape[1:]) != SHAPE:
            raise ValueError("velocity input must be [batch,50,32]")
        if z.shape[0] != self.prefix_pad_masks.shape[0]:
            raise ValueError("velocity batch and cached prefix differ")
        time = torch.as_tensor(tau, dtype=torch.float32, device=z.device)
        time = time.expand(z.shape[0])
        return self.policy.model.denoise_step(
            prefix_pad_masks=self.prefix_pad_masks, past_key_values=self.past_key_values,
            x_t=z, timestep=time)


@dataclass(frozen=True)
class SDETransition:
    step: int
    tau: float
    z: torch.Tensor
    m: torch.Tensor
    z_next: torch.Tensor


@dataclass(frozen=True)
class SDEOutput:
    chunks: torch.Tensor
    transitions: tuple[SDETransition, ...]


def _slot_generators(batch_size: int, seeds, generators) -> tuple[torch.Generator, ...]:
    if (seeds is None) == (generators is None):
        raise ValueError("provide exactly one explicit seed or generator per slot")
    if seeds is not None:
        values = tuple(torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds)
    else:
        values = tuple(generators)
    if len(values) != batch_size or any(generator.device.type != "cpu" for generator in values):
        raise ValueError("one independent CPU generator per slot is required")
    if len({id(generator) for generator in values}) != len(values):
        raise ValueError("slots cannot share a mutable SDE generator")
    return values


def sample_sde(
    velocity: NativeVelocity, noise: torch.Tensor, *, seeds: Sequence[int] | None = None,
    generators: Sequence[torch.Generator] | None = None, return_path: bool = False,
) -> SDEOutput:
    """Ten SDE steps on all 1600 coordinates; initial noise remains caller-owned."""
    if noise.ndim != 3 or tuple(noise.shape[1:]) != SHAPE:
        raise ValueError("initial noise must be [batch,50,32]")
    streams = _slot_generators(noise.shape[0], seeds, generators)
    z, records = noise, []
    for step in range(STEPS):
        tau = 1.0 - step * H
        v = velocity(z, tau)
        if v.shape != z.shape:
            raise ValueError("native velocity must retain all 32 latent coordinates")
        m = z - H * ((1.0 + 0.5 * (1.0 - tau)) * v + 0.5 * z)
        xi = torch.stack([torch.randn(SHAPE, generator=g, dtype=torch.float32)
                          for g in streams]).to(device=z.device, dtype=z.dtype)
        z_next = m + (H * tau) ** 0.5 * xi
        if return_path:
            records.append(SDETransition(step, tau, z, m, z_next))
        z = z_next
    return SDEOutput(z[:, :, :7], tuple(records))


def sample_action_chunk(policy: nn.Module, batch, *, noise: torch.Tensor, **kwargs) -> SDEOutput:
    return sample_sde(NativeVelocity(policy, batch), noise, **kwargs)


def sde_noise_seed(spec: Mapping, task: Mapping, init_state_id: int, replan_index: int) -> int:
    from ember.pi05_eval.run_contract import policy_noise_seed

    return policy_noise_seed(
        spec["sde_seed_root"], f"{spec['seed_purpose']}:{task['suite']}",
        int(task["task_id"]), int(init_state_id), int(replan_index))


@contextmanager
def evaluation_sampler(policy, contract: Mapping, slots: Sequence[dict], task: Mapping):
    """The real bank calls this override inside its existing batched LoRA context."""
    spec = contract["denoising_return_sampler"]
    seeds = [sde_noise_seed(spec, task, slot["init_state_id"], slot["replan_index"])
             for slot in slots]
    had_override = "predict_action_chunk" in policy.__dict__
    previous = policy.__dict__.get("predict_action_chunk")
    calls = 0

    def predict(batch, *, noise, num_steps=STEPS, **kwargs):
        nonlocal calls
        if num_steps != STEPS or kwargs or noise.shape[0] != len(slots):
            raise ValueError("registered SDE call changed steps, arguments or slot count")
        calls += 1
        return sample_action_chunk(policy, batch, noise=noise, seeds=seeds).chunks

    policy.predict_action_chunk = predict
    try:
        yield
        if calls != 1:
            raise ValueError("bank did not invoke the registered SDE sampler exactly once")
        for slot, seed in zip(slots, seeds, strict=True):
            slot.setdefault("denoising_return_sde_seeds", []).append(seed)
    finally:
        if had_override:
            policy.predict_action_chunk = previous
        else:
            del policy.predict_action_chunk


def episode_sampler_fields(contract: Mapping, slot: Mapping) -> dict:
    spec = contract.get("denoising_return_sampler")
    return {} if spec is None else {"denoising_return_sampler": {
        **spec, "sde_noise_seeds": list(slot.get("denoising_return_sde_seeds", ()))}}


def validate_episode_sampler(contract: Mapping, row: Mapping, task: Mapping, replans: int) -> bool:
    spec = contract.get("denoising_return_sampler")
    recorded = row.get("denoising_return_sampler")
    if spec is None:
        return recorded is None
    seeds = [sde_noise_seed(spec, task, row["init_state_id"], index)
             for index in range(replans)]
    return recorded == {**spec, "sde_noise_seeds": seeds}
