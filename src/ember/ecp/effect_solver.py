"""Exact policy-effect targets and the fixed EMBER-PECS inner solver."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.ecp.observer import ECPNativeObserver
from ember.ecp.stage0 import ECPVideoEncoder, ECPVideoEncoderOutput
from ember.ecp.stage1_data import PackedStage1Videos
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class PolicyEffectProbe:
    """Two action-hidden support frames per ordered event, one from each video."""

    prefix_embeddings: torch.Tensor
    prefix_padding: torch.Tensor
    presence: torch.Tensor
    selected_flat_indices: torch.Tensor
    selected_frame_positions: torch.Tensor

    @property
    def event_count(self) -> int:
        return int(self.prefix_embeddings.shape[0])

    @property
    def frames_per_event(self) -> int:
        return int(self.prefix_embeddings.shape[1])


@dataclass(frozen=True)
class PolicyEffectResponse:
    owner: torch.Tensor
    flow: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True)
class ExactPolicyEffectTargets:
    source_owner: torch.Tensor
    source_flow: torch.Tensor
    source_action: torch.Tensor
    shared_owner: torch.Tensor
    shared_flow: torch.Tensor
    shared_action: torch.Tensor
    mean_owner: torch.Tensor
    variance_owner: torch.Tensor
    mean_flow: torch.Tensor
    variance_flow: torch.Tensor
    mean_action: torch.Tensor
    variance_action: torch.Tensor
    presence: torch.Tensor


@dataclass(frozen=True)
class PolicyEffectLoss:
    total: torch.Tensor
    owner: torch.Tensor
    flow: torch.Tensor
    action: torch.Tensor
    shared_barrier: torch.Tensor


@dataclass(frozen=True)
class SolverStep:
    step: int
    effect: float
    owner: float
    flow: float
    action: float
    shared_barrier: float
    trust_distance: float
    trust_penalty: float
    gradient_rms: float
    applied_step_rms: float


def prepare_policy_effect_probe(
    *,
    encoder: ECPVideoEncoder,
    policy: torch.nn.Module,
    packed: PackedStage1Videos,
    encoded: ECPVideoEncoderOutput,
    language_tokens: torch.Tensor,
    language_mask: torch.Tensor,
    prefix_batch_size: int,
) -> PolicyEffectProbe:
    """Select each video's maximum-posterior frame for every ordered event."""

    video_count, _, event_count = encoded.state_posterior.shape
    if video_count != packed.video_offsets.numel() - 1 or prefix_batch_size <= 0:
        raise ValueError("PECS video/event topology changed")
    offsets = packed.video_offsets.detach().cpu().tolist()
    selected_flat = []
    selected_positions = []
    for event in range(event_count):
        for video in range(video_count):
            length = offsets[video + 1] - offsets[video]
            position = int(encoded.state_posterior[video, :length, event].argmax())
            selected_flat.append(offsets[video] + position)
            selected_positions.append(position)
    selected = torch.tensor(
        selected_flat, dtype=torch.long, device=packed.frames.device
    )
    frames = packed.frames.index_select(0, selected)
    bridge = policy.model.paligemma_with_expert
    image_rows = []
    with torch.no_grad():
        language = bridge.embed_language_tokens(language_tokens)
        for start in range(0, frames.shape[0], prefix_batch_size):
            image_rows.append(
                bridge.embed_image(
                    encoder._prepare_images(frames[start : start + prefix_batch_size])
                )
            )
        image = torch.cat(image_rows)
        repeated_language = language.expand(frames.shape[0], -1, -1)
        prefix = torch.cat((image, repeated_language), dim=1).detach()
    padding = torch.cat(
        (
            torch.ones(
                image.shape[:2], dtype=torch.bool, device=image.device
            ),
            language_mask.expand(frames.shape[0], -1),
        ),
        dim=1,
    )
    frames_per_event = video_count
    return PolicyEffectProbe(
        prefix_embeddings=prefix.reshape(
            event_count, frames_per_event, *prefix.shape[1:]
        ),
        prefix_padding=padding.reshape(
            event_count, frames_per_event, padding.shape[1]
        ),
        presence=encoded.presence.float().mean(0).detach(),
        selected_flat_indices=selected.reshape(event_count, frames_per_event),
        selected_frame_positions=torch.tensor(
            selected_positions,
            dtype=torch.long,
            device=packed.frames.device,
        ).reshape(event_count, frames_per_event),
    )


def _dct_basis(device: torch.device, count: int = 4) -> torch.Tensor:
    horizon = 50
    positions = torch.arange(horizon, device=device, dtype=torch.float32) + 0.5
    rows = [torch.ones(horizon, device=device) / horizon**0.5]
    for frequency in range(1, count):
        rows.append(
            (2.0 / horizon) ** 0.5
            * torch.cos(torch.pi * frequency * positions / horizon)
        )
    return torch.stack(rows)


def _prepare_prefix_cache(
    policy: torch.nn.Module,
    prefix_embeddings: torch.Tensor,
    prefix_padding: torch.Tensor,
) -> Any:
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    core = policy.model
    attention = torch.zeros_like(prefix_padding)
    attention_2d = make_att_2d_masks(prefix_padding, attention)
    attention_4d = core._prepare_attention_masks_4d(attention_2d)
    positions = torch.cumsum(prefix_padding, dim=1) - 1
    bridge = core.paligemma_with_expert
    bridge.paligemma.model.language_model.config._attn_implementation = "eager"
    with torch.no_grad():
        _, cache = bridge.forward(
            attention_mask=attention_4d,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=True,
        )
    return cache


def _autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def capture_policy_effect_response(
    *,
    policy: torch.nn.Module,
    observer: ECPNativeObserver,
    lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    prefix_embeddings: torch.Tensor,
    prefix_padding: torch.Tensor,
    suffix_noise: torch.Tensor,
    denoising_steps: int,
) -> PolicyEffectResponse:
    """Capture the complete official action/flow path from fixed noise."""

    frames = int(prefix_embeddings.shape[0])
    if (
        frames <= 0
        or prefix_padding.shape != prefix_embeddings.shape[:2]
        or suffix_noise.shape != (50, 32)
        or denoising_steps <= 0
    ):
        raise ValueError("PECS policy-effect probe changed shape")
    prefix = prefix_embeddings.repeat_interleave(2, dim=0)
    padding = prefix_padding.repeat_interleave(2, dim=0)
    noise = torch.stack((suffix_noise, -suffix_noise))[None].expand(
        frames, -1, -1, -1
    ).reshape(2 * frames, 50, 32)
    batch_size = 2 * frames
    needs_grad = any(value.requires_grad for value in state.values())
    state_names = tuple(state)
    state_values = tuple(state[name] for name in state_names)
    with _autocast(prefix.device):
        prefix_cache = _prepare_prefix_cache(policy, prefix, padding)

    def state_from(values: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        return dict(zip(state_names, values, strict=True))

    def first_step(
        x_t: torch.Tensor, *values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = state_from(values)
        with _autocast(prefix.device):
            observed = observer.observe_action_step(
                policy.model,
                padding,
                prefix_cache,
                x_t,
                torch.ones(batch_size, device=prefix.device),
                track_action_adapter_grad=needs_grad,
                action_adapter_context=lora.activate([active] * batch_size),
            )
        return observed.owner_lattice, observed.flow_velocity.float()

    def later_step(
        x_t: torch.Tensor, timestep: torch.Tensor, *values: torch.Tensor
    ) -> torch.Tensor:
        active = state_from(values)
        grad_context = torch.enable_grad() if needs_grad else torch.no_grad()
        with (
            grad_context,
            lora.activate([active] * batch_size),
            _autocast(prefix.device),
        ):
            return policy.model.denoise_step(
                prefix_pad_masks=padding,
                past_key_values=prefix_cache,
                x_t=x_t,
                timestep=timestep,
            ).float()

    x_t = noise
    velocities = []
    actions = []
    owner_lattice = None
    dt = -1.0 / float(denoising_steps)
    for step in range(denoising_steps):
        timestep = torch.full(
            (batch_size,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=prefix.device,
        )
        if step == 0:
            if needs_grad:
                owner_lattice, velocity = checkpoint(
                    first_step,
                    x_t,
                    *state_values,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                owner_lattice, velocity = first_step(x_t, *state_values)
        elif needs_grad:
            velocity = checkpoint(
                later_step,
                x_t,
                timestep,
                *state_values,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            velocity = later_step(x_t, timestep, *state_values)
        x_t = x_t + dt * velocity
        velocities.append(velocity)
        actions.append(x_t[..., :7])
    if owner_lattice is None:
        raise RuntimeError("PECS denoising trajectory did not run")
    owner = owner_lattice.reshape(
        frames, 2, 38, 50, 128
    )[:, 0].float()
    owner = torch.einsum("bohd,ph->bopd", owner, _dct_basis(owner.device))
    flow = torch.stack(velocities, dim=1).reshape(
        frames, 2, denoising_steps, 50, 32
    )
    action = torch.stack(actions, dim=1).reshape(
        frames, 2, denoising_steps, 50, 7
    )
    return PolicyEffectResponse(owner=owner, flow=flow, action=action)


def capture_full_probe_response(
    *,
    policy: torch.nn.Module,
    observer: ECPNativeObserver,
    lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    probe: PolicyEffectProbe,
    suffix_noise: torch.Tensor,
    denoising_steps: int,
) -> PolicyEffectResponse:
    owner = []
    flow = []
    action = []
    for event in range(probe.event_count):
        response = capture_policy_effect_response(
            policy=policy,
            observer=observer,
            lora=lora,
            state=state,
            prefix_embeddings=probe.prefix_embeddings[event],
            prefix_padding=probe.prefix_padding[event],
            suffix_noise=suffix_noise,
            denoising_steps=denoising_steps,
        )
        owner.append(response.owner)
        flow.append(response.flow)
        action.append(response.action)
    return PolicyEffectResponse(
        owner=torch.stack(owner),
        flow=torch.stack(flow),
        action=torch.stack(action),
    )


def build_exact_policy_effect_targets(
    *,
    policy: torch.nn.Module,
    observer: ECPNativeObserver,
    lora: BatchedLoRAInference,
    identity_state: Mapping[str, torch.Tensor],
    shared_state: Mapping[str, torch.Tensor],
    expert_states: Sequence[Mapping[str, torch.Tensor]],
    expert_weights: torch.Tensor,
    probe: PolicyEffectProbe,
    suffix_noise: torch.Tensor,
    denoising_steps: int,
) -> ExactPolicyEffectTargets:
    if not expert_states or expert_weights.shape != (len(expert_states),):
        raise ValueError("PECS exact teacher set changed")
    source = capture_full_probe_response(
        policy=policy,
        observer=observer,
        lora=lora,
        state=identity_state,
        probe=probe,
        suffix_noise=suffix_noise,
        denoising_steps=denoising_steps,
    )
    shared = capture_full_probe_response(
        policy=policy,
        observer=observer,
        lora=lora,
        state=shared_state,
        probe=probe,
        suffix_noise=suffix_noise,
        denoising_steps=denoising_steps,
    )
    experts = [
        capture_full_probe_response(
            policy=policy,
            observer=observer,
            lora=lora,
            state=state,
            probe=probe,
            suffix_noise=suffix_noise,
            denoising_steps=denoising_steps,
        )
        for state in expert_states
    ]
    source_owner = source.owner.mean(1)
    source_flow = source.flow.mean(1)
    source_action = source.action.mean(1)
    member_owner = torch.stack(
        [value.owner - source.owner for value in experts]
    )
    member_flow = torch.stack([value.flow - source.flow for value in experts])
    member_action = torch.stack(
        [value.action - source.action for value in experts]
    )
    weights = expert_weights.to(member_owner).float().clamp_min(1e-4)
    weights = weights / weights.sum()

    def moments(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.einsum("m,mef...->e...", weights, value) / value.shape[2]
        second = (
            torch.einsum("m,mef...->e...", weights, value.square())
            / value.shape[2]
        )
        return mean, (second - mean.square()).clamp_min(0)

    mean_owner, variance_owner = moments(member_owner)
    mean_flow, variance_flow = moments(member_flow)
    mean_action, variance_action = moments(member_action)
    return ExactPolicyEffectTargets(
        source_owner=source_owner.detach(),
        source_flow=source_flow.detach(),
        source_action=source_action.detach(),
        shared_owner=(shared.owner.mean(1) - source_owner).detach(),
        shared_flow=(shared.flow.mean(1) - source_flow).detach(),
        shared_action=(shared.action.mean(1) - source_action).detach(),
        mean_owner=mean_owner.detach(),
        variance_owner=variance_owner.detach(),
        mean_flow=mean_flow.detach(),
        variance_flow=variance_flow.detach(),
        mean_action=mean_action.detach(),
        variance_action=variance_action.detach(),
        presence=probe.presence.detach(),
    )


def _uncertainty_weighted_error(
    candidate: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduction = tuple(range(1, candidate.ndim))
    error = (candidate - target).square().mean(dim=reduction)
    baseline_error = (baseline - target).square().mean(dim=reduction)
    signal = target.square().mean(dim=reduction)
    uncertainty = variance.mean(dim=reduction)
    floor = 0.05 * signal.mean().clamp_min(1e-8)
    confidence = signal / (signal + uncertainty + floor)
    normalization = signal + floor
    normalized = error / normalization
    normalized_baseline = baseline_error / normalization
    weight = confidence / confidence.sum().clamp_min(1e-6)
    return (weight * normalized).sum(), (
        weight * torch.relu(normalized - normalized_baseline)
    ).sum()


def policy_effect_loss(
    *,
    response: PolicyEffectResponse,
    targets: ExactPolicyEffectTargets,
    event: int,
    owner_weight: float,
    flow_weight: float,
    action_weight: float,
    shared_barrier_weight: float,
) -> PolicyEffectLoss:
    owner_effect = response.owner.mean(0) - targets.source_owner[event]
    flow_effect = response.flow.mean(0) - targets.source_flow[event]
    action_effect = response.action.mean(0) - targets.source_action[event]
    owner, owner_barrier = _uncertainty_weighted_error(
        owner_effect,
        targets.mean_owner[event],
        targets.variance_owner[event],
        targets.shared_owner[event],
    )
    flow, flow_barrier = _uncertainty_weighted_error(
        flow_effect,
        targets.mean_flow[event],
        targets.variance_flow[event],
        targets.shared_flow[event],
    )
    action, action_barrier = _uncertainty_weighted_error(
        action_effect,
        targets.mean_action[event],
        targets.variance_action[event],
        targets.shared_action[event],
    )
    barrier = owner_barrier + flow_barrier + action_barrier
    total = (
        float(owner_weight) * owner
        + float(flow_weight) * flow
        + float(action_weight) * action
        + float(shared_barrier_weight) * barrier
    )
    return PolicyEffectLoss(
        total=total,
        owner=owner,
        flow=flow,
        action=action,
        shared_barrier=barrier,
    )


def relative_effective_update_distance(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    contract: LoRAContract,
) -> torch.Tensor:
    losses = []
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        candidate_a = candidate[a_name].float()
        candidate_b = candidate[b_name].float()
        reference_a = reference[a_name].float()
        reference_b = reference[b_name].float()
        candidate_energy = torch.einsum(
            "ij,ji->", candidate_b.T @ candidate_b, candidate_a @ candidate_a.T
        )
        reference_energy = torch.einsum(
            "ij,ji->", reference_b.T @ reference_b, reference_a @ reference_a.T
        )
        cross = torch.einsum(
            "ij,ji->", candidate_b.T @ reference_b, reference_a @ candidate_a.T
        )
        losses.append(
            (candidate_energy + reference_energy - 2.0 * cross).clamp_min(0)
            / reference_energy.clamp_min(1e-10)
        )
    return torch.stack(losses).mean()


def _regauge(
    state: Mapping[str, torch.Tensor], contract: LoRAContract
) -> dict[str, torch.Tensor]:
    result = {}
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        result[a_name], result[b_name] = canonicalize_low_rank_factors(
            state[a_name], state[b_name], output_rank=contract.rank
        )
    return result


def solve_policy_effects(
    *,
    initial_state: Mapping[str, torch.Tensor],
    targets: ExactPolicyEffectTargets,
    contract: LoRAContract,
    response: Callable[[Mapping[str, torch.Tensor], int], PolicyEffectResponse],
    steps: int,
    step_rms: float,
    step_decay_power: float,
    owner_weight: float,
    flow_weight: float,
    action_weight: float,
    shared_barrier_weight: float,
    trust_region: float,
    trust_weight: float,
) -> tuple[dict[str, torch.Tensor], tuple[SolverStep, ...]]:
    """Run the same stateless normalized-gradient/re-gauge update for every task."""

    if (
        steps <= 0
        or step_rms <= 0
        or step_decay_power < 0
        or trust_region <= 0
        or trust_weight < 0
    ):
        raise ValueError("invalid PECS fixed solver contract")
    state = _regauge(
        {name: value.detach().float().clone() for name, value in initial_state.items()},
        contract,
    )
    reference = {name: value.detach().float() for name, value in state.items()}
    event_weights = targets.presence.float().clamp_min(0)
    event_weights = event_weights / event_weights.sum().clamp_min(1e-6)
    history = []
    names = tuple(state)
    for step in range(steps):
        applied_step_rms = float(step_rms) / float(step + 1) ** float(
            step_decay_power
        )
        leaves = {name: value.detach().requires_grad_(True) for name, value in state.items()}
        gradients = {name: torch.zeros_like(value) for name, value in leaves.items()}
        totals = {
            name: 0.0
            for name in ("effect", "owner", "flow", "action", "barrier")
        }
        for event in range(targets.presence.numel()):
            loss = policy_effect_loss(
                response=response(leaves, event),
                targets=targets,
                event=event,
                owner_weight=owner_weight,
                flow_weight=flow_weight,
                action_weight=action_weight,
                shared_barrier_weight=shared_barrier_weight,
            )
            weighted = event_weights[event] * loss.total
            event_gradients = torch.autograd.grad(
                weighted, tuple(leaves[name] for name in names)
            )
            for name, gradient in zip(names, event_gradients, strict=True):
                gradients[name].add_(gradient.detach())
            totals["effect"] += float(weighted.detach())
            totals["owner"] += float(event_weights[event] * loss.owner.detach())
            totals["flow"] += float(event_weights[event] * loss.flow.detach())
            totals["action"] += float(
                event_weights[event] * loss.action.detach()
            )
            totals["barrier"] += float(
                event_weights[event] * loss.shared_barrier.detach()
            )
        trust = relative_effective_update_distance(leaves, reference, contract)
        trust_penalty = torch.relu(trust - float(trust_region)).square()
        if trust_weight:
            trust_gradients = torch.autograd.grad(
                float(trust_weight) * trust_penalty,
                tuple(leaves[name] for name in names),
            )
            for name, gradient in zip(names, trust_gradients, strict=True):
                gradients[name].add_(gradient.detach())
        gradient_energy = sum(value.float().square().sum() for value in gradients.values())
        gradient_count = sum(value.numel() for value in gradients.values())
        gradient_rms = (gradient_energy / gradient_count).sqrt()
        updated = {}
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            owner_gradient_rms = torch.sqrt(
                (
                    gradients[a_name].float().square().sum()
                    + gradients[b_name].float().square().sum()
                )
                / (gradients[a_name].numel() + gradients[b_name].numel())
            ).clamp_min(1e-12)
            updated[a_name] = leaves[a_name].detach() - (
                applied_step_rms * gradients[a_name] / owner_gradient_rms
            )
            updated[b_name] = leaves[b_name].detach() - (
                applied_step_rms * gradients[b_name] / owner_gradient_rms
            )
        state = _regauge(updated, contract)
        history.append(
            SolverStep(
                step=step,
                effect=totals["effect"],
                owner=totals["owner"],
                flow=totals["flow"],
                action=totals["action"],
                shared_barrier=totals["barrier"],
                trust_distance=float(trust.detach()),
                trust_penalty=float(trust_penalty.detach()),
                gradient_rms=float(gradient_rms.detach()),
                applied_step_rms=applied_step_rms,
            )
        )
    return state, tuple(history)


@torch.no_grad()
def evaluate_policy_effect_state(
    *,
    state: Mapping[str, torch.Tensor],
    targets: ExactPolicyEffectTargets,
    response: Callable[[Mapping[str, torch.Tensor], int], PolicyEffectResponse],
    owner_weight: float,
    flow_weight: float,
    action_weight: float,
    shared_barrier_weight: float,
) -> dict[str, float]:
    weights = targets.presence.float().clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-6)
    result = {
        name: 0.0
        for name in ("effect", "owner", "flow", "action", "barrier")
    }
    for event in range(targets.presence.numel()):
        loss = policy_effect_loss(
            response=response(state, event),
            targets=targets,
            event=event,
            owner_weight=owner_weight,
            flow_weight=flow_weight,
            action_weight=action_weight,
            shared_barrier_weight=shared_barrier_weight,
        )
        result["effect"] += float(weights[event] * loss.total)
        result["owner"] += float(weights[event] * loss.owner)
        result["flow"] += float(weights[event] * loss.flow)
        result["action"] += float(weights[event] * loss.action)
        result["barrier"] += float(weights[event] * loss.shared_barrier)
    return result
