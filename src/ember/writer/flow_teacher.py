"""Matched PI05 flow-field evidence for the one-shot Writer teacher audit.

This module is a deliberately disposable scientific component: CEFD may reuse
the matched flow primitive if the audit passes; otherwise the whole module is
removed with the rejected audit path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

from ember.lora import LoRAContract, functional_lora_call
from ember.writer.errors import WriterModelError
from ember.writer.functional import (
    functional_microbatch_contract,
    scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
)


@dataclass(frozen=True)
class FunctionalFlowTeacherAudit:
    """Matched PI05 flow errors and two student LoRA cotangents."""

    expert_target_loss: torch.Tensor
    student_target_loss: torch.Tensor
    comparison_target_loss: torch.Tensor
    distillation_loss: torch.Tensor
    positive_gradients: Mapping[str, torch.Tensor]
    distillation_gradients: Mapping[str, torch.Tensor]


def _pi05_flow_velocity_and_target(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
    expected_action_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the real-action PI05 velocity without duplicating model.forward."""

    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    model = getattr(policy, "model", None)
    config = getattr(policy, "config", None)
    action_out = getattr(model, "action_out_proj", None)
    if (
        model is None
        or config is None
        or not isinstance(action_out, torch.nn.Module)
        or not callable(getattr(policy, "_preprocess_images", None))
        or not callable(getattr(policy, "prepare_action", None))
        or not callable(getattr(model, "sample_noise", None))
        or not callable(getattr(model, "sample_time", None))
        or not callable(getattr(model, "forward", None))
    ):
        raise WriterModelError("flow-field audit requires a PI05 policy")
    try:
        images, image_masks = policy._preprocess_images(dict(batch))
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        action_width = int(config.output_features[ACTION].shape[0])
        actions = policy.prepare_action(batch)
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise WriterModelError("PI05 flow-field batch contract changed") from error
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(actions.shape[0], actions.device)
    captured: list[torch.Tensor] = []

    def capture_velocity(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output)

    handle = action_out.register_forward_hook(capture_velocity)
    try:
        losses = model.forward(
            images,
            image_masks,
            tokens,
            token_masks,
            actions,
            noise,
            time,
        )
    finally:
        handle.remove()
    if (
        len(captured) != 1
        or losses.ndim != 3
        or captured[0].shape != losses.shape
        or noise.shape != actions.shape
        or action_width != expected_action_width
        or not 0 < action_width <= losses.shape[-1]
    ):
        raise WriterModelError("PI05 flow-field output contract changed")
    velocity = captured[0][:, :, :action_width]
    target = (noise - actions)[:, :, :action_width]
    return velocity, target


@contextmanager
def _scoped_policy_flow_field_return(
    policy: torch.nn.Module,
    expected_action_width: int,
) -> Iterator[None]:
    """Temporarily make functional_call return PI05 velocity and target."""

    def flow_field_forward(
        owner: torch.nn.Module,
        batch: Mapping[str, Any],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reduction != "mean":
            raise WriterModelError("flow-field audit supports mean reduction")
        return _pi05_flow_velocity_and_target(owner, batch, expected_action_width)

    had_instance_value = "forward" in vars(policy)
    previous_instance_value = vars(policy).get("forward")
    policy.forward = MethodType(flow_field_forward, policy)
    try:
        yield
    finally:
        if had_instance_value:
            policy.forward = previous_instance_value
        else:
            delattr(policy, "forward")


def _functional_flow_fields(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_rng_seed: int,
    policy_rng_device: torch.device | str,
    flow_time_sampling_scheme: str,
    flow_noise_sampling_scheme: str,
    logical_batch_size: int,
    batch_offset: int,
    physical_microbatching: bool,
    expected_action_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    random_batch_size = logical_batch_size if physical_microbatching else None
    with scoped_policy_randomness(policy_rng_seed, policy_rng_device):
        with scoped_policy_flow_noise_sampling(
            policy,
            flow_noise_sampling_scheme,
            logical_batch_size=random_batch_size,
            batch_offset=batch_offset,
        ):
            with scoped_policy_flow_time_sampling(
                policy,
                flow_time_sampling_scheme,
                logical_batch_size=random_batch_size,
                batch_offset=batch_offset,
            ):
                with _scoped_policy_flow_field_return(policy, expected_action_width):
                    output = functional_lora_call(policy, state, contract, batch)
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not all(isinstance(value, torch.Tensor) for value in output)
    ):
        raise WriterModelError("functional PI05 flow-field return changed")
    return output


def _functional_flow_teacher_microbatch(
    policy: torch.nn.Module,
    student_state: Mapping[str, torch.Tensor],
    expert_state: Mapping[str, torch.Tensor],
    comparison_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    logical_batch_size: int,
    physical_microbatching: bool,
    policy_rng_seed: int,
    policy_rng_device: torch.device | str,
    flow_time_sampling_scheme: str,
    flow_noise_sampling_scheme: str,
    expected_action_width: int,
) -> FunctionalFlowTeacherAudit:
    microbatch = {
        name: (
            value[start:stop]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == logical_batch_size
            else value
        )
        for name, value in batch.items()
    }
    common = {
        "policy_rng_seed": policy_rng_seed,
        "policy_rng_device": policy_rng_device,
        "flow_time_sampling_scheme": flow_time_sampling_scheme,
        "flow_noise_sampling_scheme": flow_noise_sampling_scheme,
        "logical_batch_size": logical_batch_size,
        "batch_offset": start,
        "physical_microbatching": physical_microbatching,
        "expected_action_width": expected_action_width,
    }
    with torch.no_grad():
        expert_velocity, expert_target = _functional_flow_fields(
            policy, expert_state, contract, microbatch, **common
        )
        comparison_velocity, comparison_target = _functional_flow_fields(
            policy, comparison_state, contract, microbatch, **common
        )
    leaves = {
        name: value.detach().requires_grad_(True)
        for name, value in student_state.items()
    }
    student_velocity, student_target = _functional_flow_fields(
        policy, leaves, contract, microbatch, **common
    )
    if not (
        expert_velocity.shape
        == expert_target.shape
        == comparison_velocity.shape
        == comparison_target.shape
        == student_velocity.shape
        == student_target.shape
    ):
        raise WriterModelError("matched PI05 flow fields changed shape")
    expert_velocity_f32 = expert_velocity.float()
    student_velocity_f32 = student_velocity.float()
    comparison_velocity_f32 = comparison_velocity.float()
    expert_loss = (expert_velocity_f32 - expert_target.float()).square().mean()
    student_loss = (student_velocity_f32 - student_target.float()).square().mean()
    comparison_loss = (
        comparison_velocity_f32 - comparison_target.float()
    ).square().mean()
    distillation_loss = (
        student_velocity_f32 - expert_velocity_f32.detach()
    ).square().mean()
    names = tuple(leaves)
    positive = torch.autograd.grad(
        student_loss,
        tuple(leaves[name] for name in names),
        retain_graph=True,
    )
    distillation = torch.autograd.grad(
        distillation_loss,
        tuple(leaves[name] for name in names),
    )
    return FunctionalFlowTeacherAudit(
        expert_target_loss=expert_loss.detach(),
        student_target_loss=student_loss.detach(),
        comparison_target_loss=comparison_loss.detach(),
        distillation_loss=distillation_loss.detach(),
        positive_gradients={
            name: value.detach()
            for name, value in zip(names, positive, strict=True)
        },
        distillation_gradients={
            name: value.detach()
            for name, value in zip(names, distillation, strict=True)
        },
    )


def functional_lora_flow_teacher_audit(
    policy: torch.nn.Module,
    student_state: Mapping[str, torch.Tensor],
    expert_state: Mapping[str, torch.Tensor],
    comparison_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    policy_rng_seed: int,
    policy_rng_device: torch.device | str,
    flow_time_sampling_scheme: str,
    flow_noise_sampling_scheme: str,
    policy_microbatch_size: int,
    expected_action_width: int,
) -> FunctionalFlowTeacherAudit:
    """Audit a matched expert teacher with one differentiable student forward."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("flow-teacher audit received a trainable policy")
    if set(student_state) != set(expert_state) or set(student_state) != set(
        comparison_state
    ):
        raise WriterModelError("flow-teacher LoRA topology changed")
    logical_batch_size, microbatch_size = functional_microbatch_contract(
        batch,
        policy_microbatch_size,
        policy_rng_seed=policy_rng_seed,
        flow_time_sampling_scheme=flow_time_sampling_scheme,
        flow_noise_sampling_scheme=flow_noise_sampling_scheme,
    )
    names = tuple(student_state)
    gradient_sums = {
        component: {
            name: torch.zeros_like(
                student_state[name],
                dtype=(
                    torch.float32
                    if student_state[name].dtype in {torch.bfloat16, torch.float16}
                    else student_state[name].dtype
                ),
                memory_format=torch.preserve_format,
            )
            for name in names
        }
        for component in ("positive", "distillation")
    }
    loss_sums: dict[str, torch.Tensor] = {}
    for start in range(0, logical_batch_size, microbatch_size):
        stop = min(start + microbatch_size, logical_batch_size)
        weight = (stop - start) / logical_batch_size
        value = _functional_flow_teacher_microbatch(
            policy,
            student_state,
            expert_state,
            comparison_state,
            contract,
            batch,
            start=start,
            stop=stop,
            logical_batch_size=logical_batch_size,
            physical_microbatching=microbatch_size < logical_batch_size,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=policy_rng_device,
            flow_time_sampling_scheme=flow_time_sampling_scheme,
            flow_noise_sampling_scheme=flow_noise_sampling_scheme,
            expected_action_width=expected_action_width,
        )
        for field in (
            "expert_target_loss",
            "student_target_loss",
            "comparison_target_loss",
            "distillation_loss",
        ):
            weighted = getattr(value, field).to(dtype=torch.float32) * weight
            loss_sums[field] = loss_sums.get(field, torch.zeros_like(weighted)) + weighted
        for component, gradients in (
            ("positive", value.positive_gradients),
            ("distillation", value.distillation_gradients),
        ):
            for name in names:
                gradient_sums[component][name].add_(
                    gradients[name].to(dtype=gradient_sums[component][name].dtype),
                    alpha=weight,
                )
    if set(loss_sums) != {
        "expert_target_loss",
        "student_target_loss",
        "comparison_target_loss",
        "distillation_loss",
    }:
        raise WriterModelError("flow-teacher microbatch loop was empty")
    return FunctionalFlowTeacherAudit(
        **loss_sums,
        positive_gradients={
            name: value.to(dtype=student_state[name].dtype)
            for name, value in gradient_sums["positive"].items()
        },
        distillation_gradients={
            name: value.to(dtype=student_state[name].dtype)
            for name, value in gradient_sums["distillation"].items()
        },
    )
