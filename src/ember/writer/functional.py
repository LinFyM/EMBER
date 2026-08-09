"""Functional action loss from a Writer-generated task-local LoRA."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

import torch

from ember.lora import (
    LoRAContract,
    functional_lora_call,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.writer.errors import WriterModelError


TASK_LOGICAL_BATCH_POLICY_RNG_SCHEME = (
    "task_logical_batch_keyed_stateless_policy_cpu_cuda_splitmix64_v3"
)
LATIN_BETA_TIME_SAMPLING_SCHEME = "task_query_keyed_randomized_latin_beta15_time_v1"
ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME = (
    "task_query_keyed_randomized_antithetic_gaussian_v1"
)
INDEPENDENT_BETA_TIME_SAMPLING_SCHEME = (
    "task_logical_batch_keyed_independent_beta15_time_v2"
)
INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME = (
    "task_logical_batch_keyed_independent_gaussian_v2"
)

_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def pi05_mean_flow_loss(
    policy: torch.nn.Module,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """Compute the exact PI05 training mean without materializing host metrics."""

    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    model = getattr(policy, "model", None)
    config = getattr(policy, "config", None)
    if (
        model is None
        or config is None
        or not callable(getattr(policy, "_preprocess_images", None))
        or not callable(getattr(policy, "prepare_action", None))
        or not callable(getattr(model, "sample_noise", None))
        or not callable(getattr(model, "sample_time", None))
        or not callable(getattr(model, "forward", None))
    ):
        raise WriterModelError("loss-only path requires a PI05 policy")
    images, image_masks = policy._preprocess_images(dict(batch))
    try:
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        action_width = int(config.output_features[ACTION].shape[0])
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise WriterModelError("PI05 loss-only batch contract changed") from error
    actions = policy.prepare_action(batch)
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(actions.shape[0], actions.device)
    losses = model.forward(
        images,
        image_masks,
        tokens,
        token_masks,
        actions,
        noise,
        time,
    )
    if losses.ndim != 3 or not 0 < action_width <= losses.shape[-1]:
        raise WriterModelError("PI05 loss-only output contract changed")
    return losses[:, :, :action_width].mean()


@contextmanager
def _scoped_policy_detail_collection(
    policy: torch.nn.Module,
    collect_policy_details: bool,
) -> Iterator[None]:
    if collect_policy_details:
        yield
        return

    def loss_only_forward(
        owner: torch.nn.Module,
        batch: Mapping[str, Any],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        if reduction != "mean":
            raise WriterModelError("PI05 loss-only path supports mean reduction")
        return pi05_mean_flow_loss(owner, batch), {}

    had_instance_value = "forward" in vars(policy)
    previous_instance_value = vars(policy).get("forward")
    policy.forward = MethodType(loss_only_forward, policy)
    try:
        yield
    finally:
        if had_instance_value:
            policy.forward = previous_instance_value
        else:
            delattr(policy, "forward")


def task_logical_batch_policy_rng_seed(
    *,
    optimization_seed: int,
    task_id: int,
    task_visit: int,
    demo_indices: Sequence[int],
    frame_indices: Sequence[int],
) -> int:
    """Derive one rank/phase-independent seed from one ordered logical panel."""

    demos = tuple(int(value) for value in demo_indices)
    frames = tuple(int(value) for value in frame_indices)
    if (
        optimization_seed < 0
        or task_id < 0
        or task_visit < 0
        or not demos
        or len(demos) != len(frames)
        or any(value < 0 for value in (*demos, *frames))
    ):
        raise WriterModelError("invalid task-query policy randomness identity")
    state = 0x6A09E667F3BCC909
    values = (optimization_seed, task_id, task_visit, len(demos))
    for ordinal, value in enumerate(values):
        state = _splitmix64(state ^ _splitmix64(value + ordinal))
    for ordinal, (demo_index, frame_index) in enumerate(
        zip(demos, frames, strict=True),
        start=len(values),
    ):
        state = _splitmix64(state ^ _splitmix64(demo_index + ordinal))
        state = _splitmix64(state ^ _splitmix64(frame_index + ordinal + 1))
    return state & ((1 << 63) - 1)


@contextmanager
def scoped_policy_randomness(
    seed: int | None,
    device: torch.device | str | None,
) -> Iterator[None]:
    """Seed and restore the policy forward's local CPU and CUDA generators.

    PI05 samples flow noise on the policy device but samples its Beta flow
    timestep through the CPU default generator before moving it to the device.
    A CUDA policy therefore requires both generators to be keyed by the same
    task/query identity.
    """

    if seed is None:
        yield
        return
    if seed < 0 or device is None:
        raise WriterModelError("invalid scoped policy randomness request")
    selected = torch.device(device)
    if selected.type == "cuda":
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index], device_type="cuda"):
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.default_generators[index].manual_seed(seed)
            yield
        return
    if selected.type != "cpu":
        raise WriterModelError("policy randomness device must be CPU or CUDA")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


def _latin_beta15_time(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample one randomized Latin stratum per exact Beta(1.5, 1) query."""

    config = getattr(model, "config", None)
    if (
        batch_size <= 0
        or config is None
        or float(config.time_sampling_beta_alpha) != 1.5
        or float(config.time_sampling_beta_beta) != 1.0
        or float(config.time_sampling_scale) != 0.999
        or float(config.time_sampling_offset) != 0.001
    ):
        raise WriterModelError("Latin flow-time sampler lost PI05 Beta contract")
    jitter = torch.rand(batch_size, dtype=torch.float32, device="cpu")
    uniform = (torch.arange(batch_size, dtype=torch.float32) + jitter) / float(
        batch_size
    )
    permutation = torch.randperm(batch_size)
    beta = uniform.index_select(0, permutation).pow(2.0 / 3.0)
    time = beta * float(config.time_sampling_scale)
    time = time + float(config.time_sampling_offset)
    return time.to(dtype=torch.float32, device=device)


def _independent_beta15_time(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample independent exact Beta(1.5, 1) flow times on the CPU RNG."""

    config = getattr(model, "config", None)
    if (
        batch_size <= 0
        or config is None
        or float(config.time_sampling_beta_alpha) != 1.5
        or float(config.time_sampling_beta_beta) != 1.0
        or float(config.time_sampling_scale) != 0.999
        or float(config.time_sampling_offset) != 0.001
    ):
        raise WriterModelError("independent flow-time sampler lost PI05 Beta contract")
    uniform = torch.rand(batch_size, dtype=torch.float32, device="cpu")
    beta = uniform.pow(2.0 / 3.0)
    time = beta * float(config.time_sampling_scale)
    time = time + float(config.time_sampling_offset)
    return time.to(dtype=torch.float32, device=device)


@contextmanager
def scoped_policy_flow_time_sampling(
    policy: torch.nn.Module,
    scheme: str | None,
    *,
    logical_batch_size: int | None = None,
    batch_offset: int = 0,
) -> Iterator[None]:
    """Temporarily replace only PI05 flow-time sampling, never policy code."""

    if scheme is None:
        yield
        return
    samplers = {
        LATIN_BETA_TIME_SAMPLING_SCHEME: _latin_beta15_time,
        INDEPENDENT_BETA_TIME_SAMPLING_SCHEME: _independent_beta15_time,
    }
    if scheme not in samplers:
        raise WriterModelError("unsupported policy flow-time sampling scheme")
    full_sampler = samplers[scheme]
    model = getattr(policy, "model", None)
    if model is None or not callable(getattr(model, "sample_time", None)):
        raise WriterModelError("policy has no scoped PI05 flow-time owner")
    had_instance_value = "sample_time" in vars(model)
    previous_instance_value = vars(model).get("sample_time")
    if logical_batch_size is None:
        sampler = full_sampler
    else:
        if logical_batch_size <= 0 or not 0 <= batch_offset < logical_batch_size:
            raise WriterModelError("invalid logical flow-time microbatch")

        def sampler(
            owner: torch.nn.Module,
            batch_size: int,
            device: torch.device | str,
        ) -> torch.Tensor:
            stop = batch_offset + batch_size
            if batch_size <= 0 or stop > logical_batch_size:
                raise WriterModelError("flow-time microbatch exceeds logical batch")
            return full_sampler(owner, logical_batch_size, device)[batch_offset:stop]

    model.sample_time = MethodType(sampler, model)
    try:
        yield
    finally:
        if had_instance_value:
            model.sample_time = previous_instance_value
        else:
            delattr(model, "sample_time")


def _antithetic_gaussian_noise(
    model: torch.nn.Module,
    shape: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Sample exact-standard-normal marginals in randomized antithetic pairs."""

    dimensions = tuple(int(value) for value in shape)
    if not dimensions or dimensions[0] <= 0 or dimensions[0] % 2:
        raise WriterModelError(
            "antithetic policy flow noise requires a positive even batch"
        )
    half = torch.normal(
        mean=0.0,
        std=1.0,
        size=(dimensions[0] // 2, *dimensions[1:]),
        dtype=torch.float32,
        device=device,
    )
    paired = torch.cat((half, -half), dim=0)
    permutation = torch.randperm(dimensions[0], device=device)
    return paired.index_select(0, permutation)


def _independent_gaussian_noise(
    model: torch.nn.Module,
    shape: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Sample independent standard-normal PI05 flow noise."""

    del model
    dimensions = tuple(int(value) for value in shape)
    if not dimensions or dimensions[0] <= 0:
        raise WriterModelError("independent policy flow noise requires a batch")
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=dimensions,
        dtype=torch.float32,
        device=device,
    )


@contextmanager
def scoped_policy_flow_noise_sampling(
    policy: torch.nn.Module,
    scheme: str | None,
    *,
    logical_batch_size: int | None = None,
    batch_offset: int = 0,
) -> Iterator[None]:
    """Temporarily replace only PI05 flow-noise sampling."""

    if scheme is None:
        yield
        return
    samplers = {
        ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME: _antithetic_gaussian_noise,
        INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME: _independent_gaussian_noise,
    }
    if scheme not in samplers:
        raise WriterModelError("unsupported policy flow-noise sampling scheme")
    full_sampler = samplers[scheme]
    model = getattr(policy, "model", None)
    if model is None or not callable(getattr(model, "sample_noise", None)):
        raise WriterModelError("policy has no scoped PI05 flow-noise owner")
    had_instance_value = "sample_noise" in vars(model)
    previous_instance_value = vars(model).get("sample_noise")
    if logical_batch_size is None:
        sampler = full_sampler
    else:
        if logical_batch_size <= 0 or not 0 <= batch_offset < logical_batch_size:
            raise WriterModelError("invalid logical flow-noise microbatch")

        def sampler(
            owner: torch.nn.Module,
            shape: Sequence[int],
            device: torch.device | str,
        ) -> torch.Tensor:
            dimensions = tuple(int(value) for value in shape)
            if not dimensions:
                raise WriterModelError("flow-noise microbatch lost batch dimension")
            stop = batch_offset + dimensions[0]
            if dimensions[0] <= 0 or stop > logical_batch_size:
                raise WriterModelError("flow-noise microbatch exceeds logical batch")
            full = full_sampler(
                owner,
                (logical_batch_size, *dimensions[1:]),
                device,
            )
            return full[batch_offset:stop]

    model.sample_noise = MethodType(sampler, model)
    try:
        yield
    finally:
        if had_instance_value:
            model.sample_noise = previous_instance_value
        else:
            delattr(model, "sample_noise")


def prepare_frozen_writer_policy(
    policy: torch.nn.Module,
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    """Inject the sealed identity LoRA and freeze all physical policy state."""

    inject_task_lora(policy, contract)
    template = task_lora_state_dict(policy, clone=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("Writer functional policy must be fully frozen")
    return template


def functional_microbatch_contract(
    batch: Mapping[str, Any],
    policy_microbatch_size: int | None,
    *,
    policy_rng_seed: int | None,
    flow_time_sampling_scheme: str | None,
    flow_noise_sampling_scheme: str | None,
) -> tuple[int, int]:
    batch_sizes = {
        int(value.shape[0])
        for value in batch.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    }
    if len(batch_sizes) != 1:
        raise WriterModelError("functional policy batch has inconsistent leading axes")
    logical_batch_size = batch_sizes.pop()
    microbatch_size = (
        logical_batch_size if policy_microbatch_size is None else policy_microbatch_size
    )
    if not 0 < microbatch_size <= logical_batch_size:
        raise WriterModelError("invalid functional policy microbatch size")
    keyed_sliceable_pairs = {
        (
            LATIN_BETA_TIME_SAMPLING_SCHEME,
            ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        ),
        (
            INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        ),
    }
    if microbatch_size < logical_batch_size and (
        (flow_time_sampling_scheme, flow_noise_sampling_scheme)
        not in keyed_sliceable_pairs
        or policy_rng_seed is None
    ):
        raise WriterModelError(
            "policy microbatching requires keyed sliceable randomness"
        )
    return logical_batch_size, microbatch_size


def _functional_microbatch_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    logical_batch_size: int,
    physical_microbatching: bool,
    policy_rng_seed: int | None,
    policy_rng_device: torch.device | str | None,
    flow_time_sampling_scheme: str | None,
    flow_noise_sampling_scheme: str | None,
    collect_policy_details: bool,
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
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
    random_batch_size = logical_batch_size if physical_microbatching else None
    with scoped_policy_randomness(policy_rng_seed, policy_rng_device):
        with scoped_policy_flow_noise_sampling(
            policy,
            flow_noise_sampling_scheme,
            logical_batch_size=random_batch_size,
            batch_offset=start,
        ):
            with scoped_policy_flow_time_sampling(
                policy,
                flow_time_sampling_scheme,
                logical_batch_size=random_batch_size,
                batch_offset=start,
            ):
                with _scoped_policy_detail_collection(
                    policy,
                    collect_policy_details,
                ):
                    output = functional_lora_call(
                        policy,
                        leaves,
                        contract,
                        microbatch,
                    )
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 0
        or not isinstance(output[1], Mapping)
    ):
        raise WriterModelError("functional policy did not return a scalar loss")
    names = tuple(leaves)
    gradients = torch.autograd.grad(output[0], tuple(leaves[name] for name in names))
    return (
        output[0].detach(),
        output[1],
        {
            name: gradient.detach()
            for name, gradient in zip(names, gradients, strict=True)
        },
    )


def functional_lora_loss_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    policy_rng_seed: int | None = None,
    policy_rng_device: torch.device | str | None = None,
    flow_time_sampling_scheme: str | None = None,
    flow_noise_sampling_scheme: str | None = None,
    policy_microbatch_size: int | None = None,
    collect_policy_details: bool = True,
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    """Differentiate one policy loss only through detached LoRA leaf tensors.

    Backpropagating the returned leaf gradients through the one-video Writer is
    the exact first derivative by the chain rule; no policy parameter is
    trainable or accumulated.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("functional LoRA gradient received a trainable policy")
    logical_batch_size, microbatch_size = functional_microbatch_contract(
        batch,
        policy_microbatch_size,
        policy_rng_seed=policy_rng_seed,
        flow_time_sampling_scheme=flow_time_sampling_scheme,
        flow_noise_sampling_scheme=flow_noise_sampling_scheme,
    )

    if microbatch_size == logical_batch_size:
        return _functional_microbatch_gradient(
            policy,
            state,
            contract,
            batch,
            start=0,
            stop=logical_batch_size,
            logical_batch_size=logical_batch_size,
            physical_microbatching=False,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=policy_rng_device,
            flow_time_sampling_scheme=flow_time_sampling_scheme,
            flow_noise_sampling_scheme=flow_noise_sampling_scheme,
            collect_policy_details=collect_policy_details,
        )

    names = tuple(state)
    gradient_sum = {
        name: torch.zeros_like(
            value,
            dtype=(
                torch.float32
                if value.dtype in {torch.bfloat16, torch.float16}
                else value.dtype
            ),
            memory_format=torch.preserve_format,
        )
        for name, value in state.items()
    }
    loss_sum: torch.Tensor | None = None
    detail_sum: dict[str, Any] = {}
    for start in range(0, logical_batch_size, microbatch_size):
        stop = min(start + microbatch_size, logical_batch_size)
        weight = (stop - start) / logical_batch_size
        loss, details, gradients = _functional_microbatch_gradient(
            policy,
            state,
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
            collect_policy_details=collect_policy_details,
        )
        weighted_loss = loss.to(dtype=torch.float32) * weight
        loss_sum = weighted_loss if loss_sum is None else loss_sum + weighted_loss
        for name in names:
            gradient_sum[name].add_(
                gradients[name].to(dtype=gradient_sum[name].dtype),
                alpha=weight,
            )
        _accumulate_policy_details(detail_sum, details, weight)
    if loss_sum is None:
        raise WriterModelError("functional policy microbatch loop was empty")
    return (
        loss_sum,
        detail_sum,
        {
            name: gradient.to(dtype=state[name].dtype)
            for name, gradient in gradient_sum.items()
        },
    )


def _accumulate_policy_details(
    destination: dict[str, Any],
    source: Mapping[str, Any],
    weight: float,
) -> None:
    """Weight scalar/list PI05 diagnostics across physical microbatches."""

    for name, value in source.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            destination[name] = (
                float(destination.get(name, 0.0)) + float(value) * weight
            )
            continue
        if isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            previous = destination.setdefault(name, [0.0] * len(value))
            if not isinstance(previous, list) or len(previous) != len(value):
                raise WriterModelError("policy microbatch diagnostic shape changed")
            destination[name] = [
                float(left) + float(right) * weight
                for left, right in zip(previous, value, strict=True)
            ]
            continue
        if name in destination and destination[name] != value:
            raise WriterModelError("policy microbatch diagnostic value changed")
        destination[name] = value
