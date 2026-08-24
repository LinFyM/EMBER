"""Target-native banks and the task-local G1 rank-four capacity oracle."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    prepare_policy_effect_prefix_cache,
)


OUTPUT_BANK_TYPES = ("abs", "adj", "init", "goal")
G1_RESIDUAL_RANK = 4
G1_PROBE_COUNT = 2
G1_Q_OUTPUT_GROUPS = 8


class NativeFactorError(RuntimeError):
    """Raised when native capture or factor construction crosses its contract."""


@dataclass(frozen=True)
class NativeTargetChunk:
    """One ordered frame chunk with the probe and horizon axes intact."""

    start_frame: int
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]

    @property
    def frame_count(self) -> int:
        if not self.inputs:
            return 0
        return int(self.inputs[0].shape[0])


@dataclass(frozen=True)
class NativeVideoReadout:
    """Streaming Pass-B inputs for one independently ordered video."""

    frame_count: int
    process: torch.Tensor
    state_posterior: torch.Tensor
    final_outputs: tuple[torch.Tensor, ...]
    chunks: Callable[[], Iterable[NativeTargetChunk]]


@dataclass(frozen=True)
class NativeFactorResidual:
    """Ragged native-space factors for all 38 targets."""

    a: tuple[torch.Tensor, ...]
    b: tuple[torch.Tensor, ...]
    scales: torch.Tensor


class NativeTargetCapture(AbstractContextManager["NativeTargetCapture"]):
    """Scope hooks over the 38 native Linear or identity-wrapper base layers."""

    def __init__(
        self,
        policy: torch.nn.Module,
        owners: Sequence[TargetOwner],
    ) -> None:
        modules = dict(policy.named_modules())
        self.owners = tuple(owners)
        self.modules: list[torch.nn.Linear] = []
        self.capture_modes: list[str] = []
        for owner in self.owners:
            module = modules.get(owner.target_name)
            linear, mode = _resolve_native_linear(module, owner.target_name)
            self.modules.append(linear)
            self.capture_modes.append(mode)
        self.inputs: list[torch.Tensor | None] = [None] * len(self.owners)
        self.outputs: list[torch.Tensor | None] = [None] * len(self.owners)
        self.calls = [0] * len(self.owners)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "NativeTargetCapture":
        for index, module in enumerate(self.modules):

            def capture(
                _module: torch.nn.Module,
                inputs: tuple[object, ...],
                output: object,
                *,
                selected: int = index,
            ) -> None:
                if (
                    len(inputs) != 1
                    or not isinstance(inputs[0], torch.Tensor)
                    or not isinstance(output, torch.Tensor)
                ):
                    raise NativeFactorError("native Linear hook signature changed")
                self.calls[selected] += 1
                if self.calls[selected] != 1:
                    raise NativeFactorError("native target ran more than once")
                self.inputs[selected] = inputs[0].detach()
                self.outputs[selected] = output.detach()

            self.handles.append(module.register_forward_hook(capture))
        return self

    def chunk(
        self,
        *,
        start_frame: int,
        frame_count: int,
        probe_count: int,
    ) -> NativeTargetChunk:
        leading = frame_count * probe_count
        native_inputs: list[torch.Tensor] = []
        native_outputs: list[torch.Tensor] = []
        for owner, calls, x, y in zip(
            self.owners,
            self.calls,
            self.inputs,
            self.outputs,
            strict=True,
        ):
            if calls != 1 or x is None or y is None:
                raise NativeFactorError(
                    f"native target capture is incomplete: {owner.target_name}"
                )
            if (
                x.shape != (leading, ACTION_HORIZON, owner.in_features)
                or y.shape != (leading, ACTION_HORIZON, owner.out_features)
                or x.device != y.device
            ):
                raise NativeFactorError(
                    f"native target shape changed: {owner.target_name}"
                )
            native_inputs.append(
                x.reshape(
                    frame_count,
                    probe_count,
                    ACTION_HORIZON,
                    owner.in_features,
                )
            )
            native_outputs.append(
                y.reshape(
                    frame_count,
                    probe_count,
                    ACTION_HORIZON,
                    owner.out_features,
                )
            )
        return NativeTargetChunk(
            start_frame=start_frame,
            inputs=tuple(native_inputs),
            outputs=tuple(native_outputs),
        )

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _resolve_native_linear(
    module: object, target_name: str
) -> tuple[torch.nn.Linear, str]:
    if isinstance(module, torch.nn.Linear):
        return module, "bare_linear"
    base = getattr(module, "base_layer", None)
    lora_b = getattr(module, "lora_B", None)
    try:
        physical_b = lora_b["default"].weight
    except (KeyError, TypeError, AttributeError) as error:
        raise NativeFactorError(
            f"native target is not bare or identity-wrapped: {target_name}"
        ) from error
    if not isinstance(base, torch.nn.Linear) or torch.count_nonzero(physical_b).item():
        raise NativeFactorError(
            f"native target wrapper has an active physical delta: {target_name}"
        )
    return base, "identity_lora_base_layer"


def native_capture_modes(
    policy: torch.nn.Module, owners: Sequence[TargetOwner]
) -> tuple[str, ...]:
    """Report whether Pass B reads bare targets or identity-wrapper base layers."""

    modules = dict(policy.named_modules())
    return tuple(
        _resolve_native_linear(modules.get(owner.target_name), owner.target_name)[1]
        for owner in owners
    )


def capture_native_target_chunk(
    *,
    policy: torch.nn.Module,
    owners: Sequence[TargetOwner],
    prefix: ExecutionPolicyPrefix,
    fixed_probe: torch.Tensor,
    start_frame: int,
) -> NativeTargetChunk:
    """Capture X/Y once at flow-time one for two fixed antithetic probes."""

    frames = int(prefix.embeddings.shape[0])
    if (
        frames <= 0
        or prefix.padding.shape != prefix.embeddings.shape[:2]
        or fixed_probe.shape != (ACTION_HORIZON, 32)
        or any(parameter.requires_grad for parameter in policy.parameters())
    ):
        raise NativeFactorError("native Pass-B input or frozen-policy wall changed")
    device = prefix.embeddings.device
    repeated_padding = prefix.padding.repeat_interleave(G1_PROBE_COUNT, dim=0)
    noise = torch.stack((fixed_probe, -fixed_probe), dim=0)
    noise = (
        noise[None]
        .expand(frames, -1, -1, -1)
        .reshape(frames * G1_PROBE_COUNT, ACTION_HORIZON, 32)
    )
    cache = prepare_policy_effect_prefix_cache(policy, prefix)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), NativeTargetCapture(policy, owners) as capture, autocast:
        policy.model.denoise_step(
            prefix_pad_masks=repeated_padding,
            past_key_values=cache,
            x_t=noise,
            timestep=torch.ones(frames * G1_PROBE_COUNT, device=device),
        )
    return capture.chunk(
        start_frame=start_frame,
        frame_count=frames,
        probe_count=G1_PROBE_COUNT,
    )


@dataclass
class NativeOutputBankState:
    """Per-video temporal state that survives frame-chunk boundaries."""

    final: torch.Tensor
    first: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    next_frame: int = 0

    def build(self, raw: torch.Tensor, *, start_frame: int) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1:3] != (
            G1_PROBE_COUNT,
            ACTION_HORIZON,
        ):
            raise NativeFactorError("native output chunk lost probe/horizon axes")
        if start_frame != self.next_frame or raw.shape[0] <= 0:
            raise NativeFactorError("native output chunks are not contiguous")
        if self.final.shape != raw.shape[1:]:
            raise NativeFactorError("native video final activation changed shape")
        if self.first is None:
            self.first = raw[0]
        previous = self.first if self.previous is None else self.previous
        adjacent_previous = torch.cat((previous[None], raw[:-1]), dim=0)
        bank = torch.stack(
            (
                raw,
                raw - adjacent_previous,
                raw - self.first,
                self.final - raw,
            ),
            dim=3,
        )
        self.previous = raw[-1]
        self.next_frame += int(raw.shape[0])
        return bank


class OnlineSoftmaxAccumulator:
    """Differentiable online softmax sufficient statistics for two branches."""

    def __init__(self, *, ranks: int, width: int, device: torch.device) -> None:
        self.maximum = torch.full(
            (ranks, 2), -torch.inf, dtype=torch.float32, device=device
        )
        self.normalizer = torch.zeros(ranks, 2, dtype=torch.float32, device=device)
        self.weighted_sum = torch.zeros(
            ranks, 2, width, dtype=torch.float32, device=device
        )

    def add(self, logits: torch.Tensor, values: torch.Tensor) -> None:
        if logits.ndim < 3 or values.ndim != logits.ndim - 1:
            raise NativeFactorError("online signed-pooling ranks changed")
        if (
            logits.shape[:2] != self.maximum.shape
            or logits.shape[2:] != values.shape[:-1]
        ):
            raise NativeFactorError("online logits and native values do not align")
        flat_logits = logits.float().flatten(2)
        flat_values = values.float().reshape(-1, values.shape[-1])
        chunk_maximum = flat_logits.amax(-1)
        maximum = torch.maximum(self.maximum, chunk_maximum)
        old_scale = torch.exp(self.maximum - maximum)
        weights = torch.exp(flat_logits - maximum[..., None])
        self.weighted_sum = self.weighted_sum * old_scale[..., None] + torch.einsum(
            "rbn,nd->rbd", weights, flat_values
        )
        self.normalizer = self.normalizer * old_scale + weights.sum(-1)
        self.maximum = maximum

    def signed_mean(self) -> torch.Tensor:
        if torch.any(self.normalizer <= 0):
            raise NativeFactorError("online signed pooling received no candidates")
        means = self.weighted_sum / self.normalizer[..., None]
        return means[:, 0] - means[:, 1]


def rms_normalize(value: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize native vectors in matrix-RMS units used by frozen s_ref."""

    scale = value.float().square().mean(-1, keepdim=True).sqrt().clamp_min(epsilon)
    return value / scale.to(value.dtype)


def native_output_group_count(owner: TargetOwner) -> int:
    """Partition only native outputs with a proved whole-vector span ceiling."""

    if owner.family is TargetFamily.Q:
        groups = G1_Q_OUTPUT_GROUPS
    elif owner.family is TargetFamily.ACTION_IN:
        # action_in is 32 -> 1024.  One scalar signed measure over its complete
        # Y vector is confined to span(column_space(W), bias), at most 33/1024
        # dimensions.  Contiguous blocks no wider than the native input retain
        # genuine Y values while removing that algebraic ceiling.  This is the
        # minimal full-width partition implied by the actual Linear shape, not
        # a tunable group-count sweep.
        if owner.out_features % owner.in_features:
            raise NativeFactorError(
                f"native action-in output does not preserve input-width blocks: "
                f"{owner.target_name}"
            )
        groups = owner.out_features // owner.in_features
    else:
        groups = 1
    if owner.out_features % groups:
        raise NativeFactorError(
            f"native output width does not preserve value groups: {owner.target_name}"
        )
    return groups


class TaskLocalNativeFactorOracle(torch.nn.Module):
    """Held-task free selection over real native X/Y candidate banks."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        frame_counts: Sequence[int],
        event_slots: int = 8,
        program_width: int = 128,
        initialization_seed: int = 20260824,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.frame_counts = tuple(int(value) for value in frame_counts)
        if not self.owners or not self.frame_counts or min(self.frame_counts) <= 0:
            raise NativeFactorError("G1 free-code topology is empty")
        offsets = [0]
        for count in self.frame_counts:
            offsets.append(offsets[-1] + count)
        self.register_buffer(
            "video_offsets", torch.tensor(offsets, dtype=torch.long), persistent=True
        )
        targets = len(self.owners)
        frames = offsets[-1]
        output_group_counts = tuple(
            native_output_group_count(owner) for owner in self.owners
        )
        output_group_offsets = [0]
        for count in output_group_counts:
            output_group_offsets.append(output_group_offsets[-1] + count)
        self.output_group_slices = tuple(
            slice(start, stop)
            for start, stop in zip(
                output_group_offsets[:-1], output_group_offsets[1:], strict=True
            )
        )
        self.register_buffer(
            "output_group_counts",
            torch.tensor(output_group_counts, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "output_group_offsets",
            torch.tensor(output_group_offsets, dtype=torch.long),
            persistent=True,
        )
        self.rank_queries = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, program_width)
        )
        self.event_logits = torch.nn.Parameter(
            torch.empty(targets, G1_RESIDUAL_RANK, event_slots)
        )
        self.input_logits = torch.nn.Parameter(
            torch.empty(
                targets,
                G1_RESIDUAL_RANK,
                2,
                frames,
                G1_PROBE_COUNT,
                ACTION_HORIZON,
            )
        )
        self.output_logits = torch.nn.Parameter(
            torch.empty(
                output_group_offsets[-1],
                G1_RESIDUAL_RANK,
                2,
                frames,
                G1_PROBE_COUNT,
                ACTION_HORIZON,
                len(OUTPUT_BANK_TYPES),
            )
        )
        self.scale_logits = torch.nn.Parameter(
            torch.full((targets, G1_RESIDUAL_RANK), 0.1)
        )
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        for parameter in (
            self.rank_queries,
            self.event_logits,
            self.input_logits,
            self.output_logits,
        ):
            parameter.data.normal_(mean=0.0, std=0.01, generator=generator)

    def _frame_log_measure(self, video: NativeVideoReadout) -> torch.Tensor:
        if video.process.shape != (
            self.event_logits.shape[-1],
            len(self.owners),
            self.rank_queries.shape[-1],
        ) or video.state_posterior.shape != (
            video.frame_count,
            self.event_logits.shape[-1],
        ):
            raise NativeFactorError("G1 Stage-0 event evidence changed shape")
        query_scores = torch.einsum(
            "rw,ejw->jre", self.rank_queries, video.process.detach()
        ) / math.sqrt(self.rank_queries.shape[-1])
        event_weights = (query_scores + self.event_logits).softmax(-1)
        frame_mass = torch.einsum(
            "jre,te->jrt", event_weights, video.state_posterior.detach()
        )
        return frame_mass.clamp_min(1e-8).log()

    def _pool_video(
        self, video_index: int, video: NativeVideoReadout
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if video.frame_count != self.frame_counts[video_index]:
            raise NativeFactorError("G1 video frame count changed")
        start = int(self.video_offsets[video_index])
        log_measure = self._frame_log_measure(video)
        input_accumulators = [
            OnlineSoftmaxAccumulator(
                ranks=G1_RESIDUAL_RANK,
                width=owner.in_features,
                device=self.rank_queries.device,
            )
            for owner in self.owners
        ]
        output_accumulators = [
            tuple(
                OnlineSoftmaxAccumulator(
                    ranks=G1_RESIDUAL_RANK,
                    width=owner.out_features // native_output_group_count(owner),
                    device=self.rank_queries.device,
                )
                for _ in range(native_output_group_count(owner))
            )
            for owner in self.owners
        ]
        boundaries = [
            NativeOutputBankState(final=value.detach()) for value in video.final_outputs
        ]
        next_frame = 0
        for chunk in video.chunks():
            count = chunk.frame_count
            if (
                chunk.start_frame != next_frame
                or len(chunk.inputs) != len(self.owners)
                or len(chunk.outputs) != len(self.owners)
            ):
                raise NativeFactorError("G1 native video stream changed ownership")
            stop = next_frame + count
            measure = log_measure[:, :, next_frame:stop]
            for target, (
                x,
                y,
                input_accumulator,
                output_accumulator,
                boundary,
            ) in enumerate(
                zip(
                    chunk.inputs,
                    chunk.outputs,
                    input_accumulators,
                    output_accumulators,
                    boundaries,
                    strict=True,
                )
            ):
                input_logits = self.input_logits[
                    target, :, :, start + next_frame : start + stop
                ]
                input_logits = input_logits + measure[target, :, None, :, None, None]
                input_accumulator.add(input_logits, x)
                output_bank = boundary.build(y, start_frame=next_frame)
                output_slice = self.output_group_slices[target]
                output_logits = self.output_logits[
                    output_slice, :, :, start + next_frame : start + stop
                ]
                output_logits = (
                    output_logits
                    + measure[target, None, :, None, :, None, None, None]
                )
                groups = len(output_accumulator)
                # The candidate index is unchanged.  Grouping only restores a
                # proved native structure: q's [8 heads, 256 channels], or
                # action-in's minimal input-width output blocks.  Values remain
                # slices of the same real Y candidate; no candidate is copied.
                grouped_bank = output_bank.reshape(
                    *output_bank.shape[:-1], groups, output_bank.shape[-1] // groups
                ).movedim(-2, 0)
                for group, accumulator in enumerate(output_accumulator):
                    accumulator.add(output_logits[group], grouped_bank[group])
            next_frame = stop
        if next_frame != video.frame_count or any(
            boundary.next_frame != video.frame_count for boundary in boundaries
        ):
            raise NativeFactorError("G1 native video stream ended early")
        return (
            tuple(value.signed_mean() for value in input_accumulators),
            tuple(
                torch.cat(
                    tuple(group.signed_mean() for group in target), dim=-1
                )
                for target in output_accumulators
            ),
        )

    def forward(
        self,
        videos: Sequence[NativeVideoReadout],
        *,
        s_ref: torch.Tensor,
    ) -> NativeFactorResidual:
        if len(videos) != len(self.frame_counts) or s_ref.shape != (len(self.owners),):
            raise NativeFactorError("G1 video set or frozen target scales changed")
        pooled = [self._pool_video(index, video) for index, video in enumerate(videos)]
        beta = 1.0 / len(videos)
        a_values: list[torch.Tensor] = []
        b_values: list[torch.Tensor] = []
        scales = s_ref[:, None].to(self.scale_logits) * torch.tanh(self.scale_logits)
        for target in range(len(self.owners)):
            a = sum(value[0][target] * beta for value in pooled)
            b = sum(value[1][target] * beta for value in pooled)
            a_values.append(rms_normalize(a))
            b_values.append(rms_normalize(b) * scales[target, :, None])
        return NativeFactorResidual(
            a=tuple(a_values),
            b=tuple(b_values),
            scales=scales,
        )
